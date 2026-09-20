# Model partitioning, tracing and verification

Takes a model specification — config, inference code, weights — plus sample
inputs, and runs a specialized loop that partitions the model into modules each
runnable on one local GPU, traces real per-module IO, verifies every module
replays from its dumped artifacts, then emulates end-to-end inference and has the
sampled tokens judged.

The artifacts are the deliverable: per-module code, input feature maps, weights
and output feature maps in a portable binary format, ready to replay against a
Trainium kernel.

```bash
autohelix partition list                      # bundled model specs
autohelix partition inspect qwen3.8-27b       # structure and projected size, no work done
autohelix partition plan qwen3.8-27b          # produce the partition plan and stop
autohelix partition run qwen3.8-27b           # the full loop
autohelix partition report <run-dir>          # stage status
autohelix partition tokens <run-dir>          # sampled continuations
autohelix partition replay <run-dir> layers.0-2   # replay one module, no checkpoint needed
```

## The loop

Seven stages. Each declares a hash of its inputs, so a stage re-runs only when
something it depends on changed.

| Stage | What it does | Who does it |
|---|---|---|
| `ingest` | Resolve the spec, pin a revision, read the tensor inventory, load the tokenizer and inputs, check disk | script |
| `plan` | Group the stack into runnable modules; reconcile submodule names against the real module tree | script, agent on failure |
| `extract` | Write one implementation per structural signature: inference code, a verifier, and the real source | script |
| `trace` | Hook a real forward; dump per-module inputs, weights and outputs | script |
| `verify_modules` | Replay each module from its dumps and compare against the trace | script |
| `emulate` | Assemble the model from dumps only, generate tokens, judge them | script + LLM judge |
| `retain` | Prune tensors to a representative layer set, keeping deduplicated code | script |

The loop exits as soon as every module verifies and every sample's continuation
is judged sound, printing the sampled tokens for human review.

### Why a separate loop

AutoHelix's harness isolates each iteration in a git worktree and discards it on
rejection. That is exactly wrong here: a rejected iteration must not force a
re-trace of hundreds of gigabytes. So this loop keeps worktree-style isolation
for *code* — the plan is snapshotted and rolled back if the agent proposes
something invalid — while tensor artifacts live outside the repo and survive.
Invalidation is explicit and ordered: touching the plan invalidates everything
downstream of it and nothing upstream.

It reuses AutoHelix's Claude backend (`ClaudeCodeAgent`) for the agent stages, so
transport, flags and Bedrock routing are identical to the rest of the project.

## Key design decisions

**Hooks capture arguments, not just hidden states.** A module is verified by
calling it with the arguments it actually received during a real forward. That is
what keeps extraction and verification architecture-agnostic — rotary embeddings,
attention masks, sliding-window state and per-architecture extras need no special
handling. Tracing runs with caching disabled, because a cache object cannot be
serialized or rebuilt and would make every layer unreplayable.

**The partition graph is a DAG, not a chain.** DeepSeek V4.1 shares KV across
layers (`kv_source_layer_ids`) and feeds a sparse-attention indexer from
non-adjacent layers (`index_source_layer_ids`). A linear pipeline cannot express
that dataflow, so modules are nodes and named tensors are edges.

**Groups are signature-homogeneous.** Modules that share a structural signature
share one extracted implementation. A 64-layer hybrid stack such as Qwen3.8-27B
(linear attention 3:1 with full attention) resolves to 35 modules but only **5
implementations**, and no group ever mixes two kernel variants.

**Verification proves the dumps are sufficient.** Parameters the plan owns are
NaN-poisoned before each module's dumped weights are applied, so a forgotten
tensor fails loudly instead of quietly reusing whatever was in memory. Poisoning
is limited to plan-owned tensors: non-persistent buffers like rotary `inv_freq`
are computed at init and no dump can restore them.

**Use the GPU as far as it goes.** Module verification always runs on the GPU:
the plan sizes every module to fit, so each is moved onto the accelerator, replayed
and moved back — the whole model never needs to be resident. Tracing and
emulation do need the whole model, so a checkpoint larger than the GPU is spread
across GPU and host by layer placement rather than abandoning the GPU. On one
L40S, 33 of Qwen3.8-27B's 51.7 GiB sit on the card.

**A module replays without the checkpoint.** Structure comes from config and
code, weights and inputs from the dumps. So a module of a model far too large to
hold locally stays replayable — which is also what makes the artifacts useful to
hand to kernel development.

**Every module ships with its own verifier.** `extract` writes two runnable files
per implementation group: `inference.py`, which runs the module on its dumped
input feature map and dumped weights, and `verify.py`, which checks that output
against the dumped reference and exits non-zero when it disagrees. That pairing is
the handoff artifact — point `verify.py` at the same run directory after swapping
in a Trainium implementation and it tells you whether the port still reproduces
the reference.

```bash
cd <run-dir>/modules/04-decoder_layers-a5e57f50
python inference.py --module layers.0-3          # run it
python verify.py --all-modules --all-samples     # gate it
```

## Layout

```
partition/
├── config/
│   ├── defaults.yaml            # loop defaults, all overridable on the CLI
│   └── models/*.yaml            # model specs
├── inputs/
│   ├── short.jsonl              # complete instructions, 120-142 tokens (committed)
│   └── fetch_long_inputs.py     # generates long.jsonl at 2k/8k/16k
└── model_partition/
    ├── spec.py                  # what to partition and how to load it
    ├── ingest.py                # metadata first, weights on demand
    ├── weights_index.py         # per-tensor inventory without reading weights
    ├── sizing.py                # structural view + memory cost model
    ├── hardware.py storage.py   # GPU budget; footprint estimate and preflight
    ├── planner/                 # graph, seed planner, agent planner, reconcile
    ├── loaders/                 # vendor code preferred, transformers fallback
    ├── trace.py tensorstore.py  # IO capture; binary dump format
    ├── runtime/                 # module replay, assembly from dumps, standalone
    ├── verify/                  # numerics, module checks, emulation, judge
    ├── retention.py             # post-loop pruning
    └── loop/                    # the specialized loop
```

Artifacts live **outside** the repo, at `~/transonic_artifacts/<slug>/` by
default (`MODEL_PARTITION_ARTIFACTS` or `--artifact-root`):

```
run.yaml              resolved spec, pinned revision, storage estimate
plan/                 partition_graph.yaml, valid_submodules.txt, history/
trace/                manifest.yaml, records.yaml, weights/, activations/
modules/              one directory per implementation group:
                        inference.py  runs the module from its dumped input + weights
                        verify.py     checks inference.py against the dumped output
                        source.py     the real implementation's source, for reference
                        meta.yaml     module ids, layers, shapes
reports/              verify.json, emulate.json, summary.md, tokens.txt
```

## Binary format

Raw little-endian `.bin` per tensor, a JSON sidecar, and a `manifest.yaml` index.
Chosen over safetensors for portability: a contiguous blob plus explicit
dtype/shape metadata is readable from any toolchain a Trainium bring-up needs.
Every blob carries a sha256, and identical blobs are hardlinked, which is what
makes tied embeddings and repeated layers cheap.

dtypes without a numpy equivalent (bfloat16, fp8, packed fp4) round-trip as raw
bytes; `TensorStore.read_torch` reinterprets them.

## Long context and storage

Feature maps dominate the footprint at long context: one 16k boundary tensor for
Qwen3.8-27B is 168 MB. Short samples are dumped whole; long ones keep a
deterministic head/tail window (128 + 128 positions), which preserves the
prefix/suffix behaviour kernel bring-up needs — RoPE phase, sliding-window edges
— at a fraction of the bytes. `--full-dumps` overrides this.

`autohelix partition run` prints a storage estimate before doing any work and
refuses to start if the projection exceeds free disk, so a run fails on
arithmetic rather than ENOSPC mid-trace. `--no-cache-weights` and
`--no-cache-dequant` trade disk for recomputation; `--allow-overflow` downgrades
the refusal to a warning.

## Retention

After the loop passes, artifacts are pruned to what a kernel developer needs:
deduplicated per-module code, plus tensors for a representative layer set —
first, 1, 5, middle, last, **and a guaranteed representative per distinct
signature**. That last term matters: index-based selection alone would discard
the only copy of a kernel variant in a hybrid stack. Disable with `--no-retain`.

## Model specs

| Spec | Size | Fits one L40S? | Purpose |
|---|---|---|---|
| `qwen3.5-0.8b` | 1.6 GiB | yes | smoke test; hybrid attention, MTP, tied embeddings |
| `qwen3.8-27b` | 51.7 GiB | **no** | primary target; 64 layers, 2 signatures, hidden 5120 |
| `qwen3.5-35b-a3b` | 71.9 GiB | no | MoE rehearsal: expert groups and routing capture |
| `deepseek-v4-flash` | ~291 GB | no | fp8 + fp4 rehearsal, text-only, vendor inference code |
| `deepseek-v4.1-flash` | ~765 GB | no | the deployment target — **disabled**, see below |

A spec may name a HuggingFace repo, a local directory, or nothing but a repo id
on the command line. With `loader: auto` a repo that ships its own inference code
gets `repo_code` and its entry module is detected; otherwise `transformers`.
Vendor code is preferred because it defines the model's numerics and may cover
architectures transformers does not know — importing it requires
`trust_remote_code: true`.

### DeepSeek V4.1 Flash

Present as a specification, `enabled: false`. Blockers, in order:

1. **Storage.** ~765 GB on disk (fp8_e4m3 204 GB + int8-packed fp4 experts
   557 GB + bf16 2 GB) against 465 GB local. The shard stream-and-evict path
   (`ensure_shards` / `evict_shards`) exists but is not yet wired into tracing.
2. **Quantization.** fp8 blockwise (32×32, ue8m0 scales) and fp4 expert unpacking
   must be implemented to run anything on an L40S.
3. **Architecture.** 40 layers + 3 MTP, MLA-style attention, 384 routed fp4
   experts + 1 shared at top-6, a second 128-expert MoE, sparse attention with
   cross-layer KV reuse and a lightning indexer, engram n-gram memory, a
   32-layer vision tower. The cross-layer edges are why the graph is a DAG.
4. `encoding/tests/test_input_*.json` → `test_output_*.txt` in the repo give free
   tokenizer ground truth and are worth checking first.

## Testing

```bash
source .venv/bin/activate
pytest tests/ -k partition          # fast suite, no network, no GPU required
pytest tests/ -m slow               # adds metadata-only Hub queries
```

Fast tests run against a real toy causal LM (`tests/fixtures/tiny_llm.py`) written
out as a local repo with vendor inference code — real safetensors, real forwards,
optional MoE on odd layers to give two layer signatures. One test executes a
generated module harness as a subprocess to confirm it actually replays.

## Known limitations

- Emulation re-runs the full prefill for each generated token rather than using a
  KV cache. Quadratic, but it needs nothing from the model beyond
  `model(input_ids) -> logits`, which is what keeps it architecture-agnostic.
- A plan that splits *within* a layer produces modules with no corresponding
  submodule (an expert-combine step, for instance). Those cannot be hooked, so
  they trace and verify but are not emulatable; the seed planner only splits
  within a layer when a single layer exceeds the budget.
- Decode-step tracing is accounted for in the storage estimate but not yet
  captured; prefill IO is.
- Tracing and emulation need the whole model resident, so a checkpoint larger
  than the GPU is spread across GPU and host by layer placement. Most of the
  compute still lands on the GPU, but the host-resident layers are slow. Full
  GPU residency for such models needs sequential module streaming.
