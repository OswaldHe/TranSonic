# Model partitioning, tracing and verification

`autohelix partition` takes a model specification — config, inference code,
weights — plus sample inputs, and runs a specialized loop that cuts the model into
modules each runnable on one local GPU, traces real per-module IO, verifies every
module replays from its dumped artifacts, then emulates end-to-end inference and
has the sampled tokens judged.

The artifacts are the deliverable: per-module code, input feature maps, weights and
output feature maps in a portable binary format, plus a verifier for each module,
ready to gate a port to another backend.

```bash
autohelix partition list                    # bundled model specs
autohelix partition inspect <spec>          # structure and projected size, no work done
autohelix partition plan <spec>             # produce the partition plan and stop
autohelix partition run <spec>              # the full loop
autohelix partition report <run-dir>        # stage status
autohelix partition tokens <run-dir>        # sampled continuations
autohelix partition replay <run-dir> <mod>  # run one module's implementation alone
```

A spec is a bundled name, a path to a YAML file, a HuggingFace repo id, or a local
directory. See [`config/models/README.md`](config/models/README.md) for the
bundled ones and how to write another.

## The loop

Eight stages. Each declares a hash of its inputs, so a stage re-runs only when
something it depends on changed.

| Stage | What it does | Who does it |
|---|---|---|
| `ingest` | Resolve the spec, pin a revision, read the tensor inventory, load the tokenizer and inputs, check disk | script |
| `plan` | Group the stack into runnable modules; reconcile submodule names against the real module tree | script, agent to refine or repair |
| `extract` | Write one implementation per structural signature: inference code, a verifier, a README, and the real source | script |
| `trace` | Hook a real forward; dump per-module inputs, weights and outputs | script |
| `verify_modules` | Run each module's extracted implementation from its dumps and compare against the trace | script |
| `verify_chain` | Chain the implementations, carrying each output into the next, and measure how the error compounds | script |
| `emulate` | Assemble the model from dumps, install the implementations, generate tokens, judge them | script + LLM judge |
| `retain` | Prune tensors to a representative layer set, keeping deduplicated code | script |

The loop exits as soon as every module verifies and every sample's continuation
scores 4 or 5. `--iterations/-n` caps how many iterations it may take (default 5).

A judge that stays unconvinced never fails the run: the partition either reproduced
the model or it did not, and that is decided by the module checks and boundary
comparisons, not by an opinion about text. An unconvinced judge keeps the loop
iterating on the module implementations while iterations remain, then the run
concludes as passed with the reservation reported and the tokens printed for a human
to settle.

### Review, then change

When a stage fails, a **reviewer** agent runs first. It reads the reports and
writes a diagnosis to `reports/review.md`, changing nothing else. Only then does the
agent that makes the change run, with that review as its brief — the same split
AutoHelix's own loop uses, and for the same reason: the agent that has to produce a
fix is the wrong one to decide what went wrong. Each review is archived under
`reports/reviews/iter-N.md`.

### What the loop owns, and what the harness owns

The split is deliberate. The loop iterates on the two things that are genuinely
judgement calls:

- **`plan/partition_graph.yaml`** — where module boundaries fall: whether each
  module is convenient to write a kernel for and loadable on the device for
  tracing.
- **`modules/<group>/inference.py`** — each module's inference code, which is what
  gets fixed when a module does not reproduce its reference numerically.

Everything that decides *whether* a module is correct belongs to the harness: the
tolerances, the comparisons, the boundary checks, the verifier scripts, and the
dumped artifacts they compare against. An agent that could relax its own acceptance
test would make the loop worthless, so those files are snapshotted around every
agent call and restored afterwards; a reference tensor that changed fails the
iteration outright, since it cannot be put back.

| Failure | Editable surface the agent is pointed at |
|---|---|
| a plan submodule does not exist, or a module will not load | `plan/partition_graph.yaml` |
| a module's output does not match its reference | `modules/<group>/inference.py` |
| a module will not fit the device the plan sized it for | `plan/partition_graph.yaml` |
| drift through the chained implementations moves a token | `modules/<group>/inference.py` |
| the judge is unconvinced by the generated text | `modules/<group>/inference.py` (advisory) |

### Why a separate loop

AutoHelix's harness isolates each iteration in a git worktree and discards it on
rejection. That is exactly wrong here: a rejected iteration must not force a
re-trace of hundreds of gigabytes. So this loop keeps worktree-style isolation for
*code* — the plan and the implementations are snapshotted and rolled back if the
agent proposes something invalid — while tensor artifacts live outside the repo and
survive. Invalidation is explicit and ordered: touching the plan invalidates
everything downstream of it and nothing upstream.

It reuses AutoHelix's Claude backend (`ClaudeCodeAgent`) for the agent stages, so
transport, flags and Bedrock routing are identical to the rest of the project.

## Saying how you want the model partitioned

Two ways, and they compose. A spec can carry the instruction, so it travels with the
model:

```yaml
partition:
  split_attention_ffn: true
  prompt: |
    Partition every decoder layer so that attention and the FFN/MoE block are
    separate modules...
```

- `prompt` is free text handed to the agent whenever it plans or repairs. Supplying
  one implies plan refinement, since otherwise nothing would act on it.
- `split_attention_ffn` is the deterministic form of the most common instruction.
  The seed planner acts on it directly, so that partition does not depend on an
  agent call succeeding.

Both have command-line equivalents, which win over the spec:
`--partition-prompt TEXT`, `--partition-prompt-file PATH`, `--split-attention-ffn`.

### Finding the granularity

The planner prompt states the trade-off, and it cuts both ways. Too coarse and a
module bundles several distinct computations, so a numerical mismatch has many
possible causes and no way to isolate them. Too fine and every boundary becomes a
materialized tensor, which forecloses the fusions that make a kernel fast — an
epilogue folded into a GEMM, a norm fused into what follows it. The target is the
coarsest module that still corresponds to one thing a kernel author would sit down
and write.

## Key design decisions

**Hooks capture arguments, not just hidden states.** A module is verified by calling
it with the arguments it actually received during a real forward. That is what keeps
extraction and verification architecture-agnostic — rotary embeddings, attention
masks, sliding-window state and per-architecture extras need no special handling.
Tracing runs with caching disabled, because a cache object cannot be serialized or
rebuilt and would make every layer unreplayable.

**The partition graph is a DAG, not a chain.** Architectures route tensors across
non-adjacent layers: shared KV, sparse-attention indexers fed from earlier layers.
A linear pipeline cannot express that dataflow, so modules are nodes and named
tensors are edges.

**Groups are signature-homogeneous.** Modules that share a structural signature
share one extracted implementation, so a hybrid stack that alternates two attention
variants yields two decoder implementations rather than one per layer — and no group
ever mixes two kernel variants.

**A module says how its submodules relate.** `composition: sequential` chains them,
which is what a stack of like decoder layers is. `composition: parallel` marks
alternatives — a group of MoE experts, where each sees the tokens routed to it and
none feeds the next. The two are verified differently: a sequential group is one
computation from its first submodule's input to its last submodule's output, while a
parallel group is checked call by call.

**Submodule order comes from the trace, not from the plan.** A norm and the
projection it feeds have to run in the model's order, and no ordering of names can
say which that is, so a group's submodules are ordered by when the trace actually
observed them.

**Three checks, because one question is really three.** Per-module verification
starts every module from its *recorded* input, so an error in one module cannot show
up in another — the checks are independent by construction, which is what lets them
hold a near-exact bar. That says nothing about what happens when the modules are
chained, so `verify_chain` feeds each computed output into the next and reports the
drift as a curve, the first boundary outside tolerance, and whether the logits still
predict the same tokens. Emulation then generates through those same implementations
at a sequence length no recording covers.

An edge is only carried while the module that produced it is still reproducing its
reference. A plan that splits a layer puts the residual add between two modules and
inside neither, so that edge is not real dataflow and the chain says so instead of
blaming an implementation for it. Once a module *has* drifted, the same mismatch means
the opposite thing and the value is carried regardless — watching the error travel is
the point.

**Two different numeric bars, for two different questions.** Replaying one module
against its own recorded input is a single step, so it is held to a near-exact
elementwise bar. A module *boundary* seen during end-to-end emulation is the product
of every layer before it, where bf16 rounding accumulates: deep in a 64-layer stack
a measured-good boundary had only 64% of elements within 2e-2 yet cosine 0.9997,
while a genuine wiring error scored cosine 0.707. So boundaries are gated on cosine,
with the pass fraction kept as a coarse floor and still reported.

**Every tensor a module returns is compared.** Attention modules commonly return
several values; checking only the first would pass a module whose primary output is
right and whose auxiliary output is corrupt.

**Verification proves the dumps are sufficient.** Parameters the plan owns are
NaN-poisoned before each module's recorded weights are applied, so a forgotten
tensor fails loudly instead of quietly reusing whatever was in memory. Poisoning is
limited to plan-owned tensors: non-persistent buffers like rotary `inv_freq` are
computed at init and no dump can restore them.

**Unverifiable is reported, never assumed.** A *functional* module — one with no
submodule of its own, such as an expert-combine step — belongs in the plan because a
kernel has to implement it, but a hook-based trace has nothing to attach to. Those
are reported as skipped rather than counted as passing, and the same goes for
windowed long-context records.

**Use the GPU as far as it goes.** Module verification always runs on the GPU: the
plan sizes every module to fit, so each is moved onto the accelerator, replayed and
moved back — the whole model never needs to be resident. Tracing and emulation do
need the whole model, so a checkpoint larger than the GPU is spread across GPU and
host by layer placement rather than abandoning the GPU.

**A module replays without the checkpoint, and without the model.** The code comes
from the module directory, the weights and inputs from the dumps. So a module of a
model far too large to hold locally stays replayable — which is what makes the
artifacts useful to hand to kernel development.

**What gets checked is what gets shipped.** The implementations are the deliverable,
so they are what runs: `verify_modules`, `verify_chain` and `autohelix partition
replay` execute them, and emulation replaces every partitioned submodule with its
implementation before generating, so the tokens printed for human review are the
shipped code's tokens. The model's own modules appear in exactly one place — `trace`,
where they *are* the reference every later stage is measured against.

**A module directory is a unit you can optimize on its own.** `extract` writes one
directory per implementation group, and the important thing about it is what depends on
what:

- **`source.py` is the implementation** — this module's classes and the helpers they
  reference, taken verbatim out of the file they are defined in. Not the whole file:
  a modeling file is the whole model, and a module directory is one module, so the
  other layer variants and the vision tower are left behind. This is what you edit to
  optimize the module.
- **`config.json`** is the config the module's own subtree was constructed with, which
  for a multimodal checkpoint is the text stack's config rather than the model's. It
  is recorded because building a layer from the wrong one gives library defaults for
  every width it does not name, and because it is what makes the directory
  self-contained.
- **`inference.py` is the launcher, and the benchmark.** It imports `source.py`,
  constructs the class from the recorded config, loads the dumped weights into it, and
  returns the module. Run directly it times the module and compares it against the
  dumped reference, printing `##autohelix[latency_ms=…]`, `##autohelix[cosine=…]`,
  `##autohelix[max_abs_err=…]`, `##autohelix[max_rel_err=…]` and
  `##autohelix[passed=…]` — so a module directory is something `autohelix run` can
  optimize as it stands, and a faster module that stopped matching exits non-zero
  instead of scoring well. It reads no checkpoint and instantiates no model, so
  nothing outside the directory has to be present.
- **`verify.py` is the gate.** Same check, every module of the group and every
  traced sample, at a tolerance the agent cannot loosen.
- **`README.md`** states what the module does, its pre-conditions and its
  post-conditions.

`source.py` and `inference.py` are written once and then left alone, so work done on
them survives the next run of the loop. They are also what `verify_modules`,
`verify_chain` and `emulate` run, so optimizing one module and re-running the loop
tells you whether it is still correct — on its own, and chained with the others.

```bash
cd <run-dir>/modules/<group>
cat README.md                                    # what it does, what it needs
python inference.py --device cuda --repeat 50    # time it, check it, report metrics
python verify.py --all-modules --all-samples     # gate it
```

## Layout

```
partition/
├── config/
│   ├── defaults.yaml            # loop defaults, all overridable on the CLI
│   └── models/                  # model specs, and a README on writing one
├── inputs/                      # sample input sets, and a README on their format
└── model_partition/
    ├── spec.py                  # what to partition and how to load it
    ├── ingest.py                # metadata first, weights on demand
    ├── weights_index.py         # per-tensor inventory without reading weights
    ├── weights_source.py        # reading a module's weights from the checkpoint
    ├── sizing.py                # structural view + memory cost model
    ├── hardware.py storage.py   # GPU budget; footprint estimate and preflight
    ├── planner/                 # graph, seed planner, agent planner, reconcile
    ├── loaders/                 # vendor code preferred, transformers fallback
    ├── trace.py tensorstore.py  # IO capture; binary dump format
    ├── runtime/                 # launching source.py, assembly, standalone replay
    ├── verify/                  # numerics, module checks, emulation, judge
    ├── retention.py             # post-loop pruning
    └── loop/                    # the specialized loop, and the harness guard
```

Artifacts live **outside** the repo, at `~/transonic_artifacts/<slug>/` by default
(`MODEL_PARTITION_ARTIFACTS` or `--artifact-root`):

```
run.yaml              resolved spec, pinned revision, model config, storage estimate
plan/                 partition_graph.yaml, valid_submodules.txt, rationale.md, history/
trace/                manifest.yaml, records.yaml, weights/, activations/
modules/              one directory per implementation group:
                        source.py     the implementation: this module's classes, verbatim
                        inference.py  launches source.py; run it for latency + error
                        verify.py     checks the implementation against the dumped output
                        config.json   the config this module's subtree was built from
                        README.md     what it does, pre-conditions, post-conditions
                        meta.yaml     module ids, layers, shapes
reports/              verify.json, chain.json, emulate.json, summary.md, tokens.txt,
                      review.md, reviews/
```

## Binary format

Raw little-endian `.bin` per tensor, a JSON sidecar, and a `manifest.yaml` index.
Chosen over safetensors for portability: a contiguous blob plus explicit dtype/shape
metadata is readable from whatever toolchain a kernel bring-up uses. Every blob
carries a sha256, and identical blobs are hardlinked, which is what makes tied
embeddings and repeated layers cheap.

dtypes without a numpy equivalent (bfloat16, fp8, packed fp4) round-trip as raw
bytes; `TensorStore.read_torch` reinterprets them.

## Storage

`autohelix partition run` prints a storage estimate before doing any work and
refuses to start if the projection exceeds free disk, so a run fails on arithmetic
rather than ENOSPC mid-trace. `--allow-overflow` downgrades the refusal to a
warning. A re-trace clears the previous one first, so iterations do not accumulate.

Feature maps dominate the footprint at long context, and there are two ways to trade
that down:

- `--no-cache-weights` keeps only an index of which parameters each module owns and
  reads their values from the checkpoint on demand. Verification still runs against
  real weights; what it gives up is replaying a module on a machine the checkpoint
  never reaches.
- `--slice-long` windows long-sample activations to a head and tail (128 + 128
  positions). This costs verification coverage and is off by default: attention
  mixes every position, so a windowed output is not a function of the windowed
  input. Windowed records are kept for inspection and reported as skipped.

## Retention

After the loop passes, artifacts are pruned to what a kernel developer needs:
deduplicated per-module code, plus tensors for a representative layer set — first,
1, 5, middle, last, **and a guaranteed representative per distinct signature**. That
last term matters: index-based selection alone would discard the only copy of a
kernel variant in a hybrid stack. Disable with `--no-retain`.

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
- A functional module (an expert-combine step, for instance) has no traced
  reference, so it is reported as unverified rather than checked.
- A dense FFN that exceeds the per-module budget is left as one module and reported;
  splitting a single matmul chain apart is the operator's call or the agent's.
- Parameters a layer module owns directly, rather than through a child module, are
  not assigned to any partition module. They appear in the fill report as unclaimed.
- Decode-step tracing is accounted for in the storage estimate but not yet
  captured; prefill IO is.
- Tracing and emulation need the whole model resident, so a checkpoint larger than
  the GPU is spread across GPU and host by layer placement. Most of the compute
  still lands on the GPU, but the host-resident layers are slow. Full GPU residency
  for such models needs sequential module streaming.
- Because tracing and emulation may place layers differently, their bf16 results
  diverge slightly with depth. That is why accumulated boundaries are gated on
  cosine rather than an elementwise pass fraction.
