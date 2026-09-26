# Making one module fast on a single device

`autohelix optimize` takes the two things the other loops produce — a correct-but-slow NKI kernel per
module from `bootstrap`, and a decision about what runs where from `floorplan` — and closes the gap
between them. It cuts the module down to what one NeuronCore runs, makes that fast, then puts the
ranks back together with a collective and makes *that* fast.

```bash
autohelix optimize template > optimization.yaml   # then fill in the <FILL IN>s
autohelix optimize check                          # validate the config, show the projection
autohelix optimize all                            # all five stages
```

## Five stages, and why they are five

```
    init            submodule            run              assemble          run-full
 ┌──────────┐    ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
 │ project  │    │ agent cuts  │   │ N iters on  │   │ agent joins │   │ 5 iters on  │
 │ the      │───►│ the module  │──►│ one rank,   │──►│ N ranks with│──►│ the whole   │
 │ placement│    │ to one rank │   │ under the   │   │ nki.collect-│   │ module      │
 │ onto one │    │             │   │ constraint  │   │ ives        │   │             │
 │ device   │    │             │   │ schedule    │   │             │   │             │
 └──────────┘    └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘
      │                 │                 │                 │                 │
  projection.py   submodule_checker  loop.py +        module_checker     loop.py
  (a rule, not    (7 generic         constraints.py   (9 checks:        (same gate,
   a search)       checks)           (per-iteration    the semantic      no schedule)
                                      slots)           gate)
```

The seams are the design.

**`init` is a rule, not a search.** The ranked floorplan is written for a 16-device trn2.48xlarge and
development happens on a one-device trn2.3xlarge, so almost nothing fits. In `schemes/rank1.yaml`,
of 271 placements: 182 fit one device (every hyper-connection module, the norms, `embed`, the small
MTP heads), **87 span two devices** (all 43 `.ffn` at `expert x8`, all 43 `.attention`, `lm_head`),
and 2 span four (both Engram tables). So every module worth optimizing is oversized, and `init`
narrows the split to what one device holds — `expert x8` becomes `expert x4` — and records what that
cost.

**`submodule` is an agent, not a script.** How to divide a module into per-rank work is a judgement
about what the module computes: which tensors are replicated, what a partial result is, whether a
term belongs inside a rank's share or is added after the ranks rejoin. A script that knew how to
shard an MoE would work for MoE and nothing else. So the agent decides, and the loose end is tied
off at the far end of the pipeline rather than here.

**`run` and `run-full` are the same loop.** Ordinary `autohelix run` semantics — green baseline, a
metric, a rejected iteration discarded — plus one thing: a per-iteration constraint.

**`assemble` is where the looseness gets paid for.** Everything before it is judged against goldens an
agent chose. Here the target is the bootstrapped module's own recorded output, at the bootstrapped
module's own bar, faster than the bootstrapped module, and within 10% of the submodule. A wrong cut
cannot pass. A submodule that was fast because it did a quarter of the work cannot pass. That is what
makes the looseness upstream safe.

## Where the gate is, and where it is not

| | checked by a script | left to an agent |
|---|---|---|
| the projection | **yes** — `projection.py`, deterministic | — |
| how the module is cut | no | **the stage-2 agent** |
| that the cut *closes* | **yes** — `recipe.py`, arithmetic | — |
| whether the cut is *good* | **yes** — the +10% bound, at stage 4 | — |
| the submodule repo's integrity | **yes** — 7 module-agnostic checks | — |
| the per-iteration constraint | **yes** — a compiled checker per slot | the constraint compiler writes it |
| the whole module's correctness | **yes** — 9 checks, the semantic gate | — |

The row worth dwelling on is the third. A wrong cut cannot be caught by reading `source.py`, but it
*can* be caught arithmetically. The stage-2 agent has to declare how its ranks recombine — stage 4
needs that anyway — and a declaration of that form is checkable: take the N per-rank goldens it
dumped, apply the declared recipe, and see whether the result is the module's recorded output. A cut
that drops an expert, double-counts a shared path, shards along the wrong axis or dumps a golden from
the wrong rank fails, because none of those sum back to the recorded output. Host-side numpy over
bytes already on disk, a second to run, and entirely ignorant of what the module computes.

What it does *not* prove: a recipe declaring one rank does everything and three return zeros
reproduces the golden perfectly. That cut fails at stage 4, on the +10% bound, because three idle
ranks cannot make the whole module fast. The two checks divide the work on purpose — conflating them
would make the arithmetic one reject legitimate asymmetric cuts.

## The per-iteration constraint schedule

The novel part. Ten iterations of "do whatever you like" converge on whatever the first iteration
happened to try, so the iterations are given different constraints:

| iterations | constraint |
|---|---|
| 1-3 | NKI only, no torch — prove the fast path is reachable in NKI at all |
| 4-6 | NKI and torch-xla both allowed |
| 7-8 | nothing; explore aggressively |
| 9-10 | NKI and torch-xla — consolidate what 7-8 learned |

The constraints are prose, because the useful ones are prose. Turning prose into a predicate is a job,
and it goes to a **constraint compiler**: an agent that runs once, before iteration 1, and writes one
checker script per slot. The optimizing agent gets the prose in its prompt and never sees the script —
the same asymmetry `bootstrap` and `floorplan` use, for the same reason. A constraint whose
implementation is readable gets read for loopholes instead of followed.

Four properties, each a choice:

- **The compiler runs once, up front.** A checker written at the top of iteration 7 could be written
  around what iteration 6 already did.
- **Slots are indexed by iteration number, not accepted count.** If iteration 3 is rejected,
  iteration 4 still gets slot 4. The budget, the schedule and the transcript stay aligned.
- **The checker runs before the device run.** It is a static read of one file costing milliseconds;
  the run it saves is four minutes of hardware. A violating iteration is rejected without spending it.
- **A permissive slot still gets a checker.** "Both NKI and torch are allowed" has nothing to reject,
  so the compiler writes one that passes and says in a comment why. Otherwise "there was nothing to
  check" is indistinguishable from "the compiler failed to write a script".

Ranges live in `optimization.yaml` and are where module-specific guidance goes. A slot with no text
leaves those iterations unconstrained; `enforce: false` puts text in the prompt without checking it.

## The frozen validator

In both loop stages `scope.editable` is `[source.py]` and **`inference.py` is frozen** — written by
the preparation agent, gated once, then untouchable for the whole run. That is what makes one
iteration's number comparable to another's: the validator defines both what correct means and how
latency is measured, so a movable one makes every measurement incommensurable.

Two consequences.

The kernel's **I/O contract is pinned** too — argument shapes, dtypes, layout. An iteration cannot
pre-transpose a weight or repack the experts, even when that is what is holding the kernel back. The
reviewer is the channel for that: the loop's prompt tells the agent to say so in its notes rather than
edit the validator, and the reviewer carries it to the operator. Write that section into the reviewer
prompt — it is the only way the feedback gets out.

And the **metric is read back from the gate's verdict** rather than measured again
(`optimization.readback`). The gate is the only thing that runs the candidate. Running the validator a
second time as AutoHelix's metric command would double every iteration's device time and let the two
runs disagree about which code was measured.

## The two gates

**`submodule_checker.py`, 7 checks**, all module-agnostic: the declared entry point exists and the
validator reaches it; both files are self-contained; a fresh profile and a real latency; the baseline
passes at a pinned bar; the tensors are recorded bytes; the cut is declared and its reassembly
verified; the run used one core.

Note what is *not* there. The bar has no default — unlike bootstrap's, which can fall back to the
bfloat16 row from a known dtype, a submodule's golden is an intermediate the agent chose, so its bar
was derived when the repo was built or it does not exist. And **torch is allowed in `source.py`**,
because from iteration 4 the schedule permits it and a gate contradicting the schedule is a trap.

**`module_checker.py`, 9 checks**, and this one is semantic:

| | check |
|---|---|
| a | `inference.py` is byte-identical to what stage 4 froze, and drives `source.py` |
| b | self-containment |
| c | **the reduction is `nki.collectives`** — not `torch.distributed`, not `xm.*` |
| d | four ranks under `torchrun`, every rank exits 0, every rank reports its latency |
| e | rank 0's post-collective output matches the module's recorded output at the module's bar |
| f | a fresh 4-rank collective profile, and `latency_ms` is one of the numbers the ranks printed |
| g | data provenance, byte-for-byte |
| h | **faster than the bootstrapped single-core module** |
| i | **no slower than 1.1x the submodule** |

(c) matters because `torch.distributed.all_reduce` would work and would measure a different machine:
the point is a collective inside the traced graph, where the compiler can overlap it. (h) and (i) are
constraints rather than metric gates because they compare against numbers measured *outside* this run
— a metric gate can only compare an iteration against the best iteration of the same loop. Both are
**re-measured at assembly time** on this host rather than copied from a log.

## `nki.collectives` on this toolchain

Verified working before any of this was written: 4-rank `all_reduce` and `all_to_all`, fp32 and
bf16, all four ranks agreeing, under `torchrun` at LNC=2. Six requirements, each of which fails with
an internal compiler error that names something else:

1. **`kernel[2]`, not `kernel(...)`.** NKI defaults to `lnc=1` while this host runs LNC=2, and a
   LNC=1 NEFF leaves the collective's buffer unallocated on the logical core's second physical core
   (`NCC_ILLC059 Could not find MemoryLocation ...:src on core 1`).
2. **`name=` on the collective's `src`/`dst`**, or DRAM allocation fails (`NCC_IBIR440`).
3. **A collective may not touch IO tensors** (`NCC_INLA001`). `nisa.dma_copy` in, collective between
   two named scratch buffers, `nisa.dma_copy` out.
4. **All `src`/`dst` in `nl.shared_hbm`** — buffer kinds may not be mixed.
5. **Build `ReplicaGroup` outside the kernel** and pass it in; the tracer rejects `range` in a traced
   body.
6. **Build input tensors on CPU and `.to(device)`.** `torch.full(..., device=xla)` emits a broadcast
   HLO that hits `NCC_ISMP902` — and that error is the *broadcast*, not the collective, which is the
   most expensive way to lose a day here.

The reference pattern is `aws-neuron/nki-library` at
`src/nkilib_src/nkilib/experimental/collectives/collectives.py`. (`nki-samples` has no collective
examples.) Profiling:

```bash
neuron-explorer capture -n model.neff --io-from=runtime \
  --collectives-worker-count 4 --collectives-workers-per-node 4 \
  --collectives-worker-start-id 0 --collectives-profile-id all -s profile.ntff
neuron-explorer view -n model.neff -s profile_rank_0.ntff --output-format=summary-json --disable-ui
```

One `profile_rank_N.ntff` per rank. `total_exec_time` is in **seconds**; the summary also carries
`cc_op_count`, `cc_op_time` and `cc_op_active_time_percent`, which isolate the collective from the
compute — the number stage 5 is trying to shrink.

**This contradicts `floorplan/README.md`**, which states that NKI 0.6.0 exposes no collective
primitive and derives the intra-device link bandwidth from probed DMA efficiency instead of measuring
it. That claim is wrong: the collectives are in `nki.collectives`, absent from `nl`/`nisa`, which is
why they looked missing. The floorplan's intra-device bandwidth — and every conclusion it draws about
whether tensor parallelism belongs inside a device or across them — could now rest on a measurement.

## Why the metric is the fastest rank

It isolates compute from load imbalance, which is what makes iteration-to-iteration comparison
meaningful. It also **understates** what the module costs in a pipeline, where the slowest rank gates
the next stage. Both facts are stated at the top of the report rather than in a footnote, along with
the other uncalibrated thing: the projection diverges from the ranked plan, so a fast kernel here is a
fast *single-device* kernel and not evidence about the 16-device deployment.

For `.ffn` that divergence is precise and worth knowing. The floorplan's report chose `expert x8`
*because* it halves per-bank weight residency and decode is bank-bandwidth bound. Projecting to
`expert x4` doubles per-core residency — so this pipeline optimizes the plan the search ranked second.
That is the right thing to want when the goal is the fastest single-device kernel, and the wrong thing
to forget when reading the number.

## What is kept

Every constraint-passing candidate is archived to `.autohelix/optimization/candidates/iter-N/` with
its metrics, accepted or not. The 5% metric gate governs what the next iteration *builds on*; it
should not govern what the run *retains* — iterations 7-8 exist to try what the disciplined ones
cannot, and an experiment 6% slower is rejected while still being the most informative thing in the
run.

And the deliverable is the **best accepted commit**, not `HEAD`. With 5% of slack, `HEAD` after ten
iterations can be slower than the best iteration; handing it to stage 4 would quietly give away part
of what the loop achieved.

## Layout

```
optimization/
  README.md                this file
  cli.py                   autohelix optimize {template,check,init,submodule,compile-constraints,
                                               run,assemble,run-full,all,gate,report}
  driver.py                the five stages, the preparation agents, the manifests
  config.py                optimization.yaml -> the pipeline config, and the derived per-stage
                           AutoHelix configs (written outside the repos: they name the gate)
  projection.py            a floorplan placement -> what one device holds, and what that cost
  materialize.py           the two repos: what the preparation agents read
  recipe.py                the declared reassembly, checked arithmetically
  loop.py                  Harness + the per-iteration constraint, candidate archive, best commit
  constraints.py           the schedule, the compiled checkers, their manifest
  gate.py                  what the two gates share (reuses bootstrap/nki_checker's analysis)
  submodule_checker.py     the stage-2 gate: 7 module-agnostic checks
  module_checker.py        the stage-4/5 gate: 9 checks, the only semantic one in the pipeline
  readback.py              the metric command: the latency the gate already measured
  report.py                REPORT.md, with the two caveats at the top
  presets.py               reading the packaged templates
  templates/
    optimization.yaml      the config template the operator fills in
    loop_prompt.md         the per-iteration prompt
    submodule_prompt.md    stage 2: cut the module down
    assemble_prompt.md     stage 4: put the ranks back together
    compiler_prompt.md     the constraint compiler
```

## Gotchas

- **The pipeline config is not an AutoHelix config.** `optimization.yaml` describes the whole run;
  `config.derive_loop_config()` produces the AutoHelix config each loop stage takes, and writes it
  *beside* the repos because it names the hidden gate.
- **Tensors are hard-linked, not copied.** The MoE module's `tensors/` is 6.8 GiB and three copies is
  not a design. Same inode, so `sha256` still sees the recorded bytes and a write must unlink first.
- **`iteration_constraints` is read from the raw config, not from `Config`.** It is listed in
  `KNOWN_TOP_LEVEL_KEYS` so it does not warn as a typo, but there is no field for it on the shared
  dataclass — a pipeline-specific concept does not belong on every AutoHelix user's config.
- **A preparation stage that fails three times stops.** It is not a loop: a half-materialized repo is
  not a worse starting point than the last attempt, it is not a starting point. Read the last report
  before raising `preparation.retries`, because a repeated failure is the prompt or the module rather
  than luck.
- **The checkers are hidden, not sandboxed.** They live under `.autohelix/optimization/constraints/`,
  which `Sandbox.prepare_worktree` does not seed into a worktree — but a worktree sits *inside* the
  project, so a determined agent can walk up to them. As in `floorplan`, treat it as a speed bump
  backed by the reviewer.
