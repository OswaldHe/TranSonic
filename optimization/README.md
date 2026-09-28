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

The words this pipeline coins — *projection*, *slot*, *checker*, *advisory*, *custody*, *attic*,
*drift* — are defined in [`CONTEXT.md`](../CONTEXT.md); reach for it when a term here reads as
ambiguous, or before coining another. Three decisions are recorded with their rejected alternatives
in [`docs/adr/`](../docs/adr/), and named below at the point where each one bites.

## Six stages, and why they are six

```
    init            submodule            run              assemble          run-full        feedback
 ┌──────────┐    ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌───────────┐   ┌───────────┐
 │ project  │    │ agent cuts  │   │ N iters on  │   │ agent joins │   │ 5 iters on│   │ agent reads
 │ the      │───►│ the module  │──►│ one rank,   │──►│ N ranks with│──►│ the whole │──►│ every note
 │ placement│    │ to one rank │   │ under the   │   │ nki.collect-│   │ module    │   │ and reports
 │ onto one │    │             │   │ constraint  │   │ ives        │   │           │   │ what blocked
 │ device   │    │             │   │ schedule    │   │             │   │           │   │ it
 └──────────┘    └─────────────┘   └─────────────┘   └─────────────┘   └───────────┘   └───────────┘
      │                 │                 │                 │               │               │
  projection.py   submodule_checker  loop.py +        module_checker     loop.py       feedback.py
                                     constraints.py                   (no schedule)  (no gate)
```

**`init` is a rule, not a search.** The ranked floorplan is written for a 16-device trn2.48xlarge and
development happens on a one-device trn2.3xlarge, so almost nothing fits. In `schemes/rank1.yaml`,
of 271 placements: 182 fit one device (every hyper-connection module, the norms, `embed`, the small
MTP heads), **87 span two devices** (all 43 `.ffn` at `expert x8`, all 43 `.attention`, `lm_head`),
and 2 span four (both Engram tables). So every module worth optimizing is oversized, and `init`
narrows the split to what one device holds — `expert x8` becomes `expert x4`. That narrowing is a
divergence from the ranked plan and is priced as one, not waved through:
[ADR 0003](../docs/adr/0003-projecting-onto-one-device-diverges-from-the-plan.md).

**`submodule` is an agent, not a script.** How to divide a module into per-rank work is a judgement
about what the module computes, and a script that knew how to shard an MoE would work for MoE and
nothing else. So the agent decides and nothing here checks its semantics — deliberately, and the
loose end is tied off at `assemble` instead:
[ADR 0001](../docs/adr/0001-the-agent-chooses-the-cut.md).

**`run` and `run-full` are the same loop.** Ordinary `autohelix run` semantics — green baseline, a
metric, a rejected iteration discarded — plus one thing: a per-iteration constraint.

**`assemble` is where the looseness gets paid for.** Everything before it is judged against goldens an
agent chose. Here the target is the bootstrapped module's own recorded output, at the bootstrapped
module's own bar, faster than the bootstrapped module, and within 10% of the submodule. A wrong cut
cannot pass. A submodule that was fast because it did a quarter of the work cannot pass. That is what
makes the looseness upstream safe.

**`rerun-full` is how a second round happens.** The first round's notes are where the ideas for the
second one come from: you read them, you learn that the expert skip was blocked by a loop bound or
that the scale multiply is the binding pass, and you want the loop to carry on *from the kernel you
already have* with that written in as a constraint. So `optimize rerun-full` starts from the previous
round's **best accepted commit** — not `HEAD`, which regression slack can leave slower — re-measures
its baseline with the gate, and recompiles the checkers from whatever the prose now says. The old
round's run state is archived to `.autohelix/archive/<timestamp>/` — the same place and the same way
`autohelix clear` archives a run, so there is one convention rather than two. Its `notes/` and
`reviews/` are copied back afterwards, because the next agent reading them is the point;
`observations/`, `logs/` and `output/` are not, because they belong to the round that produced them.

**`feedback` reads what the run wrote and nobody else will.** Fifteen iterations leave ~83,000 words
of notes and reviews, written one iteration at a time by agents that did not know how the run would
end — so the corpus contradicts itself, and a reader who trusts any one file learns something false.
One agent reads all of it, reconciles the disagreements (a measurement beats an inference; recency
alone settles nothing), and writes `FEEDBACK.md`: what stopped the kernel, split by **who would have
to fix it**. It measures nothing and edits no repository, so it has no gate — `feedback.py` checks
the report's *shape*, and nothing checks whether a finding is true, because nothing could.

## What a script checks, and what an agent is trusted with

| | checked by a script | left to an agent |
|---|---|---|
| the projection | **yes** — `projection.py`, deterministic | — |
| how the module is cut | no | **the `submodule` agent** |
| that the cut *closes* | **yes** — `recipe.py`, arithmetic | — |
| whether the cut is *good* | **yes** — the +10% bound, at `assemble` | — |
| the submodule repo's integrity | **yes** — the submodule gate, 7 checks | — |
| the per-iteration constraint | **yes** — one compiled checker per slot | the constraint compiler writes it |
| the whole module's correctness | **yes** — the module gate, 9 checks | — |

The row worth dwelling on is the third. A wrong cut cannot be caught by reading `source.py`, but it
*can* be caught arithmetically. The `submodule` agent has to declare its **reassembly recipe** — how
its ranks recombine, which `assemble` needs anyway — and a declaration of that form is checkable:
take the N per-rank goldens it dumped, apply the declared recipe, and see whether the result is the
module's recorded output. A cut that drops an expert, double-counts a shared path, shards along the
wrong axis or dumps a golden from the wrong rank fails, because none of those sum back. Host-side
numpy over bytes already on disk, a second to run, and entirely ignorant of what the module computes.
What it cannot prove, and why that job belongs to the +10% bound instead, is in ADR 0001.

## Stripping: the code arrives without an earlier agent's claims

`source.py` and `inference.py` are carried into an optimization repo with their **comments and
docstrings removed** (`strip.py`). Those were written by an earlier agent, and they are its claims
about the hardware and the compiler — some hard-won and right, some wrong, and indistinguishable at a
glance. A claim in a comment reads as established fact to the agent that reads it next, which will
design around it without testing it.

This repository has already produced the worked example. `floorplan/README.md` states that NKI 0.6.0
exposes no collective primitive, and derives the floorplan's intra-device bandwidth rather than
measuring it on the strength of that. The claim is false (see below), and it went unchallenged because
it was written down confidently. A comment inside a kernel is the same hazard with less visibility.

`reference_torch.py`, `reference_numerics.py`, `reference_inference.py`, `vendor/` and `compat/` keep
theirs — that is the vendor's and the harness's own code, carried in verbatim as the specification,
and its comments are authoritative rather than inferred. The rule is by filename, so a directory
added to a repo later cannot accidentally be exposed to it. Formatting is preserved exactly
(position-based removal, not a `tokenize` round-trip), and the result is re-parsed before it is
written: a file that cannot be stripped safely is left alone and reported.

## The per-iteration constraint schedule

Ten iterations of "do whatever you like" converge on whatever the first iteration happened to try, so
the iterations are divided into **slots**, each with its own constraint:

| iterations | constraint | enforcement |
|---|---|---|
| 1-3 | NKI only, no torch — prove the fast path is reachable in NKI at all | hard |
| 4-6 | NKI and torch-xla both allowed | hard |
| 7-8 | nothing; explore aggressively | off |
| 9-10 | NKI and torch-xla — consolidate what 7-8 learned | hard |

A slot can be one iteration (`at: 5`), a range (`from`/`to`) or a set (`iterations: [1, 3, 5]`), and
each carries its own **enforcement**, which is what a violation costs:

- **`hard`** — the iteration is rejected and its work discarded.
- **`soft`** — checked and recorded, but it does not reject on its own. What it costs is the
  regression slack: a candidate that missed its constraint has to be correct and *strictly faster*
  than the best so far, where a compliant one only has to stay inside
  `acceptance.max_regression_pct`. That is what stops a soft constraint being no constraint.
- **`off`** — the text is guidance in the prompt and nothing checks it.

`soften_last` (default true) drops a **hard** slot's last iteration to `soft`. One slot per iteration
plus these two fields is per-iteration control of hard versus soft; `soften_last: false` on a
single-iteration slot is a constraint with no escape at all. The older `enforce: true|false` still
works and means `hard`|`off`.

Both loop stages read a schedule. Stage 5's is empty by default, because the first round has nothing
to go on — see `rerun-full` below for where it earns its place.

The constraints are prose, because the useful ones are prose. Turning prose into a predicate is a job,
and it goes to a **constraint compiler**: an agent that runs once, before iteration 1, and writes one
**checker** per slot. The optimizing agent gets the prose in its prompt and never sees the checker —
the same asymmetry `bootstrap` and `floorplan` use, for the same reason. A constraint whose
implementation is readable gets read for loopholes instead of followed.

Five properties, each a choice:

- **The compiler runs once, up front.** A checker written at the top of iteration 7 could be written
  around what iteration 6 already did.
- **Slots are indexed by iteration number, not accepted count.** If iteration 3 is rejected,
  iteration 4 still gets slot 4. The budget, the schedule and the transcript stay aligned.
- **The checker runs before the device run.** It is a static read of one file costing milliseconds;
  the run it saves is four minutes of hardware. A violating iteration is rejected without spending it.
- **On a slot's last iteration the constraint is checked but not fatal** — it runs **advisory**. By
  then the agent has had every iteration the slot allows, and the constraint's job, shaping the
  search, is done. A candidate that still misses it while passing the correctness gate and being
  **strictly faster** than the best so far is kept. What it gives up is the regression allowance every
  compliant iteration gets (`acceptance.max_regression_pct`, 5% by default), so the escape has to be
  earned rather than taken. `slotcheck.py` runs the checker `--advisory` there (same verdict, exit 0)
  and `loop.py`'s `_check_metric_gates` applies the stricter rule.
- **A permissive slot still gets a checker.** "Both NKI and torch are allowed" has nothing to reject,
  so the compiler writes one that passes and says in a comment why. Otherwise "there was nothing to
  check" is indistinguishable from "the compiler failed to write a checker".

Slots live in `optimization.yaml` and are where module-specific guidance goes. A slot with no text
leaves those iterations unconstrained; `enforce: false` puts text in the prompt without checking it.
Editing a slot's prose after its checker was compiled is **drift**, and the pipeline refuses to run
until the checker is recompiled.

Write that guidance as prose inside the block. A `#` line inside a `text: |` block is not a YAML
comment — the whole block reaches the agent verbatim — so the template's own fill-in hints sit
*outside* the blocks, and a placeholder that survives into a slot is reported as a warning. The first
MoE run sent `# <FILL IN: module-specific guidance for iterations 9-10, if any.>` to the agent as
part of its constraint for four iterations, which is how this was found.

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
(`optimization.readback`), because the stage's gate is the only thing that runs the candidate. Why
that is worth the one wrinkle it introduces at iteration 0:
[ADR 0002](../docs/adr/0002-a-stages-gate-is-the-only-thing-that-runs-a-candidate.md).

## The two gates

A bare "gate" is ambiguous between these two, so they are always named apart. Neither is
`acceptance.metric_gates`, which is a regression allowance and not a gate in this sense.

**The submodule gate** (`submodule_checker.py`), 7 checks, all module-agnostic: the declared entry
point exists and the validator reaches it; both files are self-contained; a fresh profile and a real
latency; the baseline passes at a pinned bar; the tensors are recorded bytes; the cut is declared and
its reassembly verified; the run used one core.

Note what is *not* there. The bar has no default — unlike bootstrap's, which can fall back to the
bfloat16 row from a known dtype, a submodule's golden is an intermediate the agent chose, so its bar
was derived when the repo was built or it does not exist. And **torch is allowed in `source.py`**,
because from iteration 4 the schedule permits it and a gate contradicting the schedule is a trap.

**The module gate** (`module_checker.py`), 9 checks, and this one is semantic:

| | check |
|---|---|
| a | `inference.py` is byte-identical to what `assemble` froze, and drives `source.py` |
| b | self-containment |
| c | **the reduction is `nki.collectives`** — not `torch.distributed`, not `xm.*` |
| d | every rank ran under `torchrun`, exited 0, and reported its latency |
| e | rank 0's post-collective output matches the module's recorded output at the module's bar |
| f | a fresh collective profile covering every rank, and `latency_ms` is one of the numbers they printed |
| g | data provenance, byte-for-byte |
| h | **faster than the bootstrapped single-core module** |
| i | **no slower than 1.1x the submodule** |

(c) matters because `torch.distributed.all_reduce` would work and would measure a different machine:
the point is a collective inside the traced graph, where the compiler can overlap it. It is checked
through import aliases, so `import torch.distributed as ncc` does not slip past. (d) and (f) take the
rank count from the manifest (`rank_count`, 4 on this host) rather than assuming four, so a two-rank
assembly is not failed for missing markers nobody asked for. (h) and (i) are constraints rather than
metric gates because they compare against numbers measured *outside* this run — a metric gate can
only compare an iteration against the best iteration of the same loop. Both are **re-measured at
assembly time** on this host rather than copied from a log.

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
compute — the number `run-full` is trying to shrink.

This is what **contradicts `floorplan/README.md`**: the collectives are in `nki.collectives`, absent
from `nl`/`nisa`, which is why they looked missing. So the floorplan's intra-device link bandwidth —
and every conclusion it draws about whether tensor parallelism belongs inside a device or across
them — could now rest on a measurement instead of a derivation.

## Why the metric is the fastest rank

It isolates compute from load imbalance, which is what makes iteration-to-iteration comparison
meaningful. It also **understates** what the module costs in a pipeline, where the slowest rank gates
the next stage. Both facts are stated at the top of the report rather than in a footnote, along with
the other uncalibrated thing: the projection diverges from the ranked plan, so a fast kernel here is a
fast *single-device* kernel and not evidence about the 16-device deployment. For `.ffn` that
divergence is precise and quantified in ADR 0003 — worth reading before quoting a number from this
pipeline at anyone.

## What is kept

Every constraint-passing candidate is archived to `.autohelix/optimization/candidates/iter-N/` with
its metrics, accepted or not. The regression allowance governs what the next iteration *builds on*; it
should not govern what the run *retains* — iterations 7-8 exist to try what the disciplined ones
cannot, and an experiment 6% slower is rejected while still being the most informative thing in the
run. The capture happens before the worktree is torn down, because teardown does `git branch -D` and
takes the commit with it.

And the deliverable is the **best accepted commit**, not `HEAD`. With 5% of slack, `HEAD` after ten
iterations can be slower than the best iteration; handing it to `assemble` would quietly give away
part of what the loop achieved.

## Layout

```
optimization/
  README.md                this file
  cli.py                   autohelix optimize {template,check,init,submodule,compile-constraints,
                                               run,assemble,run-full,all,gate,report}
  driver.py                the five stages, the preparation agents, the manifests
  config.py                optimization.yaml -> the pipeline config, and the derived per-stage
                           AutoHelix configs (written outside the repos: they name the gates)
  projection.py            a floorplan placement -> what one device holds, and what that cost
  materialize.py           the two repos: what the preparation agents read
  recipe.py                the declared reassembly, checked arithmetically
  strip.py                 removing an earlier agent's comments from the code the next one reads
  slotcheck.py             running a compiled checker, enforcing or advisory
  loop.py                  Harness + the per-iteration constraint, candidate archive, best commit
  constraints.py           the schedule, the compiled checkers, their manifest
  custody.py               holding the fields a gate reads back outside the agent's repo
  candidate.py             running a candidate repo and reading facts out of it: what both
                           gates share (reuses bootstrap/nki_checker's analysis)
  submodule_checker.py     the submodule gate: 7 module-agnostic checks
  module_checker.py        the module gate: 9 checks, the only semantic ones in the pipeline
  readback.py              the metric command: the latency the gate already measured
  report.py                REPORT.md, with the two caveats at the top
  feedback.py              FEEDBACK.md: the last stage's corpus, and the report's shape
                           (`--check` so the agent can check its own work)
  presets.py               reading the packaged templates
  templates/
    optimization.yaml      the config template the operator fills in
    loop_prompt.md         the per-iteration prompt
    submodule_prompt.md    the `submodule` stage: cut the module down
    assemble_prompt.md     the `assemble` stage: put the ranks back together
    compiler_prompt.md     the constraint compiler
    feedback_prompt.md     the `feedback` stage: reconcile the notes into toolchain feedback
```

## Gotchas

- **The pipeline config is not an AutoHelix config.** `optimization.yaml` describes the whole run;
  `config.derive_loop_config()` produces the AutoHelix config each loop stage takes, and writes it
  *beside* the repos because it names the hidden gate.
- **Tensors are hard-linked and made read-only.** The MoE module's `tensors/` is 6.8 GiB and three
  copies is not a design. The read-only part is load-bearing and an earlier version of this note had
  it wrong: a hard link shares the inode, and `open(path, "wb")` truncates it *without* unlinking, so
  a validator that opened a recorded tensor for writing by mistake would destroy the bootstrap repo's
  golden — irreplaceable, and the assembly's provenance hashes would then bless the corruption. This
  filesystem is ext4 with no reflink support, so copy-on-write is unavailable and the mode bits are
  the guard. Clearing write permission covers the shared original too, which is the right outcome.
- **`iteration_constraints` is read from the raw config, not from `Config`.** It is listed in
  `KNOWN_TOP_LEVEL_KEYS` so it does not warn as a typo, but there is no field for it on the shared
  dataclass — a pipeline-specific concept does not belong on every AutoHelix user's config.
- **Preparation agents do not own the fields a gate reads.** They finish the manifest, and the
  tensor record and the declaration are genuinely theirs — but the bar, the golden, the rank count
  and both latency bounds are written by materialization to a copy outside the repo and restored
  afterwards (`custody.py`). Otherwise the agent writes its own examination paper. A field that
  changed is reported rather than rejected: the usual cause is a manifest rewritten instead of
  edited, which is careless rather than dishonest, and the pipeline can simply put it back.
- **A failed preparation attempt is kept in the attic, not deleted.** Moved to
  `.optimization/attempts/<stage>-<n>/`. The first real run reached 5 of 7 checks on its first
  attempt and the work was overwritten before it could be read. The carried-in tensors are hard links
  and cost nothing, but the slices the agent cut for itself are its own bytes — around 1.8 GB per
  attempt for a quarter of this MoE, and `du` over-reports because it counts the hard links too. Only
  the last `preparation.retries` attempts are kept, so the ceiling is bounded.
- **A preparation stage that fails three times stops.** It is not a loop: a half-materialized repo is
  not a worse starting point than the last attempt, it is not a starting point. Read the last report
  before raising `preparation.retries`, because a repeated failure is the prompt or the module rather
  than luck.
- **Stripping never touches the originals.** The bootstrapped repo keeps its comments; only the
  copies inside an optimization repo lose them. If you want to read the bootstrap agent's reasoning,
  it is still in `bootstrap-runs/<module>/`.
- **The checkers are hidden, not sandboxed.** They live under `.autohelix/optimization/constraints/`,
  which `Sandbox.prepare_worktree` does not seed into a worktree — but a worktree sits *inside* the
  project, so a determined agent can walk up to them. As in `floorplan`, treat it as a speed bump
  backed by the reviewer.
</content>
</invoke>
