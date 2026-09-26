# Floorplanning a model onto a Trainium instance

`autohelix floorplan` answers a question the other two loops leave open: given a model cut
into modules, **what runs where**. Which of the 64 logical NeuronCores on a trn2.48xlarge holds
each of DeepSeek V4.1 Flash's 271 modules, how each module is divided and along which
dimension, which memory tier holds its weights, and in what order it all happens.

It cannot be answered by measurement, because the 16-device instance is not the machine in
front of us — development happens on a one-device trn2.3xlarge. So the loop optimizes against
a simulator, and most of the design here is about making a simulated metric worth optimizing
against.

```bash
autohelix floorplan probe                             # measure the primitives, once per host
autohelix floorplan init <project> --artifact <dir>    # materialize a project
autohelix floorplan build --path <project>            # 2 iterations: write the cost models
autohelix floorplan run   --path <project>            # 5 iterations: search for a floorplan
autohelix floorplan rank  --path <project>            # blind ranking + the final report
autohelix floorplan all   <project> --artifact <dir>   # all five, in order
```

## Four stages, and why they are four

```
     probe                    build                     run                    rank
  ┌──────────┐          ┌──────────────┐        ┌───────────────┐       ┌──────────────┐
  │ measure  │          │ agent writes │        │ agent edits   │       │ blind agent  │
  │ matmul,  │  ──────► │ sim/modules/ │ ─────► │ floorplan.yaml│ ────► │ ranks 3      │
  │ DMA,     │  systems │ sim/         │  sim/  │               │ top-3 │ schemes      │
  │ gather,  │  /probed │ constraints  │ frozen │ 5 iterations  │       │              │
  │ storage  │  .yaml   │ 2 iterations │        │ 1h each       │       │ no simulator │
  └──────────┘          └──────────────┘        └───────────────┘       └──────────────┘
       │                       │                        │                      │
   coefficients           invariants.py            checker.py            REPORT.md
   with provenance        (9 checks)               (7 checks, hidden)    + appendix
```

The seams are the design.

**`probe` before everything**, and the simulator *refuses to run* without it. A roofline at
100% of the datasheet's 1,299 fp8 TFLOPS would make arithmetic free and rank every scheme on
communication alone — so the choice is between a measured coefficient and no simulation, never
between a measured coefficient and peak.

**`build` and `run` are different loops** because the agent that writes the cost model must not
be the agent scored by it. `build` ends by hashing `sim/` into the manifest; from that moment
`run`'s editable scope is `floorplan.yaml` alone and gate check (d) verifies the bytes. An
agent that could do both would have every incentive to make the model cheap rather than the
plan good.

**`rank` runs blind** — no simulator, no traces, no latencies, in a sandbox that physically
lacks them. The three schemes arrive already ordered by the simulator, and an agent that could
see that order would be reviewing the search's answer instead of forming one. Since the
simulator is uncalibrated by construction, a second independent judgement is worth more than a
confirmation of the first.

## What is deterministic and what is an agent's

This is the split the whole thing rests on, so it is worth being precise.

| | written by | frozen |
|---|---|---|
| the hardware model (`parser.py` from `systems/*.yaml`) | a fixed rule per field, no inference | always |
| the timeline, collectives, memory ledger (`sim/engine.py`, `collectives.py`, `memory.py`) | us, once | always |
| the costing helpers (`sim/api.py`) | us, once | always |
| **per-module cost models** (`sim/modules/*.py`) | **an agent, in `build`** | after `build` |
| **the prose constraints, made executable** (`sim/constraints.py`) | **an agent, in `build`** | after `build` |
| the baseline floorplan | `baseline.py`, deterministically | it is iteration 0 |
| **the floorplan** (`floorplan.yaml`) | **an agent, in `run`** | never — it is the search |

Within the simulator the line is drawn once more, inside `api.py`: **a cost model says what
work exists; the framework says what it costs.** A cost model states that an attention shard is
a matmul of these dimensions in this dtype, then an elementwise pass over these bytes, then an
allreduce of this tensor. `matmul_seconds` converts that to time, using rates from the system
YAML and coefficients from the probe. A cost model cannot make a matmul cheaper by believing
in a better machine — it can only misstate the matmul's shape, which is arithmetic a reviewer
checks against the module's source in a minute. `invariants.py` check (h) rejects a literal
duration handed to `op()`, which is the one way around this.

## The platform description, and its prose half

`systems/trn2-16device.yaml` is the target. Every number carries `source:` and `confidence:`,
with three sources: `datasheet` (published by AWS, URLs in the file), `nki` (read out of the
installed toolchain, so it is the compiler's own constant), and `probed`.

The consequential facts, because two of them are easy to get wrong:

- **HBM is banked at 24 GiB per logical NeuronCore**, not 96 GiB flat. The 96 GiB is the
  device's four banks. A module whose shard exceeds 24 GiB is unplaceable however much device
  memory is free — and both Engram tables are 94.56 GiB each.
- **Doc terminology is inverted from informal usage.** A *logical NeuronCore* is the group of
  two; the two cores inside it are *physical NeuronCore-v3*. At LNC=2 the runtime dispatches to
  the pair, so the pair is the placement grain: 16 devices × 4 = **64 units**.
- The 4×4 torus wraps, so the far corner is 2 hops and the maximum is 4.

Then there is `constraints_text`: the half of a platform description that is not a number.
Nine numbered items in prose — "GPSIMD and the Tensor Engine cannot access SBUF simultaneously
on Trainium 2", "there is no direct accelerator-to-NVMe path", and so on. The `build` agent
turns each into a check or a cost term and cites the item number where it does; invariant (g)
reports any item nothing cites. That is bookkeeping rather than verification, but an item no
code references is unreviewable, which for a hidden-gate loop is the same problem.

Adding to `constraints_text` is the intended way to teach the simulator something that has no
field in the schema.

## What the probe measures, and what it cannot

Measured on the dev host, per logical NeuronCore — which is the placement grain, so the
coefficients transfer to the 16-device target unchanged:

| | |
|---|---|
| `matmul_bf16`, `matmul_fp8` | tensor-engine throughput at the streaming shape |
| `matmul_small_k` | the same with a 32-partition contraction: **the MoE and LoRA regime** |
| `vector_elementwise`, `scalar_activation` | vector and scalar engine throughput |
| `gpsimd_gather` | `nc_n_gather` rate — the Engram and expert-lookup primitive |
| `dma_large_contiguous`, `dma_small_strided` | HBM reads, full tiles versus scattered slices |
| `host_dram` bandwidth | the PCIe path |
| `nvme` bandwidth, IOPS and latency | raw instance-store reads at queue depth 1 |

Not measurable with one device: inter-device NeuronLink bandwidth, torus hop latency, and
collectives wider than four cores. Those stay at their datasheet values, and `rank`'s appendix
is required to name them as extrapolations.

Three more turned out not to be measurable here, and each is tagged `assumed` or
`derived_from_probe` with its reasoning in the overlay rather than quietly filled in:

- **`matmul_fp8`.** This target rejects `float8_e4m3fn` outright ("not supported on TRN2 and
  earlier"), so there is no fp8 matmul to time. Derived from the *measured bf16* throughput,
  which assumes fp8 achieves no more FLOPS than bf16 — conservative, so no scheme can win on an
  fp8 speedup the hardware was never observed to deliver.
- **The intra-device link.** The obvious approach looks like it should work and does not: NKI
  0.6.0 exposes no collective primitive, and the XLA path that would provide one fails to
  compile on neuronxcc 2.27.5334 with an internal simplifier error (`NCC_ISMP902`) — for an
  all-reduce and, as it happens, for a bare `torch.matmul` too. Bandwidth is derived from the
  probed DMA efficiency and the on-chip bound of text constraint 3; latency takes the same
  1.0 µs the system YAML assumes for a torus hop, on the grounds that an on-chip hop cannot be
  worse. Any conclusion about whether tensor parallelism belongs inside a device or across them
  rests on this, and the report says so.
- **Host-DRAM latency.** The *bandwidth* is measured and solid (12.9 GB/s, reproducible across
  runs and two different kernels). The latency is not obtainable: on this path every transfer
  is also a kernel dispatch, so the fitted intercept comes back at ~120 µs however the baseline
  is subtracted, and that is launch overhead rather than link latency.

The last two are worth dwelling on, because both were briefly believed. A 102 µs on-chip hop
makes every collective latency-bound and rules out tensor parallelism at decode; a 120 µs PCIe
access puts ~4 ms on one token's 32-row Engram lookup and rules out tiering the table at all.
Each would have removed a whole branch of the design space on the strength of a measurement
artifact — which is worse than admitting a number is assumed. Both are now constants with
reasoning attached, and `tests/test_floorplan_probe.py` is what keeps them from drifting back
into being fits.

The general rule this settles: a fitted *slope* across a size sweep is trustworthy; a fitted
*intercept* on a path where every sample pays a fixed cost is not.

### Two measurement traps, both of which produced confidently wrong numbers first

Recorded because a probe that falls into either fails silently.

**Operands must already be on the device.** A `torch_neuronx.trace` input is copied
host-to-device on every call, so a kernel that reads its input measures PCIe (12.9 GB/s), not
HBM (205 GB/s). The HBM probes fill a `private_hbm` scratch buffer inside the kernel and read
that. A 16× error.

**Operands must be in SBUF to measure arithmetic.** A textbook tiled matmul that loads its
tiles inside the innermost loop measures 6.4 TFLOPS — 3.8% of per-core peak — because it is
bound by 1.3 GB of redundant HBM traffic per call. Hoisted into SBUF, the same silicon measures
63 TFLOPS. The first number would model a machine 10× slower at arithmetic than the one we
have, making every scheme compute-bound with communication free: exactly backwards.

A third, in the storage probe: benchmarking a file on the mounted root measures **EBS**, not
instance store, and the page cache answers the sequential reads — 8.0 GB/s and 1,376 IOPS,
against the raw instance-store device's 2.45 GB/s and 33,238 IOPS. Both wrong, in opposite
directions. `probe/storage.py` reads the raw device `O_RDONLY | O_DIRECT` instead.

## Why no measured module latency is used

There are bootstrapped NKI kernels for many of these modules on this workspace, with real
measured latencies. They are **out of bounds** — not for cost modeling and not as a sanity
check, and `invariants.py` check (h) scans for any reference to them.

Those kernels are correctness baselines produced by `autohelix bootstrap`, deliberately
unoptimized. Anchoring the simulator to their latency would encode their inefficiency as a
property of the silicon, and the search would then optimize around an artifact of how far the
bootstrap loop happened to get. Primitive rates are measured instead; module costs are derived
from the computation.

The consequence is stated plainly wherever a number is reported: **absolute latencies are
uncalibrated.** What the simulator is good for is comparing floorplans against each other, and
that is all the loop asks of it.

## What the invariant suite can and cannot check

`build`'s gate cannot check whether a cost model is *right* — there is no ground truth, by the
choice above. It checks that the simulator behaves the way a cost model of a real machine has
to behave, whatever its constants are.

| | |
|---|---|
| a | every module on the inference path has a cost model |
| b | the generated baseline simulates and publishes four positive latencies |
| c | two identical runs produce identical metrics |
| d | no module costs nothing, and none emits no ops |
| e | **doubling a split halves per-shard compute and adds communication** |
| f | **longer contexts cost more, and superlinearly in prefill** |
| g | every numbered `constraints_text` item is cited |
| h | no duration is fabricated and no clock is read |
| i | simulating does not mutate the hardware model |
| j | **a larger batch costs more, and prefill scales roughly with it** |

(e), (f) and (j) are the load-bearing ones. A simulator where a 4-way split does not quarter
per-shard work cannot rank tensor-parallel schemes at all, however well calibrated its matmul
rate is; one insensitive to sequence length makes every context-parallel scheme look pointless;
and one blind to batch collapses the four batch columns into four copies of one measurement,
which would read as evidence that batch does not matter. A red invariant means the simulator
would mislead the search about the *direction* of a change, which is the only thing the search
uses it for.

## The exploration loop

An ordinary AutoHelix run: green baseline, sixteen metrics, a hidden gate.

The workload grid is phase × context length × batch size — `{phase}_{context}_b{batch}_ms`:

```
prefill_128_b{1,4,8,32}_ms    prefill_8192_b{1,4,8,32}_ms
decode_128_b{1,4,8,32}_ms     decode_8192_b{1,4,8,32}_ms
```

All lower-is-better, each with a **10% regression gate against its own best-so-far**. So the
frontier only moves outward and a scheme cannot buy decode with prefill, or batch 32 with batch
1. That is `acceptance.metric_gates` in `preset.yaml` — AutoHelix already compares each metric
against the best accepted value, so the rule is declarative rather than code.

**Batch is an axis because the answers reverse along it**, which a single-batch benchmark hides:

- **Pipeline depth.** At batch 1 a decode step has one token in flight, so every stage boundary
  is a bubble and depth is pure cost. At batch 32 there are 32 tokens to fill it and the same
  depth is nearly free. A plan tuned only at batch 1 is tuned for the worst case of a decision
  that flips.
- **KV capacity.** The cache grows linearly with batch, so 8192 tokens at batch 32 holds 32× the
  KV of batch 1. A residency choice that fits at batch 1 can be infeasible at batch 32, and
  capacity is checked at every point.
- **Expert routing.** A batch-1 decode step touches a handful of the 384 routed experts; a
  batch-32 step touches many more, moving the balance between expert parallelism and replication.

Sixteen gates is stricter than four but not four times stricter — the four batch columns of a
given phase and length move largely together. Where they diverge is exactly the tradeoff worth
gating.

Unlike `bootstrap`, a rejected iteration is **discarded** and the next starts from the
best-so-far floorplan. Bootstrap ratchets because its baseline is red by construction; here
iteration 0 is already a working scheme, and letting a regression merge would corrupt the
reference the 10% rule is measured against.

But rejected-and-discarded would leave `rank` with nothing to rank if four of five iterations
regressed. So **every gate-passing iteration's floorplan is archived** to
`schemes/candidates/`, recovered from its git branch after the loop, with its metrics. The
metric gate governs what the next iteration builds on; it does not govern what is retained.

### The gate

`floorplan/checker.py`, one constraint, hidden from the agent exactly as `bootstrap`'s is: the
file stays in the package, the preset names it, and no copy reaches the worktree. `preset.yaml`
states all seven requirements in prose, which makes the two a pair —
`tests/test_floorplan_checker.py` fails if they drift on the metric names, the gates, the
scope, or the phrases the goal has to contain.

| | check | what it means |
|---|---|---|
| a | plan well-formed | parses, and is legal for this hardware and this model |
| b | coverage | every module placed, fractions summing to 1, nothing off-path |
| c | dependencies | every placed module's producers are placed too |
| d | **frozen platform** | `sim/`, `systems/` **and the installed framework** byte-identical to the build |
| e | capacity | no tier over capacity, at any point of the workload grid |
| f | simulates | every workload completes and publishes a metric |
| g | deterministic | a second run produces identical metrics |

(d) is the load-bearing one for honesty, and it covers **three** things. Scope enforcement
already reverts out-of-scope edits, but that depends on git noticing; (d) verifies hashes
recorded at the freeze. First, the project's own files: the agent-written cost models plus
`systems/probed.yaml` — raising `matmul_bf16` from 0.30 to 0.90 would make every plan three
times faster without touching a line of `sim/modules/`. Second, the **installed framework**,
because the simulator runs as `python -m floorplan.sim.runner` out of the package: hashing only
the project would leave the code that computes every metric unverified. Third, that the
read-only copies in `sim/framework/` are byte-identical to those installed files, so what the
agent reads is what runs.

That third one exists because of a mistake worth recording. An earlier version did not copy the
framework at all and pointed the build agent at the installed package by absolute path, on the
reasoning that a copy nobody executes is worse than no copy. A build run then read the *entire*
package — including `checker.py` and `invariants.py`, the exploration gate and the build gate,
both of which the design depends on the agent not reading. Copying for reading and hashing all
three ways gets both properties: the project is self-contained, so there is no reason to look
outside it, and the copy cannot drift from what executes.

Note what that is and is not. It is "no reason to look", not "cannot look" — a determined agent
can still import the package and read `__file__`. A hard guarantee needs the filesystem
isolation `bootstrap/` gets from materializing a repo with no package imports at all, which this
pipeline cannot have while the simulator is a library. Treat the gate's secrecy as a
speed bump backed by the reviewer, not as a sandbox.

(g) is cheap and catches the one class of bug that would invalidate a whole run silently: a
cost model that reads the clock or iterates an unordered collection. Without it a 3%
"improvement" could be dictionary ordering.

## The design space

Everything the loop may change is in one file. `templates/project_readme.md` becomes the
project's `README.md` and documents the schema field by field; in outline a placement names a
module, the units it runs on, how it is split (`dim`, `factor`, `collective`), which tier holds
its weights and whether they are tiered, its schedule `stage`, and whether its collectives may
overlap.

`PARTITION_DIMS` is closed — `head`, `hidden`, `expert`, `seq`, `batch`, `layer`, `vocab`,
`ngram` — because the simulator has to know what a split *means* to cost it. A dimension absent
from that set is a dimension the loop cannot explore, whatever the prompt says, so adding one is
a change here and in `api.py`.

Two collectives are genuinely free, and both fall out of the platform rather than being special
cases: a `batch` split needs no exchange, and **any** split whose units share one logical
NeuronCore needs none either, because the two physical cores at LNC=2 share an address space
(text constraint 2).

The decisions with the most room in them, for the model this was built for:

- **Where tensor parallelism goes.** Inside a device it uses the intra-device link; across
  devices it pays torus hops on every collective step, so a parallel group's *shape* matters.
- **Pipeline depth.** Deep pipelines fill during a chunked prefill and stall during decode,
  where one token in flight makes every stage boundary a bubble. Both decode metrics say so
  immediately.
- **Expert parallelism versus replication.** 384 routed experts, 6 activated, 277 GiB of
  weights, batch 1. Splitting minimizes memory and maximizes all-to-all; replicating inverts it.
- **Engram residency.** 189 GiB across two tables — 12% of the instance's device memory for
  tables a token touches tens of rows of. HBM makes the lookup fast and memory scarce; host DRAM
  or NVMe inverts that; a tiered placement tries for both and has to justify a `hit_rate`. The
  largest single decision in the file.

## The ranking step

`rank.py` stages a sandbox holding only the three floorplans (anonymized `scheme-a/b/c` and
shuffled), `PLATFORM.md`, `MODEL.md`, the datasheet YAMLs with the probe overlay withheld, and
read access to the model's own source. No `sim/`, no traces, no latencies.

The agent is told that all three are feasible and what that gate checked — withholding that
would produce a worse ranking rather than a less biased one, since it might otherwise rank an
infeasible scheme first. It writes `REPORT.md`: the best scheme and why, how the workload is
distributed, each scheme's first limiting factor, and for every module the winner splits, the
decomposition recipe — which dimension, how the tensors divide, what must be communicated, and
which level of the hierarchy each piece lands on.

It also gives **one ranking per configuration** — sixteen of them, on `RANKING[<name>]:` lines —
because a single overall order hides the case a deployment actually faces. One scheme can be
right for long-context prefill at batch 32 and another for short-context decode at batch 1, and
asking the agent to call each point is what makes its architectural reasoning checkable where it
is most likely to be regime-dependent.

Then a script appends the simulator's latencies and two **agreement tables**: the overall
architectural ranking beside the simulated one, and the same per configuration. That comparison
is the most informative artifact the pipeline produces and it costs nothing to compute.
Agreement is a real cross-check, since a datasheet argument and a discrete-event simulation are
not the same evidence. Disagreement is not resolved automatically — and a disagreement
*concentrated* in one region of the grid (all the batch-32 points, say) is a specific, checkable
claim about which effect one side is missing.

The agent's overall ranking decides `schemes/rank{1,2,3}.yaml`. Nothing is discarded; all three
ship.

## Layout

```
floorplan/
  README.md              this file
  preset.yaml            the exploration loop's config — a fixed file, read directly
  build_preset.yaml      the cost-model loop's config, likewise
  cli.py                 autohelix floorplan {probe,init,build,run,rank,all,check,report,show}
  driver.py              the four stages, the manifest, the candidate archive
  schema.py              floorplan.yaml: the design space, and what makes a plan well-formed
  parser.py              systems/*.yaml -> the hardware model, deterministically
  baseline.py            iteration 0, generated rather than authored
  checker.py             the hidden gate (7 checks)
  invariants.py          the build gate (9 checks)
  rank.py                the blind sandbox, and the appendix appended afterwards
  systems/
    trn2-16device.yaml   the target: datasheet + prose constraints, with provenance
    trn2-1device.yaml    the dev host, inheriting the silicon so it cannot drift
    probed.yaml          written by `probe`; the only file that changes between runs
  probe/
    kernels.py           NKI kernels that exist only to be timed
    suite.py             the driver: measurements -> coefficients -> probed.yaml
    storage.py           raw-device read benchmark, runnable under sudo on its own
    collective.py        the four-process link probe (unusable on this toolchain; see above)
  sim/                   the framework, copied into each project and frozen
    engine.py            the timeline: engines, dependencies, SBUF exclusion
    collectives.py       what rejoining a split costs, by group shape
    memory.py            where the bytes are, and whether they fit
    api.py               the contract a cost model is written against
    runner.py            the executable the loop measures
  templates/
    project_readme.md    becomes the project's README: the schema, field by field
```

## Gotchas

- **The simulator refuses to run before `probe`.** That is the design, not a bug. It names the
  unmeasured values and tells you which command supplies them.
- **`run` refuses before `build` has frozen `sim/`.** Without the hashes, check (d) cannot tell
  whether a cost model was edited to make a floorplan look fast.
- **A bank is 24 GiB.** The most common wrong assumption, and it fails at 8192 tokens rather
  than at 128 — capacity is checked per workload for exactly that reason.
- **`probed.yaml` replaces lists wholesale.** `parser._deep_merge` replaces rather than merges
  lists, so a tier update has to be written as a complete tier list. `suite.merge_tier_patch`
  does this; a half-merged tier list would be much harder to reason about than a whole one.
- **`via:` is an explicit YAML field.** It was briefly inferred from the `reachable_from` prose
  by substring, and the YAML's `*through*` has asterisks — so NVMe was charged one leg instead
  of two and looked twice as cheap as it is. Prose is documentation; behaviour needs a field.
