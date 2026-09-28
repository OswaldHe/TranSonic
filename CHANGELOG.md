# Changelog

## Unreleased

### Added

- `autohelix optimize`: a five-stage pipeline that makes one bootstrapped module fast on a
  single Trainium device. `init` projects the module's floorplan placement onto the one
  device available and records what that cost; `submodule` has an agent cut the module down
  to the part one NeuronCore runs, accepted by a seven-check module-agnostic gate; `run`
  loops on that rank under a per-iteration constraint schedule; `assemble` has an agent
  rejoin the ranks with `nki.collectives`, accepted by a nine-check gate that compares the
  reassembled output against the bootstrapped module's own recorded golden and holds it to
  two latency bounds; `run-full` loops on the whole module. See `optimization/README.md`.
- `autohelix optimize feedback`, a sixth and final stage. One agent reads every note and review
  both loops wrote — about 83,000 words for the MoE run — reconciles the places where they
  contradict each other, and writes `FEEDBACK.md`: what stopped the kernel getting faster, as a
  table of blockers each classified by who would have to fix it (**L0** a toolchain bug, where the
  documentation or API claims the device supports something it does not; **L1** a missing software
  feature in the compiler or programming interface; **L2** missing silicon). Every row carries the
  profiling scenario in detail, a runnable reproduction under `feedback-repro/`, the mechanism in
  plain words, links into the Neuron documentation and issue tracker, and a specific suggestion.
  The corpus is self-contradictory by construction — each note was written before the run ended, so
  iteration 2's theory is disproved by iteration 7 — and the prompt makes the reconciliation rules
  explicit: a measurement beats an inference, recency alone settles nothing, an unsettled conflict
  is filed as unsettled with the measurement that would settle it. The agent is asked to run the
  `domain-modeling` skill and write against `CONTEXT.md`, because "core", "rank", "unit" and
  "block" each mean at least two things in this toolchain and the report is read by people who were
  not on the run. It measures nothing and edits no repository, so it has no gate:
  `feedback.validate_report` checks the report's shape (a level in range, a reproduction file that
  exists, a link per row) and nothing checks whether a finding is true, because nothing could.
- Per-iteration constraint enforcement. A slot names its iterations with `at: 5` (one),
  `from`/`to` (a range) or `iterations: [1, 3, 5]` (a set), and carries `enforcement: hard|soft|off`.
  `hard` rejects the iteration; `soft` checks and records without rejecting, but costs the
  regression slack — a candidate that missed its constraint has to be correct and *strictly faster*
  than the best so far, which is what stops a soft constraint being no constraint; `off` is prompt
  guidance nothing checks. `soften_last` (default true) drops a hard slot's last iteration to soft,
  so one slot per iteration with `soften_last: false` is a constraint with no escape at all. The
  older `enforce: true|false` still works and means `hard|off`. Both loop stages read a schedule;
  stage 5's is empty by default because the first round has nothing to go on.
- `autohelix optimize rerun-full`, another round of the whole-module loop starting from the kernel
  the last round produced. The first round's notes are where the second round's ideas come from, so
  this starts from the previous round's *best accepted commit* rather than `HEAD` (regression slack
  can leave `HEAD` slower), re-measures that kernel's baseline with the gate rather than trusting a
  verdict describing different code, and recompiles the checkers from whatever the prose now says.
  The old round's history, candidate archive and slot verdicts move to `.autohelix/rounds/round-N/`;
  its `notes/` and `reviews/` stay where they are, because the next agent reading them is the point.
- Per-iteration soft constraints, in `optimization.yaml` as `iteration_constraints:` slots
  of prose. Early iterations are held to NKI alone, middle ones opened to torch, a pair left
  unconstrained for aggressive exploration, and the last ones returned to the disciplined
  regime — because ten iterations of "do whatever you like" converge on whatever the first
  one happened to try. A *constraint compiler* agent runs once before the loop and turns
  each slot's prose into a checker script the optimizing agent never sees; the prose goes
  into that iteration's prompt, and a candidate that does not follow it is rejected by a
  static check before the device run rather than after it.
- `optimization/recipe.py` checks a declared cut arithmetically without knowing what the
  module computes: the per-rank goldens the stage-2 agent dumped, combined the way it
  declared, must reproduce the module's recorded output. That catches a cut which drops a
  shard, double-counts a shared path or shards along the wrong axis — the class of error
  that would otherwise survive a whole optimization loop — while leaving *how* to cut the
  module entirely to the agent, so the pipeline is not shaped around one architecture.
- `optimization/projection.py` narrows an oversized floorplan placement to what one device
  holds, dropping activation-only factors before narrowing weight-partitioning ones. Needed
  because in the shipped DeepSeek schemes only 182 of 271 placements fit one device: all 43
  `.ffn`, all 43 `.attention` and `lm_head` span two, and both Engram tables span four. The
  projection is reported as a divergence from the ranked plan, with the per-core weight
  residency it costs.
- `source.py` and `inference.py` are carried into an optimization repo with their comments and
  docstrings removed. They were written by an earlier agent and are its claims about the hardware
  and the compiler, indistinguishable at a glance from established fact, and the next agent will
  design around a wrong one without testing it — this repository's own `floorplan/README.md` claim
  that NKI 0.6.0 has no collective primitive is the worked example. The frozen references,
  `vendor/` and `compat/` keep theirs: that code is the specification, not inference.
- On a constraint slot's last iteration the constraint is checked but no longer fatal — it runs
  *advisory*. A candidate that still misses it, passes the correctness gate and is *strictly faster*
  than the best so far is kept — by then the agent has had every iteration that slot allows, and
  discarding something correct and faster buys nothing. It forfeits the regression slack a compliant
  iteration gets (`acceptance.max_regression_pct`), so the escape is earned rather than taken.
- The gate's own first contact with reality. A real run of stage 2 reached 5 of 7 checks and failed
  on two checker bugs rather than on the candidate: a literal `"/"` used as a `str.join` separator
  was read as an absolute path, and 586 provenance findings came from a manifest shape the prompt
  had never specified. Both are fixed — the tensor record is now read keyed by path, keyed by name
  or as a list, and the prompt states the shape — and both now have tests, because a gate that
  enforces what its prompt does not state is a trap rather than a requirement.
- Also from that run: a rank's partial legitimately has the reference's flattened `(tokens, dim)`
  shape where the module's golden is `(batch, tokens, dim)`, so the reassembly check reshapes when
  the element count matches; and `NEURON_RT_VISIBLE_CORES` is how a single-core run pins itself to a
  core, so only a widened `NEURON_RT_NUM_CORES` is refused now.
- Hardened the places where the gates could be satisfied without the work. Collective calls are
  resolved through the file's import table, so `import torch.distributed as ncc; ncc.all_reduce(...)`
  no longer reads as the NKI collective. The reassembly check requires one distinct golden per rank
  and refuses the module's own recorded output as a shard, closing the path where a declaration
  reproduces the target without involving any rank. Non-finite marker values are dropped rather than
  parsed, since `nan` defeats every comparison that guards the pipeline. The preparation agents no
  longer own the manifest fields the gates read back. And the assembly's rank count comes from the
  recorded projection instead of a hardcoded four, which a one- or two-unit placement could never
  have satisfied.
- Fixed the candidate archive, which retained nothing: it read each iteration back out of its branch
  after the loop, and `Sandbox.remove_worktree` deletes that branch inside the iteration. Candidates
  are now captured in the last moment the worktree exists.
- Reviewers get 2000 seconds. Reviewing a kernel here means reading a few hundred lines of NKI
  against a reference and forming an adversarial view of whether the iteration is real, and a
  reviewer killed mid-read leaves the iteration with no verdict at all.
- In both loop stages the agent may edit `source.py` alone; `inference.py` is written once
  by the preparation agent, gated, then frozen, so one iteration's latency is comparable to
  another's. The metric is read back from the gate's verdict rather than measured again, so
  the gate is the only thing that runs the candidate.
- `autohelix floorplan`: a four-stage pipeline that decides how to distribute a
  partitioned model across a Trainium instance's hierarchy — which of the 64 logical
  NeuronCores on a trn2.48xlarge holds each module, how each is split and along which
  dimension, which memory tier holds its weights, and in what order. `probe` measures
  the primitives on real silicon; `init` materializes a project with a deterministically
  generated baseline; `build` runs two agent iterations that write the per-module cost
  models, gated by a nine-check invariant suite, then freezes the simulator by hashing it;
  `run` runs five one-hour iterations that edit only `floorplan.yaml`, gated by a hidden
  seven-check gate and ranked by four latencies with a per-metric 10% regression gate;
  `rank` hands the top three schemes to an agent that cannot see the simulator and appends
  the numbers afterwards. See `floorplan/README.md` and `docs/floorplan.md`.
- `floorplan/systems/*.yaml` describe the platform with `source:` and `confidence:` on
  every number, plus a `constraints_text` block carrying the hardware features that are
  not numbers (engine exclusions, tier reachability, torus non-uniformity) as numbered
  prose the build agent must implement and cite.
- `floorplan/probe/` measures matmul, elementwise, gather, DMA, PCIe and raw NVMe rates
  through NKI and writes `systems/probed.yaml` with provenance. The simulator refuses to
  run while any coefficient is unmeasured, so a missing measurement is never silently
  replaced by datasheet peak.
- `autohelix bootstrap`: a loop for bootstrapping a Trainium NKI kernel for one
  module of a partition artifact. `init` materializes the module into a
  self-contained git repo (raw `.bin` tensors, three frozen references — what to
  compute, how the reference was run, how it was judged — and failing
  `source.py`/`inference.py` stubs); `run` loops an agent on it against a
  fixed six-part NKI constraint rather than a metric. Unlike `autohelix run`,
  every iteration is merged whether or not the constraint passes, the reviewer
  runs on every iteration, a failing constraint never ends the run, and a passing
  one does. See `bootstrap/README.md`.
- `bootstrap/preset.yaml` is the loop's config: one fixed, reviewable file read
  directly for every module repo, with nothing generated and no copy written into
  the repo. `bootstrap/prompt.md` is the agent's prompt template on the same
  terms.
- The numerical bar is derived from the reference tensor's dtype and recorded in the
  repo's manifest, rather than fixed at bfloat16 globally, so a module whose
  boundary is fp8 or float32 is held to the tolerance its reference was actually
  accepted at.
- The constraint is hidden from the agent: the preset's `goal` states all six
  requirements in prose, and because the preset never lands in the repo, the
  command naming the checker is never in the worktree. The reviewer additionally
  performs an anti-reward-hacking read of each iteration.

### Fixed

- `optimization`: three ways the run's report misled a reader, all found by reading the report the
  first full MoE run produced. It called the best iteration's figure the collective overhead
  "against the 10% the gate allows" and printed `-55.7%`, sending a reader to hunt for a bound that
  was never blown — the bound is tight at assembly and slack afterwards, because the submodule is
  frozen there while the whole module goes on being optimized past it. The projection section
  spliced in `FLOORPLAN.md` with its own `#` title, so "What the projection gave up" appeared as a
  top-level finding of the run. And the constraint schedule cut each slot's prose at its first
  physical line, ending a table cell mid-sentence with no ellipsis; it now cuts on a word boundary,
  marks the cut, closes a bold run the cut split, and escapes a pipe that would have broken the row.
- `optimization`: a wrong number in `projection.py`, which claimed a literal one-device check "fails
  on 85 of the 89 placements worth optimizing". Counted from the scheme: 271 placements, 182 fit,
  and all 89 that do not are exactly the interesting ones — every `.ffn`, every `.attention`,
  `lm_head` and both Engram tables.
- `optimization`: the constraint compiler was told its checkers had 30 seconds while `run_checker`
  killed them at 120, so a compiler that budgeted honestly budgeted for the wrong number. The
  timeout is now interpolated into `CHECKER_CONTRACT` from the one place it is enforced.
- `optimization`: the loop prompt stated a literal "more than 5% above it is rejected", which is
  wrong the moment an operator changes `acceptance.max_regression_pct`. It now quotes the configured
  allowance, and says nothing when no metric gate names the metric.
- `optimization`: the config template put its `<FILL IN>` hints *inside* the `text: |` blocks of
  each constraint slot, where a `#` is not a YAML comment but literal prose. A config used as
  delivered therefore sent "`# <FILL IN: module-specific guidance for iterations 9-10, if any.>`" to
  the optimizing agent as part of its constraint — the first MoE run did exactly that for four
  iterations. The hints now sit outside the blocks, and a placeholder surviving into a slot's prose
  is reported as a warning before the loop starts.
- `floorplan`: seventeen findings from the review of #5, several of which changed the
  simulator's numbers materially. The cost-model corrections: `stage` now constrains the
  schedule instead of being an inert label, so pipeline depth can move a metric;
  `overlap_collectives` governs whether a collective blocks the unit's compute engines, rather
  than only contending with other collectives; weight bytes are divided only by
  weight-partitioning dimensions, so a batch or sequence split no longer understates residency;
  `p2p` charges its whole payload instead of a ring's `(N-1)/N` share; the probed `host_dram`
  and `nvme` rates are no longer re-scaled by the HBM DMA coefficient, which had inflated a
  measured 30 us NVMe read by over 300x; `peer_hbm` pays its torus distance via a new
  `weights.backing_device`; activations count as a concurrent peak per scope rather than a sum
  across sequentially-executing modules; and KV/activation peaks are keyed per shard, so two
  shards sharing a bank are charged separately.
- `floorplan`: the gate now verifies the *installed* framework's bytes as well as the project's.
  The framework is no longer copied into a project, because the simulator runs out of the
  package — a copy was read by the agent, hashed by the gate, and executed by nobody.
- `floorplan`: candidate plans are archived by the gate during the iteration (`--archive`),
  before AutoHelix deletes the worktree. The previous post-hoc git recovery guessed at branch
  names and fell back to `HEAD`, which filed the final plan under every iteration's metrics.
- `floorplan`: the schema rejects `collective: none` on a split that crosses logical cores
  (legal only for `batch`, or within one logical core), and bounds split factors by the
  dimension's real cardinality.
- `floorplan`: the build reviewer is held read-only by snapshot-and-restore, and a final
  `VERDICT: circumventing` prevents the freeze. The hardware read-only invariant fingerprints
  the effective model, not only its source YAML. `rank` parses an explicit `RANKING:` line and
  honours `--count`.

### Changed

- `CONTEXT.md` at the repo root is the project's glossary, and `docs/adr/` holds the decisions that
  were hard to reverse and surprising without their reasons. `optimization/` coined a lot of
  vocabulary and defined none of it; worse, "gate" was doing three jobs — the submodule checker, the
  whole-module checker, and AutoHelix's `acceptance.metric_gates`. The two gates are now always named
  apart, and three ADRs record who chooses how a module is cut, why a stage's gate is the only thing
  that runs a candidate, and what projecting a 16-device placement onto one device gives up.
- `optimization/gate.py` is `optimization/candidate.py`. It holds what the two gates share and is not
  itself a gate, which the glossary's own rule forbids; it runs a candidate repository and reads
  facts out of it. `gate.json` is unchanged — a verdict is written by a gate, so that name was right.
- "interval" and "range" no longer stand in for **slot** in the code, the tests, the config template
  or this file. Both are on the term's avoid list now.
- `floorplan`: batch size is a metric axis. The workload grid is now phase x context length x
  batch size — 1, 4, 8 and 32 samples — so sixteen metrics named
  `{phase}_{context}_b{batch}_ms`, each with its own 10% regression gate. Batch is an axis
  because the decisions reverse along it: a deep pipeline is pure cost at batch-1 decode, where
  one token is in flight and every stage boundary is a bubble, and nearly free at batch 32; the
  KV cache grows linearly with batch, so a residency choice that fits at batch 1 can be
  infeasible at batch 32; and a batch-32 decode step touches far more of the 384 routed experts
  than a batch-1 step. `ctx.tokens()` is divided by any batch split, `ctx.kv_tokens()` scales
  the cache with it, and `ctx.batch_per_shard()` reports the idleness of a split wider than the
  batch rather than pretending the work got cheaper. A tenth invariant checks that a larger
  batch costs more and that prefill scales roughly with it.
- `floorplan`: the ranking report gives an overall ranking *and* one per configuration, on
  `RANKING[<name>]:` lines, because a single order hides the case a deployment faces — one
  scheme can be right for long-context prefill at batch 32 and another for short-context decode
  at batch 1. The appendix compares the agent's call against the simulator's for every point, so
  a disagreement concentrated in one region of the grid becomes a specific, checkable claim.
- `floorplan`: removed dead code (`log2_ceil`, `shard_bytes`, `Schedule.last_index`,
  `Engine.peak_flops`, `WorkloadResult.metric_name`, an unused `BUILD_PRESET` and `WITHHELD`, and
  a no-op branch in the memory ledger) and corrected comments and docs left stale by the change
  from four metrics to sixteen.

## 0.1.1 — 2026-09-02

### Added

- KernelBench GPU optimization scaffolding for all 270 upstream tasks, including
  a CUDA evaluation harness and container configuration.
- A five-iteration KernelBench aggregate: 3.12× median best-so-far speedup with
  Claude Opus 4.8 on an H200.
- The per-task aggregate and plot source behind the 154-task AlgoTune result.
- Advanced loop examples for research, implementation task queues, AI workflow
  optimization, and nested AutoHelix runs.
- CI coverage for Python 3.11 and 3.12, plus tests and wheel verification in the
  gated PyPI publishing workflow.

### Fixed

- Preserve and scope-check changes when an agent creates commits inside its
  iteration worktree.
- Fail closed when an agent reports an infrastructure error, without merging
  changes or consuming the iteration number.
- Handle Codex `turn.failed` events and valid non-object JSON output.
- Reject baselines that omit configured metrics.
- Kill constraint and metric subprocess groups when a run is interrupted.

### Changed

- Removed the older bin-packing and budget-demo examples in favor of examples
  organized around reusable loop patterns.
