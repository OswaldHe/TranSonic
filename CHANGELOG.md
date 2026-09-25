# Changelog

## Unreleased

### Added

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
