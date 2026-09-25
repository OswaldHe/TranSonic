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
