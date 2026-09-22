# Changelog

## Unreleased

### Added

- `autohelix bootstrap`: a loop for bootstrapping a Trainium NKI kernel for one
  module of a partition artifact. `init` materializes the module into a
  self-contained git repo (raw `.bin` tensors, a frozen PyTorch reference, and
  failing `source.py`/`inference.py` stubs); `run` loops an agent on it against a
  fixed six-part NKI constraint rather than a metric. Unlike `autohelix run`,
  every iteration is merged whether or not the constraint passes, the reviewer
  runs on every iteration, a failing constraint never ends the run, and a passing
  one does. See `bootstrap/README.md`.
- `bootstrap/preset.yaml` is the loop's config: one fixed, reviewable file read
  directly for every module repo, with nothing generated and no copy written into
  the repo. `bootstrap/prompt.md` is the agent's prompt template on the same
  terms.
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
