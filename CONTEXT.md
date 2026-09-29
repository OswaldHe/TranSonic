# AutoHelix

AutoHelix puts an AI agent in a verified improvement loop: each iteration is isolated, measured, and
kept only when it passes. Four pipelines build on that loop to take a model from a checkpoint to fast
kernels on AWS Trainium.

**Coverage.** This glossary currently covers the loop's shared vocabulary and the `optimization/`
pipeline. The `partition/`, `bootstrap/` and `floorplan/` pipelines have their own coined terms that
are not yet resolved here; add them as they come up, or split to a `CONTEXT-MAP.md` if the four
vocabularies stop overlapping.

## Language

### The loop

**Iteration**:
One attempt by an agent to improve the code, isolated so it can be discarded whole.
_Avoid_: round, pass, attempt, run

**Candidate**:
The code as an iteration left it, before anything has decided whether to keep it.
_Avoid_: submission, proposal

**Constraint**:
A command that must succeed for an iteration to be kept. Says nothing about how good the work is.
_Avoid_: test, check, validation

**Metric**:
A number an iteration is ranked by. Separate from a constraint: a constraint is pass/fail, a metric
is better/worse.
_Avoid_: score, benchmark, measurement

**Metric gate**:
An allowance on how far a metric may regress against the best iteration so far before the iteration
is rejected. Inherited from the loop's config (`acceptance.metric_gates`) and **not** a gate in the
sense below.
_Avoid_: regression gate, threshold

**Reviewer**:
A second agent that reads an iteration and writes an opinion, changing nothing.
_Avoid_: critic, judge, auditor

### Modules and ranks

**Module**:
One slice of a partitioned model, the unit everything downstream is built around — traced, verified,
bootstrapped, placed and optimized as a whole.
_Avoid_: layer, block, component, subgraph

**Submodule**:
The part of a module that one rank computes. Structurally identical across ranks.
_Avoid_: fragment, piece, partial module, sub-kernel

**Rank**:
One participant in a distributed execution of a module, and the work it holds. On this dev host a
rank is one logical NeuronCore.
_Avoid_: worker, core, device, process

**Shard**:
One rank's share of a tensor or of a module's weights.
_Avoid_: slice, chunk, partition (as a noun), split

**Golden**:
A recorded output that a candidate must reproduce. Per-rank goldens belong to submodules; the module
golden is the whole module's recorded output.
_Avoid_: expected output, ground truth, reference (which names the frozen source code instead)

**Reference**:
The frozen source code carried in from the artifact that says what a module computes and what counts
as matching. Read, never run by a candidate.
_Avoid_: golden, spec, baseline

**Validator**:
The program that loads a module's recorded tensors, runs a candidate on the device, measures its
latency and compares the result against the golden. Written once, then frozen.
_Avoid_: harness, runner, test, benchmark

**The numerical bar**:
The five pinned constants a validator judges a result by. Derived from a module's own recorded
output, never chosen by an agent.
_Avoid_: tolerance, threshold, accuracy target

### Gates

**Gate**:
A hidden program that decides whether a candidate repository is acceptable, by running its validator
once and answering a fixed list of checks. Two exist, and a bare "gate" is ambiguous between them —
**always name which one**.
_Avoid_: checker, validator, test suite, "the gate" unqualified

**Submodule gate**:
The gate that accepts a single-rank repository. Asks only module-agnostic questions, so it can accept
a cut it does not understand.
_Avoid_: stage-2 gate, preparation gate

**Module gate**:
The gate that accepts a whole-module repository. The only place in the pipeline that checks
semantics: it requires the reassembled output to be the module's golden, and holds the result to both
latency bounds.
_Avoid_: whole-module gate, final gate, stage-4 gate

**Check**:
One named question a gate answers about a candidate, passing or failing on its own.
_Avoid_: test, assertion, rule

**Verdict**:
What a gate concluded about one candidate: every check's outcome, and the measurement it took.
_Avoid_: result, report, outcome

### The optimization pipeline

**Projection**:
A module's planned placement narrowed to what a single device holds. A projection that changes the
plan is a divergence from it, not a restatement of it.
_Avoid_: mapping, translation, scaling, adaptation

**Reassembly recipe**:
The declared rule for combining every rank's golden back into the module's golden. A claim the
pipeline checks arithmetically, which is what lets the cut itself go unchecked.
_Avoid_: merge rule, combine step, join

**Slot**:
A run of consecutive iterations governed by one constraint written in prose. Slots do not overlap,
and an iteration belongs to at most one.
_Avoid_: interval, range, phase, stage, window, bucket, band

**Constraint schedule**:
The ordered set of slots for one loop stage.
_Avoid_: plan, policy, curriculum

**Constraint compiler**:
The agent that turns each slot's prose into a checker, once, before the loop starts. Never sees the
kernels its checkers judge.
_Avoid_: generator, translator

**Checker**:
The script compiled from one slot's prose, which decides whether a candidate followed that slot.
Distinct from a gate: a checker reads one file statically and judges compliance, a gate runs the
candidate and judges acceptability.
_Avoid_: gate, validator, linter

**Enforcement**:
What a violation of a slot's constraint costs on one iteration: `hard` rejects it, `soft` records it
and falls back to requiring a strict improvement, `off` does not check at all. A property of an
iteration, not of a slot: a hard slot's last iteration is soft by default.
_Avoid_: strictness, severity, level (which names a blocker's L0-L2 instead)

**Advisory**:
The `soft` enforcement of a checker run: it records its verdict and never rejects the iteration on
its own. Named this way at the invocation (`slotcheck --advisory`); `soft` is the word in the config.
_Avoid_: warning, non-blocking, lenient

**Round**:
One pass of a loop stage over its whole iteration budget. A second round starts from the first
round's best kernel with new constraints, and keeps its notes.
_Avoid_: run, pass, attempt, retry

**Custody**:
Holding the manifest fields a gate reads back — the bar, the golden, the bounds — outside the
repository an agent writes in, and restoring them afterwards. Without it an agent writes the terms it
is judged by.
_Avoid_: protection, locking, freezing

**Stripping**:
Removing an earlier agent's comments and docstrings from code carried into a new repository, so its
claims do not read as established fact to the agent that reads them next.
_Avoid_: cleaning, minifying, sanitising

**Drift**:
A recorded thing no longer matching the live one: a compiled checker against its slot's current
prose, or a workspace's projection against the scheme it was built from.
_Avoid_: staleness, mismatch, divergence (which names the projection's departure from the plan)

**Attic**:
Where a failed preparation attempt's repository is kept instead of deleted.
_Avoid_: archive, backup, trash

**Candidate archive**:
Where every iteration's kernel is kept with its metrics, accepted or not, so a rejected experiment is
still readable afterwards.
_Avoid_: history, attic

**Corpus**:
Every note and review both loops wrote, read as one body of evidence. Self-contradictory by
construction, because each file was written before the run ended.
_Avoid_: logs, history, transcript

**Blocker**:
Something that stopped a kernel getting faster and that this project cannot fix for itself, named by
who would have to: a toolchain **bug** (L0), a missing **software feature** (L1), or missing
**hardware** (L2).
_Avoid_: issue, limitation, problem, bottleneck (which names where the time goes, not who owns it)
