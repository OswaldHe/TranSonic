# The gate is the only thing that runs a candidate; the metric is read back from its verdict

The loop runs constraints and then, if they pass, runs the metric commands. The obvious metric command
here is the validator, and it is wrong twice over: it doubles every iteration's device time — four
minutes for a submodule, longer for four ranks under `torchrun` — and the two runs can disagree,
leaving an iteration accepted on one measurement and ranked on another. So the gate runs the validator
once, writes the latency into its verdict, and the metric command reads it back.

## Consequences

Iteration 0 has no verdict to read, because baseline capture runs the metric commands *without* the
constraints. That is not a wrinkle to work around — it aborted the first real run of stage 3 before
its first iteration. The stage's acceptance gate has already run the validator on precisely that code,
including the freshness checks proving the profile came from that run, so its verdict is copied to the
filename the loop reads. Nothing is synthesized and no extra device time is spent.

A metric a stage cannot produce is therefore fatal rather than merely absent: baseline capture requires
every declared metric. The single-rank stage must not declare the per-rank spread, because a submodule
has no ranks and nothing emits it.
