# The agent chooses how to cut a module; only the module gate checks semantics

How to divide a module into per-rank work is a judgement about what that module computes — which
tensors are replicated, what a partial result is, whether a term belongs inside a rank's share or is
added after the ranks rejoin. A script encoding that would work for MoE and nothing else, so the
agent decides the cut, the submodule gate asks only module-agnostic questions, and the loose end is
tied off at the far end: the module gate requires the reassembled ranks to reproduce the module's
golden, at the module's own bar, faster than the bootstrapped module and within 10% of the submodule.
A wrong cut cannot pass that, and neither can a submodule that was fast because it did a quarter of
the work.

## Considered options

A deterministic submodule generator was the obvious alternative and is why this is written down: a
reader finding nothing that validates the submodule's semantics will assume the check was forgotten.
It was not. Every generator we could describe needed to know the architecture — expert parallelism for
MoE, head parallelism for attention, table sharding for Engram — which is the thing the pipeline is
supposed to be indifferent to.

## Consequences

The submodule gate can accept a cut nobody has verified, and does. Two things keep that bounded. The
reassembly recipe is checked arithmetically at that stage — the per-rank goldens, combined as the
agent declared, must reproduce the module's golden — which catches a dropped shard or a
double-counted shared path for the cost of a numpy sum. And the "no slower than 1.1x the submodule"
bound at the far end catches the cut that closes but is lazy, since three idle ranks cannot make a
module fast. Neither check knows what the module computes.
