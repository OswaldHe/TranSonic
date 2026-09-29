# Projecting a placement onto one device diverges from the ranked plan, and we run anyway

The ranked floorplan targets a 16-device trn2.48xlarge; development happens on a one-device
trn2.3xlarge. Of 271 placements in the DeepSeek V4.1 Flash schemes only 182 fit one device — all 43
`.ffn`, all 43 `.attention` and `lm_head` span two, and both Engram tables span four — so a literal
"must fit one device" check refuses every module worth optimizing. Placements are therefore narrowed
to what one device holds, and the narrowing is reported as a divergence from the plan rather than as a
restatement of it.

## Consequences

The latencies this pipeline produces describe a plan the search ranked second, and for `.ffn` that is
precise: narrowing `expert x8` to `expert x4` doubles per-core expert residency, which is exactly the
trade the floorplan's own report priced and rejected — it chose the wider split because it lowers
per-bank weight residency and decode is bank-bandwidth bound.

So a fast kernel here is a fast *single-device* kernel and is not evidence about the 16-device
deployment. That is a legitimate thing to want and a misleading thing to forget, which is why the
projection's cost is stated in the repository the agent reads (`FLOORPLAN.md`) and at the top of the
report rather than in a footnote. Carrying a kernel back to the target means re-cutting it at the
planned width and re-measuring: a kernel tuned for twice the weight per core is not automatically the
right kernel for half of it.

`floorplan.on_oversized: error` refuses instead of projecting, for a run that wants fidelity over
coverage. It leaves only the hyper-connection modules, the norms and `embed`.
