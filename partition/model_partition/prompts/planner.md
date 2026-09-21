You are shaping how a large language model is cut into modules so each one can be
implemented, tested and deployed on its own.

## Model

- id: {{ model }}
- layers: {{ num_layers }}, hidden size: {{ hidden_size }}
- distinct layer signatures: {{ n_signatures }}
{% if layer_types %}- declared layer types: {{ layer_types_summary }}
{% endif %}- checkpoint: {{ checkpoint_bytes_h }}
- local GPU: {{ gpu_name }} — per-module budget {{ budget_h }}

## Current plan

`plan/partition_graph.yaml` has {{ n_modules }} partitioned modules in
{{ n_groups }} deduplicated implementation groups.

{{ module_table }}
{% if partition_prompt %}
## How this model should be partitioned

The operator asked for this specifically. It takes precedence over the general
guidance below; follow it unless it would produce an invalid plan, and say so in
`plan/rationale.md` if it would.

```
{{ partition_prompt }}
```
{% endif %}{% if failed_stage and failure_detail %}
## What failed

Stage: **{{ failed_stage }}**

```
{{ failure_detail }}
```

{% if failing_modules %}Failing modules:
{% for module in failing_modules %}- `{{ module }}`
{% endfor %}{% endif %}{% endif %}{% if review %}
## Review

A reviewer has already diagnosed this. Work from it.

```
{{ review }}
```
{% endif %}
## Your task

Edit `plan/partition_graph.yaml` in place so the partition is convenient for
writing and testing one kernel at a time. Judge convenience, not just fit — even a
model that fits whole on this GPU should be split where that makes a kernel easier
to develop and verify in isolation.

### Finding the right granularity

This is the judgement call, and it cuts both ways.

Too coarse and a module bundles several distinct computations — attention, its
normalization, routing, expert GEMMs — into one implementation. Then a numerical
mismatch has many possible causes and no way to isolate them, and the first
working version has to be correct in all of them at once.

Too fine and every boundary becomes a materialized tensor that has to be written
out and read back, which forecloses the fusions that make a kernel fast: an
epilogue folded into a GEMM, a norm fused into the operation that follows it, a
routing step folded into the expert call. Boundaries you introduce now are
boundaries a later optimization has to undo.

So aim for the coarsest module that still corresponds to *one* thing a kernel
author would sit down and write — and put a boundary wherever two computations
would be developed, tuned and verified separately even if they might later be
fused. Where they are genuinely the same piece of work, leave them together.

Also:

- Modules that share a `code_signature` share one extracted implementation, so
  never group layers whose structure differs.
- Modules must run comfortably inside the budget with room for activations.

## Rules

- Every `resident_bytes` (param + activation + kv) must stay under {{ budget_bytes }}.
- `submodules` must name real modules in the model (not tensors): the names are
  used as forward-hook targets. Current valid names are listed in
  `plan/valid_submodules.txt`.
- Each module's `inputs` must be produced by another module or be an entry
  tensor; every tensor must have exactly one producer; the graph must stay acyclic.
- Declare any new tensor you introduce in the `tensors` list.
- Set `composition: parallel` on a module whose submodules are alternatives rather
  than a pipeline — a group of MoE experts. The default is `sequential`.
- A module with no `submodules` is functional: pure tensor algebra with nothing to
  hook, so it is recorded in the plan but cannot be verified numerically. Use it
  only where no real submodule corresponds to the step.
- Do not touch anything under `trace/` — those artifacts are the reference and are
  expensive to rebuild.
- Leave `partitioned: false` nodes alone; they are out of scope for this run.

## If a stage failed

Common causes, in rough order of likelihood:

- A `submodules` entry names a tensor or a path that does not exist, so the module
  was never hooked and has no trace records.
- A module boundary sits mid-computation, so its recorded arguments include a
  value that cannot be replayed.
- A module's `resident_bytes` exceeds the budget.
- A layer group mixes two layer signatures, so one implementation cannot serve
  every layer in it.

Useful reading: `plan/valid_submodules.txt`, `reports/verify.json` (per-module max
absolute and relative error, cosine, pass fraction, worst index),
`reports/emulate.json` (end-to-end boundaries), `modules/index.yaml`.

When you are done, write a short rationale to `plan/rationale.md`: the boundaries
you chose, and for a repair, what was wrong and what you changed.
