You are refining a model partition plan for AWS Trainium kernel development.

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

## Your task

Edit `plan/partition_graph.yaml` in place so the partition is convenient for
writing and testing one kernel at a time. Judge convenience, not just fit — even
a model that fits whole on this GPU should be split where that makes a kernel
easier to develop and verify in isolation.

Aim for:

- One module per distinct computation a kernel author would write separately.
  Attention variants, MoE routing, and expert GEMMs are usually separate work.
- Modules that share a `code_signature` share one extracted implementation, so
  never group layers whose structure differs.
- Modules small enough to run comfortably inside the budget with room for
  activations, but not so small that boundaries multiply without benefit.

## Rules

- Every `resident_bytes` (param + activation + kv) must stay under {{ budget_bytes }}.
- `submodules` must name real modules in the model (not tensors): the names are
  used as forward-hook targets. Current valid names are listed in
  `plan/valid_submodules.txt`.
- Each module's `inputs` must be produced by another module or be an entry
  tensor; every tensor must have exactly one producer; the graph must stay acyclic.
- Declare any new tensor you introduce in the `tensors` list.
- Do not touch anything under `trace/` — those artifacts are expensive to rebuild.
- Leave `partitioned: false` nodes alone; they are out of scope for this run.

When you are done, write a one-paragraph rationale to `plan/rationale.md`
explaining the boundaries you chose.
