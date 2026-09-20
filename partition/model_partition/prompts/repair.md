A model partition run failed verification. Diagnose it and fix the plan.

## Model

- id: {{ model }}
- layers: {{ num_layers }}, hidden size: {{ hidden_size }}
- per-module budget: {{ budget_h }} on {{ gpu_name }}

## What failed

Stage: **{{ failed_stage }}**

```
{{ failure_detail }}
```

{% if failing_modules %}Failing modules:
{% for module in failing_modules %}- `{{ module }}`
{% endfor %}{% endif %}

## What you can change

Only `plan/partition_graph.yaml`. The trace artifacts under `trace/` are
expensive to regenerate — do not edit or delete them. Changing the plan
invalidates the downstream stages automatically, and they will re-run.

## How to investigate

- `plan/partition_graph.yaml` — the current plan
- `plan/valid_submodules.txt` — module names that exist in the instantiated model
- `reports/verify.json` — per-module comparison metrics (max abs/rel error,
  cosine, pass fraction, worst index)
- `reports/emulate.json` — end-to-end boundaries and the judge's verdict
- `modules/index.yaml` — the deduplicated implementation groups

## Common causes

- A `submodules` entry names a tensor or a path that does not exist, so the
  module was never hooked and has no trace records.
- A module boundary was placed mid-computation, so its recorded arguments include
  a value that cannot be replayed.
- A module's `resident_bytes` exceeds the budget.
- A layer group mixes two different layer signatures, so one implementation
  cannot serve every layer in it.

Fix the root cause in the plan, keeping the graph valid: single producer per
tensor, acyclic, every input produced or an entry tensor, every module within
budget. Then write what you changed and why to `plan/rationale.md`.
