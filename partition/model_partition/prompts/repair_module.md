A partition module's inference code does not reproduce its reference output. Fix
the implementation.

## Model

- id: {{ model }}
- layers: {{ num_layers }}, hidden size: {{ hidden_size }}
- per-module budget: {{ budget_h }} on {{ gpu_name }}

## What failed

```
{{ failure_detail }}
```

{% if failing_modules %}Failing modules:
{% for module in failing_modules %}- `{{ module }}`
{% endfor %}{% endif %}{% if review %}
## Review

A reviewer has already diagnosed this. Work from it.

```
{{ review }}
```
{% endif %}
## What you can change

Only `source.py` and `inference.py` of a failing module's group, under
`modules/<group>/`. Each group's `meta.yaml` lists which module ids it covers, so map
a failing module id to its directory there.

- `source.py` **is** the implementation: the module's classes, copied out of the
  model's own code. This is where the arithmetic lives, and editing it is what
  changes what runs.
- `inference.py` launches it: it imports `source.py`, constructs the class from the
  recorded config, loads the dumped weights into it and returns it. Fix it here when
  the module is built or wired wrongly rather than computed wrongly.

Do **not** touch:

- `verify.py` — it decides whether your implementation is correct. Loosening it
  defeats the purpose of the run.
- anything under `trace/` — those artifacts are the reference and are expensive
  to rebuild.
- `plan/partition_graph.yaml` — if the partition itself is wrong rather than the
  arithmetic, say so in `plan/rationale.md` and change nothing else.

## The contract your implementation must satisfy

```python
def build_module(config, weights, device="cpu", submodule=None):
    """Return a callable computing this module's forward."""
```

`config` is the model's resolved config; `weights` is keyed by original parameter
name. The returned callable is invoked with exactly the arguments the module
received during tracing, and every tensor it returns is compared against the dumped
output feature map.

`submodule` only matters for a group whose `composition` is `parallel` — a set of
MoE experts, where one recorded call exercises one expert. Leave it off the
signature if the module has a single submodule.

## How to investigate

- `reports/verify.json` — per-module metrics: max absolute and relative error,
  cosine similarity, pass fraction, and the index of the worst element
- `modules/<group>/source.py` — the implementation itself; add a print or an
  assertion to it and `verify.py` will run that code
- `modules/<group>/verify.py` — run it directly to iterate:
  `python verify.py --all-modules --all-samples`

## Common causes

- an operation reimplemented with the wrong axis, transpose, or scaling factor
- a normalization computed in the wrong dtype, or its epsilon dropped
- weights indexed by the wrong name, so a tensor is silently left at its
  initialized value
- for a group spanning several layers, feeding the wrong tensor from one layer to
  the next

Iterate until `verify.py` passes for every module in the group. Then write what
was wrong and what you changed to `modules/<group>/NOTES.md`.
