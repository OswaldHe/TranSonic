You are reviewing a failed stage of a model partition run. Diagnose it. Do not fix
it — another agent will make the change, working only from what you write.

## Model

- id: {{ model }}
- layers: {{ num_layers }}, hidden size: {{ hidden_size }}
- distinct layer signatures: {{ n_signatures }}
- per-module budget: {{ budget_h }} on {{ gpu_name }}
- run directory: `{{ run_root }}`

## What failed

Stage: **{{ failed_stage }}**

```
{{ failure_detail }}
```

{% if failing_modules %}Failing modules:
{% for module in failing_modules %}- `{{ module }}`
{% endfor %}{% endif %}
## The plan

{{ n_modules }} partitioned modules in {{ n_groups }} implementation groups.

{{ module_table }}
{% if partition_prompt %}
The operator asked for this partition specifically:

```
{{ partition_prompt }}
```
{% endif %}
## What to read

- `plan/partition_graph.yaml` — the plan, and `plan/valid_submodules.txt` for the
  module names that actually exist in the instantiated model
- `reports/verify.json` — per-module comparison metrics: max absolute and relative
  error, cosine similarity, pass fraction, index of the worst element, and whether
  a check was skipped
- `reports/emulate.json` — end-to-end module boundaries and the sampled output
- `modules/index.yaml` and each group's `meta.yaml` — which module ids share an
  implementation
- `modules/<group>/inference.py` — the implementation as it stands
- `modules/<group>/source.py` — the real implementation's source

Read the numbers before forming a view. A cosine near 1 with a poor pass fraction
is accumulated rounding; a cosine well below 1 is a wiring error. A module that
errored never ran at all, which is a different problem from one that ran and
disagreed.

## What to write

Write your review to `{{ review_path }}`. Change nothing else. Keep it short and
concrete — four sections, no preamble:

1. **What failed** — the specific module, tensor and number, quoted from the
   reports rather than paraphrased.
2. **Root cause** — your best single explanation, and what in the artifacts
   supports it.
3. **Where the fix belongs** — the partition plan (a boundary in the wrong place,
   a name that does not exist, a module over budget) or a module's `inference.py`
   (the arithmetic is wrong). Say which, and why it is that one and not the other.
4. **What to change** — the concrete edit you would make. If you are unsure
   between two causes, say so and give the cheapest way to tell them apart.

If the evidence does not support a diagnosis, say that instead of guessing. A
review that says "insufficient evidence, here is what to instrument" is more useful
than a confident wrong answer.
