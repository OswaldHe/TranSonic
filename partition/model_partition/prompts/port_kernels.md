This model's own inference code does not run on this machine's GPU. Port the parts
that do not, so the model can be traced here.

## Model

- id: {{ model }}
- layers: {{ num_layers }}, hidden size: {{ hidden_size }}
- vendor code: `{{ snapshot }}` (read-only — the reference implementation)

## This GPU

```
{{ device }}
```

## What failed

```
{{ failure_detail }}
```

{% if review %}
## Review

A reviewer has already diagnosed this. Work from it.

```
{{ review }}
```
{% endif %}
## What you can change

Only files under `compat/`. Each is a small python module that replaces what will not
run, and nothing else:

```python
# compat/sparse_attn_sm89.py
"""`sparse_attn` asks for 141 312 bytes of shared memory; this card allows 101 376."""

import torch


def _sparse_attn(q, kv, attn_sink, topk_idxs, ...):
    ...


def apply(vendor, device):
    """Replace what does not run here; return the names replaced."""
    vendor.sparse_attn = _sparse_attn
    return ["sparse_attn"]
```

`vendor` is the imported entry module, so assigning to a name on it replaces that
function or class everywhere the model uses it. `device` describes the card, so branch
on what it supports rather than on its name. Patches are applied in sorted filename
order, before the model is constructed.

Do **not** touch:

- the vendor snapshot — it is the specification you are porting *from*, and the diff
  between it and your patch is what a reader checks;
- `plan/`, `trace/`, `modules/*/verify.py` — the plan, the reference and the gate.

## The bar

**Same arithmetic.** You are changing how a computation is expressed, not what it
computes. A patch that reorders a reduction is fine; one that drops a term, a mask or
a scale is not. The trace taken after this becomes the reference every later stage is
measured against, so an error here is an error nothing downstream can see.

**Same precision where the card supports it.** Keep the original's dtypes when this
GPU has them. Where it does not, move to the closest it does — fp4 to fp8 to bfloat16,
in that order — and say so in a comment at the top of the patch, naming the tensor and
the reason. Accumulate in float32 wherever the original does.

**Prefer the framework to a new kernel.** A `torch` implementation of a fused kernel is
slower and still correct, and correctness is what tracing needs; speed is what the
extracted modules are for afterwards. Write a custom kernel only when the framework
cannot express the computation.

**One patch per thing you replace**, named after it, with a docstring saying what the
original asked for and what this card allows.

## How to check yourself

The vendor's code is in front of you: read the function you are replacing and match it
term for term, including the details that look incidental — a finite lower bound instead
of `-inf` to keep an all-masked row from becoming NaN, an index of `-1` marking an empty
slot, a scale applied before rather than after a cast. Those are the parts a rewrite
silently gets wrong.

Then check it runs:

```bash
python -c "
import sys; sys.path.insert(0, '{{ snapshot }}/inference')
import model
"          # the vendor module imports
```

The loop re-runs the trace after you return. If the model still does not run, the next
iteration hands you the new failure.
