# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The files `autohelix bootstrap init` writes into a module repo.

The stubs are deliberately incomplete: the first iteration must fail the gate, because a
loop that starts green has nothing to bootstrap. What they *do* carry is the shape of the
answer — the kernel's name and decorator, the pinned tolerance constants, and the table of
which `.bin` holds which tensor at which dtype and shape. That information is mechanical,
error-prone to rediscover, and not what the loop is for; the kernel and the validator are.

Everything raised as `NotImplementedError` is the agent's work.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bootstrap.nki_checker import CEILING_NAME, PINNED_TOLERANCE

if TYPE_CHECKING:
    from bootstrap.materialize import Materialized, TensorRecord


GITIGNORE = """\
# Run state: config, manifest, notes, logs. Never committed.
.autohelix/

# Profiling output, produced by inference.py on every run.
*.neff
*.ntff
*.ntff.json
neuron_profile/
log-neuron-cc.txt
__pycache__/
*.pyc
"""


SOURCE_STUB = '''\
"""The NKI kernel for {module_id}.

{summary}

This is a stub: `kernel` has the shape the loop requires — a single top-level function
under `@nki.jit` — and computes nothing. Implementing it is the task.

NKI only. torch, numpy and scipy may not appear in this file, not even in a comment;
`reference_torch.py` holds the PyTorch specification to read instead, and it may not be
imported from here.

Reference: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/index.html
"""

import nki
import nki.language as nl


@nki.jit
def kernel({params}):
    """{one_line}

    Inputs, in order:
{param_docs}

    Returns the group's output: {returns}.
    """
    raise NotImplementedError(
        "the kernel is not implemented yet: compute {module_id} in NKI"
    )
'''


INFERENCE_STUB = '''\
"""Run and validate the NKI kernel for {module_id}.

Loads the recorded tensors out of `tensors/`, runs `source.kernel` on the device through
`torch_neuronx.trace`, profiles it, and compares the result against the recorded
reference.

Four things this file must end up doing, none of which it does yet:

1. load every tensor in TENSORS from its `.bin`, at the dtype and shape given there
2. trace and run `kernel` with `torch_neuronx.trace`, leaving a `.neff` and a `.ntff`
3. run `neuron-explorer` on those, read `total_exec_time`, and print it as
   `##autohelix[latency_ms=...]`
4. compare against the reference at the tolerance below, print the worst element as
   `##autohelix[max_abs_err=...]`, and print `##autohelix[passed=1]` exiting 0 only when
   every element is within tolerance *and* none exceeds MAX_ABS_ERR

The tolerances are the ones the reference was recorded at. They are the bar, not a
suggestion, and raising any of them is not an available way to pass.
"""

import json
import subprocess
import sys
from pathlib import Path

import torch
import torch_neuronx

from source import kernel

HERE = Path(__file__).resolve().parent

# The numerical bar. Do not change these values.
RTOL = {rtol:g}
ATOL = {atol:g}
MIN_COSINE = {min_cosine:g}
MIN_PASS_FRACTION = {min_pass_fraction:g}
#: A hard ceiling: no single element may be off by more than this, whatever the pass
#: fraction says. Report the worst element as ##autohelix[max_abs_err=...] and fail on it.
MAX_ABS_ERR = {max_abs_err:g}

#: The profiler this reads its latency from, and the field it reads.
NEURON_EXPLORER = "neuron-explorer"
LATENCY_FIELD = "total_exec_time"

#: Every recorded tensor: (name, file, dtype, shape). The bytes are raw little-endian,
#: C-contiguous, with no header — dtype and shape here are the whole description.
TENSORS = [
{tensor_rows}
]

#: Non-tensor arguments the recorded forward was called with.
SCALAR_ARGS = {scalar_args!r}


def load(name):
    """The named tensor, from its `.bin`.

    Read the file listed for `name` in TENSORS and reinterpret its bytes at the recorded
    dtype and shape. These files are the ground truth: nothing here may generate,
    randomize or substitute data.
    """
    raise NotImplementedError("load the recorded tensors from tensors/*.bin")


def run_on_device(inputs):
    """Trace `kernel` with torch_neuronx and run it, leaving a .neff and a .ntff behind."""
    raise NotImplementedError("trace and run the kernel with torch_neuronx.trace")


def measure_latency(neff, ntff):
    """Milliseconds, from neuron-explorer's total_exec_time for this run."""
    raise NotImplementedError("read total_exec_time out of neuron-explorer")


def compare(actual, expected):
    """Whether `actual` matches the recorded reference at the pinned bar.

    All five constants apply: elementwise closeness at RTOL/ATOL over at least
    MIN_PASS_FRACTION of elements, cosine similarity of at least MIN_COSINE, and no single
    element off by more than MAX_ABS_ERR.
    """
    raise NotImplementedError("compare against the reference at RTOL/ATOL/MIN_COSINE")


def main():
    raise NotImplementedError("wire load -> run_on_device -> measure_latency -> compare")


if __name__ == "__main__":
    sys.exit(main())
'''


README = """\
# Bootstrap repo: `{module_id}`

{summary}

Generated by `autohelix bootstrap init` from partition artifact group
`{group}`, recorded pass `{sample_id}#{step}` (invocation {call_index}).

## The computation

The kernel must reproduce this whole group, end to end — every submodule in the chain,
not just the last one:

{chain}

## Frozen references

Carried in from the artifact, verbatim. Read them; you may not import, call, open or edit
any of them, and an edit is reverted before your work is judged.

| file | what it tells you |
|---|---|
| `reference_torch.py` | **what to compute** — the original PyTorch implementation of this module. The specification. |
| `reference_inference.py` | **how it was run** — the artifact's own launcher: load weights and inputs, run, time, compare against the dumped output, report `##autohelix[...]` metrics. The shape your `inference.py` has to take. It cannot run here (it needs the harness runtime and the artifact's `calls.json`), so take the structure, not the imports. |
| `reference_numerics.py` | **how it was judged** — `Tolerance.for_dtype` is where the four tolerance constants come from and `compare_outputs` is how they are applied. Reimplement this self-contained in `inference.py`; do not import it. |
| `compat/*.py` | **what the reference was actually recorded with.** The GPU that produced these feature maps could not run the vendor's version of some kernel, so this replaced it before tracing. For any name one of these rebinds, it — not `vendor/kernel.py` — is the semantics the reference has. Check here first. |
| `vendor/kernel.py` | **the primitives** `reference_torch.py` imports but does not contain — `act_quant`, `fp8_gemm`, `fp4_act_quant`, `fp4_gemm`, `sparse_attn`. The arithmetic bottoms out here; read it rather than inferring quantization or masking from the tensors. |
| `vendor/model.py` | **the whole model** the slice came from, for when the slice does not say how a value reaching it was produced. |

`vendor/` is the artifact's own package and imports cleanly with
`sys.path.insert(0, "vendor")`; each `compat/*.py` exposes `apply(vendor_module, device)` to
rebind what it replaced. So the reference can be *run*, not just read — build a host-side
torch version in a scratch file and check intermediates against `tensors/` before committing
to NKI. The compat `sparse_attn` runs on CPU; the tilelang fp8/fp4 primitives need an NVIDIA
GPU this machine does not have and raise `No registered target detector found an available
target`, so reimplement those in torch if you want to execute that part.

Neither `source.py` nor `inference.py` may import or open any of it.

`config.json` is the config the module was built from; `MODULE.md` is the artifact's own
description of its pre- and post-conditions.

{scalar_section}
## Tensors

Raw little-endian bytes, C-contiguous, no header. The dtype and shape below are the
complete description of each file; there is no sidecar to read.

| tensor | role | file | dtype | shape | bytes |
|---|---|---|---|---|---|
{tensor_table}

Total: {total_mib:.1f} MiB across {tensor_count} file(s).

`input` is what flows into the group; `reference` is the output the recorded forward
produced and what your result is compared against. `weight` entries are parameters;
`buffer` entries are non-persistent state the module registered.

{compaction_section}`state` entries are cross-module state: tensors the recorded forward read off a shared
object rather than through its arguments, so they reach your kernel as arguments instead.
Read them with care, because they are a *snapshot of that object* taken as this group was
entered, not a list of what it reads:

- An entry may be something this group **produces** rather than consumes. Computing it and
  then finding the recorded answer sitting in an argument is the trap — using that value
  instead of computing it reproduces nothing, and the review is looking for exactly this.
- An entry may be **left over from an earlier pass**, because the shared object is not reset
  between recorded samples. A shape that does not fit this module's own is the tell.

`reference_torch.py` is what settles both: whichever names it reads before writing are
inputs, and the rest are not yours to use.
{notes_section}
## The numerical bar

Declare these five in `inference.py` as module-level number literals, under exactly these
names and with exactly these values. They are derived from this module's own recorded output
— from the coarsest number format anywhere in its chain, and from the largest reference
value — so they are these numbers for this module and not for every module. Do not change
any of them in either direction, and do not compute them from an expression.

`MAX_ABS_ERR` is a hard ceiling: **no single element** may differ from the reference by more
than it, whatever the pass fraction says. Print the worst element as
`##autohelix[max_abs_err=...]` and fail when it exceeds the ceiling.

```python
{tolerance_block}
```

`reference_numerics.py` is where they come from and how they are applied.

## Your job

Write `source.py` (the NKI kernel) and `inference.py` (the validator that loads these
tensors, runs the kernel on the device, profiles it, and checks it against `reference`).
Both start as stubs that fail on purpose.

Only those two files are yours. Everything else here is frozen and edits to it are
reverted.

```bash
python inference.py        # what the gate runs
```
"""


def _one_line(result: "Materialized") -> str:
    kind = result.module_id.rsplit(".", 1)[-1]
    return f"Compute {result.module_id} ({kind}) in NKI."


def _summary(result: "Materialized") -> str:
    """One sentence naming the module, read from the artifact rather than assumed.

    The model and the composition both come from what was published: hardcoding either
    would put a false statement in the file the agent reads as its specification, and
    "sequential" in particular is wrong for a parallel group — an expert shard runs beside
    its siblings, not after them.
    """
    names = (" -> " if result.composition == "sequential" else " | ").join(
        s for s in result.submodules if s
    )
    model = f" of {result.model}" if result.model else ""
    return (
        f"One partition module{model}: `{result.module_id}`, a {result.composition} "
        f"group over {len(result.submodules)} submodule(s) ({names})."
    )


def _bar(result: "Materialized", name: str) -> float:
    """One of the four tolerance constants for this repo.

    The stub, the README and the gate all read it from the same place, so they cannot
    disagree about the bar.
    """
    return result.tolerance.get(name, PINNED_TOLERANCE.get(name, 0.0))


def _kernel_params(result: "Materialized") -> list["TensorRecord"]:
    """The tensors the kernel takes: the input and every weight, reference excluded."""
    return [t for t in result.tensors if t.role != "golden"]


def _param_name(record: "TensorRecord") -> str:
    """The kernel parameter this tensor arrives as.

    A lone input reads better as `hidden_states`; where a module takes several — a mask, a
    rotary embedding alongside the hidden state — each keeps its recorded name, because
    collapsing them all to one label would produce a signature with repeated parameters.
    """
    from bootstrap.materialize import _safe_name

    return "hidden_states" if record.name == "input" else _safe_name(record.name)


def render_source_stub(result: "Materialized") -> str:
    params = _kernel_params(result)
    references = [t for t in result.tensors if t.role == "golden"]
    param_docs = "\n".join(
        f"      {_param_name(p)}: {p.dtype}{p.shape}"
        + (f"  # {p.note}" if p.note else "")
        for p in params
    )
    returns = ", ".join(f"{r.dtype}{r.shape}" for r in references) or "unknown"
    if len(references) > 1:
        returns = f"{len(references)} tensors, in order: {returns}"
    return SOURCE_STUB.format(
        module_id=result.module_id,
        summary=_summary(result),
        one_line=_one_line(result),
        params=", ".join(_param_name(p) for p in params),
        param_docs=param_docs or "      (none recorded)",
        returns=returns,
    )


def render_inference_stub(result: "Materialized") -> str:
    rows = "\n".join(
        f'    ("{t.name}", "{t.file}", "{t.dtype}", {tuple(t.shape)!r}),'
        + (f"  # {t.note}" if t.note else "")
        for t in result.tensors
    )
    return INFERENCE_STUB.format(
        module_id=result.module_id,
        tensor_rows=rows,
        scalar_args=result.scalar_args,
        max_abs_err=_bar(result, "MAX_ABS_ERR"),
        rtol=_bar(result, "RTOL"),
        atol=_bar(result, "ATOL"),
        min_cosine=_bar(result, "MIN_COSINE"),
        min_pass_fraction=_bar(result, "MIN_PASS_FRACTION"),
    )


def _compaction_section(result: "Materialized") -> str:
    """What `init --compact-tables` did, stated where the agent cannot miss it."""
    rows = [t for t in result.tensors if "COMPACTED" in (t.note or "")]
    if not rows:
        return ""
    lines = [
        "**This module's lookup table is compacted, and one input is remapped onto it.**",
        "",
    ]
    for t in rows:
        lines.append(f"- `{t.name}` holds {t.shape[0]:,} rows here. {t.note}")
    lines += [
        "",
        "The table does not fit host or device memory at its recorded size, so the rows the",
        "recorded pass indexes were read out of the checkpoint and the integer input was",
        "replaced by positions in the compacted table. Every value gathered is the value the",
        "recorded forward gathered, and `reference` is unchanged — so the comparison is the",
        "real one. What this repo does *not* exercise is the model's full address space.",
        "",
        "Write the kernel against the shapes in the table above, not against the",
        "`num_embeddings` in `config.json`.",
        "",
    ]
    return "\n".join(lines) + "\n"


def render_readme(result: "Materialized") -> str:
    table = "\n".join(
        f"| `{t.name}` | {t.role} | `{t.file}` | {t.dtype} | {tuple(t.shape)} | {t.nbytes:,} |"
        for t in result.tensors
    )
    chain = "\n".join(f"{n}. `{s}`" for n, s in enumerate(result.submodules, start=1) if s)

    notes = {t.note for t in result.tensors if t.note}
    notes_section = ""
    if notes:
        notes_section = "\n" + "\n".join(f"Note: {n}." for n in sorted(notes)) + "\n"

    scalar_section = ""
    if result.scalar_args:
        lines = "\n".join(
            f"- `{s.get('submodule')}` "
            + (f"keyword `{s['keyword']}`" if "keyword" in s else f"positional {s['position']}")
            + f" = `{s['value']!r}`"
            for s in result.scalar_args
        )
        scalar_section = (
            "## Non-tensor arguments\n\n"
            "Values the recorded forward was called with, reproduced in `inference.py` as\n"
            f"`SCALAR_ARGS`:\n\n{lines}\n\n"
        )

    tolerance_block = "\n".join(
        f"{name} = {_bar(result, name):g}"
        for name in (*PINNED_TOLERANCE, CEILING_NAME) if name in result.tolerance
    )
    return README.format(
        tolerance_block=tolerance_block,
        module_id=result.module_id,
        summary=_summary(result),
        group=result.group,
        sample_id=result.sample_id,
        step=result.step,
        call_index=result.call_index,
        chain=chain or "(none recorded)",
        scalar_section=scalar_section,
        tensor_table=table,
        compaction_section=_compaction_section(result),
        total_mib=result.total_bytes / (1 << 20),
        tensor_count=len(result.tensors),
        notes_section=notes_section,
    )
