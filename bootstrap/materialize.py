# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn one module of a partition artifact into a standalone git repo to bootstrap in.

A published module directory is not self-contained: it imports the harness runtime, the
vendor's modeling package and the compatibility patches, and it reads its weights out of
a 475 GiB checkpoint. None of that can survive the NKI constraint, which wants two files
and a directory of raw tensors.

So this materializes the module instead of copying it. Every tensor the recorded call
touched — the group's input, its reference output, and each weight, whether it was dumped
beside the module or read from the checkpoint — is written out as raw little-endian bytes
with its dtype and shape recorded in a manifest. The original implementation is kept as
`reference_torch.py`, frozen, because the agent needs to read what the computation *is*
and is forbidden from importing torch to do it.

The result is a git repo whose first commit is a stub that deliberately fails the gate:
bootstrapping is the work of making it pass.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bootstrap import templates
from bootstrap.preset import dump_manifest

#: Where the tensors go inside the repo, and where the gate expects to find them.
TENSOR_DIR = "tensors"

#: Run state lives here: gitignored, outside the agent's editable scope, and not copied into
#: the iteration worktree by `Sandbox.prepare_worktree` (which carries only notes, logs and
#: observations). The manifest is the only thing `init` writes here — the config and the
#: prompt template are fixed files in the package, read directly.
#:
#: Must agree with `nki_checker.MANIFEST_REL`, which is what the gate searches for.
STATE_DIR = ".autohelix"
MANIFEST_REL = f"{STATE_DIR}/bootstrap/manifest.json"

#: Where the gate drops its verdict inside the iteration worktree. The reviewer runs in
#: that same worktree and reads it there; it dies with the worktree, so one iteration's
#: verdict never leaks into the next agent's context except through the review.
CHECKS_REL = f"{STATE_DIR}/nki_checks.json"

#: complex64 has no NKI equivalent and no clean raw representation for a traced kernel, so
#: a complex tensor is written as its real view: the same bytes, one extra trailing
#: dimension of 2, and a README that says so.
COMPLEX_SPLIT_NOTE = "complex64 written as float32 with a trailing [real, imag] dimension"


@dataclass
class TensorRecord:
    """One materialized tensor, as the manifest records it."""

    name: str
    role: str
    file: str
    dtype: str
    shape: list[int]
    nbytes: int
    sha256: str
    required: bool = False
    source: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "name": self.name, "role": self.role, "file": self.file, "dtype": self.dtype,
            "shape": self.shape, "nbytes": self.nbytes, "sha256": self.sha256,
            "required": self.required,
        }
        if self.source:
            payload["source"] = self.source
        if self.note:
            payload["note"] = self.note
        return payload


@dataclass
class Materialized:
    """What was written, so the caller can report it without re-reading the repo."""

    repo: Path
    group: str
    module_id: str
    sample_id: str
    step: int
    call_index: int
    tensors: list[TensorRecord] = field(default_factory=list)
    scalar_args: list[dict[str, Any]] = field(default_factory=list)
    submodules: list[str] = field(default_factory=list)
    total_bytes: int = 0
    tolerance: dict[str, float] = field(default_factory=dict)
    model: str = ""
    composition: str = "sequential"


class MaterializeError(RuntimeError):
    """The artifact cannot supply what was asked for."""


def _safe_name(name: str) -> str:
    """`layers.0.attn.wq_a.weight` -> `layers_0_attn_wq_a_weight`."""
    return "".join(c if c.isalnum() else "_" for c in name).strip("_")


def _raw_bytes(tensor: Any) -> tuple[bytes, str, list[int], str]:
    """A tensor's little-endian payload, plus the dtype and shape to read it back with.

    Flattened before the uint8 view because fp8 and bfloat16 have no numpy equivalent, and a
    1-D tensor of any dtype reinterprets as bytes.
    """
    import torch

    note = ""
    if tensor.dtype in (torch.complex64, torch.complex128):
        tensor = torch.view_as_real(tensor.contiguous())
        note = COMPLEX_SPLIT_NOTE
    tensor = tensor.detach().cpu().contiguous()
    shape = list(tensor.shape)
    dtype = str(tensor.dtype).removeprefix("torch.")
    flat = tensor.reshape(-1)
    raw = flat if flat.dtype == torch.uint8 else flat.view(torch.uint8)
    return raw.numpy().tobytes(), dtype, shape, note


def _write_tensor(
    directory: Path, name: str, role: str, tensor: Any, *,
    required: bool = False, source: str = "",
) -> TensorRecord:
    import hashlib

    payload, dtype, shape, note = _raw_bytes(tensor)
    filename = f"{TENSOR_DIR}/{_safe_name(name)}.bin"
    path = directory / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return TensorRecord(
        name=name, role=role, file=filename, dtype=dtype, shape=shape,
        nbytes=len(payload), sha256=hashlib.sha256(payload).hexdigest(),
        required=required, source=source, note=note,
    )


def find_group(artifact: Path, module_id: str) -> str:
    """The implementation group holding a module id.

    A module id names a place in the model; a group names the deduplicated implementation
    that serves it, and the two do not resemble each other — `layers.0.attention` lives in
    `00-Attention` alongside thirty other layers. Searching for it means the caller does not
    have to know the mapping.
    """
    modules = artifact / "modules"
    if not modules.is_dir():
        raise MaterializeError(f"{artifact} holds no modules/ directory")
    candidates: list[str] = []
    for calls in sorted(modules.glob("*/calls.json")):
        try:
            payload = json.loads(calls.read_text())
        except json.JSONDecodeError:
            continue
        if module_id in (payload.get("modules") or {}):
            candidates.append(calls.parent.name)
    if not candidates:
        raise MaterializeError(
            f"no group in {modules} publishes {module_id!r}; pass --group explicitly"
        )
    if len(candidates) > 1:
        raise MaterializeError(
            f"{module_id} appears in several groups ({', '.join(candidates)}); pass --group"
        )
    return candidates[0]


def _artifact_reader(artifact: Path):
    """The reader the artifact vendored, loaded from the artifact itself.

    Not the installed `model_partition.runtime.artifact`: a published run carries its own
    copy, and the two can differ. This artifact's copy resolves checkpoint-backed weights
    out of `hf/`, which the installed one does not — with the wrong reader every parameter
    that was never dumped silently goes missing. Reading a run with the reader it shipped
    with is the only version-safe way to read it.

    The file is standard-library-only and imports torch lazily, so loading it by path
    costs nothing and pulls in no package.
    """
    import importlib.util

    candidate = artifact / "runtime" / "model_partition" / "runtime" / "artifact.py"
    if not candidate.is_file():
        # No fallback to the installed reader. The two versions differ in the API used here
        # — `ModuleCalls.runs`, `checkpoint_weights` — so falling back would either raise
        # AttributeError halfway through or, worse, silently read only the dumped weights and
        # produce a repo missing most of its parameters.
        raise MaterializeError(
            f"{artifact} ships no runtime/model_partition/runtime/artifact.py. A published "
            f"run carries the reader it was written with, and reading it with another "
            f"version is not safe; re-download the artifact."
        )

    import sys

    name = "_bootstrap_artifact_reader"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, candidate)
    if spec is None or spec.loader is None:
        raise MaterializeError(f"cannot load the artifact's reader at {candidate}")
    module = importlib.util.module_from_spec(spec)
    # Registered before execution because the file defines dataclasses, and
    # `@dataclass` resolves annotations through `sys.modules[cls.__module__]`.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        del sys.modules[name]
        raise
    return module


def _load_module_calls(artifact: Path, group: str, module_id: str):
    """One module's recorded calls, via the artifact's own reader."""
    reader = _artifact_reader(artifact)
    directory = artifact / "modules" / group
    if not (directory / "calls.json").is_file():
        raise MaterializeError(f"{directory} is not a published module group")
    recorded = reader.load_calls(directory)
    if module_id not in recorded:
        raise MaterializeError(
            f"{module_id} is not in {group}; it holds: {', '.join(sorted(recorded))}"
        )
    return reader, directory, recorded[module_id]


def materialize(
    artifact: Path,
    group: str,
    module_id: str,
    repo: Path,
    *,
    sample_id: str | None = None,
    step: int | None = None,
    call_index: int = 0,
) -> Materialized:
    """Write the module out as a bootstrap repo. Returns what was written."""
    reader, directory, calls = _load_module_calls(artifact, group, module_id)
    samples = _sample_order(directory)
    key, recorded = calls.select(sample_id, step, samples)
    invocations = calls.runs(recorded)
    if not 0 <= call_index < len(invocations):
        raise MaterializeError(
            f"{module_id} recorded {len(invocations)} invocation(s) for {key}; "
            f"call_index must be 0..{len(invocations) - 1}"
        )
    chain = invocations[call_index]
    resolved_sample, _, resolved_step = key.partition("#")

    repo.mkdir(parents=True, exist_ok=True)
    records: list[TensorRecord] = []

    # The group's boundary: everything the first submodule was called with going in, and
    # everything the last one returned coming out. Anything between is internal to the kernel.
    #
    # Both sides can hold more than one tensor. A module is handed a mask, a rotary
    # embedding, a position alongside its hidden state, and can return a tuple or a dict —
    # taking only the first of each would drop inputs the kernel needs and check only part of
    # what it produced, and both would go unnoticed on a module where the first happens to be
    # the only one.
    inputs = _flatten_tensors(reader, directory, chain[0].get("args") or [], "input")
    inputs += _flatten_tensors(reader, directory, chain[0].get("kwargs") or {}, "input")
    if not inputs:
        raise MaterializeError(f"{module_id} records no tensor among its arguments")
    outputs = _flatten_tensors(reader, directory, chain[-1].get("output"), "reference")
    if not outputs:
        raise MaterializeError(f"{module_id} records no tensor as its output")

    for name, tensor, origin in inputs:
        records.append(_write_tensor(
            directory=repo, name=name, role="input", tensor=tensor, required=True,
            source=origin,
        ))
    reference_max = 0.0
    for name, tensor, origin in outputs:
        records.append(_write_tensor(
            directory=repo, name=name, role="golden", tensor=tensor, required=True,
            source=origin,
        ))
        reference_max = max(reference_max, float(tensor.detach().float().abs().max()))

    # Every weight the module needs, dumped ones and checkpoint-backed ones alike. The
    # reader resolves both, so this is where the fetched shards are paid for.
    weights = calls.weights("cpu")
    missing = set(calls.payload.get("weights") or {}) | set(calls.checkpoint_weights)
    absent = missing - set(weights)
    if absent:
        raise MaterializeError(
            f"{module_id} is short {len(absent)} weight(s) — run fetch_weights.py in the "
            f"artifact first (e.g. {sorted(absent)[:2]})"
        )
    dumped = set(calls.payload.get("weights") or {})
    for name in sorted(weights):
        records.append(_write_tensor(
            directory=repo, name=name,
            role="buffer" if name in dumped else "weight",
            tensor=weights[name],
            source="dumped" if name in dumped else _shard_of(calls, name),
        ))
    del weights
    tolerance = _tolerance_for(artifact, records, reference_max)

    # Non-tensor arguments (a decode position, a flag) are values, not files. They belong
    # in the generated stub as literals and in the README as a table.
    scalars: list[dict[str, Any]] = []
    for call in chain:
        for index, value in enumerate(call.get("args") or []):
            if not _is_tensor(value):
                scalars.append({
                    "submodule": call.get("submodule"), "position": index, "value": value,
                })
        for kw, value in (call.get("kwargs") or {}).items():
            if not _is_tensor(value):
                scalars.append({
                    "submodule": call.get("submodule"), "keyword": kw, "value": value,
                })

    result = Materialized(
        repo=repo, group=group, module_id=module_id, sample_id=resolved_sample,
        step=int(resolved_step or 0), call_index=call_index, tensors=records,
        scalar_args=scalars,
        submodules=[c.get("submodule", "") for c in chain],
        total_bytes=sum(r.nbytes for r in records),
        tolerance=tolerance,
        model=_model_name(artifact),
        composition=_composition(directory),
    )

    _write_frozen_files(artifact, directory, repo, result)
    (repo / "source.py").write_text(templates.render_source_stub(result))
    (repo / "inference.py").write_text(templates.render_inference_stub(result))
    (repo / "README.md").write_text(templates.render_readme(result))
    (repo / ".gitignore").write_text(templates.GITIGNORE)
    return result


def _literal_from_inference(directory: Path, name: str, fallback: Any) -> Any:
    """A module-level constant out of the group's own `inference.py`.

    By AST rather than by import: that file puts the harness runtime on `sys.path` when
    imported, which is the coupling this whole materialization exists to avoid.
    """
    import ast

    path = directory / "inference.py"
    if not path.is_file():
        return fallback
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return fallback
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            continue
        try:
            return ast.literal_eval(node.value)
        except (ValueError, SyntaxError):
            return fallback
    return fallback


def _sample_order(directory: Path) -> list[str]:
    """Sample ids in the order the run took them — short prompts first, so the default
    selection is the cheap one."""
    value = _literal_from_inference(directory, "SAMPLE_IDS", [])
    return [str(v) for v in value] if isinstance(value, (list, tuple)) else []


def _is_tensor(value: Any) -> bool:
    return isinstance(value, dict) and "__tensor__" in value


def _bin_of(value: Any) -> str:
    return value["__tensor__"].get("bin", "") if _is_tensor(value) else ""


def _shard_of(calls: Any, name: str) -> str:
    ref = calls.checkpoint_weights.get(name) or {}
    shard, key = ref.get("shard"), ref.get("key")
    return f"{shard}:{key}" if shard else ""


def _decode(reader: Any, directory: Path, value: Any) -> Any:
    if not _is_tensor(value):
        return None
    return reader.decode(directory, value, "cpu")


def _flatten_tensors(
    reader: Any, directory: Path, value: Any, prefix: str,
) -> list[tuple[str, Any, str]]:
    """Every recorded tensor inside a value, named for where it sat.

    A boundary is not always one tensor: a list of arguments, a returned tuple, a dict of
    outputs. Names follow the shape — `reference`, or `reference_0`/`reference_1` for a
    tuple, or `reference_logits` for a dict — so a reader of the manifest can see which
    position each file came from.
    """
    found: list[tuple[str, Any, str]] = []

    def walk(node: Any, name: str) -> None:
        if _is_tensor(node):
            found.append((name, _decode(reader, directory, node), _bin_of(node)))
        elif isinstance(node, (list, tuple)):
            for index, item in enumerate(node):
                walk(item, f"{name}_{index}")
        elif isinstance(node, dict):
            for key, item in node.items():
                walk(item, f"{name}_{key}")

    walk(value, prefix)
    # One tensor at the boundary is the common case and reads better unsuffixed.
    if len(found) == 1 and found[0][0] != prefix:
        found = [(prefix, found[0][1], found[0][2])]
    return found


def _numerics(artifact: Path) -> tuple[dict[str, tuple[float, float]], tuple[float, float], dict[str, float]]:
    """The artifact's own tolerance table, default row and pass fractions.

    Loaded from the artifact rather than retyped, so the bar is the one the run used.
    """
    import importlib.util

    table: dict[str, tuple[float, float]] = {}
    default = (2e-2, 2e-2)
    fractions = {"MIN_COSINE": 0.9999, "MIN_PASS_FRACTION": 0.999}
    path = artifact / "runtime" / "model_partition" / "verify" / "numerics.py"
    if not path.is_file():
        return table, default, fractions
    spec = importlib.util.spec_from_file_location("_bootstrap_numerics", path)
    if spec is None or spec.loader is None:
        return table, default, fractions
    module = importlib.util.module_from_spec(spec)
    sys.modules["_bootstrap_numerics"] = module
    try:
        spec.loader.exec_module(module)
        table = dict(getattr(module, "TOLERANCES", {}))
        default = getattr(module, "DEFAULT_TOLERANCE", default)
        bar = getattr(module, "Tolerance", None)
        if bar is not None:
            fractions = {"MIN_COSINE": float(bar.min_cosine),
                         "MIN_PASS_FRACTION": float(bar.min_pass_fraction)}
    finally:
        sys.modules.pop("_bootstrap_numerics", None)
    return table, default, fractions


def _tolerance_for(
    artifact: Path, records: list[TensorRecord], reference_max: float,
) -> dict[str, float]:
    """The bar this module is held to: five numbers, all derived.

    **rtol/atol come from the coarsest number format anywhere in the module, not from the
    output's dtype.** That distinction is the whole point. `layers.0.attention` returns
    bfloat16, whose row is 2e-2 — but its weights are fp8 and its `kv` is fp8-quantized
    mid-chain, so one adjacent-grid step is a ~9% relative change, four times that bar. An
    implementation that is correct but not bit-identical through an fp8 rounding therefore
    fails elementwise at every value sitting near a grid midpoint, however small the
    underlying error. Keying on the output dtype asks for bit-exactness with a CUDA fp8
    tensor-core MMA, which no other hardware can give; keying on the narrowest format in the
    chain asks for the precision the computation actually carries.

    Non-float dtypes are skipped. An fp4 expert weight is stored packed in `uint8`, which the
    table has no row for and which says nothing about precision — so a module whose real
    floor is fp4 currently gets fp8's bar. That is a known open question, not an oversight:
    the loosening is measured on fp8 and unmeasured on fp4.

    **MAX_ABS_ERR is a ceiling no element may cross**, set at the most generous allowance the
    elementwise test gives anywhere in the tensor — `atol + rtol * max|reference|`. It closes
    the gap a pass-fraction leaves open: at 0.999, 0.1% of elements may be wrong by *any*
    amount, and cosine notices one wild value but not a few dozen merely-bad ones. A small
    element normally gets a small allowance; this stops it borrowing a large one.
    """
    table, default, fractions = _numerics(artifact)

    present = {r.dtype for r in records}
    rows = {dt: table[dt] for dt in present if dt in table}
    # Coarsest row wins: the largest rtol among the formats actually in the chain.
    rtol, atol = max(rows.values(), key=lambda pair: pair[0]) if rows else default

    bar = {"RTOL": rtol, "ATOL": atol, **fractions}
    bar["MAX_ABS_ERR"] = _legible(atol + rtol * reference_max)
    return bar


def _legible(value: float) -> float:
    """A value that survives being written as a literal and read back.

    The bar has to be typed into `inference.py` as a bare number and then compared for
    equality against this one. `atol + rtol * max|reference|` is an arbitrary float —
    0.28046875000000004 for one module — and `f"{v:g}"` prints six significant digits, so the
    literal and the bar would differ in the last bits and the gate would report
    "MAX_ABS_ERR is 0.280469, but the bar is 0.280469".

    So the bar is rounded to what six digits can express, upward: a ceiling must never come
    out stricter than the one that was derived.
    """
    rounded = float(f"{value:.6g}")
    if rounded < value:
        rounded = float(f"{value + abs(value) * 1e-6:.6g}")
    return rounded


def _model_name(artifact: Path) -> str:
    """What the artifact says it was traced from, for the generated README.

    Empty rather than guessed when the artifact says nothing: a wrong model name in a file
    the agent reads as its specification is worse than no model name.
    """
    import yaml

    run = artifact / "run.yaml"
    if run.is_file():
        try:
            spec = (yaml.safe_load(run.read_text()) or {}).get("spec") or {}
        except yaml.YAMLError:
            spec = {}
        for key in ("source", "name"):
            value = spec.get(key)
            if isinstance(value, str) and value:
                return value
    return ""


def _composition(directory: Path) -> str:
    """Whether the group's submodules run in sequence or in parallel.

    From the group's own `inference.py`, which records it as `COMPOSITION` — the same reason
    `SAMPLE_IDS` is read there, and again by AST so importing it cannot drag the harness
    runtime onto `sys.path`.
    """
    return _literal_from_inference(directory, "COMPOSITION", "sequential")


#: The artifact files carried in verbatim as frozen references, and why each is here.
#: Renaming is the point: left under their original names, the first thing the agent would
#: have to do is delete the only description of the job it has.
FROZEN_REFERENCES: tuple[tuple[str, str, str], ...] = (
    (
        "source.py", "reference_torch.py",
        "the original PyTorch implementation of this module — what to compute.\n"
        "Read it as the specification. Do not import it, call it, or copy torch out of it:\n"
        "the kernel in source.py must be NKI only.",
    ),
    (
        "inference.py", "reference_inference.py",
        "the artifact's own launcher — how the recorded reference was produced.\n"
        "It loads the module's weights and recorded inputs, runs it, times it, compares the\n"
        "result against the dumped output and reports the `##autohelix[...]` metrics. Read it\n"
        "to see the shape your own inference.py has to take. It cannot run here: it needs the\n"
        "harness runtime and the artifact's calls.json, neither of which is in this repo, and\n"
        "importing it would fail self-containment. Take the structure, not the imports.",
    ),
)

#: The artifact's own code, copied in as directories under their original names rather than
#: flattened and renamed. That is what makes the reference *runnable*: `reference_torch.py`
#: does `from kernel import act_quant, ...`, which resolves once `vendor/` is on `sys.path`,
#: and each `compat/*.py` exposes `apply(vendor, device)` to rebind what it replaced.
#:
#: Being able to execute it matters as much as reading it. The first real run watched an
#: agent spend an iteration reconstructing quantization semantics by hand because it had
#: neither the source nor anything to run.
#:
#: `nki_checker.FORBIDDEN_PATH_MARKERS` matches both names, so the deliverables still cannot
#: import or open them — they are for the agent's own experiments, not for `inference.py`.
VENDOR_DIR = "vendor"
COMPAT_DIR = "compat"

#: The comparison the reference was judged by, vendored from the artifact's runtime. Its
#: path inside an artifact, and the name it gets here.
NUMERICS_SOURCE = ("runtime", "model_partition", "verify", "numerics.py")
NUMERICS_TARGET = "reference_numerics.py"
NUMERICS_WHY = (
    "the exact definition of the numerical bar this module was accepted at.\n"
    "`Tolerance.for_dtype` is where RTOL/ATOL/MIN_COSINE/MIN_PASS_FRACTION come\n"
    "from, and `compare_outputs` is how they are applied — elementwise closeness over a\n"
    "minimum fraction of elements, plus cosine similarity. Your inference.py has to reach the\n"
    "same verdict on the same numbers without importing this: reimplement it, self-contained,\n"
    "with the four constants as literals."
)


def _frozen_header(why: str, origin: str) -> str:
    """The banner every frozen reference carries.

    Each one says what it is *for*, because a file the agent may read but not import, call
    or edit is an unusual thing to be handed and the reason has to travel with it.
    """
    return (
        f'"""FROZEN REFERENCE — {why}\n\n'
        f"Frozen: any edit to this file is reverted before your work is judged. Copied\n"
        f"verbatim from the artifact's {origin}.\n"
        '"""\n\n'
    )


def _write_frozen_files(artifact: Path, directory: Path, repo: Path, result: Materialized) -> None:
    """The specification the agent reads: what to compute, how it was run, how it was judged.

    Three files rather than one. The implementation alone says what the answer is but not
    what a validator looks like, and neither says what "passes" means — which matters here
    more than usual, because the gate takes the candidate's own comparison at its word
    rather than recomputing it. An ambiguous bar would leave both the agent and the reviewer
    guessing at the thing they are respectively meeting and auditing.
    """
    for original_name, target_name, why in FROZEN_REFERENCES:
        original = directory / original_name
        body = (
            original.read_text() if original.is_file()
            else f"# (the artifact published no {original_name})\n"
        )
        origin = f"modules/{result.group}/{original_name}"
        (repo / target_name).write_text(_frozen_header(why, origin) + body)

    numerics = artifact.joinpath(*NUMERICS_SOURCE)
    if numerics.is_file():
        (repo / NUMERICS_TARGET).write_text(
            _frozen_header(NUMERICS_WHY, "/".join(NUMERICS_SOURCE)) + numerics.read_text()
        )

    # Verbatim, original filenames, no injected header: the point is that these import and
    # run, and rewriting them would be rewriting the specification.
    for name in (VENDOR_DIR, COMPAT_DIR):
        source = artifact / name
        if source.is_dir():
            shutil.copytree(source, repo / name, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__"))

    config = directory / "config.json"
    if config.is_file():
        shutil.copy2(config, repo / "config.json")

    readme = directory / "README.md"
    if readme.is_file():
        (repo / "MODULE.md").write_text(readme.read_text())


def write_manifest(result: Materialized) -> Path:
    """Record what was put in the repo, for the gate to check it against.

    This is the only per-repo state `init` writes. The config and the prompt template are
    fixed files in the package that the loop reads directly, so there is nothing else to
    install and nothing that can drift between a repo and the preset it runs under.

    The manifest lands under `.autohelix/`, which is gitignored and outside the agent's
    editable scope. It is not a secret — an iteration worktree is nested inside the repo, so
    it is reachable from there — but it is not handed to the agent either, and check (f)
    compares the tensors against it, so a repo whose data was edited is caught whether or
    not the edit was noticed.
    """
    manifest_path = result.repo / MANIFEST_REL
    dump_manifest(manifest_path, {
        "artifact_group": result.group,
        "module_id": result.module_id,
        "sample_id": result.sample_id,
        "step": result.step,
        "call_index": result.call_index,
        "submodules": result.submodules,
        "composition": result.composition,
        "scalar_args": result.scalar_args,
        # The bar check (e) holds this repo to, derived from the reference's dtype rather
        # than fixed globally: an fp8 boundary and a float32 one are not the same question.
        "tolerance": result.tolerance,
        "tensors": [t.to_dict() for t in result.tensors],
    })
    return manifest_path


def clear(repo: Path) -> None:
    """Empty a directory so `init --force` starts clean.

    Overwriting file by file would leave whatever the previous init wrote and this one does
    not — another module's tensors, a stale manifest — and the repo would then be a mix of
    two modules that nothing detects. Removing git history is part of that: the baseline
    commit has to be this module's stub.
    """
    for entry in repo.iterdir():
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def git_init(repo: Path) -> str:
    """Make the repo a repo, with the stub as its first commit.

    The loop needs a clean tree and a committed baseline before it will start, and the
    baseline has to be the failing stub: the first iteration's diff is then exactly the
    bootstrap work and nothing else.
    """
    def run(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, check=False,
        )

    if not (repo / ".git").is_dir():
        result = run("init", "-q")
        if result.returncode != 0:
            raise MaterializeError(f"git init failed: {result.stderr.strip()}")
        run("config", "user.name", "autohelix-bootstrap")
        run("config", "user.email", "autohelix-bootstrap@localhost")

    run("add", "-A")
    status = run("status", "--porcelain")
    if status.stdout.strip():
        committed = run("commit", "-q", "-m", "Bootstrap baseline: failing NKI stub")
        if committed.returncode != 0:
            raise MaterializeError(f"git commit failed: {committed.stderr.strip()}")
    head = run("rev-parse", "--short", "HEAD")
    return head.stdout.strip()


def load_manifest(repo: Path) -> dict[str, Any]:
    """The manifest for an already-initialized repo."""
    path = repo / MANIFEST_REL
    if not path.is_file():
        raise MaterializeError(
            f"{repo} has no bootstrap manifest; run `autohelix bootstrap init` first"
        )
    return json.loads(path.read_text())
