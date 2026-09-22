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


class MaterializeError(RuntimeError):
    """The artifact cannot supply what was asked for."""


def _safe_name(name: str) -> str:
    """`layers.0.attn.wq_a.weight` -> `layers_0_attn_wq_a_weight`."""
    return "".join(c if c.isalnum() else "_" for c in name).strip("_")


def _raw_bytes(tensor: Any) -> tuple[bytes, str, list[int], str]:
    """A tensor's little-endian payload, plus the dtype and shape to read it back with.

    Flattening before the uint8 view is what makes this work for every dtype the
    checkpoint holds: fp8 and bfloat16 have no numpy equivalent, but a 1-D tensor of any
    dtype reinterprets as bytes, and that is all the file needs to be.
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
        from model_partition.runtime import artifact as installed
        return installed

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

    # The group's boundary: the first submodule's first argument in, the last submodule's
    # output out. Anything in between is internal to the kernel being written.
    group_input = _decode(reader, directory, chain[0]["args"][0])
    if group_input is None:
        raise MaterializeError(f"{module_id} records no tensor as its first argument")
    records.append(_write_tensor(
        directory=repo, name="input", role="input", tensor=group_input, required=True,
        source=_bin_of(chain[0]["args"][0]),
    ))
    group_output = _decode(reader, directory, chain[-1]["output"])
    if group_output is None:
        raise MaterializeError(f"{module_id} records no tensor as its output")
    records.append(_write_tensor(
        directory=repo, name="reference", role="golden", tensor=group_output, required=True,
        source=_bin_of(chain[-1]["output"]),
    ))

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
    )

    _write_frozen_files(artifact, directory, repo, result)
    (repo / "source.py").write_text(templates.render_source_stub(result))
    (repo / "inference.py").write_text(templates.render_inference_stub(result))
    (repo / "README.md").write_text(templates.render_readme(result))
    (repo / ".gitignore").write_text(templates.GITIGNORE)
    return result


def _sample_order(directory: Path) -> list[str]:
    """Sample ids in the order the run took them.

    Read out of the group's own `inference.py`, which records them as `SAMPLE_IDS` for
    exactly this purpose — short prompts first, so selecting the default selects the cheap
    one. Taken by AST rather than by import: that file bootstraps the harness runtime onto
    `sys.path` when imported, which is precisely what this is trying to avoid.
    """
    import ast

    path = directory / "inference.py"
    if not path.is_file():
        return []
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "SAMPLE_IDS" for t in node.targets):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, SyntaxError):
            return []
        if isinstance(value, (list, tuple)):
            return [str(v) for v in value]
    return []


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


def _write_frozen_files(artifact: Path, directory: Path, repo: Path, result: Materialized) -> None:
    """The specification the agent reads: the original implementation and its config.

    `reference_torch.py` keeps the vendor's code verbatim below a header that says it is
    frozen and unimportable. Renaming it is the point — leaving it as `source.py` would
    mean the first thing the agent must do is delete the only description of the
    computation it has.
    """
    original = directory / "source.py"
    header = (
        '"""FROZEN REFERENCE — the original PyTorch implementation of this module.\n\n'
        "Read this to learn what to compute. Do not import it, call it, or copy torch out\n"
        "of it: the kernel in source.py must be NKI only, and any edit to this file is\n"
        f"reverted. Copied verbatim from the artifact's modules/{result.group}/source.py.\n"
        '"""\n\n'
    )
    body = original.read_text() if original.is_file() else "# (the artifact published no source.py)\n"
    (repo / "reference_torch.py").write_text(header + body)

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
        "scalar_args": result.scalar_args,
        "tensors": [t.to_dict() for t in result.tensors],
    })
    return manifest_path


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
