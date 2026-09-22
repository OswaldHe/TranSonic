# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extract per-module code, one implementation per structural signature.

Modules that share a signature share an implementation, so a 64-layer model with
two attention variants yields two decoder implementations rather than 64 copies.
Each group gets the real source of the classes involved plus a runnable replay
harness.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from model_partition import yamlio

from model_partition.hardware import format_bytes
from model_partition.planner.graph import PartitionGraph
from model_partition.trace import _lookup
# Named by the launcher, which is what reads them in a published artifact and must
# import nothing from here: this module renders templates, and a downloaded module
# directory should not need a template engine to run.
#
# ``VENDOR_DIR`` is where a run keeps the model's own sibling modules, at its root
# beside ``modules/``: ``source.py`` is a slice of one file of the vendor's package and
# keeps that file's imports — DeepSeek's attention needs ``kernel`` for its GEMMs and
# ``engram`` for the n-gram layout — so without these a module directory needs the
# model's repo present to import at all.
#
# ``COMPUTE_DTYPE_KEY`` is where the dtype the model's activations flow in is recorded.
# Not a field of any framework config: a quantized checkpoint names its *storage* dtype
# and computes in another, and a module built in the wrong one has its kernels reject
# their own output buffers.
from model_partition.runtime.launcher import (  # noqa: F401  (re-exported)
    CALLS_FILENAME,
    COMPUTE_DTYPE_KEY,
    CONFIG_FILENAME,
    RUNTIME_DIR,
    SOURCE_FILENAME,
    VENDOR_DIR,
)

#: The implementation and its launcher are the loop's to edit, so they are written
#: once and then left alone — an optimization made to a module has to survive the
#: next extraction. Everything else belongs to the harness and is regenerated.
EDITABLE_TEMPLATES = {"inference.py": "module_inference.py.tmpl"}

#: Config fields a framework keeps private and ``to_dict()`` leaves out, but which
#: decide what a module computes. The attention implementation is the one that
#: matters: a recorded call carries ``attention_mask=None`` when the traced kernel
#: applied causality itself, and rebuilding the module as eager attention then lets
#: every position see the future — a module that looks close and is wrong.
PRIVATE_CONFIG_KEYS = ("_attn_implementation",)
HARNESS_TEMPLATES = {
    "verify.py": "module_verify.py.tmpl",
    "README.md": "module_readme.md.tmpl",
}

#: Module kind -> what the module is for, in one sentence. Read by a kernel author
#: before anything else in the directory.
KIND_PURPOSE = {
    "embed": "Turns token ids into the initial hidden state.",
    "decoder_layers": "Runs one or more complete decoder layers of the stack: "
                      "attention and the feed-forward block, residuals included.",
    "attention": "The attention side of a decoder layer: its normalization, the "
                 "projections, the attention itself, and the output projection.",
    "moe_router": "Scores each token against the experts and decides which ones "
                  "it is routed to.",
    "moe_experts": "Runs a contiguous group of MoE experts over the tokens routed "
                   "to them.",
    "mlp": "The feed-forward side of a decoder layer.",
    "norm": "A normalization applied to the residual stream.",
    "lm_head": "Projects the final hidden state to vocabulary logits.",
    "other": "A component of a decoder layer that is neither attention nor the "
             "feed-forward block.",
}

#: Prepended to the implementation, as a comment so the file still imports. This is
#: the file `inference.py` loads and the file to optimize.
SOURCE_BANNER = """\
# This is the implementation for partition group `{signature}`, {scope}
#   {origin}
# `inference.py` beside it imports this file and calls `{class_name}` — so editing
# here is what changes what runs, and `verify.py` tells you whether the result still
# reproduces the dumped reference. Nothing here reads the checkpoint.
#
# Its imports resolve against the installed framework and against `vendor/` at the root
# of this run, which holds the model's own sibling modules. The class bodies are this
# file's.

"""

#: How much of the originating file ``source.py`` holds. A module directory is one
#: module, so what it needs is taken and the rest of the model is left behind.
SLICED_SCOPE = ("{kept} of the {total} definitions — this module's classes and what "
                "they reference — copied verbatim from")
WHOLE_SCOPE = "copied verbatim from"

#: Written when there is no importable original to copy: the collected class bodies,
#: which document the module but cannot be launched.
EXCERPT_HEADER = '''\
"""Excerpt of the implementation for partition group `{signature}`.

Collected with inspect.getsource. Provenance:
{provenance}

Not importable — ``inspect.getsource`` returns class bodies without their module's
imports and helpers — so ``inference.py`` cannot launch this and says so. Kept
because it still documents what the module computes.
"""

'''

@dataclass
class ExtractedGroup:
    """One deduplicated implementation."""

    signature: str
    module_ids: list[str] = field(default_factory=list)
    layer_indices: list[int] = field(default_factory=list)
    class_name: str = ""
    classes: list[str] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)
    #: Dotted name the implementation's classes were defined under, so ``source.py``
    #: can be imported with its own relative imports intact.
    source_module: str = ""
    #: Class of the config recorded in ``config.json``, which is the config this
    #: module's subtree was built from rather than the model's top-level one.
    config_class: str = ""
    #: True when ``source.py`` is the real, importable implementation rather than an
    #: excerpt. False means the launcher has nothing to import and says so.
    launchable: bool = False
    directory: Path | None = None
    source_lines: int = 0
    #: True when an existing implementation was left in place.
    preserved: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "signature": self.signature,
            "module_ids": list(self.module_ids),
            "layer_indices": list(self.layer_indices),
            "class_name": self.class_name,
            "classes": list(self.classes),
            "source_files": list(self.source_files),
            "source_module": self.source_module,
            "config_class": self.config_class,
            "launchable": self.launchable,
            "source_lines": self.source_lines,
            "preserved": self.preserved,
        }


def _render_template(name: str, context: dict[str, Any]) -> str:
    from jinja2 import Environment, FileSystemLoader, StrictUndefined

    templates = Path(__file__).resolve().parent / "templates"
    env = Environment(loader=FileSystemLoader(str(templates)), undefined=StrictUndefined,
                      keep_trailing_newline=True)
    return env.get_template(name).render(**context)


def collect_sources(module: Any, max_classes: int = 40) -> tuple[list[str], list[str], list[str]]:
    """Source of a module's class and its children's classes.

    Returns ``(sources, class_names, files)``, deduplicated and ordered from the root
    class outward. ``files`` names the modules the classes came from, which is both
    the provenance a reader needs and what gets copied verbatim.
    """
    seen: set[type] = set()
    sources: list[str] = []
    names: list[str] = []
    files: list[str] = []

    def visit(obj: Any) -> None:
        if len(seen) >= max_classes:
            return
        cls = type(obj)
        if cls in seen or cls.__module__ in ("builtins", "torch.nn.modules.container"):
            return
        seen.add(cls)
        try:
            sources.append(inspect.getsource(cls))
            names.append(f"{cls.__module__}.{cls.__qualname__}")
            path = str(inspect.getfile(cls))
            if path not in files:
                files.append(path)
        except (OSError, TypeError):
            names.append(f"{cls.__module__}.{cls.__qualname__} (source unavailable)")
        if hasattr(obj, "children"):
            for child in obj.children():
                visit(child)

    visit(module)
    return sources, names, files


def collect_group_sources(targets: list[Any]) -> tuple[list[str], list[str], list[str]]:
    """Source of every submodule in a group, deduplicated across them.

    A group spans more than one submodule — a normalization and the attention it
    feeds — and the reference is only useful if it covers all of them. Taking the
    first submodule's source alone left the attention groups documented by a
    28-line RMSNorm.
    """
    sources: list[str] = []
    names: list[str] = []
    files: list[str] = []
    for target in targets:
        for collected, into in zip(collect_sources(target), (sources, names, files)):
            for item in collected:
                if item not in into:
                    into.append(item)
    return sources, names, files


#: Modules whose classes are the framework rather than the model. A group's source
#: comes from the model's own file: ``torch.nn.Linear`` is resolved from the installed
#: framework, not vendored, and picking its file would leave the model's own classes
#: out of ``source.py``.
FRAMEWORK_MODULES = ("torch.", "torch")


def is_framework(cls: type) -> bool:
    return str(cls.__module__).startswith(FRAMEWORK_MODULES)


def _principal(targets: list[Any]) -> Any:
    """The submodule that carries the group's computation, by parameter count.

    Restricted to the model's own classes where there are any, so ``source.py`` is the
    model's file. What a reader wants the group called is ``Qwen3_5Attention``, not the
    norm in front of it and not ``Linear``.
    """
    def weight(target: Any) -> int:
        try:
            return sum(p.numel() for p in target.parameters())
        except (AttributeError, TypeError):
            return 0

    own = [t for t in targets if not is_framework(type(t))]
    return max(own or targets, key=weight)


def extract(
    graph: PartitionGraph,
    model: Any,
    out_dir: str | Path,
    run_root: str | Path = ".",
    sample_ids: list[str] | None = None,
    weight_tensors: dict[str, list[str]] | None = None,
    param_names: dict[str, list[str]] | None = None,
    regenerate: bool = False,
    config: dict[str, Any] | None = None,
    bundle: Any = None,
) -> list[ExtractedGroup]:
    """Write one implementation directory per signature group.

    ``param_names`` maps a module id to the original parameter names its
    implementation will be handed, so the generated file can document them. ``config``
    is the model's own, written beside a module whose classes keep no config object of
    their own — DeepSeek's copy their fields onto themselves and drop it — so the
    directory is a unit someone can build from without this run.

    An existing ``source.py`` and ``inference.py`` are preserved: they are the loop's
    editable surface and may carry the agent's numerical fixes, or someone's kernel
    work. ``regenerate=True`` overwrites them.

    ``bundle`` is the trace, and with it each directory also gets the recorded calls and
    weight paths its ``inference.py`` reads. Without it the directories are written all
    the same, so ``extract`` still works before anything has been traced.
    """
    param_names = param_names or {}
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    groups: list[ExtractedGroup] = []
    vendored: set[Path] = set()
    vendor_runtime(run_root)

    for index, (signature, module_ids) in enumerate(sorted(graph.signature_groups().items())):
        modules = [graph.by_id(mid) for mid in module_ids]
        representative = modules[0]
        targets = [t for t in (_lookup(model, name) for name in representative.submodules)
                   if t is not None]

        group = ExtractedGroup(
            signature=signature,
            module_ids=list(module_ids),
            layer_indices=sorted({i for m in modules for i in m.layer_indices}),
        )
        if not targets:
            group.class_name = "(not instantiated)"
            sources, names, files = [], [], []
        else:
            sources, names, files = collect_group_sources(targets)
            group.class_name = type(_principal(targets)).__name__
            vendored |= vendor_code(run_root, files)
        group.classes = names
        group.source_files = files

        directory = root / _group_dirname(index, representative, signature)
        directory.mkdir(parents=True, exist_ok=True)
        group.directory = directory

        principal = _principal(targets) if targets else None
        group.source_module = type(principal).__module__ if principal is not None else ""
        settings, config_class = _module_config(model, representative.submodules,
                                               fallback=config)
        group.config_class = config_class
        (directory / CONFIG_FILENAME).write_text(
            json.dumps(settings, indent=2, sort_keys=True, default=str) + "\n")
        # Whether a previous extraction has been here, asked before this one writes
        # anything. It decides what counts as somebody's work to keep.
        had_source = (directory / SOURCE_FILENAME).is_file()
        if had_source and not regenerate:
            group.preserved = True
        group.launchable = _write_source(directory, principal, sources, signature, files,
                                         classes=names, regenerate=regenerate)
        # What source.py actually holds, so the reported figure is the code someone has
        # to read rather than the size of the file it was taken from.
        source_path = directory / SOURCE_FILENAME
        group.source_lines = (len(source_path.read_text().splitlines())
                              if source_path.is_file() else 0)
        weight_names = list(param_names.get(representative.id, []))
        # One class name per submodule, positionally. Groups are signature-homogeneous,
        # so submodule i of any module in the group has submodule i's class.
        class_names = [type(t).__name__ for t in targets]
        context = {
            "signature": signature,
            "module_ids": list(module_ids),
            "layer_map": {m.id: list(m.layer_indices) for m in modules},
            "submodule_map": {m.id: list(m.submodules) for m in modules},
            "class_name": group.class_name,
            "sample_ids": list(sample_ids or []),
            "weight_tensors": {mid: list((weight_tensors or {}).get(mid, [])) for mid in module_ids},
            "submodules": list(representative.submodules),
            "weight_names": weight_names,
            "weight_count": len(weight_names),
            "run_root": str(run_root),
            "kind": representative.kind,
            "class_names": class_names,
            "config_class": config_class,
            "source_module": group.source_module,
            "launchable": group.launchable,
            "module_submodules": {m.id: list(m.submodules) for m in modules},
            "purpose": KIND_PURPOSE.get(representative.kind, "A partition module."),
            "notes": representative.notes,
            "composition": representative.composition,
            "verifiable": not representative.functional,
            "inputs": list(representative.inputs),
            "outputs": list(representative.outputs),
            "param_bytes_h": format_bytes(representative.param_bytes),
            "kv_bytes_h": format_bytes(representative.kv_bytes) if representative.kv_bytes else "",
        }
        for filename, template in HARNESS_TEMPLATES.items():
            (directory / filename).write_text(_render_template(template, context))
        for filename, template in EDITABLE_TEMPLATES.items():
            path = directory / filename
            # Kept only when there is an implementation for it to launch. A launcher
            # beside no `source.py` is not somebody's work to preserve: it is what an
            # agent left in a directory the harness had not written yet, and preserving
            # it would let that survive every later extraction.
            if path.is_file() and not regenerate and had_source:
                group.preserved = True
                continue
            path.write_text(_render_template(template, context))
        if bundle is not None:
            write_calls(bundle, directory, list(module_ids),
                        {m.id: list(m.submodules) for m in modules},
                        class_names, {m.id: list(m.layer_indices) for m in modules})
        (directory / "meta.yaml").write_text(yamlio.dumps({
            **group.to_dict(),
            "kind": representative.kind,
            "param_bytes": representative.param_bytes,
            "inputs": list(representative.inputs),
            "outputs": list(representative.outputs),
            "submodules": list(representative.submodules),
        }, sort_keys=False))
        groups.append(group)

    (root / "index.yaml").write_text(yamlio.dumps({
        "n_groups": len(groups),
        "n_modules": len(graph.partitioned_modules),
        "groups": [
            {**g.to_dict(), "directory": g.directory.name if g.directory else None}
            for g in groups
        ],
    }, sort_keys=False))
    return groups


def _module_config(model: Any, submodule_names: list[str],
                   fallback: dict[str, Any] | None = None) -> tuple[dict[str, Any], str]:
    """The config a module's own subtree was built from, and that config's class.

    The model's top-level config is not always what the module was constructed with.
    A multimodal checkpoint keeps the text stack's widths and head counts under
    ``text_config`` and hands *that* to the decoder layers, so building a layer from
    the top-level config gives a module whose shapes are library defaults — twice the
    heads the recorded weights have, in the case that found this. Walking up from the
    submodule to the nearest config gets the one the weights actually match.
    """
    from dataclasses import asdict, is_dataclass

    def settings_of(holder: Any) -> tuple[dict[str, Any], str] | None:
        candidate = getattr(holder, "config", None) if holder is not None else None
        if candidate is None:
            return None
        if hasattr(candidate, "to_dict"):
            payload = dict(candidate.to_dict())
            for private in PRIVATE_CONFIG_KEYS:
                value = getattr(candidate, private, None)
                if value is not None:
                    payload[private] = value
            return payload, type(candidate).__name__
        if is_dataclass(candidate) and not isinstance(candidate, type):
            return asdict(candidate), type(candidate).__name__
        if isinstance(candidate, dict):
            return dict(candidate), ""
        return None

    for name in submodule_names:
        parts = name.split(".")
        while parts:
            found = settings_of(_lookup(model, ".".join(parts)))
            if found is not None:
                return found
            parts.pop()
    return settings_of(model) or (dict(fallback or {}), "")


def _wanted_classes(principal: Any, classes: list[str]) -> list[str]:
    """Names, defined in the principal's own file, that this group's code needs.

    ``classes`` is dotted and spans the group's submodules and their children, so the
    framework's classes are in there too; those come from the installed framework and
    are not part of this file.
    """
    module = type(principal).__module__
    wanted = [type(principal).__name__]
    for dotted in classes:
        head, _, leaf = dotted.rpartition(".")
        if head == module and leaf not in wanted:
            wanted.append(leaf)
    return wanted


def vendor_code(run_root: str | Path, origin_files: list[str]) -> set[Path]:
    """Copy the model's own package into the run, beside ``modules/``.

    Returns the files copied. ``source.py`` is a slice of one file of that package and
    keeps its imports, so the sibling modules have to travel with it: DeepSeek's
    attention imports its GEMMs from ``kernel`` and its n-gram layout from ``engram``,
    and neither is anything this loop could vendor by slicing. Only the ``.py`` files
    beside each originating file are taken — not the weights, not the examples.
    """
    import shutil

    target = Path(run_root) / VENDOR_DIR
    copied: set[Path] = set()
    for origin in origin_files:
        source_dir = Path(origin).parent
        if not source_dir.is_dir():
            continue
        target.mkdir(parents=True, exist_ok=True)
        for path in sorted(source_dir.glob("*.py")):
            destination = target / path.name
            if not destination.is_file() or destination.stat().st_mtime < path.stat().st_mtime:
                shutil.copy2(path, destination)
            copied.add(destination)
    return copied


def write_calls(bundle: Any, directory: str | Path, module_ids: list[str],
                submodules: dict[str, list[str]], classes: list[str],
                layer_map: dict[str, list[int]]) -> Path:
    """Write the recorded calls and weight paths for one group, as plain JSON.

    This is what lets ``inference.py`` read its own inputs with nothing but ``json`` and
    ``torch``: every tensor it needs — arguments, cross-module state, the reference
    output, each weight — is named here by a path relative to this directory and by the
    dtype and shape to read those bytes as. The run's ``records.yaml`` says the same
    thing for the whole run at once, which is the harness's view; a module directory
    needs only its own slice and gets it without a YAML parser or this package.

    Written per group rather than per module because that is the unit that shares an
    implementation, and a group's modules differ in the tensors they point at.
    """
    directory = Path(directory)
    by_name = {entry.name: entry for entry in bundle.store.entries}

    def reference(name: str) -> dict[str, Any] | None:
        return _tensor_ref(bundle, directory, by_name.get(name))

    def rewrite(value: Any) -> Any:
        """The recorded tree with each tensor's store name replaced by its path."""
        from model_partition.trace import LIST_KEY, TENSOR_KEY, TUPLE_KEY, UNSUPPORTED_KEY

        if not isinstance(value, dict):
            return value
        if TENSOR_KEY in value:
            found = reference(value[TENSOR_KEY])
            # A tensor that was not dumped stays named: `inference.py` reports which one
            # is missing, which is more use than a null that fails somewhere later.
            return {TENSOR_KEY: found or value[TENSOR_KEY]}
        if UNSUPPORTED_KEY in value:
            return dict(value)
        for key in (TUPLE_KEY, LIST_KEY):
            if key in value:
                return {key: [rewrite(item) for item in value[key]]}
        return {key: rewrite(item) for key, item in value.items()}

    modules: dict[str, Any] = {}
    for module_id in module_ids:
        weights = _weight_refs(bundle, directory, module_id)
        steps: dict[str, list[dict[str, Any]]] = {}
        for record in sorted(bundle.select(module_id=module_id), key=lambda r: r.order):
            key = f"{record.sample_id}#{record.step}"
            steps.setdefault(key, []).append({
                "submodule": record.submodule,
                "sample_id": record.sample_id,
                "step": record.step,
                "sliced": record.sliced,
                "args": [rewrite(item) for item in record.args],
                "kwargs": rewrite(record.kwargs),
                "state": rewrite(record.state),
                "output": rewrite(record.output),
            })
        modules[module_id] = {
            "submodules": list(submodules.get(module_id) or []),
            "classes": list(classes),
            "layers": list(layer_map.get(module_id) or []),
            "weights": weights,
            # Parameters the run did not dump: read from the checkpoint during the run,
            # and materialized into the artifact before it is published.
            "weights_missing": [p for p in (bundle.weight_params.get(module_id) or [])
                                if p not in weights],
            "calls": steps,
        }

    path = directory / CALLS_FILENAME
    path.write_text(json.dumps({"version": 1, "modules": modules}, indent=1) + "\n")
    return path


def _tensor_ref(bundle: Any, directory: Path, entry: Any) -> dict[str, Any] | None:
    """Where one dumped tensor is, relative to a module directory, and how to read it."""
    import os

    if entry is None:
        return None
    return {
        "bin": os.path.relpath(bundle.store.root / entry.path, directory),
        "dtype": entry.dtype,
        "shape": list(entry.shape),
    }


def _weight_refs(bundle: Any, directory: Path, module_id: str) -> dict[str, Any]:
    """One module's dumped weights, keyed by original parameter name."""
    by_name = {entry.name: entry for entry in bundle.store.entries}
    weights: dict[str, Any] = {}
    for name in bundle.weights.get(module_id) or []:
        entry = by_name.get(name)
        found = _tensor_ref(bundle, directory, entry)
        if found is not None:
            weights[(entry.extra or {}).get("param") or name] = found
    return weights


def refresh_calls(bundle: Any, directory: str | Path) -> int:
    """Re-point one group's ``calls.json`` at the weights the run now holds.

    Publishing a run that traced without caching weights materializes them first — reads
    each one out of the checkpoint and dumps it beside the activations. Until this is
    called the module directory still says it has none, which is the difference between
    an artifact somebody can verify and one that needs the original 475 GiB checkpoint.
    """
    directory = Path(directory)
    path = directory / CALLS_FILENAME
    if not path.is_file():
        return 0
    payload = json.loads(path.read_text())
    updated = 0
    for module_id, entry in (payload.get("modules") or {}).items():
        weights = _weight_refs(bundle, directory, module_id)
        if weights == entry.get("weights"):
            continue
        entry["weights"] = weights
        entry["weights_missing"] = [p for p in (bundle.weight_params.get(module_id) or [])
                                    if p not in weights]
        updated += 1
    if updated:
        path.write_text(json.dumps(payload, indent=1) + "\n")
    return updated


#: The harness files a module directory needs to build and check itself, by package
#: path. ``artifact`` reads the recorded calls beside it; the launcher constructs the
#: module out of ``source.py``; ``compat`` applies the run's kernel replacements;
#: ``hardware`` is how a patch learns which card it is on; ``numerics`` decides whether
#: the result matches, and must be the same code the run judged by. Each of these imports only the others, which is the property that keeps
#: this list short — see the note at the top of the launcher.
RUNTIME_FILES = (
    "__init__.py",
    "hardware.py",
    "runtime/__init__.py",
    "runtime/artifact.py",
    "runtime/launcher.py",
    "runtime/compat.py",
    "verify/__init__.py",
    "verify/numerics.py",
)


def vendor_runtime(run_root: str | Path) -> Path:
    """Copy the launcher and its neighbours into the run, and return the import root.

    A module's ``inference.py`` builds from these. Copying them in is what makes the
    claim on the tin true — the artifact is the deliverable, and running a module out of
    it must not need this harness installed, only ``torch``. They are copied under their
    own package path so their imports of each other resolve unchanged: putting a path
    into ``sys.path`` is the whole of what ``inference.py`` has to do.
    """
    import shutil

    root = Path(run_root) / RUNTIME_DIR
    package = root / "model_partition"
    here = Path(__file__).resolve().parent
    for relative in RUNTIME_FILES:
        source = here / relative
        destination = package / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.is_file() or destination.stat().st_mtime < source.stat().st_mtime:
            shutil.copy2(source, destination)
    return root


def _is_main_guard(node: Any) -> bool:
    """True for a ``if __name__ == "__main__":`` block."""
    import ast

    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    left = node.test.left
    return (isinstance(left, ast.Name) and left.id == "__name__"
            and any(isinstance(c, ast.Constant) and c.value == "__main__"
                    for c in node.test.comparators))


def _slice_source(origin: Path, wanted: list[str]) -> tuple[str, int, int] | None:
    """The part of ``origin`` this module needs: its classes, and what they reference.

    Returns ``(text, definitions kept, definitions in the file)``.

    A modeling file holds the whole model — every layer variant, the vision tower, the
    generation wrapper. A module directory is one module, so copying all of it gives a
    kernel author thousands of lines to read past and makes the file look shared when
    it is not. This keeps the module-level statements every version of the file needs
    (its imports, its constants) plus the transitive closure of definitions the wanted
    classes actually reference, in their original order and text.

    Returns None when the result would not be trustworthy — a class that is not defined
    at the top level, a file that does not parse — in which case the caller copies the
    whole file rather than shipping something incomplete.
    """
    import ast

    try:
        text = origin.read_text()
        tree = ast.parse(text)
    except (OSError, SyntaxError):
        return None

    lines = text.splitlines(keepends=True)
    definitions: dict[str, Any] = {}
    order: list[tuple[int, int, str]] = []
    preamble: list[tuple[int, int]] = []
    kept_statements: list[Any] = []
    for node in tree.body:
        if _is_main_guard(node):
            # A demo, not part of the module. Left out both ways: keeping it would put
            # the vendor's whole-model script in a one-module directory, and seeding the
            # dependency closure from it drags in every class it touches — for DeepSeek
            # that is the entire file, which is how 31 of 31 definitions came along.
            continue
        start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        end = node.end_lineno or start
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            definitions[node.name] = node
            order.append((start, end, node.name))
        else:
            # Imports, constants, logger setup, `if TYPE_CHECKING` — cheap to keep and
            # the file does not import without them.
            preamble.append((start, end))
            kept_statements.append(node)

    missing = [name for name in wanted if name not in definitions]
    if not wanted or missing:
        return None

    def referenced(node: Any) -> set[str]:
        """Top-level definitions this node mentions by name."""
        found: set[str] = set()
        for inner in ast.walk(node):
            name = inner.id if isinstance(inner, ast.Name) else None
            if name is None and isinstance(inner, ast.Attribute):
                root = inner
                while isinstance(root, ast.Attribute):
                    root = root.value
                name = root.id if isinstance(root, ast.Name) else None
            if name in definitions:
                found.add(name)
        return found

    needed = set(wanted)
    # The preamble is kept whole, so whatever it constructs has to be kept too. A
    # module-level `shared_attn = SharedAttentionRuntime()` is the case that found this:
    # keeping the statement without the class gives a file that will not import.
    frontier = list(wanted)
    for node in kept_statements:
        frontier.extend(referenced(node) - needed)
    needed.update(frontier)
    while frontier:
        for name in referenced(definitions[frontier.pop()]) - needed:
            needed.add(name)
            frontier.append(name)

    # In file order, preamble and definitions together: a module-level statement can
    # depend on a class defined above it, and emitting every statement first would put
    # that use before its definition in a file that has to import.
    kept = sorted(preamble + [(start, end) for start, end, name in order if name in needed])
    parts = ["".join(lines[start - 1:end]) for start, end in kept]
    sliced = "\n".join(part.rstrip("\n") for part in parts) + "\n"
    try:
        ast.parse(sliced)
    except SyntaxError:
        return None
    return sliced, len(needed), len(definitions)


def _write_source(directory: Path, principal: Any, sources: list[str],
                  signature: str, files: list[str], classes: list[str] | None = None,
                  regenerate: bool = False) -> bool:
    """Write ``source.py``: the implementation if there is one, else an excerpt.

    Returns whether the result is importable. What makes the module directory a unit
    you can optimize on its own is that the code that runs is the code in front of you,
    so this takes the module's own classes out of the model's file verbatim — not the
    whole file, which is the entire model. An existing one is left alone for the same
    reason ``inference.py`` is: it is the file being optimized, and re-extracting must
    not throw that away.
    """
    import inspect

    destination = directory / SOURCE_FILENAME
    if destination.is_file() and not regenerate:
        previous = yamlio.load_path(directory / "meta.yaml") or {}
        return bool(previous.get("launchable", True))

    origin = None
    if principal is not None:
        try:
            candidate = Path(inspect.getfile(type(principal)))
            origin = candidate if candidate.is_file() else None
        except (OSError, TypeError):
            origin = None

    if origin is not None:
        result = _slice_source(origin, _wanted_classes(principal, classes or []))
        if result is None:
            body, scope = origin.read_text(), WHOLE_SCOPE
        else:
            body, kept, total = result
            scope = SLICED_SCOPE.format(kept=kept, total=total)
        destination.write_text(
            SOURCE_BANNER.format(signature=signature, origin=origin, scope=scope,
                                 class_name=type(principal).__name__)
            + body
        )
        return True

    provenance = ("\n".join(f"  {path}" for path in sorted(files))
                  or "  (not present in the loaded model)")
    destination.write_text(
        EXCERPT_HEADER.format(signature=signature, provenance=provenance)
        + "\n\n".join(sources)
    )
    return False


def _group_dirname(index: int, module: Any, signature: str) -> str:
    """Readable directory name: kind plus a short signature tag."""
    tag = signature.removeprefix("sig-")[:8] if signature.startswith("sig-") else signature[:8]
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in f"{module.kind}-{tag}")
    return f"{index:02d}-{safe}"
