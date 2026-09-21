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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from model_partition import yamlio

from model_partition.hardware import format_bytes
from model_partition.planner.graph import PartitionGraph
from model_partition.trace import _lookup

#: The implementation is the loop's to edit, so it is written once and then left
#: alone. Everything else belongs to the harness and is regenerated every run.
EDITABLE_TEMPLATES = {"inference.py": "module_inference.py.tmpl"}
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

#: Prepended to a verbatim implementation file, as a comment so the file still
#: imports. This is the file `inference.py` loads and the file to optimize.
SOURCE_BANNER = """\
# This is the implementation for partition group `{signature}`, copied verbatim from
#   {origin}
# `inference.py` beside it imports this file and calls `{class_name}` — so editing
# here is what changes what runs, and `verify.py` tells you whether the result still
# reproduces the dumped reference. Nothing here reads the checkpoint.
#
# Its own imports resolve against the installed framework; the class bodies are this
# file's.

"""

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

#: Where the verbatim originals go, shared across groups since several of them come
#: from the same file.
SOURCE_FILES_DIR = "source_files"


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
) -> list[ExtractedGroup]:
    """Write one implementation directory per signature group.

    ``param_names`` maps a module id to the original parameter names its
    implementation will be handed, so the generated file can document them.

    An existing ``inference.py`` is preserved: it is the loop's editable surface
    and may carry the agent's numerical fixes. ``regenerate=True`` overwrites it.
    """
    param_names = param_names or {}
    # The class the model's own config is, rather than whichever name in the source
    # file happens to end in "Config" — a modeling file imports several.
    config_class = type(getattr(model, "config", None)).__name__
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    groups: list[ExtractedGroup] = []

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
        group.classes = names
        group.source_files = files
        group.source_lines = sum(s.count("\n") for s in sources)

        directory = root / _group_dirname(index, representative, signature)
        directory.mkdir(parents=True, exist_ok=True)
        group.directory = directory

        principal = _principal(targets) if targets else None
        group.source_module = type(principal).__module__ if principal is not None else ""
        group.launchable = _write_source(directory, principal, sources, signature, files)
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
            if path.is_file() and not regenerate:
                group.preserved = True
                continue
            path.write_text(_render_template(template, context))
        (directory / "meta.yaml").write_text(yamlio.dumps({
            **group.to_dict(),
            "kind": representative.kind,
            "param_bytes": representative.param_bytes,
            "inputs": list(representative.inputs),
            "outputs": list(representative.outputs),
            "submodules": list(representative.submodules),
        }, sort_keys=False))
        groups.append(group)

    _copy_source_files(groups, root)
    (root / "index.yaml").write_text(yamlio.dumps({
        "n_groups": len(groups),
        "n_modules": len(graph.partitioned_modules),
        "groups": [
            {**g.to_dict(), "directory": g.directory.name if g.directory else None}
            for g in groups
        ],
    }, sort_keys=False))
    return groups


def _write_source(directory: Path, principal: Any, sources: list[str],
                  signature: str, files: list[str]) -> bool:
    """Write ``source.py``: the implementation if there is one, else an excerpt.

    Returns whether the result is importable. Copying the defining file verbatim is
    what makes the module directory a unit you can optimize on its own — the code that
    runs is the code in front of you, not whatever happens to be installed.
    """
    import inspect

    origin = None
    if principal is not None:
        try:
            candidate = Path(inspect.getfile(type(principal)))
            origin = candidate if candidate.is_file() else None
        except (OSError, TypeError):
            origin = None

    if origin is not None:
        (directory / "source.py").write_text(
            SOURCE_BANNER.format(signature=signature, origin=origin,
                                 class_name=type(principal).__name__)
            + origin.read_text()
        )
        return True

    provenance = ("\n".join(f"  {path}" for path in sorted(files))
                  or "  (not present in the loaded model)")
    (directory / "source.py").write_text(
        EXCERPT_HEADER.format(signature=signature, provenance=provenance)
        + "\n\n".join(sources)
    )
    return False


def _copy_source_files(groups: list[ExtractedGroup], root: Path) -> int:
    """Copy each implementation's originating file verbatim; returns how many.

    The extracted ``source.py`` is a readable subset and does not import. A porter
    wants the real file, and the files are shared between groups, so they are written
    once beside them.
    """
    import shutil

    target = root / SOURCE_FILES_DIR
    wanted = {path for group in groups for path in group.source_files}
    if not wanted:
        return 0
    target.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in sorted(wanted):
        origin = Path(path)
        if not origin.is_file():
            continue
        destination = target / origin.name
        if destination.exists() and destination.read_bytes() == origin.read_bytes():
            copied += 1
            continue
        shutil.copy2(origin, destination)
        copied += 1
    return copied


def _group_dirname(index: int, module: Any, signature: str) -> str:
    """Readable directory name: kind plus a short signature tag."""
    tag = signature.removeprefix("sig-")[:8] if signature.startswith("sig-") else signature[:8]
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in f"{module.kind}-{tag}")
    return f"{index:02d}-{safe}"
