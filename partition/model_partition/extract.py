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

import yaml

from model_partition.planner.graph import PartitionGraph
from model_partition.trace import _lookup

TEMPLATE_NAME = "module_harness.py.tmpl"

SOURCE_HEADER = '''\
"""Source of the real implementation for partition group `{signature}`.

Collected with inspect.getsource from the loaded model. Provenance:
{provenance}

This is a reading and porting reference, not a standalone module — imports and
helpers from the original package are not inlined.
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
    directory: Path | None = None
    source_lines: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "signature": self.signature,
            "module_ids": list(self.module_ids),
            "layer_indices": list(self.layer_indices),
            "class_name": self.class_name,
            "classes": list(self.classes),
            "source_files": list(self.source_files),
            "source_lines": self.source_lines,
        }


def _render_template(context: dict[str, Any]) -> str:
    from jinja2 import Environment, FileSystemLoader, StrictUndefined

    templates = Path(__file__).resolve().parent / "templates"
    env = Environment(loader=FileSystemLoader(str(templates)), undefined=StrictUndefined,
                      keep_trailing_newline=True)
    return env.get_template(TEMPLATE_NAME).render(**context)


def collect_sources(module: Any, max_classes: int = 40) -> tuple[list[str], list[str], str]:
    """Source of a module's class and its children's classes.

    Returns ``(sources, class_names, provenance)``, deduplicated and ordered from
    the root class outward.
    """
    seen: set[type] = set()
    sources: list[str] = []
    names: list[str] = []
    files: set[str] = set()

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
            files.add(str(inspect.getfile(cls)))
        except (OSError, TypeError):
            names.append(f"{cls.__module__}.{cls.__qualname__} (source unavailable)")
        if hasattr(obj, "children"):
            for child in obj.children():
                visit(child)

    visit(module)
    provenance = "\n".join(f"  {path}" for path in sorted(files)) or "  (unknown)"
    return sources, names, provenance


def extract(
    graph: PartitionGraph,
    model: Any,
    out_dir: str | Path,
    run_root: str | Path = ".",
    sample_ids: list[str] | None = None,
    weight_tensors: dict[str, list[str]] | None = None,
) -> list[ExtractedGroup]:
    """Write one implementation directory per signature group."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    groups: list[ExtractedGroup] = []

    for index, (signature, module_ids) in enumerate(sorted(graph.signature_groups().items())):
        modules = [graph.by_id(mid) for mid in module_ids]
        representative = modules[0]
        target = _lookup(model, representative.submodules[0]) if representative.submodules else None

        group = ExtractedGroup(
            signature=signature,
            module_ids=list(module_ids),
            layer_indices=sorted({i for m in modules for i in m.layer_indices}),
        )
        if target is None:
            group.class_name = "(not instantiated)"
            sources, names, provenance = [], [], "  (module not present in the loaded model)"
        else:
            sources, names, provenance = collect_sources(target)
            group.class_name = type(target).__name__
        group.classes = names
        group.source_lines = sum(s.count("\n") for s in sources)

        directory = root / _group_dirname(index, representative, signature)
        directory.mkdir(parents=True, exist_ok=True)
        group.directory = directory

        (directory / "source.py").write_text(
            SOURCE_HEADER.format(signature=signature, provenance=provenance)
            + "\n\n".join(sources)
        )
        (directory / "module.py").write_text(_render_template({
            "signature": signature,
            "module_ids": list(module_ids),
            "layer_map": {m.id: list(m.layer_indices) for m in modules},
            "class_name": group.class_name,
            "sample_ids": list(sample_ids or []),
            "weight_tensors": {mid: list((weight_tensors or {}).get(mid, [])) for mid in module_ids},
            "run_root": str(run_root),
        }))
        (directory / "meta.yaml").write_text(yaml.safe_dump({
            **group.to_dict(),
            "kind": representative.kind,
            "param_bytes": representative.param_bytes,
            "inputs": list(representative.inputs),
            "outputs": list(representative.outputs),
            "submodules": list(representative.submodules),
        }, sort_keys=False))
        groups.append(group)

    (root / "index.yaml").write_text(yaml.safe_dump({
        "n_groups": len(groups),
        "n_modules": len(graph.partitioned_modules),
        "groups": [
            {**g.to_dict(), "directory": g.directory.name if g.directory else None}
            for g in groups
        ],
    }, sort_keys=False))
    return groups


def _group_dirname(index: int, module: Any, signature: str) -> str:
    """Readable directory name: kind plus a short signature tag."""
    tag = signature.removeprefix("sig-")[:8] if signature.startswith("sig-") else signature[:8]
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in f"{module.kind}-{tag}")
    return f"{index:02d}-{safe}"
