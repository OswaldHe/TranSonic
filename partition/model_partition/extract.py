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

#: The implementation and its launcher are the loop's to edit, so they are written
#: once and then left alone — an optimization made to a module has to survive the
#: next extraction. Everything else belongs to the harness and is regenerated.
EDITABLE_TEMPLATES = {"inference.py": "module_inference.py.tmpl"}
SOURCE_FILENAME = "source.py"

#: The config the module's own subtree was built from, written beside it so the
#: directory carries everything needed to construct the module.
CONFIG_FILENAME = "config.json"

#: Config fields a framework keeps private and ``to_dict()`` leaves out, but which
#: decide what a module computes. The attention implementation is the one that
#: matters: a recorded call carries ``attention_mask=None`` when the traced kernel
#: applied causality itself, and rebuilding the module as eager attention then lets
#: every position see the future — a module that looks close and is wrong.
PRIVATE_CONFIG_KEYS = ("_attn_implementation",)

#: Where the dtype the model's activations flow in is recorded. Not a field of any
#: framework config: a quantized checkpoint names its *storage* dtype and computes in
#: another, and a module built in the wrong one has its kernels reject their own output
#: buffers. Read by the launcher before it constructs anything.
COMPUTE_DTYPE_KEY = "_compute_dtype"
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
# Its own imports resolve against the installed framework; the class bodies are this
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
    """
    param_names = param_names or {}
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
    for node in tree.body:
        start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        end = node.end_lineno or start
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            definitions[node.name] = node
            order.append((start, end, node.name))
        else:
            # Imports, constants, logger setup, `if TYPE_CHECKING` — cheap to keep and
            # the file does not import without them.
            preamble.append((start, end))

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
    for node in tree.body:
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
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
