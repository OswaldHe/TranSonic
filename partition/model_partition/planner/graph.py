# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The partition graph: modules as nodes, named tensors as edges.

Deliberately a DAG rather than a chain. Modern architectures route tensors
across non-adjacent layers — DeepSeek V4.1 shares KV via ``kv_source_layer_ids``
and feeds a sparse-attention indexer from ``index_source_layer_ids`` — so a
linear pipeline cannot express the real dataflow.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

#: Recommended module kinds. ``decoder_layers`` is the repeating stack; the rest are
#: either one-off (embed, lm_head) or architecture-specific. Not a closed set: an
#: architecture the list does not anticipate should be nameable, and rejecting a
#: whole plan over an unlisted kind would throw away a good partition. A kind
#: outside this list is reported as a warning instead.
KINDS = (
    "embed", "decoder_layers", "attention", "rope", "moe_router", "moe_experts",
    "moe_shared", "mlp", "norm", "lm_head", "vision", "mtp", "engram", "other",
)

#: A kind must at least look like one: lowercase, no spaces.
KIND_RE = re.compile(r"^[a-z][a-z0-9_]*$")

#: How a module's submodules relate to each other. ``sequential`` chains them —
#: a stack of like decoder layers. ``parallel`` gives each the same position in
#: the dataflow — a group of MoE experts, which all read the routed activation
#: and none of which feeds the next.
COMPOSITIONS = ("sequential", "parallel")


class GraphError(ValueError):
    """Raised when a graph is malformed or violates the budget."""


@dataclass
class TensorRef:
    """A logical tensor flowing between modules.

    ``shape`` entries are ints or symbolic names (``"batch"``, ``"seq"``).
    """

    name: str
    dtype: str = "bfloat16"
    shape: list[int | str] = field(default_factory=list)
    kind: str = "activation"  # activation | kv | routing | logits | index

    def bytes_per_token(self, dtype_bytes: int | None = None) -> int:
        """Bytes for one sequence position, ignoring symbolic batch/seq dims."""
        width = dtype_bytes if dtype_bytes is not None else DTYPE_BYTES.get(self.dtype, 2)
        product = 1
        for dim in self.shape:
            if isinstance(dim, int):
                product *= dim
        return product * width

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "dtype": self.dtype, "shape": list(self.shape), "kind": self.kind}


DTYPE_BYTES = {
    "float32": 4, "float16": 2, "bfloat16": 2, "float8_e4m3fn": 1,
    "float8_e5m2": 1, "int8": 1, "uint8": 1, "int32": 4, "int64": 8, "bool": 1,
    "fp4_packed": 1,
}


@dataclass
class ModuleNode:
    """One partition unit: what it contains, costs, and consumes/produces."""

    id: str
    kind: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    layer_indices: list[int] = field(default_factory=list)
    submodules: list[str] = field(default_factory=list)
    param_bytes: int = 0
    activation_bytes: int = 0
    kv_bytes: int = 0
    #: Dedup key: modules sharing a signature get one extracted implementation.
    code_signature: str | None = None
    #: False for nodes discovered but out of scope (vision, MTP, engram).
    partitioned: bool = True
    expert_range: list[int] = field(default_factory=list)
    #: One of :data:`COMPOSITIONS`.
    composition: str = "sequential"
    notes: str = ""

    @property
    def resident_bytes(self) -> int:
        return self.param_bytes + self.activation_bytes + self.kv_bytes

    @property
    def functional(self) -> bool:
        """True when this node is pure tensor algebra with no submodule of its own.

        An expert-combine step is the example: it sums partials into the residual
        stream. It belongs in the plan — a kernel has to implement it — but a
        hook-based trace has nothing to attach to, so it carries no reference of
        its own and is reported as unverified rather than silently passing.
        """
        return self.partitioned and not self.submodules

    @property
    def is_parallel(self) -> bool:
        return self.composition == "parallel" and len(self.submodules) > 1

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "param_bytes": self.param_bytes,
            "activation_bytes": self.activation_bytes,
            "partitioned": self.partitioned,
        }
        if self.composition != "sequential":
            data["composition"] = self.composition
        for key, value in (
            ("layer_indices", self.layer_indices),
            ("submodules", self.submodules),
            ("expert_range", self.expert_range),
        ):
            if value:
                data[key] = list(value)
        if self.kv_bytes:
            data["kv_bytes"] = self.kv_bytes
        if self.code_signature:
            data["code_signature"] = self.code_signature
        if self.notes:
            data["notes"] = self.notes
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModuleNode:
        if not isinstance(data, dict):
            raise GraphError(f"Module entry must be a mapping, got {type(data).__name__}")
        missing = {"id", "kind"} - set(data)
        if missing:
            raise GraphError(f"Module missing required key(s): {', '.join(sorted(missing))}")
        if not KIND_RE.match(str(data["kind"])):
            raise GraphError(
                f"Module {data['id']!r} has a malformed kind {data['kind']!r}; "
                f"use a lowercase name, ideally one of {KINDS}"
            )
        composition = str(data.get("composition", "sequential"))
        if composition not in COMPOSITIONS:
            raise GraphError(
                f"Module {data['id']!r} has unknown composition {composition!r}; "
                f"expected one of {COMPOSITIONS}"
            )
        return cls(
            id=str(data["id"]),
            kind=str(data["kind"]),
            inputs=list(data.get("inputs") or []),
            outputs=list(data.get("outputs") or []),
            layer_indices=list(data.get("layer_indices") or []),
            submodules=list(data.get("submodules") or []),
            param_bytes=int(data.get("param_bytes") or 0),
            activation_bytes=int(data.get("activation_bytes") or 0),
            kv_bytes=int(data.get("kv_bytes") or 0),
            code_signature=data.get("code_signature"),
            partitioned=bool(data.get("partitioned", True)),
            expert_range=list(data.get("expert_range") or []),
            composition=composition,
            notes=str(data.get("notes") or ""),
        )


@dataclass
class PartitionGraph:
    """A complete partition plan."""

    model: str
    modules: list[ModuleNode] = field(default_factory=list)
    tensors: dict[str, TensorRef] = field(default_factory=dict)
    entry_tensors: list[str] = field(default_factory=list)
    output_tensors: list[str] = field(default_factory=list)
    revision: str | None = None
    dtype: str = "bfloat16"
    budget_bytes: int = 0
    num_layers: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- lookups ---------------------------------------------------------------

    def by_id(self, module_id: str) -> ModuleNode:
        for module in self.modules:
            if module.id == module_id:
                return module
        raise KeyError(module_id)

    @property
    def partitioned_modules(self) -> list[ModuleNode]:
        return [m for m in self.modules if m.partitioned]

    def producers(self) -> dict[str, str]:
        """Map tensor name -> producing module id."""
        result: dict[str, str] = {}
        for module in self.modules:
            for name in module.outputs:
                result[name] = module.id
        return result

    def signature_groups(self) -> dict[str, list[str]]:
        """Map code signature -> module ids sharing it (dedup groups)."""
        groups: dict[str, list[str]] = defaultdict(list)
        for module in self.partitioned_modules:
            groups[module.code_signature or module.id].append(module.id)
        return dict(groups)

    @property
    def functional_modules(self) -> list[ModuleNode]:
        return [m for m in self.partitioned_modules if m.functional]

    def topological_order(self) -> list[str]:
        """Module ids in dependency order. Raises :class:`GraphError` on a cycle.

        A module listing one of its own outputs as an input is a cycle of length
        one and is reported as such: it cannot be scheduled, and dropping the
        self-edge would hide the mistake instead of surfacing it.
        """
        producers = self.producers()
        entry = set(self.entry_tensors)
        pending = {m.id: {producers[t] for t in m.inputs if t in producers}
                   for m in self.modules}
        for module in self.modules:
            for name in module.inputs:
                if name not in producers and name not in entry:
                    raise GraphError(f"Module {module.id!r} consumes undefined tensor {name!r}")

        ordered: list[str] = []
        remaining = dict(pending)
        while remaining:
            ready = sorted(mid for mid, deps in remaining.items() if not deps - set(ordered))
            if not ready:
                raise GraphError(f"Cycle detected among modules: {', '.join(sorted(remaining))}")
            ordered.extend(ready)
            for mid in ready:
                remaining.pop(mid)
        return ordered

    # -- validation ------------------------------------------------------------

    def validate(self, budget_bytes: int | None = None) -> list[str]:
        """Check structural integrity and the memory budget.

        Raises :class:`GraphError` for structural faults; returns warnings for
        things that are suspicious but runnable.
        """
        warnings: list[str] = []
        if not self.modules:
            raise GraphError("Graph has no modules")

        seen: set[str] = set()
        for module in self.modules:
            if module.id in seen:
                raise GraphError(f"Duplicate module id {module.id!r}")
            seen.add(module.id)

        produced = defaultdict(list)
        for module in self.modules:
            for name in module.outputs:
                produced[name].append(module.id)
        for name, owners in produced.items():
            if len(owners) > 1:
                raise GraphError(f"Tensor {name!r} produced by multiple modules: {', '.join(owners)}")

        self.topological_order()

        for name in self.output_tensors:
            if name not in produced and name not in self.entry_tensors:
                raise GraphError(f"Declared output tensor {name!r} is never produced")

        unusual = sorted({m.kind for m in self.modules if m.kind not in KINDS})
        if unusual:
            warnings.append(
                f"Module kind(s) outside the recommended set: {', '.join(unusual)}"
            )

        functional = [m.id for m in self.functional_modules]
        if functional:
            warnings.append(
                f"Functional module(s) with no submodule of their own, so no traced "
                f"reference and no numeric verification: {', '.join(functional)}"
            )

        known = set(self.tensors)
        referenced = {n for m in self.modules for n in (*m.inputs, *m.outputs)} | set(self.entry_tensors)
        undeclared = referenced - known
        if undeclared:
            warnings.append(f"Tensors referenced without a declaration: {', '.join(sorted(undeclared))}")

        ceiling = budget_bytes if budget_bytes is not None else self.budget_bytes
        if ceiling:
            for module in self.partitioned_modules:
                if module.resident_bytes > ceiling:
                    raise GraphError(
                        f"Module {module.id!r} needs {module.resident_bytes} bytes, "
                        f"over the {ceiling}-byte budget"
                    )

        if self.num_layers:
            covered: dict[int, list[str]] = defaultdict(list)
            for module in self.partitioned_modules:
                for index in module.layer_indices:
                    covered[index].append(module.id)
            missing = sorted(set(range(self.num_layers)) - set(covered))
            if missing:
                warnings.append(f"Layers not covered by any partitioned module: {missing}")
            doubled = {i: ids for i, ids in covered.items() if len(ids) > 1}
            if doubled:
                # Legitimate when a layer is split (attention | router | experts).
                kinds = {self.by_id(mid).kind for ids in doubled.values() for mid in ids}
                if kinds <= {"decoder_layers"}:
                    raise GraphError(f"Layers assigned to multiple decoder modules: {sorted(doubled)}")
        return warnings

    # -- (de)serialization -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "revision": self.revision,
            "dtype": self.dtype,
            "budget_bytes": self.budget_bytes,
            "num_layers": self.num_layers,
            "entry_tensors": list(self.entry_tensors),
            "output_tensors": list(self.output_tensors),
            "tensors": [t.to_dict() for t in self.tensors.values()],
            "modules": [m.to_dict() for m in self.modules],
            "metadata": dict(self.metadata),
        }

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_yaml())
        return target

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PartitionGraph:
        if not isinstance(data, dict):
            raise GraphError(f"Graph must be a mapping, got {type(data).__name__}")
        if "model" not in data:
            raise GraphError("Graph missing required key 'model'")
        raw_tensors = data.get("tensors") or []
        if isinstance(raw_tensors, dict):
            raw_tensors = [{"name": k, **(v or {})} for k, v in raw_tensors.items()]
        tensors: dict[str, TensorRef] = {}
        for entry in raw_tensors:
            if "name" not in entry:
                raise GraphError("Tensor entry missing 'name'")
            tensors[entry["name"]] = TensorRef(
                name=str(entry["name"]),
                dtype=str(entry.get("dtype", "bfloat16")),
                shape=list(entry.get("shape") or []),
                kind=str(entry.get("kind", "activation")),
            )
        return cls(
            model=str(data["model"]),
            modules=[ModuleNode.from_dict(m) for m in (data.get("modules") or [])],
            tensors=tensors,
            entry_tensors=list(data.get("entry_tensors") or []),
            output_tensors=list(data.get("output_tensors") or []),
            revision=data.get("revision"),
            dtype=str(data.get("dtype", "bfloat16")),
            budget_bytes=int(data.get("budget_bytes") or 0),
            num_layers=int(data.get("num_layers") or 0),
            metadata=dict(data.get("metadata") or {}),
        )

    @classmethod
    def load(cls, path: str | Path) -> PartitionGraph:
        source = Path(path)
        if not source.is_file():
            raise GraphError(f"Partition graph not found: {source}")
        try:
            data = yaml.safe_load(source.read_text())
        except yaml.YAMLError as exc:
            raise GraphError(f"Invalid YAML in {source}: {exc}") from exc
        return cls.from_dict(data or {})
