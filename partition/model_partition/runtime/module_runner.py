# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay a single module from its dumped inputs and weights.

Verification reloads weights *from the dump* rather than reusing the live model,
so a passing module proves the dumped artifacts are sufficient to reproduce it —
which is the property that matters when the same data is later replayed against
another implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from model_partition.trace import CallRecord, _lookup, decode_value
from model_partition.tensorstore import TensorStore

RECORDS_NAME = "records.yaml"


class ReplayError(RuntimeError):
    """Raised when a module cannot be replayed from its artifacts."""


@dataclass
class TraceBundle:
    """Everything tracing produced: records plus the tensor store."""

    store: TensorStore
    records: list[CallRecord] = field(default_factory=list)
    #: module id -> dumped tensor names in the store. Empty when weights were not
    #: cached, in which case ``weight_params`` says what to read instead.
    weights: dict[str, list[str]] = field(default_factory=dict)
    #: module id -> original parameter names the module owns. Always populated.
    weight_params: dict[str, list[str]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    #: Set at runtime when the dumps were skipped, never serialized.
    checkpoint: Any = field(default=None, repr=False)

    @property
    def root(self) -> Path:
        return self.store.root

    @property
    def config(self) -> dict[str, Any]:
        """The model config tracing resolved, for implementations that need it."""
        return dict(self.metadata.get("config") or {})

    def sample_ids(self) -> list[str]:
        seen: list[str] = []
        for record in self.records:
            if record.sample_id not in seen:
                seen.append(record.sample_id)
        return seen

    def module_ids(self) -> list[str]:
        seen: list[str] = []
        for record in self.records:
            if record.module_id not in seen:
                seen.append(record.module_id)
        return seen

    def select(self, module_id: str | None = None, sample_id: str | None = None,
               step: int | None = None) -> list[CallRecord]:
        return sorted(
            (r for r in self.records
             if (module_id is None or r.module_id == module_id)
             and (sample_id is None or r.sample_id == sample_id)
             and (step is None or r.step == step)),
            key=lambda r: r.order,
        )

    def save(self) -> Path:
        self.store.save_manifest(self.metadata)
        path = self.root / RECORDS_NAME
        path.write_text(yaml.safe_dump({
            "metadata": self.metadata,
            "weights": self.weights,
            "weight_params": self.weight_params,
            "records": [r.to_dict() for r in self.records],
        }, sort_keys=False))
        return path

    @classmethod
    def load(cls, root: str | Path) -> TraceBundle:
        store = TensorStore.load(root)
        path = Path(root) / RECORDS_NAME
        if not path.is_file():
            raise ReplayError(f"No trace records at {path}")
        payload = yaml.safe_load(path.read_text()) or {}
        return cls(
            store=store,
            records=[CallRecord.from_dict(r) for r in payload.get("records", [])],
            weights=payload.get("weights") or {},
            weight_params=payload.get("weight_params") or {},
            metadata=payload.get("metadata") or {},
        )


def tensor_loader(store: TensorStore, device: str = "cpu"):
    """Build a ``name -> tensor`` loader backed by the store's manifest."""
    by_name = {entry.name: entry for entry in store.entries}

    def load(name: str):
        entry = by_name.get(name)
        if entry is None:
            raise ReplayError(f"Tensor {name!r} is not in the manifest")
        return store.read_torch(entry, device=device)

    return load


def decode_call(record: CallRecord, store: TensorStore, device: str = "cpu") -> tuple[tuple, dict]:
    """Rebuild a record's positional args and kwargs."""
    load = tensor_loader(store, device)
    args = tuple(decode_value(item, load) for item in record.args)
    kwargs = decode_value(record.kwargs, load)
    return args, kwargs


def load_dumped_weights(bundle: TraceBundle, module_id: str, device: str = "cpu") -> dict[str, Any]:
    """Load a module's dumped weights, keyed by their filesystem-safe names."""
    load = tensor_loader(bundle.store, device)
    return {name: load(name) for name in bundle.weights.get(module_id, [])}


def load_named_weights(bundle: TraceBundle, module_id: str, device: str = "cpu") -> dict[str, Any]:
    """Load a module's weights keyed by their original parameter names.

    This is what an extracted implementation is handed, so its code reads
    ``weights["model.layers.0.self_attn.q_proj.weight"]`` rather than a flattened
    filesystem name. Values come from the dumps, or from the checkpoint when the
    run chose not to cache them.
    """
    dumped = bundle.weights.get(module_id)
    if dumped:
        load = tensor_loader(bundle.store, device)
        by_name = {entry.name: entry for entry in bundle.store.entries}
        result: dict[str, Any] = {}
        for name in dumped:
            entry = by_name.get(name)
            original = (entry.extra or {}).get("param") if entry else None
            result[original or name] = load(name)
        return result

    params = bundle.weight_params.get(module_id) or []
    if params and bundle.checkpoint is not None:
        return bundle.checkpoint.load(params, device=device)
    return {}


def apply_named_weights(model: Any, weights: dict[str, Any], graph_module: Any) -> int:
    """Overwrite a module's live parameters from ``{original name: tensor}``.

    Returns the number of tensors applied. One place decides how a module's
    parameters are matched, whether they came from the dumps or the checkpoint.
    """
    import torch

    applied = 0
    for submodule_name in graph_module.submodules:
        submodule = _lookup(model, submodule_name)
        if submodule is None or not hasattr(submodule, "named_parameters"):
            continue
        items = list(submodule.named_parameters()) + list(submodule.named_buffers())
        for param_name, tensor in items:
            full = f"{submodule_name}.{param_name}" if param_name else submodule_name
            candidate = weights.get(full)
            if candidate is None:
                continue
            with torch.no_grad():
                tensor.copy_(candidate.reshape(tensor.shape).to(tensor.dtype))
            applied += 1
    return applied


def apply_dumped_weights(model: Any, bundle: TraceBundle, module_id: str, graph_module: Any,
                         device: str = "cpu") -> int:
    """Overwrite a module's live parameters with its recorded ones."""
    weights = load_named_weights(bundle, module_id, device=device)
    return apply_named_weights(model, weights, graph_module)


def replay_record(model: Any, record: CallRecord, store: TensorStore, device: str = "cpu") -> Any:
    """Call the recorded submodule with its recorded arguments."""
    import torch

    submodule = _lookup(model, record.submodule)
    if submodule is None:
        raise ReplayError(f"Submodule {record.submodule!r} is absent from this model")
    if record.has_unsupported():
        raise ReplayError(
            f"{record.module_id}: recorded arguments include an unserializable value; "
            "trace with use_cache disabled or extend the encoder"
        )
    args, kwargs = decode_call(record, store, device)
    with torch.no_grad():
        return submodule(*args, **kwargs)


def expected_output(record: CallRecord, store: TensorStore, device: str = "cpu") -> Any:
    """The output the trace recorded for this call."""
    load = tensor_loader(store, device)
    return decode_value(record.output, load)


def first_tensor(value: Any) -> Any | None:
    """The first tensor in a possibly nested output structure."""
    from model_partition.trace import is_tensor

    if is_tensor(value):
        return value
    if isinstance(value, dict):
        for item in value.values():
            found = first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, (list, tuple)):
        for item in value:
            found = first_tensor(item)
            if found is not None:
                return found
    return None
