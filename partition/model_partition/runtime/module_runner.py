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

from model_partition import yamlio

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
        return self._distinct("sample_id")

    def module_ids(self) -> list[str]:
        return self._distinct("module_id")

    def _distinct(self, field_name: str) -> list[str]:
        """Values of one record field, in the order the trace first saw them."""
        seen: list[str] = []
        for record in self.records:
            value = getattr(record, field_name)
            if value not in seen:
                seen.append(value)
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
        path.write_text(yamlio.dumps({
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
        payload = yamlio.load_path(path) or {}
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


def decode_group_call(records: list[CallRecord], store: TensorStore,
                      device: str = "cpu") -> tuple[tuple, dict]:
    """Arguments for a whole sequential group, not just its first submodule.

    A group can be heterogeneous — a normalization followed by attention — and then
    the first submodule's recorded call says nothing about the rotary embeddings and
    masks the attention needs. The value flowing in comes from the group's entry point;
    everything beyond it is the union over the group, so whatever the module does
    internally, it was given everything.

    Extra *positional* arguments count too, and not every model passes its extras by
    keyword: DeepSeek's attention takes ``(x, start_pos, ...)``, so a group of
    ``[attn_norm, attn]`` driven by the norm's one argument alone cannot call the
    attention at all. They come from whichever recorded call carries the most, and
    :func:`~model_partition.runtime.launcher._chain` hands each submodule as many as its
    own signature takes.

    First occurrence wins, so a keyword that changes down the group keeps the value
    the group's entry point saw.
    """
    if not records:
        return (), {}
    args, kwargs = decode_call(records[0], store, device)
    merged = dict(kwargs)
    extras: tuple = args[1:]
    for record in records[1:]:
        other, extra = decode_call(record, store, device)
        for key, value in extra.items():
            merged.setdefault(key, value)
        if len(other) - 1 > len(extras):
            extras = other[1:]
    return (args[:1] + extras if args else extras), merged


def apply_state(source: Any, record: Any, store: TensorStore, device: str = "cpu") -> int:
    """Put the cross-module state a recorded call ran on back where it reads it.

    A model that passes tensors between modules through a module-level object — the
    compressed KV DeepSeek's attention layers share — leaves a consumer with no argument
    naming the largest thing it reads. The trace records it per call; this restores it on
    the module's own ``source.py``, so the check is of the module against its reference
    and not of one module against another having run first.
    """
    from model_partition.trace import decode_value

    if not getattr(record, "state", None):
        return 0
    load = tensor_loader(store, device)
    applied = 0
    for dotted, encoded in record.state.items():
        holder_name, _, attribute = dotted.rpartition(".")
        holder = getattr(source, holder_name, None)
        if holder is None:
            continue
        setattr(holder, attribute, decode_value(encoded, load))
        applied += 1
    return applied


def load_named_weights(bundle: TraceBundle, module_id: str, device: str = "cpu") -> dict[str, Any]:
    """Load a module's weights keyed by their original parameter names.

    This is what an extracted implementation is handed, so its code reads
    ``weights["model.layers.0.self_attn.q_proj.weight"]`` rather than a flattened
    filesystem name.

    Both sources contribute, and the dumps win. A run that does not cache weights still
    dumps the derived buffers — a rotary table is in no checkpoint — so neither source
    is complete on its own: the recording holds what the checkpoint cannot, and the
    checkpoint holds what the run chose not to copy.
    """
    result: dict[str, Any] = {}
    dumped = bundle.weights.get(module_id) or []
    if dumped:
        load = tensor_loader(bundle.store, device)
        by_name = {entry.name: entry for entry in bundle.store.entries}
        for name in dumped:
            entry = by_name.get(name)
            original = (entry.extra or {}).get("param") if entry else None
            result[original or name] = load(name)

    wanted = [p for p in (bundle.weight_params.get(module_id) or []) if p not in result]
    if wanted and bundle.checkpoint is not None:
        result.update(bundle.checkpoint.load(wanted, device=device))
    return result


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
