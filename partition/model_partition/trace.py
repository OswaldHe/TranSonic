# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture real per-module IO by hooking a full-model forward.

Hooks record the exact positional args and kwargs each submodule received, not
just its hidden state. That is what keeps extraction and verification
architecture-agnostic: replaying a module means calling it with the arguments it
actually got, so rotary embeddings, attention masks and other per-architecture
extras need no special handling.

Tensors are dumped to the tensor store and referenced by name; non-tensor values
are inlined; anything unserializable is marked so replay can report honestly
instead of silently substituting a default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from model_partition.planner.graph import PartitionGraph
from model_partition.storage import DumpPolicy
from model_partition.tensorstore import SliceInfo, TensorStore

TENSOR_KEY = "__tensor__"
LIST_KEY = "__list__"
TUPLE_KEY = "__tuple__"
UNSUPPORTED_KEY = "__unsupported__"


class TraceError(RuntimeError):
    """Raised when tracing cannot produce a usable record."""


class DumpBudgetExceeded(TraceError):
    """Raised when tracing would write past ``DumpPolicy.max_total_bytes``."""


@dataclass
class CallRecord:
    """One submodule invocation: its arguments and its output."""

    module_id: str
    submodule: str
    sample_id: str
    step: int = 0
    args: list[Any] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)
    output: Any = None
    order: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_id": self.module_id, "submodule": self.submodule,
            "sample_id": self.sample_id, "step": self.step, "order": self.order,
            "args": self.args, "kwargs": self.kwargs, "output": self.output,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CallRecord:
        return cls(
            module_id=data["module_id"], submodule=data["submodule"],
            sample_id=data["sample_id"], step=int(data.get("step", 0)),
            args=data.get("args") or [], kwargs=data.get("kwargs") or {},
            output=data.get("output"), order=int(data.get("order", 0)),
        )

    def tensor_names(self) -> list[str]:
        names: list[str] = []
        _collect_tensor_names(self.args, names)
        _collect_tensor_names(self.kwargs, names)
        _collect_tensor_names(self.output, names)
        return names

    def has_unsupported(self) -> bool:
        return _contains_key(self.args, UNSUPPORTED_KEY) or _contains_key(self.kwargs, UNSUPPORTED_KEY)


def _collect_tensor_names(value: Any, out: list[str]) -> None:
    if isinstance(value, dict):
        if TENSOR_KEY in value:
            out.append(value[TENSOR_KEY])
            return
        for item in value.values():
            _collect_tensor_names(item, out)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect_tensor_names(item, out)


def _contains_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(v, key) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_key(v, key) for v in value)
    return False


def forward_no_cache(model: Any, input_ids: Any) -> Any:
    """Run a forward with caching disabled, falling back if unsupported.

    A KV/hybrid cache object reaches each decoder layer's kwargs and cannot be
    serialized or rebuilt, which would make every layer unreplayable. Disabling
    the cache keeps recorded arguments to tensors and plain values. Nothing is
    lost for tracing: the emulator re-runs the full prefill per step anyway.
    """
    try:
        return model(input_ids, use_cache=False)
    except TypeError:
        return model(input_ids)


def is_tensor(value: Any) -> bool:
    try:
        import torch
    except ImportError:  # pragma: no cover
        return False
    return isinstance(value, torch.Tensor)


def slice_for_dump(tensor: Any, seq_len: int, policy: DumpPolicy) -> tuple[Any, SliceInfo | None]:
    """Cut a long-context tensor down to a head and tail window.

    Only an axis whose length equals the sample's sequence length is sliced, so
    weights and per-head shapes are never touched.
    """
    if policy.full_dumps or not policy.is_long(seq_len):
        return tensor, None
    axis = next((i for i, size in enumerate(tensor.shape) if size == seq_len), None)
    if axis is None:
        return tensor, None
    head, tail = policy.slice_head, policy.slice_tail
    if head + tail >= seq_len:
        return tensor, None
    import torch

    kept = torch.cat([
        tensor.index_select(axis, torch.arange(head, device=tensor.device)),
        tensor.index_select(axis, torch.arange(seq_len - tail, seq_len, device=tensor.device)),
    ], dim=axis)
    return kept, SliceInfo(axis=axis, head=head, tail=tail, original_length=seq_len)


class Tracer:
    """Records per-module IO for a graph over a set of sample inputs."""

    def __init__(
        self,
        model: Any,
        graph: PartitionGraph,
        store: TensorStore,
        policy: DumpPolicy | None = None,
    ):
        self.model = model
        self.graph = graph
        self.store = store
        self.policy = policy or DumpPolicy()
        self.records: list[CallRecord] = []
        self.bytes_written = 0
        self._targets = self._resolve_targets()

    def _account(self, nbytes: int) -> None:
        """Track dumped bytes and stop at the policy ceiling."""
        self.bytes_written += nbytes
        ceiling = self.policy.max_total_bytes
        if ceiling is not None and self.bytes_written > ceiling:
            raise DumpBudgetExceeded(
                f"tracing has written {self.bytes_written} bytes, past the "
                f"{ceiling}-byte max_total_bytes ceiling; raise it, reduce the "
                "input set, or disable cache_weights"
            )

    def _resolve_targets(self) -> dict[str, list[str]]:
        """Map submodule qualified name -> module ids that own it."""
        named = {name for name, _ in self.model.named_modules() if name}
        targets: dict[str, list[str]] = {}
        for module in self.graph.partitioned_modules:
            for submodule in module.submodules:
                if submodule in named:
                    targets.setdefault(submodule, []).append(module.id)
        return targets

    def unresolved_submodules(self) -> list[str]:
        """Graph submodules with no counterpart in the instantiated model."""
        named = {name for name, _ in self.model.named_modules() if name}
        return sorted({
            submodule
            for module in self.graph.partitioned_modules
            for submodule in module.submodules
            if submodule not in named
        })

    # -- weights ---------------------------------------------------------------

    def dump_weights(self, module_ids: list[str] | None = None) -> dict[str, list[str]]:
        """Dump each module's real parameters; returns module id -> tensor names."""
        wanted = set(module_ids) if module_ids else None
        dumped: dict[str, list[str]] = {}
        for module in self.graph.partitioned_modules:
            if wanted is not None and module.id not in wanted:
                continue
            names: list[str] = []
            for submodule_name in module.submodules:
                submodule = _lookup(self.model, submodule_name)
                if submodule is None:
                    continue
                if hasattr(submodule, "named_parameters"):
                    items = list(submodule.named_parameters()) + list(submodule.named_buffers())
                    for param_name, tensor in items:
                        full = f"{submodule_name}.{param_name}" if param_name else submodule_name
                        meta = self.store.write(
                            _safe_name(full), tensor, role="weight",
                            module_id=module.id, subdir=f"weights/{module.id}",
                        )
                        self._account(meta.nbytes)
                        names.append(meta.name)
                else:
                    meta = self.store.write(
                        _safe_name(submodule_name), submodule, role="weight",
                        module_id=module.id, subdir=f"weights/{module.id}",
                    )
                    self._account(meta.nbytes)
                    names.append(meta.name)
            dumped[module.id] = names
        return dumped

    # -- activations -----------------------------------------------------------

    def trace_sample(self, sample_id: str, input_ids: Any, step: int = 0) -> list[CallRecord]:
        """Run one forward with hooks installed; return the records produced."""
        import torch

        produced: list[CallRecord] = []
        counter = {"n": 0}
        seq_len = int(input_ids.shape[-1])

        def make_hook(submodule_name: str, module_ids: list[str]):
            def hook(_module, args, kwargs, output):
                for module_id in module_ids:
                    prefix = f"{sample_id}/s{step}/{_safe_name(submodule_name)}"
                    record = CallRecord(
                        module_id=module_id, submodule=submodule_name,
                        sample_id=sample_id, step=step, order=counter["n"],
                        args=[self._encode(a, f"{prefix}/arg{i}", seq_len, module_id, sample_id, step)
                              for i, a in enumerate(args)],
                        kwargs=self._encode(kwargs, f"{prefix}/kw", seq_len, module_id, sample_id, step),
                        output=self._encode(output, f"{prefix}/out", seq_len, module_id, sample_id,
                                            step, role="output"),
                    )
                    produced.append(record)
                counter["n"] += 1
            return hook

        handles = []
        for submodule_name, module_ids in self._targets.items():
            submodule = _lookup(self.model, submodule_name)
            if submodule is None:
                continue
            handles.append(submodule.register_forward_hook(
                make_hook(submodule_name, module_ids), with_kwargs=True,
            ))
        try:
            with torch.no_grad():
                forward_no_cache(self.model, input_ids)
        finally:
            for handle in handles:
                handle.remove()

        self.records.extend(produced)
        return produced

    def _encode(
        self, value: Any, path: str, seq_len: int,
        module_id: str, sample_id: str, step: int, role: str = "activation",
    ) -> Any:
        """Encode an argument tree, dumping tensors as it goes."""
        if is_tensor(value):
            sliced, slice_info = slice_for_dump(value, seq_len, self.policy)
            meta = self.store.write(
                _safe_name(path), sliced, role=role, module_id=module_id,
                sample_id=sample_id, step=step, subdir=f"activations/{module_id}",
                slice_info=slice_info,
            )
            self._account(meta.nbytes)
            return {TENSOR_KEY: meta.name}
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, dict):
            return {
                str(key): self._encode(item, f"{path}.{key}", seq_len, module_id,
                                       sample_id, step, role)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            encoded = [
                self._encode(item, f"{path}.{i}", seq_len, module_id, sample_id, step, role)
                for i, item in enumerate(value)
            ]
            return {TUPLE_KEY if isinstance(value, tuple) else LIST_KEY: encoded}
        return {UNSUPPORTED_KEY: type(value).__name__}


def decode_value(value: Any, load: Any) -> Any:
    """Rebuild an encoded argument tree, loading tensors via ``load(name)``."""
    if isinstance(value, dict):
        if TENSOR_KEY in value:
            return load(value[TENSOR_KEY])
        if UNSUPPORTED_KEY in value:
            return None
        if TUPLE_KEY in value:
            return tuple(decode_value(item, load) for item in value[TUPLE_KEY])
        if LIST_KEY in value:
            return [decode_value(item, load) for item in value[LIST_KEY]]
        return {key: decode_value(item, load) for key, item in value.items()}
    return value


def _lookup(model: Any, qualified_name: str) -> Any | None:
    current = model
    for part in qualified_name.split("."):
        if part.isdigit() and hasattr(current, "__getitem__"):
            try:
                current = current[int(part)]
                continue
            except (IndexError, KeyError, TypeError):
                return None
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def _safe_name(name: str) -> str:
    """Filesystem-safe tensor name."""
    return name.replace("/", "__").replace(".", "_")
