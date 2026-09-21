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

from contextlib import contextmanager
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
    #: At least one tensor in this record was windowed rather than dumped whole.
    #: Such a record documents the module but cannot verify it: attention mixes
    #: every position, so the windowed output is not a function of the windowed
    #: input.
    sliced: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = {
            "module_id": self.module_id, "submodule": self.submodule,
            "sample_id": self.sample_id, "step": self.step, "order": self.order,
            "args": self.args, "kwargs": self.kwargs, "output": self.output,
        }
        if self.sliced:
            data["sliced"] = True
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CallRecord:
        return cls(
            module_id=data["module_id"], submodule=data["submodule"],
            sample_id=data["sample_id"], step=int(data.get("step", 0)),
            args=data.get("args") or [], kwargs=data.get("kwargs") or {},
            output=data.get("output"), order=int(data.get("order", 0)),
            sliced=bool(data.get("sliced", False)),
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

    Every axis whose length equals the sample's sequence length is windowed — an
    attention mask is square in the sequence, and windowing only its query axis
    would leave a tensor that describes no real computation. Weights and per-head
    shapes have no such axis and are never touched.
    """
    if not policy.slices(seq_len):
        return tensor, None
    axes = [i for i, size in enumerate(tensor.shape) if size == seq_len]
    head, tail = policy.slice_head, policy.slice_tail
    if not axes or head + tail >= seq_len:
        return tensor, None
    import torch

    index = torch.cat([
        torch.arange(head, device=tensor.device),
        torch.arange(seq_len - tail, seq_len, device=tensor.device),
    ])
    kept = tensor
    for axis in axes:
        kept = kept.index_select(axis, index)
    return kept, SliceInfo(axes=axes, head=head, tail=tail, original_length=seq_len)


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
        #: What the last traced forward returned, and the hidden state its decoder
        #: stack produced — which is what an extra pass continues from.
        self.last_output: Any = None
        self.last_hidden: Any = None
        #: Set while encoding one record, read back onto it afterwards.
        self._sliced = False
        #: Non-persistent buffers as the traced forward saw them, by original name.
        self._derived: dict[str, Any] = {}
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

    def _capture_derived(self, submodule_name: str, submodule: Any) -> None:
        """Keep the non-persistent buffers as this forward saw them.

        No checkpoint holds them: the class computes them at construction. Which means
        the values depend on how the model was built, and the weights are dumped from a
        second, single-device copy — whose rotary ``inv_freq`` was bfloat16 while the
        placed copy that actually ran kept float32. A module rebuilt from the wrong one
        drifts a tenth of a radian by position 2000: right on short samples, wrong on
        long ones. So they are captured here, during the call whose output is the
        reference they will be checked against.
        """
        for name, buffer in _non_persistent(submodule).items():
            if getattr(buffer, "is_meta", False):
                continue
            full = f"{submodule_name}.{name}" if name else submodule_name
            self._derived.setdefault(full, buffer.detach().clone())

    def _module_tensors(self, module: Any, weights_only: bool = False):
        """Yield ``(original name, tensor)`` for everything a module owns.

        ``weights_only`` leaves out the non-persistent buffers, which are dumped from
        the traced model by :meth:`dump_derived` rather than from this one.
        """
        for submodule_name in module.submodules:
            submodule = _lookup(self.model, submodule_name)
            if submodule is None:
                continue
            if hasattr(submodule, "named_parameters"):
                skip = set(_non_persistent(submodule)) if weights_only else set()
                items = list(submodule.named_parameters()) + list(submodule.named_buffers())
                for param_name, tensor in items:
                    if param_name in skip:
                        continue
                    yield (f"{submodule_name}.{param_name}" if param_name else submodule_name), tensor
            else:
                yield submodule_name, submodule

    def dump_derived(self) -> dict[str, list[str]]:
        """Dump the buffers captured during tracing; returns module id -> names.

        Separate from :meth:`dump_weights` because these are not weights, and the copy
        of the model that ran is the only one that can be asked for them.
        """
        dumped: dict[str, list[str]] = {}
        for module in self.graph.partitioned_modules:
            names: list[str] = []
            for full, tensor in self._derived.items():
                if not any(full == s or full.startswith(f"{s}.") for s in module.submodules):
                    continue
                meta = self.store.write(
                    _safe_name(full), tensor.detach().cpu(), role="weight",
                    module_id=module.id, subdir=f"weights/{module.id}",
                    extra={"param": full},
                )
                self._account(meta.nbytes)
                names.append(meta.name)
            if names:
                dumped[module.id] = names
        return dumped

    def dump_weights(self, module_ids: list[str] | None = None,
                     weights_only: bool = True) -> dict[str, list[str]]:
        """Dump each module's real parameters; returns module id -> tensor names.

        Non-persistent buffers are left to :meth:`dump_derived`, which takes them from
        the model that traced rather than from this one.
        """
        wanted = set(module_ids) if module_ids else None
        dumped: dict[str, list[str]] = {}
        for module in self.graph.partitioned_modules:
            if wanted is not None and module.id not in wanted:
                continue
            names: list[str] = []
            for full, tensor in self._module_tensors(module, weights_only=weights_only):
                meta = self.store.write(
                    _safe_name(full), tensor, role="weight",
                    module_id=module.id, subdir=f"weights/{module.id}",
                    extra={"param": full},
                )
                self._account(meta.nbytes)
                names.append(meta.name)
            dumped[module.id] = names
        return dumped

    def index_weights(self) -> dict[str, list[str]]:
        """Which checkpoint tensors each module owns, writing nothing.

        What ``cache_weights=false`` leaves behind: the mapping is all a run needs
        to read those tensors back out of the checkpoint on demand, so
        verification still has real weights without a second copy on disk.

        Non-persistent buffers are left out because no checkpoint holds them — a rotary
        table and a KV cache are built at construction, and asking a checkpoint for one
        fails on a model that has them. :meth:`dump_derived` is where they come from.
        """
        return {module.id: [name for name, _ in self._module_tensors(module, weights_only=True)]
                for module in self.graph.partitioned_modules}

    # -- activations -----------------------------------------------------------

    def trace_sample(self, sample_id: str, input_ids: Any, step: int = 0) -> list[CallRecord]:
        """Run one forward with hooks installed; return the records produced."""
        import torch

        produced: list[CallRecord] = []
        hidden, handle = self._capture_stack_output()
        try:
            with self._recording(sample_id, step, int(input_ids.shape[-1]), produced):
                with torch.no_grad():
                    self.last_output = forward_no_cache(self.model, input_ids)
        finally:
            handle.remove()
        self.last_hidden = hidden.get("value")

        self.records.extend(produced)
        return produced

    def trace_extra_passes(
        self, sample_id: str, input_ids: Any, spec: Any,
        only: set[str] | None = None, step: int = 1,
    ) -> list[CallRecord]:
        """Drive the entry points ``forward`` does not reach, recording what they call.

        Must follow :meth:`trace_sample` for the same input: a pass reads what the main
        forward returned, and a decode pass continues from the KV state it left.

        Only the submodules in ``only`` are recorded. A decode pass runs the whole model
        again to reach its second entry point, and those layers already have their
        reference from the prefill; recording them again would double the artifacts and
        give the chain two unrelated steps to reconcile.
        """
        import torch

        produced: list[CallRecord] = []
        for offset, extra in enumerate(spec.extra_passes):
            target = _lookup(self.model, extra.entry)
            if target is None or not callable(target):
                raise TraceError(
                    f"trace.extra_passes names {extra.entry!r}, which this model has no "
                    f"callable for"
                )
            with torch.no_grad():
                pool, ids = self._pass_pool(input_ids, extra, tuple(spec.returns))
                missing = [name for name in extra.args if pool.get(name) is None]
                if missing:
                    raise TraceError(
                        f"{extra.entry} needs {', '.join(missing)}, which this forward did "
                        f"not produce; check trace.returns against what it returns"
                    )
                with self._recording(sample_id, step + offset, int(ids.shape[-1]),
                                     produced, only=only):
                    target(*[pool[name] for name in extra.args])

        self.records.extend(produced)
        return produced

    def _pass_pool(self, input_ids: Any, extra: Any,
                   returns: tuple[str, ...]) -> tuple[dict[str, Any], Any]:
        """The values an extra pass can ask for, and the ids it runs on."""
        ids, start_pos, output = input_ids, 0, self.last_output
        hidden_state = self.last_hidden
        if extra.decode:
            # The prefill above filled the cache through the last prompt position, so a
            # step at the next one is a real decode. The token fed there is the prompt's
            # last one again: a draft stack's reference call does not depend on which
            # token it is, and reusing a prompt token keeps the trace independent of
            # what the model sampled.
            ids = input_ids[..., -1:]
            start_pos = int(input_ids.shape[-1])
            hidden, handle = self._capture_stack_output()
            try:
                output = self.model(ids, start_pos)
            except TypeError as exc:
                raise TraceError(
                    f"a decode pass calls model(ids, start_pos) and this model's forward "
                    f"does not take that: {exc}"
                ) from exc
            finally:
                handle.remove()
            hidden_state = hidden.get("value")
        pool: dict[str, Any] = {"input_ids": ids, "start_pos": start_pos,
                                "hidden": hidden_state}
        if returns:
            values = output if isinstance(output, (tuple, list)) else (output,)
            pool.update(dict(zip(returns, values)))
        return pool, ids

    def _capture_stack_output(self) -> tuple[dict[str, Any], Any]:
        """Hook the decoder stack's last layer to keep what it produced.

        An extra pass that continues the backbone rather than reading its return value
        needs the hidden state entering the head, which no forward returns.
        """
        stack = _decoder_stack(self.model)
        captured: dict[str, Any] = {}

        def hook(_module, _args, output):
            captured["value"] = output

        if stack is None:
            return captured, _NullHandle()
        return captured, stack[-1].register_forward_hook(hook)

    @contextmanager
    def _recording(self, sample_id: str, step: int, seq_len: int,
                   produced: list[CallRecord], only: set[str] | None = None):
        """Install per-submodule hooks for the duration of one pass."""
        counter = {"n": 0}

        def make_hook(submodule_name: str, module_ids: list[str]):
            def hook(_module, args, kwargs, output):
                self._capture_derived(submodule_name, _module)
                for module_id in module_ids:
                    prefix = f"{sample_id}/s{step}/{_safe_name(submodule_name)}"
                    self._sliced = False
                    record = CallRecord(
                        module_id=module_id, submodule=submodule_name,
                        sample_id=sample_id, step=step, order=counter["n"],
                        args=[self._encode(a, f"{prefix}/arg{i}", seq_len, module_id, sample_id, step)
                              for i, a in enumerate(args)],
                        kwargs=self._encode(kwargs, f"{prefix}/kw", seq_len, module_id, sample_id, step),
                        output=self._encode(output, f"{prefix}/out", seq_len, module_id, sample_id,
                                            step, role="output"),
                    )
                    record.sliced = self._sliced
                    produced.append(record)
                counter["n"] += 1
            return hook

        handles = []
        for submodule_name, module_ids in self._targets.items():
            if only is not None and submodule_name not in only:
                continue
            submodule = _lookup(self.model, submodule_name)
            if submodule is None:
                continue
            handles.append(submodule.register_forward_hook(
                make_hook(submodule_name, module_ids), with_kwargs=True,
            ))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def _encode(
        self, value: Any, path: str, seq_len: int,
        module_id: str, sample_id: str, step: int, role: str = "activation",
    ) -> Any:
        """Encode an argument tree, dumping tensors as it goes."""
        if is_tensor(value):
            sliced, slice_info = slice_for_dump(value, seq_len, self.policy)
            self._sliced = self._sliced or slice_info is not None
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


class _NullHandle:
    """Stands in for a hook handle when there was nothing to hook."""

    def remove(self) -> None:
        pass


def _decoder_stack(model: Any) -> Any | None:
    """The repeating decoder stack: the longest ``ModuleList`` in the model."""
    import torch

    best = None
    for _name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) > 1:
            if best is None or len(module) > len(best):
                best = module
    return best


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


def _non_persistent(root: Any) -> dict[str, Any]:
    """Non-persistent buffers under ``root``, keyed relative to it."""
    found: dict[str, Any] = {}
    for prefix, module in root.named_modules():
        for leaf in getattr(module, "_non_persistent_buffers_set", set()) or set():
            buffer = module._buffers.get(leaf)
            if buffer is not None:
                found[f"{prefix}.{leaf}" if prefix else leaf] = buffer
    return found


def _safe_name(name: str) -> str:
    """Filesystem-safe tensor name."""
    return name.replace("/", "__").replace(".", "_")
