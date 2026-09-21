# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a model whose weights do not fit, by holding one module's at a time.

A traced forward needs the whole model to *execute*; it does not need it resident. At
any moment only the module being called needs its weights, so the model is built on the
meta device and each module reads its own parameters out of the checkpoint in a
pre-forward hook, then drops them in a post-forward hook. Peak memory becomes the
largest single module plus the activations — which is what makes a 478 GiB checkpoint
traceable on a 44 GiB card.

Tensors are taken in the dtype the model asked for, which is usually the dtype the
checkpoint stores: casting an fp8/fp4 mixture to bfloat16 is what made the model not
fit in the first place. Three cases come up, in this order:

- same dtype: copied as is;
- same width, different dtype: reinterpreted, which is how a checkpoint's ``int8`` pair
  of fp4 values becomes ``float4_e2m1fn_x2``;
- the model wants a wider float than the checkpoint holds: dequantized with the
  block scale stored beside it, which is what a vendor's own conversion script does
  for the few weights its kernels want in bfloat16.

Small tensors stay resident. Norms, biases and block scales are a rounding error in
bytes and are read on almost every call, and some of them are consumed outside the
module that owns them, where a hook would never fire.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from model_partition.loaders.base import LoaderError

#: A tensor at or below this may stay resident once read: a norm or a block scale, not
#: a weight matrix.
RESIDENT_LIMIT = 8 << 20

#: Ceiling on everything kept resident. Needed because "small" says nothing about how
#: many there are: this model has 47 546 block scales, each a few kilobytes and 44 GiB
#: together, which filled the card before a single forward ran.
RESIDENT_BUDGET = 1 << 30

#: Suffix of the block-scale tensor that sits beside a quantized weight.
SCALE_SUFFIX = ".scale"

#: Bytes one module may materialize on the accelerator. A module over this runs on the
#: host instead: this model's n-gram memory holds a 94 GiB table, which no card here
#: takes, and the alternative to running it slowly is not running the model at all.
MODULE_BUDGET = 8 << 30

#: A module whose whole subtree is at most this reads the subtree when it is called,
#: rather than leaving each descendant to read its own. Needed because a parent may use a
#: child's weight *without calling the child*: this model's attention does
#: ``self.wo_a.weight.view(...)`` and then an einsum, so no hook of ``wo_a``'s ever fires
#: and a placeholder propagates silently — einsum on a meta operand returns meta. Kept
#: small so a whole MoE layer is not pulled in for the six experts a token routes to.
SUBTREE_BUDGET = 1 << 30

#: Allocations larger than this are placeholders during construction; smaller ones are
#: made for real. Above a gigabyte a single tensor is a weight, and weights come from the
#: checkpoint. Below it a tensor is either small enough not to matter or something the
#: model *computes* — and a computed table is not always small: this model's rotary
#: frequencies are a 256 MiB table that every layer keeps a 1 MiB slice of, so a lower
#: line left 43 buffers as placeholders that no checkpoint could fill. Anything the
#: checkpoint does supply is released again the moment the streaming hooks go on.
PLACEHOLDER_THRESHOLD = 1 << 30

#: Dtypes that only ever come out of a checkpoint. A model computes a rotary table or a
#: hash map in float or int; it does not compute an fp4 weight or an e8m0 block scale, so
#: a request for one is a request for something the checkpoint will supply. This matters
#: because size alone does not separate the two: this model's expert weights are 5.9 MB
#: each and its block scales 0.35 MB, all of them from the checkpoint, and 94 000 of them
#: fill a card that a 0.07 GiB set of derived tables would not notice.
QUANTIZED_DTYPES = frozenset({
    "torch.float8_e4m3fn", "torch.float8_e5m2", "torch.float8_e8m0fnu",
    "torch.float4_e2m1fn_x2", "torch.uint8", "torch.int8",
})


def sparse_allocation(threshold: int = PLACEHOLDER_THRESHOLD):
    """Context manager: allocate what a model computes, placehold what it will be given.

    ``torch.device("meta")`` would make everything a placeholder, including the tables a
    model derives from its config or its tokenizer — rotary frequencies, an n-gram hash
    map, a zeroed cache — which no checkpoint can restore and which are then silently
    garbage. This draws the line twice, and the dtype is the sharper of the two: a
    quantized tensor comes from a quantized checkpoint, because nothing computes an fp4
    weight or an e8m0 block scale at construction. Size catches the rest, since a
    derived table is small where a weight matrix is not.
    """
    import torch
    from torch.overrides import TorchFunctionMode

    factories = {
        torch.empty, torch.zeros, torch.ones, torch.full, torch.rand, torch.randn,
        torch.empty_like, torch.zeros_like, torch.ones_like,
    }

    class _Sparse(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = dict(kwargs or {})
            if func in factories and kwargs.get("device") is None:
                dtype = _requested_dtype(args, kwargs)
                if (str(dtype) in QUANTIZED_DTYPES
                        or _requested_bytes(args, kwargs) > threshold):
                    kwargs["device"] = "meta"
            return func(*args, **kwargs)

    return _Sparse()


def _requested_dtype(args: tuple, kwargs: dict) -> Any:
    """The dtype a factory call asks for, explicitly or by default."""
    import torch

    if kwargs.get("dtype") is not None:
        return kwargs["dtype"]
    if args and isinstance(args[0], torch.Tensor):
        return args[0].dtype
    return torch.get_default_dtype()


def _requested_bytes(args: tuple, kwargs: dict) -> int:
    """Bytes a factory call is asking for, as far as its arguments say."""
    import torch

    shape: tuple[int, ...] = ()
    if args and isinstance(args[0], torch.Tensor):
        shape = tuple(args[0].shape)
    elif args and isinstance(args[0], (list, tuple)):
        shape = tuple(int(d) for d in args[0] if isinstance(d, int))
    else:
        dimensions = [a for a in args if isinstance(a, int)]
        shape = tuple(dimensions)
    width = getattr(_requested_dtype(args, kwargs), "itemsize", 4) or 4
    count = 1
    for dimension in shape:
        count *= max(int(dimension), 0)
    return count * width


@dataclass
class ShardEntry:
    """Where one tensor lives, and what has to happen to it on the way in."""

    shard: Path
    key: str
    scale_key: str | None = None


@dataclass
class StreamReport:
    """What installing the streaming hooks set up."""

    streamed_modules: int = 0
    #: Modules too large for the accelerator, materialized and run on the host.
    host_modules: list[str] = field(default_factory=list)
    resident_tensors: int = 0
    streamed_tensors: int = 0
    resident_bytes: int = 0
    largest_module_bytes: int = 0
    #: Tensors the model computed for itself, which no checkpoint holds.
    derived: list[str] = field(default_factory=list)
    #: Tensors the checkpoint should have held and did not. A placeholder is not a
    #: weight: a forward on one validates nothing, so the caller refuses.
    unresolved: list[str] = field(default_factory=list)

    def summary(self) -> str:
        from model_partition.hardware import format_bytes

        text = (f"{self.streamed_tensors} tensor(s) streamed across "
                f"{self.streamed_modules} module(s), largest "
                f"{format_bytes(self.largest_module_bytes)}; "
                f"{self.resident_tensors} small tensor(s) resident "
                f"({format_bytes(self.resident_bytes)})")
        if self.host_modules:
            text += (f"; {len(self.host_modules)} too large for the device, on the host "
                     f"({', '.join(self.host_modules[:3])})")
        if self.derived:
            text += f"; {len(self.derived)} computed at construction"
        if self.unresolved:
            text += (f"; {len(self.unresolved)} parameter(s) not in the checkpoint: "
                     f"{', '.join(self.unresolved[:4])}")
        return text


def rename(key: str, rules: tuple[tuple[str, str], ...]) -> str:
    """Apply a spec's rename rules to a checkpoint key, in order.

    The rules exist because a vendor's own inference code and its published checkpoint
    do not always agree on names — ``self_attn`` against ``attn``, a scale called
    ``weight_scale_inv`` — and the mapping belongs with the model's other oddities in
    its spec rather than hard-coded here.
    """
    for pattern, replacement in rules:
        key = re.sub(pattern, replacement, key)
    return key


def build_index(shards: list[Path],
                rules: tuple[tuple[str, str], ...] = ()) -> dict[str, ShardEntry]:
    """Map model parameter names to their place in the checkpoint.

    Reads headers only: a shard's keys and nothing else, so indexing a 478 GiB
    checkpoint costs a few seconds and no memory.
    """
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - dependency present in practice
        raise LoaderError("safetensors is required to stream weights") from exc

    entries: dict[str, ShardEntry] = {}
    scales: dict[str, tuple[Path, str]] = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                name = rename(key, rules)
                if name.endswith(SCALE_SUFFIX):
                    scales[name] = (shard, key)
                entries[name] = ShardEntry(shard=shard, key=key)
    # A weight the model wants dequantized is stored beside its block scale, and the
    # scale is then not a parameter of its own.
    for name, entry in entries.items():
        if not name.endswith(".weight"):
            continue
        scale = scales.get(name[: -len(".weight")] + SCALE_SUFFIX)
        if scale is not None:
            entry.scale_key = scale[1]
    return entries


def clear_caches(module: Any) -> list[str]:
    """Empty any memoized function in a vendor module; returns what was cleared.

    Vendor code memoizes what it computes once — rotary frequencies, window index
    tables — and a tensor cached during one build is handed to the next. Build a probe on
    the meta device and the real model afterwards receives that probe's placeholders,
    which is how 43 rotary buffers came out empty in a model that computes them
    correctly.
    """
    cleared: list[str] = []
    for name, value in list(vars(module).items()):
        clear = getattr(value, "cache_clear", None)
        if callable(clear):
            clear()
            cleared.append(name)
    return cleared


def derived_bytes(model: Any, index: dict[str, ShardEntry]) -> int:
    """The largest tensor this model computes for itself rather than being given.

    Sizes the placeholder threshold: a derived tensor has to be allocated for real
    whatever its size, because no checkpoint can restore it, and rotary frequencies for
    a million-token context are not small.
    """
    return max((tensor.numel() * tensor.element_size()
                for name, tensor in _all_tensors(model) if name not in index), default=0)


def _all_tensors(model: Any):
    for prefix, module in model.named_modules():
        for leaf in list(module._parameters) + list(module._buffers):
            tensor = _tensor_of(module, leaf)
            if tensor is not None:
                yield (f"{prefix}.{leaf}" if prefix else leaf), tensor


def install_streaming(model: Any, index: dict[str, ShardEntry], device: str = "cuda",
                      keep_bytes: int = RESIDENT_LIMIT,
                      resident_budget: int = RESIDENT_BUDGET,
                      module_budget: int = MODULE_BUDGET,
                      subtree_budget: int = SUBTREE_BUDGET) -> StreamReport:
    """Give each module a hook that reads the weights its call needs, and drops them.

    Walking top-down, a module whose whole subtree fits ``subtree_budget`` takes
    responsibility for that subtree — because a parent may reach into a child's weight
    without calling the child, and only the parent's own call is a reliable moment to
    have it. Anything bigger is left to its descendants, so a MoE layer is not
    materialized whole for the few experts a token visits.

    A module whose weights exceed ``module_budget`` runs on the host: its tensors are
    materialized there, its inputs are moved across for the call and its output moved
    back. Slow, and the honest alternative to refusing the model outright.
    """
    report = StreamReport()
    claimed: set[int] = set()
    for prefix, module in model.named_modules():
        if id(module) in claimed:
            continue
        subtree = _subtree_tensors(module, prefix)
        wanted = sum(tensor.numel() * tensor.element_size()
                     for _, _, _, tensor in subtree)
        own_only = wanted > subtree_budget
        covered = [row for row in subtree if row[0] is module] if own_only else subtree
        if not own_only:
            claimed.update(id(child) for child in module.modules())
        streamed = _classify(covered, index, report, device, keep_bytes, resident_budget)
        if not streamed:
            continue
        report.streamed_modules += 1
        report.streamed_tensors += len(streamed)
        held = sum(owner._parameters.get(leaf, owner._buffers.get(leaf)).numel()
                   * owner._parameters.get(leaf, owner._buffers.get(leaf)).element_size()
                   for owner, leaf, _ in streamed)
        report.largest_module_bytes = max(report.largest_module_bytes, held)
        place = device
        if held > module_budget and not device.startswith("cpu"):
            place = "cpu"
            report.host_modules.append(prefix or "(root)")
        _install_hooks(module, streamed, index, place, moves=place != device)
    return report


def _subtree_tensors(root: Any, prefix: str) -> list[tuple[Any, str, str, Any]]:
    """``(owner, leaf, full name, tensor)`` for everything under ``root``, root included."""
    rows: list[tuple[Any, str, str, Any]] = []
    for name, module in root.named_modules():
        path = f"{prefix}.{name}" if prefix and name else (prefix or name)
        for leaf in list(module._parameters) + list(module._buffers):
            tensor = _tensor_of(module, leaf)
            if tensor is not None:
                rows.append((module, leaf, f"{path}.{leaf}" if path else leaf, tensor))
    return rows


def _classify(rows: list[tuple[Any, str, str, Any]], index: dict[str, ShardEntry],
              report: StreamReport, device: str, keep_bytes: int,
              resident_budget: int) -> list[tuple[Any, str, str]]:
    """Read the small ones now, mark the rest for streaming, account for the others."""
    import torch

    streamed: list[tuple[Any, str, str]] = []
    for owner, leaf, full, tensor in rows:
        aliases = _aliases(owner)
        entry = index.get(full)
        if entry is None:
            # Not in the checkpoint. Either the model computed it at construction — fine,
            # it is real and stays — or it is a weight the checkpoint should have had, and
            # running on a placeholder would validate nothing.
            (report.unresolved if tensor.is_meta else report.derived).append(full)
            continue
        nbytes = tensor.numel() * tensor.element_size()
        if nbytes <= keep_bytes and report.resident_bytes + nbytes <= resident_budget:
            _set(owner, leaf, _read(entry, tensor, device))
            report.resident_tensors += 1
            report.resident_bytes += nbytes
            _rebind(owner, aliases)
            continue
        # Released now rather than at the first forward: a tensor the checkpoint supplies
        # may have been allocated during construction, and holding it until something
        # calls this module defeats the point of streaming.
        if not tensor.is_meta:
            _set(owner, leaf, torch.empty(tensor.shape, dtype=tensor.dtype, device="meta"))
            _rebind(owner, aliases)
        streamed.append((owner, leaf, full))
    return streamed


def _install_hooks(module: Any, streamed: list[tuple[Any, str, str]],
                   index: dict[str, ShardEntry], device: str,
                   moves: bool = False) -> None:
    """Read the weights this call needs, from anywhere in the subtree, and drop them after.

    ``moves`` is for a module placed off the accelerator: its inputs are brought to
    where its weights are and its output sent back, so the rest of the model does not
    have to know.
    """
    import torch

    shapes = {(id(owner), leaf): (_tensor_of(owner, leaf).shape,
                                  _tensor_of(owner, leaf).dtype)
              for owner, leaf, _ in streamed}
    aliases = {id(owner): _aliases(owner) for owner, _, _ in streamed}
    origin: list[Any] = [None]

    def materialize(_module: Any, args: tuple, kwargs: dict):
        for owner, leaf, full in streamed:
            current = _tensor_of(owner, leaf)
            if current.is_meta:
                _set(owner, leaf, _read(index[full], current, device))
        for owner, _, _ in streamed:
            _rebind(owner, aliases[id(owner)])
        if not moves:
            return None
        origin[0] = _device_of(args) or _device_of(tuple(kwargs.values()))
        return _moved(args, device), _moved(kwargs, device)

    def release(_module: Any, args: tuple, kwargs: dict, output: Any):
        for owner, leaf, _ in streamed:
            shape, dtype = shapes[(id(owner), leaf)]
            _set(owner, leaf, torch.empty(shape, dtype=dtype, device="meta"))
        for owner, _, _ in streamed:
            _rebind(owner, aliases[id(owner)])
        if moves and origin[0] is not None:
            return _moved(output, origin[0])
        return None

    module.register_forward_pre_hook(materialize, with_kwargs=True)
    module.register_forward_hook(release, with_kwargs=True)


def _device_of(value: Any) -> Any:
    """The device of the first tensor in an argument tree."""
    import torch

    if isinstance(value, torch.Tensor):
        return value.device
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _device_of(item)
            if found is not None:
                return found
    if isinstance(value, dict):
        return _device_of(tuple(value.values()))
    return None


def _moved(value: Any, device: Any) -> Any:
    """The same argument tree with every tensor on ``device``."""
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_moved(item, device) for item in value)
    if isinstance(value, list):
        return [_moved(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _moved(item, device) for key, item in value.items()}
    return value


def _tensor_of(module: Any, leaf: str) -> Any:
    if leaf in module._parameters:
        return module._parameters[leaf]
    return module._buffers.get(leaf)


def _set(module: Any, leaf: str, value: Any) -> None:
    import torch

    if leaf in module._parameters:
        module._parameters[leaf] = torch.nn.Parameter(value, requires_grad=False)
    else:
        module._buffers[leaf] = value


def _aliases(module: Any) -> list[tuple[str, str, str]]:
    """Attributes on one tensor that point at another of the same module's tensors.

    A vendor's quantized ``Linear`` hangs its block scale off the weight so its kernel
    can reach it — ``self.weight.scale = self.scale`` — and replacing either tensor
    breaks the link unless it is put back.
    """
    found: list[tuple[str, str, str]] = []
    names = list(module._parameters) + list(module._buffers)
    for leaf in names:
        holder = _tensor_of(module, leaf)
        if holder is None:
            continue
        for attribute, value in list(vars(holder).items()):
            for other in names:
                if other != leaf and _tensor_of(module, other) is value:
                    found.append((leaf, attribute, other))
    return found


def _rebind(module: Any, aliases: list[tuple[str, str, str]]) -> None:
    for leaf, attribute, other in aliases:
        holder, target = _tensor_of(module, leaf), _tensor_of(module, other)
        if holder is not None and target is not None:
            try:
                setattr(holder, attribute, target)
            except Exception:  # pragma: no cover - a read-only attribute
                continue


def _read(entry: ShardEntry, target: Any, device: str) -> Any:
    """The checkpoint's tensor, in the dtype and shape this parameter wants."""
    import torch
    from safetensors import safe_open

    with safe_open(str(entry.shard), framework="pt", device="cpu") as handle:
        raw = handle.get_tensor(entry.key)
        scale = handle.get_tensor(entry.scale_key) if entry.scale_key else None
    if raw.dtype != target.dtype:
        if scale is not None and target.dtype in (torch.bfloat16, torch.float16,
                                                  torch.float32):
            raw = _dequantize(raw, scale, target.dtype)
        elif raw.element_size() == target.element_size():
            # Same bytes, different reading of them: an int8 pair of fp4 values.
            raw = raw.view(target.dtype)
        else:
            raw = raw.to(target.dtype)
    if tuple(raw.shape) != tuple(target.shape):
        raw = raw.reshape(target.shape)
    return raw.to(device)


def _dequantize(weight: Any, scale: Any, dtype: Any) -> Any:
    """Expand a block-scaled weight to ``dtype``, one scale per block.

    The arithmetic a vendor conversion script does for the weights its kernels want in
    bfloat16: each block of the weight is multiplied by its own scale.
    """
    if weight.dim() != 2 or scale.dim() != 2:
        return weight.to(dtype)
    out_block = weight.shape[0] // scale.shape[0]
    in_block = weight.shape[1] // scale.shape[1]
    if out_block < 1 or in_block < 1:
        return weight.to(dtype)
    expanded = (weight.unflatten(0, (-1, out_block)).unflatten(-1, (-1, in_block)).float()
                * scale[:, None, :, None].float())
    return expanded.flatten(2, 3).flatten(0, 1).to(dtype)
