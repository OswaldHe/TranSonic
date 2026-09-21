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

#: Allocations larger than this are placeholders during construction; smaller ones are
#: made for real. A weight matrix comes from the checkpoint, so allocating it would be
#: waste. Everything a model *computes* at construction — rotary frequencies, an n-gram
#: hash table's primes, a token map built from the vocabulary — is small, is in no
#: checkpoint, and is wrong if left as a placeholder.
PLACEHOLDER_THRESHOLD = 16 << 20


def sparse_allocation(threshold: int = PLACEHOLDER_THRESHOLD):
    """Context manager: build big tensors as placeholders, small ones for real.

    ``torch.device("meta")`` would make everything a placeholder, including the tables a
    model derives from its config or tokenizer, which no checkpoint can restore. This
    draws the line at size instead, which is the line between "comes from the
    checkpoint" and "computed here".
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
                if _requested_bytes(args, kwargs) > threshold:
                    kwargs["device"] = "meta"
            return func(*args, **kwargs)

    return _Sparse()


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
    dtype = kwargs.get("dtype") or (
        args[0].dtype if args and isinstance(args[0], torch.Tensor)
        else torch.get_default_dtype())
    width = getattr(dtype, "itemsize", 4) or 4
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


def install_streaming(model: Any, index: dict[str, ShardEntry], device: str = "cuda",
                      keep_bytes: int = RESIDENT_LIMIT,
                      resident_budget: int = RESIDENT_BUDGET) -> StreamReport:
    """Make every module read its own weights when called, and release them after."""
    import torch

    report = StreamReport()
    owned = _owned_tensors(model)
    for module, prefix, names in owned:
        streamed: list[tuple[str, str]] = []
        for leaf, full in names:
            entry = index.get(full)
            tensor = _tensor_of(module, leaf)
            if entry is None:
                # Not in the checkpoint. Either the model computed it at construction —
                # fine, it is real and stays — or it is a weight the checkpoint should
                # have had, and running on a placeholder would validate nothing.
                (report.unresolved if tensor.is_meta else report.derived).append(full)
                continue
            nbytes = tensor.numel() * tensor.element_size()
            if entry is None:
                continue
            if nbytes <= keep_bytes and report.resident_bytes + nbytes <= resident_budget:
                _set(module, leaf, _read(entry, tensor, device))
                report.resident_tensors += 1
                report.resident_bytes += nbytes
                continue
            streamed.append((leaf, full))
        if not streamed:
            continue
        report.streamed_modules += 1
        report.streamed_tensors += len(streamed)
        report.largest_module_bytes = max(report.largest_module_bytes, sum(
            _tensor_of(module, leaf).numel() * _tensor_of(module, leaf).element_size()
            for leaf, _ in streamed))
        _install_hooks(module, streamed, index, device)
    del torch
    return report


def _install_hooks(module: Any, streamed: list[tuple[str, str]],
                   index: dict[str, ShardEntry], device: str) -> None:
    import torch

    shapes = {leaf: (_tensor_of(module, leaf).shape, _tensor_of(module, leaf).dtype)
              for leaf, _ in streamed}
    aliases = _aliases(module)

    def materialize(*_args: Any) -> None:
        for leaf, full in streamed:
            current = _tensor_of(module, leaf)
            if not current.is_meta:
                continue
            _set(module, leaf, _read(index[full], current, device))
        _rebind(module, aliases)

    def release(*_args: Any) -> None:
        for leaf, _ in streamed:
            shape, dtype = shapes[leaf]
            _set(module, leaf, torch.empty(shape, dtype=dtype, device="meta"))
        _rebind(module, aliases)

    module.register_forward_pre_hook(lambda *args: materialize())
    module.register_forward_hook(lambda *args: release())


def _owned_tensors(model: Any) -> list[tuple[Any, str, list[tuple[str, str]]]]:
    """Every module with tensors of its own, and their names relative to the model."""
    found: list[tuple[Any, str, list[tuple[str, str]]]] = []
    for prefix, module in model.named_modules():
        names = [(leaf, f"{prefix}.{leaf}" if prefix else leaf)
                 for leaf in list(module._parameters) + list(module._buffers)
                 if _tensor_of(module, leaf) is not None]
        if names:
            found.append((module, prefix, names))
    return found


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
