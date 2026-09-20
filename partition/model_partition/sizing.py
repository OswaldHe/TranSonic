# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structural view of a checkpoint plus the memory cost model.

:class:`ModelInventory` turns a raw tensor inventory into layers, subtrees, and
structural signatures. :class:`CostModel` turns those bytes into a fit decision
against the GPU budget.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from model_partition.weights_index import TensorEntry, WeightIndex

#: Name fragments identifying optional subtrees, checked in order.
SUBTREE_MARKERS: dict[str, tuple[str, ...]] = {
    "vision": ("visual", "vision_tower", "vision_model", ".vision.", "vit."),
    "mtp": (".mtp.", "mtp_", "nextn", "next_n", "eh_proj"),
    "engram": ("engram",),
}

#: Activations per module boundary, as a multiple of one hidden-state tensor.
#: Covers the residual stream plus transient MLP/attention intermediates.
DEFAULT_ACTIVATION_MULTIPLIER = 6.0


def config_get(config: dict[str, Any], key: str, default: Any = None) -> Any:
    """Look up ``key`` at the top level, then inside ``text_config``.

    Modern multimodal configs nest the language model under ``text_config``.
    """
    if key in config:
        return config[key]
    text = config.get("text_config")
    if isinstance(text, dict) and key in text:
        return text[key]
    return default


@dataclass
class QuantProfile:
    """Quantization scheme, as far as sizing is concerned."""

    method: str | None = None
    block_size: tuple[int, ...] | None = None
    scale_fmt: str | None = None
    expert_dtype: str | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> QuantProfile:
        raw = config.get("quantization_config") or {}
        block = raw.get("weight_block_size")
        return cls(
            method=raw.get("quant_method"),
            block_size=tuple(block) if isinstance(block, list) else None,
            scale_fmt=raw.get("scale_fmt"),
            expert_dtype=raw.get("expert_dtype"),
        )

    @property
    def quantized(self) -> bool:
        return self.method is not None

    def elements_per_byte(self, entry: TensorEntry) -> int:
        """How many logical values one stored byte holds.

        Packed fp4 experts arrive as int8 tensors holding two values per byte.
        """
        if self.expert_dtype == "fp4" and entry.dtype in ("int8", "uint8") and "expert" in entry.name:
            return 2
        return 1

    def dequant_bytes(self, entry: TensorEntry, target_width: int = 2) -> int:
        """Bytes this tensor occupies once dequantized to ``target_width``."""
        if entry.dtype in ("float8_e4m3fn", "float8_e5m2"):
            return entry.nbytes * target_width
        if self.elements_per_byte(entry) == 2:
            return entry.nbytes * 2 * target_width
        return 0


@dataclass
class LayerProfile:
    """One layer of the repeating stack."""

    index: int
    signature: str
    param_bytes: int
    tensor_names: list[str] = field(default_factory=list)
    shards: list[str] = field(default_factory=list)
    expert_bytes: int = 0
    n_experts: int = 0
    dequant_bytes: int = 0

    @property
    def has_experts(self) -> bool:
        return self.n_experts > 0

    @property
    def non_expert_bytes(self) -> int:
        return self.param_bytes - self.expert_bytes


@dataclass
class ModelInventory:
    """Structural view of a checkpoint: layers, subtrees, signatures."""

    index: WeightIndex
    config: dict[str, Any]
    quant: QuantProfile
    layers: list[LayerProfile] = field(default_factory=list)
    global_bytes: int = 0
    global_tensor_names: list[str] = field(default_factory=list)
    subtree_bytes: dict[str, int] = field(default_factory=dict)
    #: Bytes in repeating stacks other than the main one, keyed by prefix.
    aux_stack_bytes: dict[str, int] = field(default_factory=dict)
    stack_prefix: str | None = None
    dequant_bytes: int = 0

    @property
    def hidden_size(self) -> int:
        return int(config_get(self.config, "hidden_size", 0) or 0)

    @property
    def num_layers(self) -> int:
        declared = int(config_get(self.config, "num_hidden_layers", 0) or 0)
        return declared or len(self.layers)

    @property
    def vocab_size(self) -> int:
        return int(config_get(self.config, "vocab_size", 0) or 0)

    @property
    def layer_types(self) -> list[str]:
        """Per-layer type when the config declares one (hybrid attention stacks)."""
        types = config_get(self.config, "layer_types") or []
        return [str(t) for t in types] if isinstance(types, list) else []

    def total_param_bytes(self, include_excluded: bool = True) -> int:
        total = (sum(layer.param_bytes for layer in self.layers)
                 + self.global_bytes + sum(self.aux_stack_bytes.values()))
        if include_excluded:
            total += sum(self.subtree_bytes.values())
        return total

    def signature_groups(self) -> dict[str, list[int]]:
        """Map structural signature -> layer indices sharing it."""
        groups: dict[str, list[int]] = defaultdict(list)
        for layer in self.layers:
            groups[layer.signature].append(layer.index)
        return dict(groups)

    def representative_layers(self, preferred: tuple[int, ...] = (1, 5)) -> list[int]:
        """Layers whose tensors are kept after pruning.

        The union of ``preferred``, a midpoint, the last layer, and one
        representative per distinct signature. The signature term is what stops a
        hybrid stack (Qwen3.5 alternates linear and full attention) from losing
        the only copy of a kernel variant.
        """
        if not self.layers:
            return []
        indices = sorted(layer.index for layer in self.layers)
        keep = {i for i in preferred if i in set(indices)}
        keep.add(indices[len(indices) // 2])
        keep.add(indices[-1])
        keep.add(indices[0])
        for layer_indices in self.signature_groups().values():
            if not keep & set(layer_indices):
                keep.add(min(layer_indices))
        return sorted(keep)

    @classmethod
    def build(
        cls,
        index: WeightIndex,
        config: dict[str, Any],
        include: tuple[str, ...] = (),
    ) -> ModelInventory:
        """Group a tensor inventory structurally.

        Subtrees not named in ``include`` are accounted separately and excluded
        from layers and globals, so their bytes never enter the module budget.
        """
        quant = QuantProfile.from_config(config)
        excluded = {name: markers for name, markers in SUBTREE_MARKERS.items() if name not in include}

        def subtree_of(entry: TensorEntry) -> str | None:
            for name, markers in excluded.items():
                if any(marker in entry.name for marker in markers):
                    return name
            return None

        subtree_bytes: dict[str, int] = defaultdict(int)
        aux_stack_bytes: dict[str, int] = defaultdict(int)
        layer_entries: dict[int, list[TensorEntry]] = defaultdict(list)
        globals_: list[TensorEntry] = []
        dequant_total = 0

        kept = [e for e in index.entries if subtree_of(e) is None]
        main_stack = WeightIndex(entries=kept).main_stack_prefix()

        for entry in index.entries:
            subtree = subtree_of(entry)
            if subtree:
                subtree_bytes[subtree] += entry.nbytes
                continue
            dequant_total += quant.dequant_bytes(entry)
            layer_index = entry.layer_index
            if layer_index is None:
                globals_.append(entry)
            elif entry.layer_prefix == main_stack:
                layer_entries[layer_index].append(entry)
            else:
                # A second repeating stack (e.g. an in-scope vision tower) keeps
                # its own index space rather than colliding with the main one.
                aux_stack_bytes[entry.layer_prefix or "?"] += entry.nbytes

        layers: list[LayerProfile] = []
        for layer_index in sorted(layer_entries):
            entries = layer_entries[layer_index]
            expert_entries = [e for e in entries if e.expert_index is not None or ".experts" in e.name]
            signature = "|".join(sorted(
                f"{e.normalized()}:{e.dtype}:{','.join(str(d) for d in e.shape)}" for e in entries
            ))
            layers.append(LayerProfile(
                index=layer_index,
                signature=signature,
                param_bytes=sum(e.nbytes for e in entries),
                tensor_names=[e.name for e in entries],
                shards=sorted({e.shard for e in entries}),
                expert_bytes=sum(e.nbytes for e in expert_entries),
                n_experts=len({e.expert_index for e in expert_entries if e.expert_index is not None}),
                dequant_bytes=sum(quant.dequant_bytes(e) for e in entries),
            ))

        return cls(
            index=index,
            config=config,
            quant=quant,
            layers=layers,
            global_bytes=sum(e.nbytes for e in globals_),
            global_tensor_names=[e.name for e in globals_],
            subtree_bytes=dict(subtree_bytes),
            aux_stack_bytes=dict(aux_stack_bytes),
            stack_prefix=main_stack,
            dequant_bytes=dequant_total,
        )


@dataclass
class CostModel:
    """Turns parameter bytes into a resident-memory figure for a module."""

    hidden_size: int
    dtype_bytes: int = 2
    seq_len: int = 16384
    batch: int = 1
    activation_multiplier: float = DEFAULT_ACTIVATION_MULTIPLIER
    num_kv_heads: int = 0
    head_dim: int = 0
    kv_dtype_bytes: int = 2

    def activation_bytes(self, seq_len: int | None = None) -> int:
        length = seq_len if seq_len is not None else self.seq_len
        one = self.batch * length * self.hidden_size * self.dtype_bytes
        return int(one * self.activation_multiplier)

    def kv_bytes(self, n_layers: int, seq_len: int | None = None) -> int:
        """KV cache for ``n_layers`` layers; 0 when the shape is unknown."""
        if not (self.num_kv_heads and self.head_dim):
            return 0
        length = seq_len if seq_len is not None else self.seq_len
        per_layer = 2 * self.batch * length * self.num_kv_heads * self.head_dim * self.kv_dtype_bytes
        return per_layer * n_layers

    def resident_bytes(self, param_bytes: int, n_layers: int = 1, seq_len: int | None = None) -> int:
        return param_bytes + self.activation_bytes(seq_len) + self.kv_bytes(n_layers, seq_len)

    def max_layers_per_module(self, layer_bytes: int, budget_bytes: int, seq_len: int | None = None) -> int:
        """Largest layer count whose resident footprint stays under budget."""
        if layer_bytes <= 0:
            return 1
        count = 0
        while self.resident_bytes(layer_bytes * (count + 1), count + 1, seq_len) <= budget_bytes:
            count += 1
            if count > 4096:
                break
        return count

    @classmethod
    def from_config(cls, config: dict[str, Any], dtype_bytes: int = 2, **overrides: Any) -> CostModel:
        head_dim = int(config_get(config, "head_dim", 0) or 0)
        hidden = int(config_get(config, "hidden_size", 0) or 0)
        heads = int(config_get(config, "num_attention_heads", 0) or 0)
        if not head_dim and hidden and heads:
            head_dim = hidden // heads
        params: dict[str, Any] = {
            "hidden_size": hidden,
            "dtype_bytes": dtype_bytes,
            "num_kv_heads": int(config_get(config, "num_key_value_heads", 0) or 0),
            "head_dim": head_dim,
        }
        params.update(overrides)
        return cls(**params)
