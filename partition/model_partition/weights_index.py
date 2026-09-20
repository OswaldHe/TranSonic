# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inventory of every tensor in a checkpoint: name, dtype, shape, shard.

Built without reading weight data — remotely via the Hub's safetensors metadata
endpoint, locally by parsing safetensors headers. That keeps sizing and planning
architecture-agnostic: we measure the checkpoint that exists instead of
reimplementing each architecture's parameter-count formula.
"""

from __future__ import annotations

import json
import re
import struct
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

#: safetensors dtype tag -> (name in tensorstore.DTYPES, bytes per element)
ST_DTYPES: dict[str, tuple[str, int]] = {
    "F64": ("float64", 8), "F32": ("float32", 4), "F16": ("float16", 2),
    "BF16": ("bfloat16", 2), "F8_E4M3": ("float8_e4m3fn", 1), "F8_E5M2": ("float8_e5m2", 1),
    "I8": ("int8", 1), "U8": ("uint8", 1), "I16": ("int16", 2), "U16": ("uint16", 2),
    "I32": ("int32", 4), "U32": ("uint32", 4), "I64": ("int64", 8), "U64": ("uint64", 8),
    "BOOL": ("bool", 1),
}

#: Matches the repeating-stack index in a parameter name, e.g. ``...layers.12.``
LAYER_RE = re.compile(r"^(?P<prefix>.*\.(?:layers|blocks|h)\.)(?P<index>\d+)(?P<suffix>\..*)$")

#: Matches a per-expert parameter, e.g. ``...experts.37.gate_proj.weight``
EXPERT_RE = re.compile(r"^(?P<prefix>.*\.experts\.)(?P<index>\d+)(?P<suffix>\..*)$")


class WeightIndexError(RuntimeError):
    """Raised when a checkpoint's tensor inventory cannot be read."""


@dataclass(frozen=True)
class TensorEntry:
    """One tensor in the checkpoint."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    shard: str

    @property
    def layer_index(self) -> int | None:
        match = LAYER_RE.match(self.name)
        return int(match.group("index")) if match else None

    @property
    def layer_prefix(self) -> str | None:
        """The stack this tensor belongs to, e.g. ``model.layers.``.

        Distinguishes the text stack from a vision tower's blocks, which would
        otherwise share an index space.
        """
        match = LAYER_RE.match(self.name)
        return match.group("prefix") if match else None

    @property
    def expert_index(self) -> int | None:
        match = EXPERT_RE.match(self.name)
        return int(match.group("index")) if match else None

    def normalized(self) -> str:
        """Name with layer and expert indices replaced by ``{i}``/``{e}``.

        Two tensors with the same normalized name are the same role in different
        repetitions — the basis for code deduplication.
        """
        name = LAYER_RE.sub(lambda m: f"{m.group('prefix')}{{i}}{m.group('suffix')}", self.name)
        return EXPERT_RE.sub(lambda m: f"{m.group('prefix')}{{e}}{m.group('suffix')}", name)


@dataclass
class WeightIndex:
    """Every tensor in a checkpoint, plus shard sizes."""

    entries: list[TensorEntry] = field(default_factory=list)
    shard_bytes: dict[str, int] = field(default_factory=dict)
    source: str = ""

    # -- aggregates ------------------------------------------------------------

    @property
    def total_bytes(self) -> int:
        return sum(e.nbytes for e in self.entries)

    @property
    def largest_shard_bytes(self) -> int:
        return max(self.shard_bytes.values(), default=0)

    @property
    def num_layers(self) -> int:
        indices = self.layer_indices()
        return max(indices) + 1 if indices else 0

    def bytes_by_dtype(self) -> dict[str, int]:
        totals: dict[str, int] = defaultdict(int)
        for entry in self.entries:
            totals[entry.dtype] += entry.nbytes
        return dict(totals)

    def layer_indices(self) -> list[int]:
        return sorted({i for e in self.entries if (i := e.layer_index) is not None})

    def expert_indices(self) -> list[int]:
        return sorted({i for e in self.entries if (i := e.expert_index) is not None})

    def stack_prefixes(self) -> dict[str, int]:
        """Map each repeating-stack prefix to its number of distinct layers."""
        seen: dict[str, set[int]] = defaultdict(set)
        for entry in self.entries:
            prefix, index = entry.layer_prefix, entry.layer_index
            if prefix is not None and index is not None:
                seen[prefix].add(index)
        return {prefix: len(indices) for prefix, indices in seen.items()}

    def main_stack_prefix(self) -> str | None:
        """The stack with the most layers; ties broken toward the shorter name."""
        prefixes = self.stack_prefixes()
        if not prefixes:
            return None
        return min(prefixes.items(), key=lambda kv: (-kv[1], len(kv[0]), kv[0]))[0]

    # -- selection -------------------------------------------------------------

    def by_layer(self) -> dict[int, list[TensorEntry]]:
        groups: dict[int, list[TensorEntry]] = defaultdict(list)
        for entry in self.entries:
            index = entry.layer_index
            if index is not None:
                groups[index].append(entry)
        return dict(groups)

    def global_tensors(self) -> list[TensorEntry]:
        """Tensors outside the repeating stack (embeddings, final norm, head)."""
        return [e for e in self.entries if e.layer_index is None]

    def matching(self, *substrings: str) -> list[TensorEntry]:
        return [e for e in self.entries if any(s in e.name for s in substrings)]

    def excluding(self, *substrings: str) -> list[TensorEntry]:
        return [e for e in self.entries if not any(s in e.name for s in substrings)]

    def layer_bytes(self, index: int) -> int:
        return sum(e.nbytes for e in self.by_layer().get(index, []))

    def layer_signature(self, index: int) -> str:
        """Structural fingerprint of a layer: its normalized tensor names and shapes.

        Layers with equal signatures can share one extracted implementation.
        """
        parts = sorted(
            f"{e.normalized()}:{e.dtype}:{','.join(str(d) for d in e.shape)}"
            for e in self.by_layer().get(index, [])
        )
        return "|".join(parts)

    def shards_for(self, entries: list[TensorEntry]) -> list[str]:
        return sorted({e.shard for e in entries})

    # -- construction ----------------------------------------------------------

    @classmethod
    def from_hub(cls, repo_id: str, revision: str | None = None, token: str | None = None) -> WeightIndex:
        """Read the inventory from the Hub without downloading weight data."""
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:  # pragma: no cover - dependency present in practice
            raise WeightIndexError("huggingface_hub is required to index a remote repo") from exc

        api = HfApi(token=token)
        try:
            meta = api.get_safetensors_metadata(repo_id, revision=revision)
        except Exception as exc:
            raise WeightIndexError(f"Could not read safetensors metadata for {repo_id}: {exc}") from exc

        entries: list[TensorEntry] = []
        shard_bytes: dict[str, int] = {}
        for shard, file_meta in meta.files_metadata.items():
            shard_total = 0
            for name, info in file_meta.tensors.items():
                dtype, _ = ST_DTYPES.get(info.dtype, (info.dtype.lower(), 1))
                nbytes = info.data_offsets[1] - info.data_offsets[0]
                entries.append(TensorEntry(name, dtype, tuple(info.shape), nbytes, shard))
                shard_total += nbytes
            shard_bytes[shard] = shard_total
        return cls(entries=entries, shard_bytes=shard_bytes, source=f"hf:{repo_id}@{revision or 'main'}")

    @classmethod
    def from_local(cls, directory: str | Path) -> WeightIndex:
        """Read the inventory from safetensors files in ``directory``."""
        root = Path(directory)
        files = sorted(root.glob("*.safetensors"))
        if not files:
            raise WeightIndexError(f"No .safetensors files in {root}")
        entries: list[TensorEntry] = []
        shard_bytes: dict[str, int] = {}
        for path in files:
            header = read_safetensors_header(path)
            shard_total = 0
            for name, info in header.items():
                if name == "__metadata__":
                    continue
                dtype, _ = ST_DTYPES.get(info["dtype"], (info["dtype"].lower(), 1))
                start, end = info["data_offsets"]
                nbytes = end - start
                entries.append(TensorEntry(name, dtype, tuple(info["shape"]), nbytes, path.name))
                shard_total += nbytes
            shard_bytes[path.name] = shard_total
        return cls(entries=entries, shard_bytes=shard_bytes, source=str(root))


def read_safetensors_header(path: str | Path) -> dict:
    """Parse a safetensors header: u64 length prefix followed by JSON."""
    file_path = Path(path)
    with file_path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise WeightIndexError(f"{file_path} is too short to be a safetensors file")
        (length,) = struct.unpack("<Q", raw_length)
        if length <= 0 or length > 256 * 1024 * 1024:
            raise WeightIndexError(f"{file_path} has an implausible header length {length}")
        payload = handle.read(length)
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise WeightIndexError(f"{file_path} has a corrupt safetensors header: {exc}") from exc
