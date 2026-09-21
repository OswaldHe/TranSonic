# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Binary tensor dumps: raw ``.bin`` + JSON sidecar + a manifest index.

Chosen over safetensors for portability: a contiguous little-endian blob plus
explicit dtype/shape metadata is readable from whatever toolchain a kernel
bring-up uses. Identical blobs are hardlinked, which is what makes tied
embeddings and repeated layers cheap.

dtypes with no numpy equivalent (bfloat16, fp8, packed fp4) round-trip as raw
bytes; :func:`TensorStore.read_torch` reinterprets them correctly.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

MANIFEST_NAME = "manifest.yaml"

#: dtype name -> (bytes per element, numpy dtype string or None)
DTYPES: dict[str, tuple[int, str | None]] = {
    "float32": (4, "<f4"),
    "float16": (2, "<f2"),
    "bfloat16": (2, None),
    "float64": (8, "<f8"),
    "float8_e4m3fn": (1, None),
    "float8_e5m2": (1, None),
    "int8": (1, "<i1"),
    "uint8": (1, "|u1"),
    "int16": (2, "<i2"),
    "int32": (4, "<i4"),
    "int64": (8, "<i8"),
    "bool": (1, "|b1"),
    "fp4_packed": (1, "|u1"),
}


class TensorStoreError(RuntimeError):
    """Raised on a corrupt or missing tensor blob."""


def dtype_name(dtype: Any) -> str:
    """Normalize a torch or numpy dtype to a name in :data:`DTYPES`."""
    text = str(dtype)
    text = text.removeprefix("torch.").removeprefix("dtype('").removesuffix("')")
    aliases = {"float": "float32", "double": "float64", "half": "float16", "bfloat": "bfloat16"}
    text = aliases.get(text, text)
    if text not in DTYPES:
        raise TensorStoreError(f"Unsupported dtype {dtype!r}")
    return text


@dataclass
class SliceInfo:
    """Records that a dump covers only a head/tail window of the sequence.

    ``axes`` lists every windowed axis: a square attention mask has two.
    """

    axes: list[int]
    head: int
    tail: int
    original_length: int


@dataclass
class TensorMeta:
    """Sidecar metadata for one dumped tensor."""

    name: str
    dtype: str
    shape: list[int]
    nbytes: int
    sha256: str
    path: str
    role: str = "activation"       # activation | weight | output | routing | logits
    module_id: str | None = None
    sample_id: str | None = None
    step: int | None = None
    slice_info: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def element_count(self) -> int:
        count = 1
        for dim in self.shape:
            count *= dim
        return count

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in (None, {}, [])or k in ("shape", "nbytes")}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TensorMeta:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


def contiguous_bytes(tensor: Any) -> tuple[bytes, str, list[int]]:
    """Extract (raw bytes, dtype name, shape) from a torch tensor or numpy array."""
    if hasattr(tensor, "detach"):  # torch
        import torch

        t = tensor.detach().to("cpu").contiguous()
        name = dtype_name(t.dtype)
        raw = t.view(torch.uint8).numpy().tobytes() if name in ("bfloat16", "float8_e4m3fn", "float8_e5m2") \
            else t.numpy().tobytes()
        return raw, name, list(t.shape)
    import numpy as np

    array = np.ascontiguousarray(tensor)
    return array.tobytes(), dtype_name(array.dtype), list(array.shape)


class TensorStore:
    """Writes and reads tensor dumps beneath ``root``."""

    def __init__(self, root: str | Path, dedupe: bool = True):
        self.root = Path(root)
        self.dedupe = dedupe
        self._by_hash: dict[str, Path] = {}
        self.entries: list[TensorMeta] = []
        self.metadata: dict[str, Any] = {}

    # -- writing ---------------------------------------------------------------

    def write(
        self,
        name: str,
        tensor: Any,
        *,
        role: str = "activation",
        module_id: str | None = None,
        sample_id: str | None = None,
        step: int | None = None,
        subdir: str | Path = "",
        slice_info: SliceInfo | None = None,
        extra: dict[str, Any] | None = None,
    ) -> TensorMeta:
        raw, dtype, shape = contiguous_bytes(tensor)
        digest = hashlib.sha256(raw).hexdigest()
        directory = self.root / subdir if subdir else self.root
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{name}.bin"

        existing = self._by_hash.get(digest) if self.dedupe else None
        if existing is not None and existing.exists() and not target.exists():
            try:
                target.hardlink_to(existing)
            except OSError:
                target.write_bytes(raw)
        else:
            target.write_bytes(raw)
            self._by_hash.setdefault(digest, target)

        meta = TensorMeta(
            name=name,
            dtype=dtype,
            shape=shape,
            nbytes=len(raw),
            sha256=digest,
            path=str(target.relative_to(self.root)),
            role=role,
            module_id=module_id,
            sample_id=sample_id,
            step=step,
            slice_info=asdict(slice_info) if slice_info else None,
            extra=dict(extra or {}),
        )
        target.with_suffix(".json").write_text(json.dumps(meta.to_dict(), indent=2))
        self.entries.append(meta)
        return meta

    # -- reading ---------------------------------------------------------------

    def blob_path(self, meta: TensorMeta) -> Path:
        return self.root / meta.path

    def verify(self, meta: TensorMeta) -> None:
        """Raise unless the blob exists with the recorded size and digest."""
        path = self.blob_path(meta)
        if not path.is_file():
            raise TensorStoreError(f"Missing blob for {meta.name}: {path}")
        raw = path.read_bytes()
        if len(raw) != meta.nbytes:
            raise TensorStoreError(f"{meta.name}: expected {meta.nbytes} bytes, found {len(raw)}")
        digest = hashlib.sha256(raw).hexdigest()
        if digest != meta.sha256:
            raise TensorStoreError(f"{meta.name}: sha256 mismatch ({digest} != {meta.sha256})")

    def read_numpy(self, meta: TensorMeta):
        """Read as a numpy array. Raw-byte dtypes come back as uint8."""
        import numpy as np

        _, np_dtype = DTYPES[meta.dtype]
        raw = self.blob_path(meta).read_bytes()
        if np_dtype is None:
            return np.frombuffer(raw, dtype=np.uint8)
        return np.frombuffer(raw, dtype=np.dtype(np_dtype)).reshape(meta.shape)

    def read_torch(self, meta: TensorMeta, device: str = "cpu"):
        """Read as a torch tensor, reinterpreting bfloat16/fp8 correctly."""
        import torch

        raw = bytearray(self.blob_path(meta).read_bytes())
        torch_dtype = getattr(torch, meta.dtype) if hasattr(torch, meta.dtype) else torch.uint8
        flat = torch.frombuffer(raw, dtype=torch_dtype)
        return flat.reshape(meta.shape).to(device)

    # -- manifest --------------------------------------------------------------

    @property
    def manifest_path(self) -> Path:
        return self.root / MANIFEST_NAME

    def save_manifest(self, metadata: dict[str, Any] | None = None) -> Path:
        payload = {
            "version": 1,
            "metadata": dict(metadata or {}),
            "total_bytes": sum(e.nbytes for e in self.entries),
            "unique_blobs": len({e.sha256 for e in self.entries}),
            "tensors": [e.to_dict() for e in self.entries],
        }
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False))
        return self.manifest_path

    @classmethod
    def load(cls, root: str | Path) -> TensorStore:
        store = cls(root)
        path = store.manifest_path
        if not path.is_file():
            raise TensorStoreError(f"No manifest at {path}")
        payload = yaml.safe_load(path.read_text()) or {}
        store.entries = [TensorMeta.from_dict(e) for e in payload.get("tensors", [])]
        store.metadata = payload.get("metadata", {})
        for entry in store.entries:
            store._by_hash.setdefault(entry.sha256, store.root / entry.path)
        return store

    def find(self, **filters: Any) -> list[TensorMeta]:
        """Select manifest entries by exact field match."""
        return [e for e in self.entries
                if all(getattr(e, key, None) == value for key, value in filters.items())]
