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

from model_partition import yamlio

MANIFEST_NAME = "manifest.yaml"

#: dtype name -> (bytes per element, numpy dtype string or None)
DTYPES: dict[str, tuple[int, str | None]] = {
    "float32": (4, "<f4"),
    "float16": (2, "<f2"),
    "bfloat16": (2, None),
    "float64": (8, "<f8"),
    "float8_e4m3fn": (1, None),
    "float8_e5m2": (1, None),
    # A block scale's dtype and a pair of fp4 values packed into a byte. Both are what a
    # quantized checkpoint stores, so both appear the moment a module's weights are
    # dumped rather than read back out of the checkpoint.
    "float8_e8m0fnu": (1, None),
    "float4_e2m1fn_x2": (1, None),
    "int8": (1, "<i1"),
    "uint8": (1, "|u1"),
    "int16": (2, "<i2"),
    "int32": (4, "<i4"),
    "int64": (8, "<i8"),
    "bool": (1, "|b1"),
    "fp4_packed": (1, "|u1"),
    # A precomputed rotary table is complex: DeepSeek's `precompute_freqs_cis` returns
    # `torch.polar` output, and it is a derived buffer, so the recording is the only
    # place a module can get it from.
    "complex64": (8, "<c8"),
    "complex128": (16, "<c16"),
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


def contiguous_view(tensor: Any) -> tuple[Any, str, list[int]]:
    """A zero-copy view of a tensor's bytes, its dtype name, and its shape.

    A ``memoryview`` rather than ``bytes``: asking for the bytes copies the whole tensor,
    so a tensor larger than half of memory could not be written at all — and DeepSeek
    V4.1's n-gram table is 94.4 GiB on a 124 GiB host. A view can be hashed and written in
    chunks with nothing duplicated.

    For a dtype numpy has no name for — bfloat16, fp8, a packed fp4 pair — the bytes are
    reinterpreted as ``uint8`` first, which loses nothing: :meth:`TensorStore.read_torch`
    reads them back as what they are.
    """
    if hasattr(tensor, "detach"):  # torch
        import torch

        t = tensor.detach().to("cpu").contiguous()
        name = dtype_name(t.dtype)
        array = (t.view(torch.uint8) if DTYPES[name][1] is None else t).numpy()
        return memoryview(array).cast("B"), name, list(t.shape)
    import numpy as np

    array = np.ascontiguousarray(tensor)
    return memoryview(array).cast("B"), dtype_name(array.dtype), list(array.shape)


#: Bytes moved per read and per hash update. Large enough that a hundred-gigabyte
#: tensor is not a hundred million calls, small enough to stay in cache.
CHUNK = 32 << 20


def _digest(view: memoryview) -> str:
    """The sha256 of a tensor's bytes, without copying them."""
    digest = hashlib.sha256()
    for start in range(0, view.nbytes, CHUNK):
        digest.update(view[start:start + CHUNK])
    return digest.hexdigest()


def _write(target: Path, view: memoryview) -> None:
    """Write a tensor's bytes, streaming so nothing is held twice."""
    with open(target, "wb") as handle:
        for start in range(0, view.nbytes, CHUNK):
            handle.write(view[start:start + CHUNK])


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
        view, dtype, shape = contiguous_view(tensor)
        digest = _digest(view)
        nbytes = view.nbytes
        directory = self.root / subdir if subdir else self.root
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{name}.bin"

        existing = self._by_hash.get(digest) if self.dedupe else None
        if existing is not None and existing.exists() and not target.exists():
            try:
                target.hardlink_to(existing)
            except OSError:
                _write(target, view)
        else:
            _write(target, view)
            self._by_hash.setdefault(digest, target)

        meta = TensorMeta(
            name=name,
            dtype=dtype,
            shape=shape,
            nbytes=nbytes,
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
        size = path.stat().st_size
        if size != meta.nbytes:
            raise TensorStoreError(f"{meta.name}: expected {meta.nbytes} bytes, found {size}")
        # Digested in blocks. The largest blob in a run is tens of gigabytes, and holding
        # one whole to hash it costs as much host memory as reading the tensor does.
        running = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
                running.update(block)
        digest = running.hexdigest()
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
        """Read as a torch tensor, reinterpreting bfloat16/fp8 correctly.

        Read straight into one tensor. ``bytearray(path.read_bytes())`` holds the blob
        twice — an immutable copy and a mutable one, neither releasable until the other
        is — which for the largest tensors in a run is the difference between a check
        that starts and one the kernel kills.

        Always read as ``uint8`` and reinterpret: neither ``from_file`` nor
        ``frombuffer`` takes every dtype torch has, and bfloat16, fp8 and a packed fp4
        pair are among the ones they do not.
        """
        import torch

        path = self.blob_path(meta)
        torch_dtype = getattr(torch, meta.dtype) if hasattr(torch, meta.dtype) else torch.uint8
        flat = torch.from_file(str(path), shared=False, size=path.stat().st_size,
                               dtype=torch.uint8)
        if torch_dtype is not torch.uint8:
            flat = flat.view(torch_dtype)
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
        self.manifest_path.write_text(yamlio.dumps(payload))
        return self.manifest_path

    @classmethod
    def load(cls, root: str | Path) -> TensorStore:
        store = cls(root)
        path = store.manifest_path
        if not path.is_file():
            raise TensorStoreError(f"No manifest at {path}")
        payload = yamlio.load_path(path) or {}
        store.entries = [TensorMeta.from_dict(e) for e in payload.get("tensors", [])]
        store.metadata = payload.get("metadata", {})
        for entry in store.entries:
            store._by_hash.setdefault(entry.sha256, store.root / entry.path)
        return store

    def find(self, **filters: Any) -> list[TensorMeta]:
        """Select manifest entries by exact field match."""
        return [e for e in self.entries
                if all(getattr(e, key, None) == value for key, value in filters.items())]
