# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read a module directory's own recorded calls, weights and reference outputs.

A module directory is the deliverable, and it says what it needs: ``calls.json`` names
every tensor one of its modules was called with by a path relative to the directory,
plus the dtype and shape to read those bytes as. This reads that, so ``inference.py`` and
``verify.py`` inside a published artifact need nothing but ``torch`` and this file —
which travels with the artifact under ``runtime/``, copied in by
:func:`~model_partition.extract.vendor_runtime`.

The harness has a second, richer view of the same tracing: ``trace/records.yaml`` indexed
by :class:`~model_partition.runtime.module_runner.TraceBundle`, which covers a whole run
at once and is what the loop's own stages use. This is one module's slice of it, in JSON,
so reading it needs no YAML parser and no part of the harness.

Imports nothing from the harness beyond its own neighbours — keep it that way.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CALLS_FILENAME = "calls.json"
CONFIG_FILENAME = "config.json"

#: Where ``fetch_weights.py`` puts the checkpoint shards, relative to the run root. A run
#: that read its weights from the checkpoint rather than dumping them — which is what makes
#: a 475 GiB model traceable at all — leaves its modules short of the numbers they need,
#: and this is where they are looked for instead.
CHECKPOINT_DIR = "hf"

#: Read in blocks when pulling a tensor out of a shard, so one tensor-sized allocation is
#: enough. A single n-gram table is 94.4 GiB and the host holds 124.
CHUNK_BYTES = 16 * 1024 * 1024

#: safetensors' dtype names, which are not torch's. ``F8_E8M0`` is the exponent-only
#: block scale a blockwise-quantized checkpoint stores beside each weight — 47589 of the
#: tensors here are one — and it needs a torch new enough to have ``float8_e8m0fnu``.
SAFETENSORS_DTYPES = {
    "F64": "float64", "F32": "float32", "F16": "float16", "BF16": "bfloat16",
    "F8_E4M3": "float8_e4m3fn", "F8_E5M2": "float8_e5m2",
    "F8_E8M0": "float8_e8m0fnu", "F4_E2M1": "float4_e2m1fn_x2",
    "I64": "int64", "I32": "int32", "I16": "int16", "I8": "int8",
    "U8": "uint8", "BOOL": "bool",
}

#: Keys the trace encodes a value tree with. A tensor is named by where its bytes are.
TENSOR_KEY = "__tensor__"
LIST_KEY = "__list__"
TUPLE_KEY = "__tuple__"
UNSUPPORTED_KEY = "__unsupported__"


class ArtifactError(RuntimeError):
    """Raised when a module directory cannot be read on its own terms."""


def read_tensor(directory: Path, spec: dict[str, Any], device: str = "cpu") -> Any:
    """One dumped tensor, from the ``.bin`` its recorded path points at.

    The bytes are read as ``uint8`` and reinterpreted, which is how they were written and
    the only way that works for every dtype in play: ``torch.frombuffer`` does not take
    bfloat16, fp8, a packed fp4 pair or a complex rotary table, and all four are here.

    Read straight into one tensor rather than through ``bytes``. ``read_bytes()`` holds
    the whole blob as an immutable object and a ``bytearray`` of it is a second full copy
    that cannot be released until the first is, so the supported 94.4 GiB n-gram table
    needed about 189 GiB of transient host memory just to begin a check — on hosts with
    124 GiB, which is the class this is meant to run on.
    """
    import torch

    path = (Path(directory) / spec["bin"]).resolve()
    if not path.is_file():
        raise ArtifactError(f"{path} is missing from the artifact")
    dtype = getattr(torch, spec["dtype"], None)
    if not isinstance(dtype, torch.dtype):
        raise ArtifactError(f"this torch has no dtype {spec['dtype']!r}")
    flat = torch.from_file(str(path), shared=False, size=path.stat().st_size,
                           dtype=torch.uint8)
    if dtype is not torch.uint8:
        flat = flat.view(dtype)
    return flat.reshape(spec["shape"]).to(device)


def read_shard_tensors(path: Path, wanted: list[tuple[str, str]],
                       device: str = "cpu") -> dict[str, Any]:
    """Named tensors out of one safetensors shard, opened once.

    The format is read here rather than with the ``safetensors`` package on purpose: a
    module directory's promise is that it runs with ``torch`` and nothing else installed,
    and the format is an 8-byte little-endian header length, that many bytes of JSON
    naming each tensor's dtype, shape and byte range, then the data. Reading it costs
    twenty lines and keeps the promise.

    ``wanted`` is ``(name to return it under, key inside the shard)``.
    """
    import torch

    with path.open("rb") as handle:
        length = int.from_bytes(handle.read(8), "little")
        if not length:
            raise ArtifactError(f"{path} has no safetensors header")
        header = json.loads(handle.read(length))
        start_of_data = 8 + length

        loaded: dict[str, Any] = {}
        for name, key in wanted:
            entry = header.get(key)
            if not isinstance(entry, dict):
                raise ArtifactError(f"{path} does not hold {key!r}")
            torch_name = SAFETENSORS_DTYPES.get(entry["dtype"])
            dtype = getattr(torch, torch_name, None) if torch_name else None
            if not isinstance(dtype, torch.dtype):
                raise ArtifactError(
                    f"{key}: this torch ({torch.__version__}) has no dtype for "
                    f"{entry['dtype']!r}"
                    + (f" (expected torch.{torch_name})" if torch_name else "")
                    + ". A blockwise-quantized checkpoint needs a torch new enough for "
                      "its scale format.")
            begin, end = entry["data_offsets"]
            nbytes = end - begin
            # One tensor-sized allocation, filled in blocks. Reading the whole slice as
            # `bytes` first would hold the largest tensors twice.
            flat = torch.empty(nbytes, dtype=torch.uint8)
            handle.seek(start_of_data + begin)
            at = 0
            while at < nbytes:
                block = handle.read(min(CHUNK_BYTES, nbytes - at))
                if not block:
                    raise ArtifactError(f"{path} ended early reading {key!r}")
                flat[at:at + len(block)] = torch.frombuffer(
                    bytearray(block), dtype=torch.uint8)
                at += len(block)
            if dtype is not torch.uint8:
                flat = flat.view(dtype)
            loaded[name] = flat.reshape(entry["shape"]).to(device)
    return loaded


def decode(directory: Path, value: Any, device: str = "cpu") -> Any:
    """Rebuild a recorded argument tree, reading each tensor it names."""
    if isinstance(value, dict):
        if TENSOR_KEY in value:
            spec = value[TENSOR_KEY]
            if not isinstance(spec, dict):
                # A tensor the run chose not to dump. Named rather than nulled, so the
                # failure says which one instead of surfacing somewhere later.
                raise ArtifactError(
                    f"tensor {spec!r} was not dumped; publish materializes the weights "
                    "a module needs, or trace with weight caching on")
            return read_tensor(directory, spec, device)
        if UNSUPPORTED_KEY in value:
            return None
        if TUPLE_KEY in value:
            return tuple(decode(directory, item, device) for item in value[TUPLE_KEY])
        if LIST_KEY in value:
            return [decode(directory, item, device) for item in value[LIST_KEY]]
        return {key: decode(directory, item, device) for key, item in value.items()}
    return value


@dataclass
class ModuleCalls:
    """One module's recorded calls and weights, as its directory records them."""

    directory: Path
    module_id: str
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def submodules(self) -> list[str]:
        return list(self.payload.get("submodules") or [])

    @property
    def layers(self) -> list[int]:
        return list(self.payload.get("layers") or [])

    @property
    def missing_weights(self) -> list[str]:
        """Parameters the run did not dump, whether or not they can still be read."""
        return list(self.payload.get("weights_missing") or [])

    @property
    def fetched_weights(self) -> list[str]:
        """Undumped parameters the fetched checkpoint beside this artifact does supply."""
        root = self.checkpoint_root()
        if root is None:
            return []
        return sorted(name for name, ref in self.checkpoint_weights.items()
                      if ref.get("shard") and ref.get("key")
                      and (root / ref["shard"]).is_file())

    @property
    def absent_weights(self) -> list[str]:
        """Parameters nothing here can supply: not dumped, and not fetched either.

        The distinction matters to anyone reading the output. "Not in the artifact" was
        true of a parameter waiting for ``fetch_weights.py`` and of one that is simply
        gone, and only the second is a reason the numbers will not match.
        """
        return sorted(set(self.missing_weights) - set(self.fetched_weights))

    def keys(self, sample_order: list[str] | None = None) -> list[str]:
        """The recorded passes, prefill first and in the run's own sample order."""
        order = list(sample_order or [])

        def rank(key: str) -> tuple[int, int, str]:
            sample, _, step = key.rpartition("#")
            known = order.index(sample) if sample in order else len(order)
            return (int(step), known, sample)

        return sorted(self.payload.get("calls") or {}, key=rank)

    def select(self, sample: str | None = None, step: int | None = None,
               sample_order: list[str] | None = None) -> tuple[str, list[dict[str, Any]]]:
        """One pass: its key, and its calls in the order the submodules ran."""
        keys = [k for k in self.keys(sample_order)
                if (sample is None or k.rpartition("#")[0] == sample)
                and (step is None or int(k.rpartition("#")[2]) == step)]
        if not keys:
            raise ArtifactError(
                f"{self.module_id}: nothing recorded for sample {sample!r} step {step!r}; "
                f"have {self.keys(sample_order)}")
        return keys[0], list((self.payload.get("calls") or {})[keys[0]])

    @property
    def checkpoint_weights(self) -> dict[str, dict[str, str]]:
        """Parameters to read from the checkpoint, each with its shard and its key there.

        Written down so the artifact is self-describing: ``fetch_weights.py`` can pull
        exactly the shards these names live in instead of the whole checkpoint, and reading
        one needs no rename rules here — the key it has *inside* the shard is recorded.
        Older artifacts recorded only the names, which is a list rather than a mapping.
        """
        entry = self.payload.get("weights_missing") or {}
        return entry if isinstance(entry, dict) else {name: {} for name in entry}

    def weights(self, device: str = "cpu") -> dict[str, Any]:
        """This module's weights, keyed by original parameter name.

        The dumped ones come from the ``.bin`` files beside this directory. Anything the
        run read from the checkpoint instead is read from the checkpoint too, out of
        ``<run>/hf/`` — which ``fetch_weights.py`` fills. Without that directory the module
        still builds from whatever was dumped, and :attr:`missing_weights` says what it is
        short of, because a partial build that reports its own gap beats an import error.
        """
        found = {name: read_tensor(self.directory, spec, device)
                 for name, spec in (self.payload.get("weights") or {}).items()}
        root = self.checkpoint_root()
        if root is None:
            return found
        by_shard: dict[str, list[tuple[str, str]]] = {}
        for name, ref in self.checkpoint_weights.items():
            shard, key = ref.get("shard"), ref.get("key")
            if shard and key and (root / shard).is_file():
                by_shard.setdefault(shard, []).append((name, key))
        for shard, pairs in by_shard.items():
            found.update(read_shard_tensors(root / shard, pairs, device))
        return found

    def checkpoint_root(self) -> Path | None:
        """``<run>/hf/``, if the checkpoint has been fetched beside this artifact."""
        for base in (self.directory.parent.parent, self.directory.parent, self.directory):
            candidate = base / CHECKPOINT_DIR
            if candidate.is_dir():
                return candidate
        return None

    def runs(self, calls: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        """One pass's calls, split into the group's separate invocations.

        A sequential group is one computation from its first submodule's input to its last
        submodule's output — across *distinct* submodules. A submodule the pass called more
        than once is not a longer chain, it is the group run again: DeepSeek's draft head
        calls `markov_head` once per drafted position, each on the token the previous call
        sampled. Pairing the first call's input with the last call's output checks neither,
        and since that module is pure the whole mismatch would be the pairing's.
        """
        runs: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        seen: set[Any] = set()
        for call in calls:
            name = call.get("submodule")
            if name in seen:
                runs.append(current)
                current, seen = [], set()
            current.append(call)
            seen.add(name)
        if current:
            runs.append(current)
        return runs

    def arguments(self, calls: list[dict[str, Any]], device: str = "cpu") -> tuple[tuple, dict]:
        """Arguments for the whole group, not just its first submodule.

        A group can be heterogeneous — a normalization followed by attention — and then
        the first submodule's recorded call says nothing about the positions and shared
        state the attention needs. The value flowing in comes from the group's entry
        point; everything beyond it is the union over the group, so whatever the module
        does internally, it was given everything. First occurrence wins, so a keyword
        that changes down the group keeps the value the entry point saw.
        """
        args = [decode(self.directory, item, device) for item in calls[0]["args"]]
        kwargs = {k: decode(self.directory, v, device)
                  for k, v in (calls[0].get("kwargs") or {}).items()}
        extras = tuple(args[1:])
        for call in calls[1:]:
            for key, value in (call.get("kwargs") or {}).items():
                if key not in kwargs:
                    kwargs[key] = decode(self.directory, value, device)
            other = [decode(self.directory, item, device) for item in call["args"]]
            if len(other) - 1 > len(extras):
                extras = tuple(other[1:])
        return (tuple(args[:1]) + extras if args else extras), kwargs

    def reference(self, calls: list[dict[str, Any]], parallel: bool = False,
                  device: str = "cpu") -> Any:
        """The output the trace recorded for this pass.

        A parallel group's calls are independent — one expert each — so the first call's
        own output is the reference. A sequential group is one computation, from the first
        submodule's input to the last submodule's output.
        """
        return decode(self.directory, (calls[0] if parallel else calls[-1])["output"], device)

    def apply_state(self, source: Any, calls: list[dict[str, Any]],
                    device: str = "cpu") -> int:
        """Put back the cross-module state the recorded call ran on.

        A model that passes tensors between modules through a module-level object — the
        compressed KV DeepSeek's attention layers share — leaves a consumer with no
        argument naming the largest thing it reads. Restoring it on this module's own
        ``source.py`` is what makes the check one of this module against its reference,
        rather than of this module against another having run first.
        """
        applied = 0
        for call in calls:
            for dotted, encoded in (call.get("state") or {}).items():
                holder_name, _, attribute = dotted.rpartition(".")
                holder = getattr(source, holder_name, None)
                if holder is None:
                    continue
                setattr(holder, attribute, decode(self.directory, encoded, device))
                applied += 1
        return applied


def load_calls(directory: str | Path) -> dict[str, ModuleCalls]:
    """Every module recorded beside this implementation, by module id."""
    directory = Path(directory).resolve()
    path = directory / CALLS_FILENAME
    if not path.is_file():
        raise ArtifactError(f"{path} is missing; re-run extraction over the trace")
    payload = json.loads(path.read_text())
    return {module_id: ModuleCalls(directory, module_id, entry)
            for module_id, entry in (payload.get("modules") or {}).items()}


def load_config(directory: str | Path) -> dict[str, Any]:
    """The config this module's subtree was constructed with."""
    path = Path(directory) / CONFIG_FILENAME
    return json.loads(path.read_text()) if path.is_file() else {}
