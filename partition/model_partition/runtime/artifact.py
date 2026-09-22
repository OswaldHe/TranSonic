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
    """
    import torch

    path = (Path(directory) / spec["bin"]).resolve()
    if not path.is_file():
        raise ArtifactError(f"{path} is missing from the artifact")
    dtype = getattr(torch, spec["dtype"], None)
    if not isinstance(dtype, torch.dtype):
        raise ArtifactError(f"this torch has no dtype {spec['dtype']!r}")
    flat = torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.uint8)
    if dtype is not torch.uint8:
        flat = flat.view(dtype)
    return flat.reshape(spec["shape"]).to(device)


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
        """Parameters the run did not dump, so this module cannot be built from it."""
        return list(self.payload.get("weights_missing") or [])

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

    def weights(self, device: str = "cpu") -> dict[str, Any]:
        """This module's dumped weights, keyed by original parameter name."""
        return {name: read_tensor(self.directory, spec, device)
                for name, spec in (self.payload.get("weights") or {}).items()}

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
