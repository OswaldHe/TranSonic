# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common loader interface.

Two modes matter. ``build_meta`` instantiates structure only, on the meta device,
so the module tree can be discovered for a model far larger than local memory.
``build`` instantiates for real, optionally with a partial state dict when only
one module's weights are resident.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

TORCH_DTYPES = {
    "float32": "float32", "float16": "float16", "bfloat16": "bfloat16",
    "fp32": "float32", "fp16": "float16", "bf16": "bfloat16",
}


class LoaderError(RuntimeError):
    """Raised when a model's code or config cannot produce a module."""


def torch_dtype(name: str):
    import torch

    resolved = TORCH_DTYPES.get(name, name)
    dtype = getattr(torch, resolved, None)
    if dtype is None:
        raise LoaderError(f"Unknown dtype {name!r}")
    return dtype


@dataclass
class LoadedModel:
    """An instantiated model plus its discovered module tree."""

    model: Any
    config: dict[str, Any]
    dtype: str = "bfloat16"
    device: str = "cpu"
    meta: bool = False
    #: ``"single"`` when every parameter is on ``device``; ``"auto"`` when layers
    #: are spread across GPU and host. An auto-placed model must never be moved
    #: with ``.to()`` — that would undo the placement.
    placement: str = "single"
    metadata: dict[str, Any] = field(default_factory=dict)

    def named_modules(self) -> dict[str, Any]:
        return {name: module for name, module in self.model.named_modules() if name}

    def submodule(self, qualified_name: str):
        module = self.model
        for part in qualified_name.split("."):
            if part.isdigit() and hasattr(module, "__getitem__"):
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module

    def layer_container(self) -> tuple[str, Any] | None:
        """Find the repeating decoder stack, e.g. ``model.layers``."""
        import torch

        best: tuple[str, Any] | None = None
        for name, module in self.named_modules().items():
            if isinstance(module, torch.nn.ModuleList) and len(module) > 1:
                if best is None or len(module) > len(best[1]):
                    best = (name, module)
        return best


class ModelLoader(Protocol):
    """What every loader provides."""

    def build_meta(self) -> LoadedModel: ...

    def build(self, state_dict: dict[str, Any] | None = None, device: str = "cpu") -> LoadedModel: ...
