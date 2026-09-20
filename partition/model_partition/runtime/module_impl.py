# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load and run a module's extracted inference implementation.

This is the boundary between what the loop owns and what the harness owns. The
implementation under ``modules/<group>/inference.py`` is the loop's: the agent
edits it to make a module numerically correct and convenient to write a kernel
for. Everything here — how it is invoked and how its output is judged — is the
harness's, so the agent cannot make a module pass by weakening the check.

The contract an implementation must satisfy:

    def build_module(config: dict, weights: dict[str, Tensor], device: str) -> Callable

``weights`` is keyed by original parameter name. The returned callable is invoked
with the arguments the trace recorded for that module.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

IMPL_FILENAME = "inference.py"
BUILD_FUNCTION = "build_module"


class ImplError(RuntimeError):
    """Raised when an extracted implementation is missing or unusable."""


@dataclass
class ExtractedImpl:
    """An imported ``inference.py`` and the group it belongs to."""

    path: Path
    module: Any
    module_ids: list[str]

    def build(self, config: dict[str, Any], weights: dict[str, Any], device: str = "cpu") -> Any:
        builder = getattr(self.module, BUILD_FUNCTION, None)
        if not callable(builder):
            raise ImplError(f"{self.path} defines no callable {BUILD_FUNCTION}()")
        try:
            return builder(config, weights, device)
        except TypeError:
            return builder(config, weights)


def load_impl(directory: str | Path) -> ExtractedImpl:
    """Import the ``inference.py`` in an extracted group directory."""
    path = Path(directory) / IMPL_FILENAME
    if not path.is_file():
        raise ImplError(f"No {IMPL_FILENAME} in {directory}")
    name = f"_model_partition_impl_{abs(hash(str(path.resolve())))}"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImplError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(name, None)
        raise ImplError(f"{path} failed to import: {exc}") from exc
    return ExtractedImpl(path=path, module=module,
                         module_ids=list(getattr(module, "MODULE_IDS", [])))


def find_impl_dirs(modules_dir: str | Path) -> dict[str, Path]:
    """Map each module id to the group directory implementing it."""
    root = Path(modules_dir)
    index = root / "index.yaml"
    if not index.is_file():
        return {}
    import yaml

    payload = yaml.safe_load(index.read_text()) or {}
    mapping: dict[str, Path] = {}
    for group in payload.get("groups", []):
        directory = group.get("directory")
        if not directory:
            continue
        for module_id in group.get("module_ids", []):
            mapping[module_id] = root / directory
    return mapping


def run_impl(
    impl: ExtractedImpl,
    config: dict[str, Any],
    weights: dict[str, Any],
    args: tuple,
    kwargs: dict[str, Any],
    device: str = "cpu",
) -> Any:
    """Build the implementation and call it with one recorded invocation."""
    import torch

    callable_module = impl.build(config, weights, device)
    if not callable(callable_module):
        raise ImplError(f"{impl.path}:{BUILD_FUNCTION}() returned a non-callable")
    with torch.no_grad():
        return callable_module(*args, **kwargs)
