# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load and run a module's extracted inference implementation.

This is the boundary between what the loop owns and what the harness owns. The
implementation under ``modules/<group>/inference.py`` is the loop's: the agent
edits it to make a module numerically correct and convenient to write a kernel
for. Everything here — how it is invoked and how its output is judged — is the
harness's, so the agent cannot make a module pass by weakening the check.

The contract an implementation must satisfy:

    def build_module(config: dict, weights: dict[str, Tensor], device: str,
                     submodule: str | None = None) -> Callable

``weights`` is keyed by original parameter name. The returned callable is invoked
with the arguments the trace recorded for that module. ``submodule`` is optional
and only matters for a parallel group — a set of MoE experts, where one recorded
call exercises one expert — so an implementation that needs no such distinction
can leave it off its signature entirely.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

IMPL_FILENAME = "inference.py"
BUILD_FUNCTION = "build_module"

#: What ``build_module`` may declare beyond ``config`` and ``weights``. Each is passed
#: only when the implementation declares it, so an older one keeps working unchanged.
OPTIONAL_ARGUMENTS = ("device", "submodule", "module_id")


class ImplError(RuntimeError):
    """Raised when an extracted implementation is missing or unusable."""


@dataclass
class ExtractedImpl:
    """An imported ``inference.py`` and the group it belongs to."""

    path: Path
    module: Any
    module_ids: list[str]

    def build(self, config: dict[str, Any], weights: dict[str, Any], device: str = "cpu",
              submodule: str | None = None, module_id: str | None = None) -> Any:
        """Call the implementation's ``build_module``, passing what it accepts.

        The optional arguments are tried widest first so an implementation is free
        to declare only what it uses; a ``TypeError`` from inside the builder is
        distinguished from one raised by the call itself by inspecting the
        signature rather than by catching it.
        """
        import inspect

        builder = getattr(self.module, BUILD_FUNCTION, None)
        if not callable(builder):
            raise ImplError(f"{self.path} defines no callable {BUILD_FUNCTION}()")
        try:
            parameters = inspect.signature(builder).parameters
            accepted = (
                set(OPTIONAL_ARGUMENTS)
                if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())
                else set(parameters)
            )
        except (TypeError, ValueError):  # pragma: no cover - builtins only
            accepted = set(OPTIONAL_ARGUMENTS)
        kwargs: dict[str, Any] = {}
        if "device" in accepted:
            kwargs["device"] = device
        if "submodule" in accepted and submodule is not None:
            kwargs["submodule"] = submodule
        if "module_id" in accepted and module_id is not None:
            kwargs["module_id"] = module_id
        return builder(config, weights, **kwargs)


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
    # No `__pycache__` beside it: the directory is a deliverable, and the file is one
    # the agent edits between runs.
    written = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(name, None)
        raise ImplError(f"{path} failed to import: {exc}") from exc
    finally:
        sys.dont_write_bytecode = written
    return ExtractedImpl(path=path, module=module,
                         module_ids=list(getattr(module, "MODULE_IDS", [])))


def find_impl_dirs(modules_dir: str | Path) -> dict[str, Path]:
    """Map each module id to the group directory implementing it.

    The index names the directory, but a directory that has been renamed or moved since
    the index was written would otherwise map every one of its modules to a path with no
    ``inference.py`` — and that reads as "every module is unusable" rather than as "the
    index is stale". So a name the index gives that is not there is looked up again by the
    signature in each group's own ``meta.yaml``, which travels with the directory.
    """
    root = Path(modules_dir)
    index = root / "index.yaml"
    if not index.is_file():
        return {}
    from model_partition import yamlio

    payload = yamlio.load_path(index) or {}
    mapping: dict[str, Path] = {}
    by_signature: dict[str, Path] | None = None
    for group in payload.get("groups", []):
        directory = group.get("directory")
        if directory and (root / directory).is_dir():
            target = root / directory
        else:
            if by_signature is None:
                by_signature = {}
                for meta in sorted(root.glob("*/meta.yaml")):
                    found = (yamlio.load_path(meta) or {}).get("signature")
                    if found:
                        by_signature[str(found)] = meta.parent
            target = by_signature.get(str(group.get("signature")))
        if target is None:
            continue
        for module_id in group.get("module_ids", []):
            mapping[module_id] = target
    return mapping


def run_impl(
    impl: ExtractedImpl,
    config: dict[str, Any],
    weights: dict[str, Any],
    args: tuple,
    kwargs: dict[str, Any],
    device: str = "cpu",
    submodule: str | None = None,
) -> Any:
    """Build the implementation and call it with one recorded invocation."""
    import torch

    callable_module = impl.build(config, weights, device, submodule=submodule)
    if not callable(callable_module):
        raise ImplError(f"{impl.path}:{BUILD_FUNCTION}() returned a non-callable")
    with torch.no_grad():
        return callable_module(*args, **kwargs)
