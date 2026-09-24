# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Make a model's own code run on the GPU in front of us.

A vendor's reference implementation is written for the hardware the vendor has. Its
kernels can ask for more shared memory than this card allows, or a dtype whose tensor
cores only exist a generation later — and then the model does not run at all, which
means there is no trace, no reference and nothing to partition.

So that becomes work for the loop rather than a wall: a *compatibility patch* is a small
python file under ``<run>/compat/`` that replaces the parts that will not run. It is the
agent's to write and a reader's to check:

    def apply(vendor, device):
        '''Replace what does not run here. Return what was replaced.'''
        vendor.sparse_attn = _sparse_attn_sm89
        return ["sparse_attn"]

``vendor`` is the imported entry module, so a patch rebinds a function or a class by
name. ``device`` is a :class:`~model_partition.hardware.GPUInfo`, so a patch can ask
what it is adapting to instead of assuming.

Two things a patch owes the reader. It runs *before* tracing, so whatever it computes
becomes the reference every later stage is measured against — the run manifest and the
summary say so, because "reproduces the model" would otherwise quietly mean "reproduces
the adaptation". And it should keep the precision the original used where the card
supports it, and say in a comment which precision it moved to where it could not.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from model_partition.runtime import private_module_name

#: Where a run keeps its compatibility patches, applied in sorted order.
COMPAT_DIR = "compat"

#: The function a patch exposes.
ENTRY_POINT = "apply"

#: Messages that mean "this kernel cannot run on this GPU", as opposed to a bug in the
#: model or in the loop. Matched case-insensitively against the exception text.
HARDWARE_LIMITS = (
    "shared memory",
    "invalid device function",
    "no kernel image is available",
    "device_type mismatch",
    "not supported on this architecture",
    "requires sm_",
    "unsupported dtype",
    "cuda error: invalid argument",
)


class CompatError(RuntimeError):
    """Raised when a compatibility patch is unusable."""


@dataclass
class CompatReport:
    """What the patches replaced."""

    files: list[str] = field(default_factory=list)
    replaced: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def applied(self) -> bool:
        return bool(self.replaced) and not self.errors

    def summary(self) -> str:
        if not self.files:
            return "no compatibility patches"
        text = (f"{len(self.files)} compatibility patch(es) replaced "
                f"{len(self.replaced)}: {', '.join(self.replaced[:6])}")
        if self.errors:
            text += f"; {len(self.errors)} failed: {'; '.join(list(self.errors)[:3])}"
        return text

    def to_dict(self) -> dict[str, Any]:
        return {"files": list(self.files), "replaced": list(self.replaced),
                "errors": dict(self.errors)}


def is_hardware_limit(exc: BaseException) -> bool:
    """Whether a failure is the GPU refusing the code, not the code being wrong.

    The distinction decides who gets the failure: a numeric mismatch goes to whoever
    owns the arithmetic, while a kernel this card cannot launch is a porting job.
    """
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in HARDWARE_LIMITS)


def patch_paths(run_dir: str | Path) -> list[Path]:
    """The compatibility patches of a run, in the order they are applied."""
    directory = Path(run_dir) / COMPAT_DIR
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.glob("*.py") if not p.name.startswith("_"))


def apply_patches(vendor: Any, paths: list[Path], device: Any = None) -> CompatReport:
    """Run each patch against the imported vendor module."""
    report = CompatReport()
    for path in paths:
        report.files.append(path.name)
        try:
            patch = _import(path)
            entry = getattr(patch, ENTRY_POINT, None)
            if not callable(entry):
                raise CompatError(f"{path.name} defines no callable {ENTRY_POINT}()")
            replaced = entry(vendor, device)
            report.replaced.extend(
                [str(name) for name in replaced] if isinstance(replaced, (list, tuple))
                else [str(replaced or path.stem)]
            )
        except Exception as exc:
            report.errors[path.name] = f"{type(exc).__name__}: {exc}"
    return report


def _import(path: Path) -> Any:
    name = private_module_name("_model_partition_compat_", path.resolve())
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise CompatError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    written = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(name, None)
        raise CompatError(f"{path.name} failed to import: {exc}") from exc
    finally:
        sys.dont_write_bytecode = written
    return module
