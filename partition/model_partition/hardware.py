# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hardware discovery and the per-module memory budget.

A module is locally runnable when its weights, activations, and KV share fit in
the usable fraction of one GPU. Importable without torch or a GPU — detection
degrades to an empty list.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

GIB = 1024 ** 3

#: Fraction of GPU memory one module may occupy; the rest absorbs allocator
#: fragmentation, cuBLAS workspaces, and CUDA context overhead.
DEFAULT_HEADROOM = 0.60


@dataclass(frozen=True)
class GPUInfo:
    """One visible GPU."""

    index: int
    name: str
    total_bytes: int
    free_bytes: int
    capability: tuple[int, int] | None = None
    #: Dynamic shared memory one block may opt into. A kernel written for a bigger
    #: card asks for more than this and fails to launch — the most common reason a
    #: vendor's own kernels do not run on the GPU in front of you.
    shared_memory_per_block: int = 0
    multiprocessors: int = 0

    def supports_fp8(self) -> bool:
        """True on sm_89+ (L40S, H100).

        Hardware fp8 is not the same as a usable block-scaled fp8 GEMM: the kernel
        still has to fit the card. DeepSeek V4.1's does on sm_89, and its sparse
        attention does not — which is what ``runtime/compat.py`` exists for.
        """
        return self.capability is not None and self.capability >= (8, 9)

    def supports_fp4(self) -> bool:
        """True on sm_100+ (Blackwell). Below that, fp4 has to be widened to compute."""
        return self.capability is not None and self.capability >= (10, 0)

    def architecture(self) -> str:
        return f"sm_{self.capability[0]}{self.capability[1]}" if self.capability else "?"

    def describe(self) -> str:
        """What a kernel author needs to know about this card, in one line."""
        return (f"{self.name} ({self.architecture()}), {format_bytes(self.total_bytes)}, "
                f"{self.multiprocessors} SMs, {self.shared_memory_per_block} bytes of "
                f"dynamic shared memory per block, fp8 "
                f"{'yes' if self.supports_fp8() else 'no'}, fp4 "
                f"{'yes' if self.supports_fp4() else 'no'}")


@dataclass(frozen=True)
class HostInfo:
    """Host-side resources that bound streaming and artifact storage."""

    cpu_count: int
    ram_total_bytes: int
    ram_available_bytes: int
    disk_total_bytes: int
    disk_free_bytes: int
    artifact_root: Path


@dataclass(frozen=True)
class MemoryBudget:
    """The resolved per-module memory ceiling."""

    gpu: GPUInfo | None
    headroom: float
    usable_bytes: int
    #: Set when no GPU was found and the budget is a caller-supplied fallback.
    synthetic: bool = False

    def fits(self, nbytes: int) -> bool:
        return nbytes <= self.usable_bytes


def _detect_gpus_torch() -> list[GPUInfo]:
    try:
        import torch
    except Exception:
        return []
    if not torch.cuda.is_available():
        return []
    gpus: list[GPUInfo] = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        try:
            free, total = torch.cuda.mem_get_info(i)
        except Exception:
            free, total = props.total_memory, props.total_memory
        gpus.append(
            GPUInfo(
                index=i,
                name=props.name,
                total_bytes=int(total),
                free_bytes=int(free),
                capability=(props.major, props.minor),
                shared_memory_per_block=int(getattr(
                    props, "shared_memory_per_block_optin",
                    getattr(props, "shared_memory_per_block", 0))),
                multiprocessors=int(getattr(props, "multi_processor_count", 0)),
            )
        )
    return gpus


def _detect_gpus_smi() -> list[GPUInfo]:
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return []
    gpus: list[GPUInfo] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            gpus.append(GPUInfo(
                index=int(parts[0]),
                name=parts[1],
                total_bytes=int(float(parts[2])) * 1024 * 1024,
                free_bytes=int(float(parts[3])) * 1024 * 1024,
            ))
        except ValueError:
            continue
    return gpus


def detect_gpus() -> list[GPUInfo]:
    """Enumerate visible GPUs, preferring torch (richer info) over nvidia-smi."""
    return _detect_gpus_torch() or _detect_gpus_smi()


def _read_meminfo() -> tuple[int, int]:
    """Return (total, available) host RAM in bytes; (0, 0) if unreadable."""
    try:
        fields: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            value = rest.strip().split()
            if value and value[0].isdigit():
                fields[key] = int(value[0]) * 1024
        return fields.get("MemTotal", 0), fields.get("MemAvailable", 0)
    except OSError:
        return 0, 0


def detect_host(artifact_root: str | Path) -> HostInfo:
    """Probe host CPU/RAM and the filesystem backing ``artifact_root``.

    ``artifact_root`` need not exist; we walk up to the nearest existing ancestor.
    """
    root = Path(artifact_root).expanduser()
    probe = root
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        usage = shutil.disk_usage(probe)
        disk_total, disk_free = usage.total, usage.free
    except OSError:
        disk_total = disk_free = 0
    ram_total, ram_available = _read_meminfo()
    return HostInfo(
        cpu_count=os.cpu_count() or 1,
        ram_total_bytes=ram_total,
        ram_available_bytes=ram_available,
        disk_total_bytes=disk_total,
        disk_free_bytes=disk_free,
        artifact_root=root,
    )


def resolve_budget(
    gpus: list[GPUInfo] | None = None,
    headroom: float = DEFAULT_HEADROOM,
    device_index: int = 0,
    fallback_bytes: int | None = None,
) -> MemoryBudget:
    """Pick the GPU to plan against and compute the per-module ceiling.

    Uses ``total_bytes``, not ``free_bytes``: a plan should describe what this GPU
    can run, not what happened to be free when it was made.
    """
    if not 0 < headroom <= 1:
        raise ValueError(f"headroom must be in (0, 1], got {headroom}")
    gpus = detect_gpus() if gpus is None else gpus
    if not gpus:
        if fallback_bytes is None:
            raise RuntimeError(
                "No GPU detected and no fallback_bytes given. Pass --gpu-memory-gib "
                "to plan for a target GPU without one attached."
            )
        return MemoryBudget(gpu=None, headroom=headroom,
                            usable_bytes=int(fallback_bytes * headroom), synthetic=True)
    chosen = next((g for g in gpus if g.index == device_index), gpus[0])
    return MemoryBudget(gpu=chosen, headroom=headroom,
                        usable_bytes=int(chosen.total_bytes * headroom))


def move_to_device(model, device: str):
    """Move a model, falling back to the host on OOM. Returns ``(model, device)``.

    ``torch.cuda.OutOfMemoryError`` subclasses ``RuntimeError``, so one except
    covers it and the allocator's other placement failures.
    """
    if device == "cpu":
        return model.to("cpu"), "cpu"
    try:
        return model.to(device), device
    except RuntimeError:
        return model.to("cpu"), "cpu"


def format_bytes(nbytes: float) -> str:
    """Human-readable byte count (base-1024), e.g. ``55.6 GiB``."""
    step = 1024.0
    value = float(nbytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < step or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= step
    return f"{value:.1f} TiB"
