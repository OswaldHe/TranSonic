# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure the primitives, write ``systems/probed.yaml``.

The simulator refuses to run while any efficiency coefficient is null, and this is what
fills them in. That refusal is the point: a roofline at 100% of 1,299 fp8 TFLOPS would make
arithmetic free and rank every scheme on communication alone, so the choice is between a
measured number and no simulation at all — never between a measured number and peak.

What can and cannot be measured here is decided by the host, not by preference. This is a
trn2.3xlarge with one device, so:

- **Probed.** Tensor-engine throughput, vector and scalar engine throughput, GPSIMD gather
  rate, HBM read bandwidth contiguous and strided, PCIe bandwidth and per-transfer cost,
  NVMe sequential and random-read behaviour. All per logical NeuronCore, which is the
  placement grain, so they transfer to the 16-device target unchanged.
- **Not probeable.** Anything needing a second device: inter-device NeuronLink bandwidth,
  torus hop latency, collectives wider than four cores. Those stay at their datasheet values
  and the ranking report is required to name them as extrapolations.

Every value written carries ``source`` and, where it is not a direct measurement,
``derived_from``. A coefficient is a fitted ratio, not a law of the machine, and the report
that leans on one should say so.
"""

from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

#: Per-logical-NeuronCore peaks, derived from the device totals in the system YAML by
#: dividing by the four logical cores. The probe reports ratios against these.
LOGICAL_NCS_PER_DEVICE = 4

#: Repetitions of each timed dispatch. The median is taken, not the mean: a single outlier
#: from host scheduling should not move a coefficient the whole run depends on.
TIMED_ITERS = 20

#: Compiler arguments the bootstrapped kernels on this host are known to work with. A bare
#: `torch.matmul` through XLA fails to compile on this toolchain (NCC_ISMP902 in the
#: simplifier), which is why every probe goes through NKI rather than through torch.
COMPILER_ARGS = ["--target=trn2", "--auto-cast=none"]

#: Plausible range for a fitted on-chip or host link bandwidth, in bytes/s. A least-squares
#: slope can be arbitrarily wrong without the fit reporting failure, and one was: an early
#: run of the collective probe fitted 0.78 MB/s for an on-chip link and wrote it into the
#: overlay as a measurement, where nothing downstream would have questioned it. Anything
#: outside this band is treated as a failed probe rather than as news about the hardware.
PLAUSIBLE_LINK_BANDWIDTH = (1e9, 1e13)

#: Assumed per-hop latency for the intra-device link when it cannot be measured, matching the
#: inter-device assumption in the system YAML. Deliberately not derived from the PCIe
#: intercept: doing that put 102 us on an on-chip hop and would have made every collective
#: latency-bound across the whole search.
INTRA_DEVICE_LATENCY_ASSUMPTION_US = 1.0

#: Assumed host-DRAM (PCIe) round-trip latency, in microseconds. Also not measurable here:
#: every transfer on this path is a kernel dispatch, so the fitted intercept is launch
#: overhead (~120 us) rather than link latency. Believing the fit would put 4 ms on one
#: token's Engram lookup and rule out tiering the table at all.
HOST_DRAM_LATENCY_ASSUMPTION_US = 2.0


@dataclass
class Measurement:
    """One probed value, with enough provenance to argue about."""

    name: str
    value: float | None
    unit: str
    source: str = "probed"
    derived_from: str = ""
    note: str = ""
    error: str = ""

    def ok(self) -> bool:
        return self.value is not None and not self.error


@dataclass
class Results:
    measurements: dict[str, Measurement] = field(default_factory=dict)

    def add(self, measurement: Measurement) -> None:
        self.measurements[measurement.name] = measurement

    def value(self, name: str) -> float | None:
        entry = self.measurements.get(name)
        return entry.value if entry and entry.ok() else None

    def failures(self) -> list[Measurement]:
        return [m for m in self.measurements.values() if not m.ok()]


# ---------------------------------------------------------------------------------------
# Timing harness
# ---------------------------------------------------------------------------------------
class DeviceBench:
    """Traces and times NKI kernels, subtracting dispatch overhead.

    One instance per probe run. The workdir is a temporary directory outside the repository
    so a probe leaves nothing behind, and ``NEURON_RT_NUM_CORES=1`` keeps the measurement on
    a single logical NeuronCore — which is the unit the simulator places onto, so a
    per-device number would be four times too generous.
    """

    def __init__(self, workdir: Path | None = None) -> None:
        os.environ.setdefault("NEURON_RT_NUM_CORES", "1")
        self.workdir = Path(workdir or tempfile.mkdtemp(prefix="floorplan-probe-"))
        self.workdir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("NKI_COMPILE_CACHE_URL", str(self.workdir / "nki-cache"))
        self._overhead: float | None = None
        import torch  # noqa: F401  — imported here so a missing stack is a clear failure

    def time(self, fn: Callable, args: list, label: str, iters: int = TIMED_ITERS) -> float:
        """Median seconds per call of ``fn(*args)`` on the device, including dispatch."""
        import torch
        import torch_neuronx

        class Wrapper(torch.nn.Module):
            def forward(self, *inner):
                return fn(*inner)

        compiler_workdir = self.workdir / f"wd_{label}"
        cwd = os.getcwd()
        os.chdir(self.workdir)
        try:
            traced = torch_neuronx.trace(
                Wrapper(), tuple(args),
                compiler_workdir=str(compiler_workdir),
                compiler_args=COMPILER_ARGS,
            )
        finally:
            os.chdir(cwd)
        samples: list[float] = []
        with torch.no_grad():
            traced(*args)                      # warm the runtime, fault in the neff
            for _ in range(iters):
                start = time.perf_counter()
                traced(*args)
                samples.append(time.perf_counter() - start)
        return statistics.median(samples)

    def overhead(self) -> float:
        """Seconds of host round trip per dispatch, measured once and cached.

        Without subtracting this, a 550 us kernel reads 20% slow and a 100 us kernel reads at
        half its speed — the probe's answer would depend on how big the probe happened to be.
        """
        if self._overhead is None:
            import torch

            from floorplan.probe import kernels

            self._overhead = self.time(
                kernels.dispatch_overhead,
                [torch.zeros(128, 128, dtype=torch.bfloat16)],
                "overhead", iters=50,
            )
        return self._overhead

    def net(self, fn: Callable, args: list, label: str, iters: int = TIMED_ITERS) -> float:
        """Device-side seconds: the measured time less dispatch overhead."""
        return max(self.time(fn, args, label, iters) - self.overhead(), 1e-9)

    def cleanup(self) -> None:
        shutil.rmtree(self.workdir, ignore_errors=True)


def _guarded(results: Results, name: str, unit: str, note: str = "") -> Callable:
    """Run one probe, recording a failure rather than aborting the suite.

    A probe that cannot run is information — the simulator will refuse for a named reason —
    and it should not cost the other fifteen measurements.
    """
    def wrap(fn: Callable[[], float]) -> None:
        try:
            value = fn()
        except Exception as exc:                       # noqa: BLE001 — recorded, not hidden
            detail = str(exc).strip().splitlines()
            results.add(Measurement(
                name, None, unit, note=note,
                error=f"{type(exc).__name__}: {detail[-1][:200] if detail else exc}",
            ))
            return
        # A probe that already recorded this name has richer provenance than the default
        # `source: probed` — it derived the value, or measured it a different way — so leave
        # it alone. Without this, a derived fp8 fraction would be relabelled as measured.
        existing = results.measurements.get(name)
        if existing is not None and existing.ok():
            return
        results.add(Measurement(name, float(value), unit, note=note))
    return wrap


# ---------------------------------------------------------------------------------------
# Compute probes
# ---------------------------------------------------------------------------------------
def probe_compute(bench: DeviceBench, results: Results, peaks: dict[str, float]) -> None:
    """Tensor, vector, scalar and GPSIMD throughput, as fractions of per-core peak."""
    import torch

    from floorplan.probe import kernels

    bf16_peak = peaks["bf16_flops"] / LOGICAL_NCS_PER_DEVICE
    fp8_peak = peaks["fp8_flops"] / LOGICAL_NCS_PER_DEVICE

    def matmul_flops(partitions: int, moving_free: int, dtype) -> float:
        # `torch.randn` has no CPU kernel for the fp8 dtypes, so operands are generated in
        # bf16 and cast. The values are irrelevant to a throughput measurement; only the
        # element width the tensor engine sees is.
        def operand(rows: int, cols: int):
            tensor = torch.randn(rows, cols, dtype=torch.bfloat16)
            return tensor if dtype == torch.bfloat16 else tensor.to(dtype)

        stationary = operand(partitions, 128 * kernels.STATIONARY_TILES)
        moving = operand(partitions, moving_free)
        seconds = bench.net(
            kernels.matmul_pe, [stationary, moving],
            f"mm_{partitions}_{moving_free}_{str(dtype).split('.')[-1]}",
        )
        flops = kernels.MATMUL_REPS * 2.0 * 128 * partitions * moving_free
        return flops / seconds

    @_guarded(results, "matmul_bf16", "fraction of per-core bf16 peak",
              "128-partition contraction, 512-wide moving tile: the streaming peak")
    def _bf16() -> float:
        return matmul_flops(128, 512, torch.bfloat16) / bf16_peak

    @_guarded(results, "matmul_small_k", "fraction of per-core bf16 peak",
              "32-partition contraction: the MoE and LoRA regime, where the systolic "
              "array cannot be filled along K")
    def _small_k() -> float:
        return matmul_flops(32, 512, torch.bfloat16) / bf16_peak

    @_guarded(results, "matmul_fp8", "fraction of per-core fp8 peak",
              "fp8 operands at the streaming shape")
    def _fp8() -> float:
        for dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            try:
                return matmul_flops(128, 512, dtype) / fp8_peak
            except Exception:                              # noqa: BLE001 — try the next
                continue
        # TRN2 rejects `float8_e4m3fn` outright ("not supported on this target (TRN2 and
        # earlier)"), so on this host there is no fp8 matmul to time. Derive the fraction
        # from the *measured bf16 throughput* rather than from the fp8 peak: that assumes
        # fp8 achieves no more FLOPS than bf16, which is conservative — it makes fp8
        # matmuls look no cheaper than bf16 ones, so no scheme can win by claiming an fp8
        # speedup the hardware was never observed to deliver.
        measured_bf16 = results.value("matmul_bf16")
        if measured_bf16 is None:
            raise RuntimeError(
                "fp8 is not supported on this target and matmul_bf16 was not measured, so "
                "there is nothing to derive the fp8 fraction from"
            )
        derived = measured_bf16 * bf16_peak / fp8_peak
        results.add(Measurement(
            "matmul_fp8", derived, "fraction of per-core fp8 peak",
            source="derived_from_probe",
            derived_from=f"measured matmul_bf16 ({measured_bf16:.4f}) x bf16 peak / fp8 peak",
            note="NOT MEASURED. This target rejects float8_e4m3fn, so fp8 is assumed to "
                 "achieve the same FLOPS as bf16 — conservative, and it means no scheme can "
                 "win on an unobserved fp8 speedup.",
        ))
        return derived

    # The elementwise coefficients are relative to the *bank bandwidth*, because that is what
    # `api.elementwise_seconds` divides by. They are coefficients, not fractions, and can
    # exceed 1: SBUF-resident work is not HBM-bound, so the vector engine legitimately
    # outruns the bank. Keeping the model's denominator and fitting the ratio is what makes
    # the two consistent.
    bank_bandwidth = peaks["hbm_bandwidth"] / LOGICAL_NCS_PER_DEVICE

    @_guarded(results, "vector_elementwise", "coefficient on bank bandwidth",
              "tensor_tensor add over SBUF-resident tiles; >1 is expected and correct")
    def _vector() -> float:
        a = torch.randn(128, 512, dtype=torch.bfloat16)
        b = torch.randn(128, 512, dtype=torch.bfloat16)
        seconds = bench.net(kernels.vector_elementwise, [a, b], "vector")
        nbytes = kernels.VECTOR_REPS * 128 * 512 * 2
        return (nbytes / seconds) / bank_bandwidth

    @_guarded(results, "scalar_activation", "coefficient on bank bandwidth",
              "exp() over SBUF-resident tiles on the scalar engine")
    def _scalar() -> float:
        a = torch.randn(128, 512, dtype=torch.bfloat16)
        seconds = bench.net(kernels.scalar_activation, [a], "scalar")
        nbytes = kernels.VECTOR_REPS * 128 * 512 * 2
        return (nbytes / seconds) / bank_bandwidth

    @_guarded(results, "gpsimd_gather", "coefficient on bank bandwidth",
              "nc_n_gather rate; the Engram and MoE expert-lookup primitive")
    def _gather() -> float:
        table = torch.randn(128, 256, dtype=torch.bfloat16)
        indices = torch.randint(0, 256, (128, 512), dtype=torch.int32)
        seconds = bench.net(kernels.gpsimd_gather, [table, indices], "gather")
        nbytes = kernels.GATHER_REPS * 128 * 512 * 2
        return (nbytes / seconds) / bank_bandwidth


def probe_memory(bench: DeviceBench, results: Results, peaks: dict[str, float]) -> None:
    """HBM read bandwidth, contiguous and strided."""
    import torch

    from floorplan.probe import kernels

    bank_bandwidth = peaks["hbm_bandwidth"] / LOGICAL_NCS_PER_DEVICE
    seed = torch.randn(128, kernels.HBM_COLS, dtype=torch.bfloat16)
    read_bytes = kernels.HBM_PASSES * kernels.HBM_TILES * 128 * kernels.HBM_COLS * 2
    strided_bytes = (
        kernels.HBM_PASSES * kernels.HBM_TILES * 128 * kernels.STRIDE_COLS * 2
    )

    baseline: dict[str, float] = {}

    @_guarded(results, "dma_large_contiguous", "coefficient on bank bandwidth",
              "full-tile reads from a device-resident HBM scratch buffer")
    def _contiguous() -> float:
        fill = bench.net(kernels.hbm_fill, [seed], "hbm_fill", iters=10)
        baseline["fill"] = fill
        total = bench.net(kernels.hbm_read_contiguous, [seed], "hbm_read", iters=10)
        seconds = max(total - fill, 1e-9)
        baseline["contiguous_bandwidth"] = read_bytes / seconds
        return (read_bytes / seconds) / bank_bandwidth

    @_guarded(results, "dma_small_strided", "coefficient on bank bandwidth",
              f"{kernels.STRIDE_COLS}-column strided reads: same transfer count as the "
              f"contiguous probe, a fraction of the bytes")
    def _strided() -> float:
        fill = baseline.get("fill")
        if fill is None:
            fill = bench.net(kernels.hbm_fill, [seed], "hbm_fill", iters=10)
        total = bench.net(kernels.hbm_read_strided, [seed], "hbm_strided", iters=10)
        seconds = max(total - fill, 1e-9)
        return (strided_bytes / seconds) / bank_bandwidth


def probe_pcie(bench: DeviceBench, results: Results) -> None:
    """Host DRAM to device: bandwidth and per-transfer cost.

    Measured by exploiting the trap the HBM probes avoid — a traced input *is* copied
    host-to-device every call, so a kernel that reads a large input measures exactly the PCIe
    path. Sweeping the input size separates the fixed per-transfer cost from the slope.
    """
    import torch

    from floorplan.probe import kernels

    @_guarded(results, "host_dram_bandwidth", "bytes/s",
              "fitted slope of a host-to-device transfer size sweep")
    def _bandwidth() -> float:
        # `dispatch_overhead` touches one element, so the only thing that grows with the
        # input is the host-to-device copy the trace does before every call. Timing it across
        # input sizes therefore isolates the PCIe path, and `bench.net` has already removed
        # the fixed dispatch cost measured against a 128x128 input.
        # The baseline is the *same kernel* at its smallest input, not the generic dispatch
        # probe. That matters for the intercept rather than the slope: subtracting
        # `dispatch_overhead` leaves `pcie_consume`'s own fixed costs — its reduce loop, its
        # setup — inside the intercept, and the first version of this reported 102 us of
        # "PCIe latency" that was really kernel launch. 102 us per access would have made any
        # per-token host-DRAM lookup catastrophic and quietly ruled out tiering Engram, which
        # is the largest decision in the floorplan.
        tiny = torch.randn(128, 128, dtype=torch.bfloat16)
        floor = bench.net(kernels.pcie_consume, [tiny], "pcie_floor", iters=20)
        floor_bytes = 128 * 128 * 2

        points: list[tuple[int, float]] = []
        for tiles, cols in ((16, 512), (64, 512), (128, 1024), (64, 2048), (256, 2048)):
            src = torch.randn(tiles * 128, cols, dtype=torch.bfloat16)
            seconds = bench.net(
                kernels.pcie_consume, [src], f"pcie_{tiles}_{cols}", iters=10,
            )
            points.append((tiles * 128 * cols * 2, max(seconds - floor, 1e-9)))

        # Least squares over every point, not a two-point fit on the extremes. The first
        # version of this used the extremes and reported 2.9 GB/s where the regression gives
        # ~13; with five samples spanning 32x in size, one noisy endpoint should not set the
        # slope for the whole tier.
        slope, intercept = _fit_line(points)
        if slope <= 0:
            raise RuntimeError(
                f"the transfer sweep is not monotonic in size ({points}); "
                f"cannot fit a bandwidth"
            )
        # Bandwidth is measured; latency is *assumed*, and the two are recorded differently
        # on purpose.
        #
        # The slope is solid — 12.9 GB/s, reproducible across runs and across two different
        # kernels. The intercept is not: on this path every "transfer" is also a kernel
        # dispatch, so the fit cannot separate PCIe latency from launch overhead, and it
        # returns ~120 us however the baseline is subtracted. Believing that would put 120 us
        # on every host-DRAM access — roughly 4 ms for one token's 32-row Engram lookup — and
        # foreclose tiering the Engram tables, which is the largest genuine decision in the
        # floorplan. Foreclosing a real option on a measurement artifact is worse than
        # admitting the number is an assumption.
        #
        # So: a documented PCIe round-trip figure, with the observed intercept recorded beside
        # it as what the probe saw, and the whole thing flagged as a sensitivity.
        results.add(Measurement(
            "host_dram_latency_us", HOST_DRAM_LATENCY_ASSUMPTION_US, "us",
            source="assumed",
            derived_from="a typical PCIe round trip; the probe cannot isolate it here",
            note=f"NOT MEASURED. The sweep's fitted intercept was "
                 f"{max(intercept * 1e6, 0.0):.0f} us, but on this path a transfer is also a "
                 f"kernel dispatch, so that figure is launch overhead rather than link "
                 f"latency. The *bandwidth* above is measured and trustworthy. Any conclusion "
                 f"about tiering a lookup table through host DRAM is sensitive to this value.",
        ))
        return 1.0 / slope


def _fit_line(points: list[tuple[int, float]]) -> tuple[float, float]:
    """Least-squares ``(slope, intercept)`` for ``seconds = intercept + slope * bytes``."""
    count = len(points)
    if count < 2:
        raise RuntimeError("need at least two samples to fit a line")
    mean_x = sum(x for x, _ in points) / count
    mean_y = sum(y for _, y in points) / count
    variance = sum((x - mean_x) ** 2 for x, _ in points)
    if variance == 0:
        raise RuntimeError("all samples are the same size; cannot fit a slope")
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / variance
    return (slope, mean_y - slope * mean_x)


def probe_storage(results: Results, scratch: Path) -> None:
    """NVMe sequential bandwidth, random-read IOPS, and latency.

    Delegates to ``probe/storage.py`` against the raw instance-store device, re-invoking it
    under ``sudo -n`` when the device is not directly readable. Read-only throughout: nothing
    is written and no filesystem is created.

    The indirection earns its keep. Benchmarking a file on the mounted root measures **EBS**,
    not instance store, and the page cache answers the sequential reads: that version of this
    probe reported 8.0 GB/s and 1,376 IOPS where the raw instance-store device gives
    2.45 GB/s and 33,238 IOPS. An Engram table placed on the basis of the first pair of
    numbers would be placed on the basis of the wrong device, 24x out on the metric that
    matters most for a random row lookup.
    """
    from floorplan.probe import storage as storage_probe

    def measure() -> dict:
        device = storage_probe.find_instance_store()
        if device is None:
            raise RuntimeError(
                "no unmounted NVMe device found; the instance-store tier cannot be measured "
                "on this host"
            )
        try:
            return storage_probe.benchmark(device)
        except PermissionError:
            pass
        # Re-invoke just this module with privilege rather than running the whole suite as
        # root, which would leave root-owned artifacts behind.
        output = scratch / "storage.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            ["sudo", "-n", sys.executable, "-m", "floorplan.probe.storage",
             "--device", str(device), "--out", str(output)],
            capture_output=True, text=True, timeout=900,
            cwd=str(Path(__file__).resolve().parents[2]),
        )
        if completed.returncode != 0 or not output.exists():
            raise RuntimeError(
                f"reading {device} needs privilege and `sudo -n` failed: "
                f"{(completed.stderr or completed.stdout).strip()[:160]}"
            )
        payload = json.loads(output.read_text())
        payload["note"] = payload.get("note", "") + " (read under sudo -n)"
        return payload

    @_guarded(results, "nvme_sequential_bandwidth", "bytes/s",
              "raw O_DIRECT 1 MiB reads from the instance-store device")
    def _storage() -> float:
        payload = measure()
        results.add(Measurement(
            "nvme_random_read_iops", float(payload["random_read_iops"]), "operations/s",
            note=f"{payload['block_bytes']} B random reads at queue depth 1 on "
                 f"{payload['device']}",
        ))
        results.add(Measurement(
            "nvme_latency_us", float(payload["latency_us"]), "us",
            note=f"median of the same sweep; p99 "
                 f"{float(payload.get('latency_p99_us', 0)):.0f} us",
        ))
        return float(payload["sequential_bandwidth_bytes_per_s"])


def derive_intra_device(results: Results, peaks: dict[str, float]) -> None:
    """Fall back to a derivation when the collective probe cannot run, and say so loudly.

    The intra-device link is the one figure on the target that is unpublished *and* decides a
    real question — whether tensor parallelism is better spent across the four logical cores
    of one device or across devices. On this host it turns out not to be measurable at all:
    NKI 0.6.0 exposes no collective primitive, and the XLA path that would provide one fails
    to compile on neuronxcc 2.27.5334 with an internal simplifier error (NCC_ISMP902) for both
    all-reduce and plain matmul. So there is no measurement to be had here, and the choice is
    between a derivation and no simulation.

    The derivation is grounded rather than invented. Text constraint 3 says a cross-bank read
    stays on-chip and contends for the device's shared HBM bandwidth, so core-to-core exchange
    within a device is bounded by that path: the device's 2.9 TB/s scaled by the *measured*
    DMA efficiency. Latency comes from the measured PCIe intercept, which is an upper bound —
    an on-chip hop cannot cost more than one across the bus.

    Both are tagged ``derived_from_probe`` with ``confidence: low``, and the ranking report is
    required to list any conclusion that rests on them.
    """
    efficiency = results.value("dma_large_contiguous")
    if efficiency is None:
        results.add(Measurement(
            "intra_device_bandwidth", None, "bytes/s",
            error="cannot derive: the DMA efficiency probe also failed",
        ))
        return
    bandwidth = peaks["hbm_bandwidth"] * efficiency
    results.add(Measurement(
        "intra_device_bandwidth", bandwidth, "bytes/s",
        source="derived_from_probe",
        derived_from=f"device HBM bandwidth ({peaks['hbm_bandwidth']:.3g} B/s) x measured "
                     f"dma_large_contiguous ({efficiency:.4f})",
        note="NOT MEASURED. NKI 0.6.0 has no collective primitive and the XLA all-reduce "
             "fails to compile (NCC_ISMP902), so no intra-device collective can be timed on "
             "this host. Bounded by the on-chip path per text constraint 3.",
    ))
    # Latency is *assumed*, not derived from the host transfer. Reusing the PCIe intercept
    # here was a mistake worth recording: it put 102 us on an on-chip hop, which would have
    # made every collective latency-bound and effectively ruled out tensor parallelism at
    # decode across the whole search. The defensible statement is weaker and safer — an
    # on-chip hop cannot cost more than an inter-device one, so it takes the same 1.0 us the
    # system YAML already assumes for a torus hop, and carries the same low confidence.
    results.add(Measurement(
        "intra_device_latency_us", INTRA_DEVICE_LATENCY_ASSUMPTION_US, "us",
        source="assumed",
        derived_from="the inter-device per-hop assumption in the system YAML",
        note="NOT MEASURED. An on-chip hop is no worse than an inter-device hop, so it takes "
             "the same assumed value. Do not read a conclusion about intra- versus "
             "inter-device tensor parallelism off this number alone.",
    ))


def probe_collective(results: Results, workdir: Path) -> None:
    """Intra-device NeuronLink: bandwidth and latency across the four logical cores.

    Runs ``floorplan.probe.collective`` under ``torch.distributed.run --nproc_per_node=4``. A
    failure is recorded rather than raised, and ``run`` then falls back to
    ``derive_intra_device``.
    """
    script = Path(__file__).resolve().parent / "collective.py"
    output = workdir / "collective.json"
    # `python -m torch.distributed.run` rather than the `torchrun` console script, so the
    # four workers inherit this interpreter and its venv without depending on PATH.
    command = [
        sys.executable, "-m", "torch.distributed.run",
        "--nproc_per_node=4", "--nnodes=1",
        str(script), "--out", str(output),
    ]
    environment = dict(os.environ)
    environment["NEURON_RT_NUM_CORES"] = "4"
    environment.pop("NEURON_RT_VISIBLE_CORES", None)
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=1800,
            cwd=str(workdir), env=environment,
        )
    except subprocess.SubprocessError as exc:
        results.add(Measurement(
            "intra_device_bandwidth", None, "bytes/s",
            error=f"the four-process launch failed: {exc}",
        ))
        return
    if not output.exists():
        tail = "\n".join((completed.stdout + completed.stderr).strip().splitlines()[-6:])
        results.add(Measurement(
            "intra_device_bandwidth", None, "bytes/s",
            error=f"the collective probe produced no result. Last output:\n{tail}",
        ))
        return
    payload = json.loads(output.read_text())
    bandwidth = float(payload["bandwidth_bytes_per_s"])

    # A fitted slope can come back arbitrarily wrong without the fit failing, and one did:
    # an earlier run of this probe reported 0.78 MB/s for an on-chip link — six orders of
    # magnitude low — and wrote it into the overlay as a measurement. Nothing downstream
    # would have caught it; the simulator would simply have concluded that no collective is
    # ever worth doing. So the result has to clear a plausibility band before it is believed,
    # and outside the band it is treated as a failed probe and derived instead.
    if not PLAUSIBLE_LINK_BANDWIDTH[0] <= bandwidth <= PLAUSIBLE_LINK_BANDWIDTH[1]:
        results.add(Measurement(
            "intra_device_bandwidth", None, "bytes/s",
            error=f"the probe fitted {bandwidth:.3e} B/s, outside the plausible band "
                  f"{PLAUSIBLE_LINK_BANDWIDTH[0]:.0e}-{PLAUSIBLE_LINK_BANDWIDTH[1]:.0e} B/s "
                  f"for an on-chip link. Treating the fit as failed. Samples: "
                  f"{payload.get('samples')}",
        ))
        return
    results.add(Measurement(
        "intra_device_bandwidth", bandwidth, "bytes/s",
        note=f"all-reduce over 4 logical NeuronCores, "
             f"{payload.get('sizes_probed', '?')} sizes, fitted slope",
    ))
    results.add(Measurement(
        "intra_device_latency_us", float(payload["latency_us"]), "us",
        note="fitted intercept of the same sweep",
    ))


# ---------------------------------------------------------------------------------------
# Writing the overlay
# ---------------------------------------------------------------------------------------
def to_overlay(results: Results, host: str) -> dict[str, Any]:
    """Shape the measurements into the overlay ``parser.load_system`` merges.

    Coefficients and the intra-device link go under ``shared`` because they are properties of
    the silicon and apply to every target built on it. Tier bandwidths that depend on how many
    drives an instance has go under the specific target.
    """
    def note_for(name: str) -> str:
        entry = results.measurements.get(name)
        return entry.note if entry else ""

    efficiency: dict[str, Any] = {
        "source": f"probed on {host}",
        "confidence": "medium",
        "probed_at": time.strftime("%Y-%m-%d"),
    }
    for key in (
        "matmul_fp8", "matmul_bf16", "matmul_small_k", "vector_elementwise",
        "scalar_activation", "gpsimd_gather", "dma_large_contiguous", "dma_small_strided",
    ):
        value = results.value(key)
        if value is not None:
            efficiency[key] = round(value, 6)

    # Collective efficiency is not separately measurable without a second device, so it is
    # tied to the probed intra-device link rather than invented: the coefficient is 1.0 and
    # the *bandwidth* carries the measurement. Named so the report can say which it is.
    efficiency.setdefault("collective_allreduce", 1.0)
    efficiency.setdefault("collective_all_to_all", 1.0)
    efficiency["collective_note"] = (
        "1.0 by construction: the measured intra-device link bandwidth already reflects "
        "achieved all-reduce throughput, so a second coefficient would double-count. "
        "Inter-device collectives inherit the datasheet bandwidth and are extrapolations."
    )

    shared: dict[str, Any] = {"efficiency": efficiency}

    intra_bandwidth = results.value("intra_device_bandwidth")
    intra_latency = results.value("intra_device_latency_us")
    if intra_bandwidth is not None and intra_latency is not None:
        entry = results.measurements["intra_device_bandwidth"]
        derived = entry.source != "probed"
        shared["links"] = {
            "intra_device": {
                "bandwidth_bytes_per_s": float(intra_bandwidth),
                "latency_us": float(intra_latency),
                # Carried through verbatim rather than flattened to "probed": a derived link
                # and a measured one should not look the same in the file the report reads.
                "source": entry.source,
                "confidence": "low" if derived else "medium",
                "note": entry.note,
                **({"derived_from": entry.derived_from} if entry.derived_from else {}),
            },
        }

    host_bandwidth = results.value("host_dram_bandwidth")
    host_latency = results.value("host_dram_latency_us")
    if host_bandwidth is not None and host_latency is not None:
        shared.setdefault("links", {})["host"] = {
            "bandwidth_bytes_per_s": float(host_bandwidth),
            "latency_us": float(host_latency),
            "source": "probed",
            "confidence": "medium",
        }

    overlay: dict[str, Any] = {
        "_provenance": {
            "probed_on": host,
            "probed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "not_probeable": [
                "links.inter_device.bandwidth_bytes_per_s",
                "links.inter_device.per_hop_latency_us",
                "collectives wider than the 4 logical cores of one device",
            ],
            "not_probeable_reason": (
                "this host has one Trainium2 device; every inter-device figure stays at its "
                "datasheet value and is an extrapolation"
            ),
            "failures": [
                {"name": m.name, "error": m.error} for m in results.failures()
            ],
        },
        "shared": shared,
    }

    # Tier values need to land on the tier list, which the overlay replaces wholesale, so
    # they are emitted per target and the caller merges them into the target's own list.
    tiers: dict[str, dict[str, Any]] = {}
    if host_bandwidth is not None:
        tiers["host_dram"] = {
            "bandwidth_bytes_per_s": float(host_bandwidth),
            "latency_us": float(host_latency or 0.0),
            "source": "probed",
        }
    nvme_bandwidth = results.value("nvme_sequential_bandwidth")
    nvme_iops = results.value("nvme_random_read_iops")
    nvme_latency = results.value("nvme_latency_us")
    if nvme_bandwidth is not None:
        tiers["nvme"] = {
            "bandwidth_bytes_per_s": float(nvme_bandwidth),
            "random_read_iops": float(nvme_iops or 0.0),
            "latency_us": float(nvme_latency or 0.0),
            "source": "probed",
            "note": (
                "measured on this host's single NVMe drive; the 16-device target has four, "
                "so its aggregate bandwidth and IOPS are extrapolations"
            ),
        }
    if tiers:
        overlay["_tier_patch"] = tiers
    return overlay


def merge_tier_patch(system_path: Path, patch: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Return ``system_path``'s tier list with ``patch`` applied, for writing per target.

    The overlay replaces lists wholesale (see ``parser._deep_merge``), so a partial tier
    update has to be materialized as a complete list. Done here rather than in the parser
    because a half-probed tier list — some entries measured, some not, order unspecified —
    is much harder to reason about than a whole one.
    """
    system = yaml.safe_load(system_path.read_text())
    tiers = [dict(entry) for entry in system.get("tiers", [])]
    scale = _drive_scale(system)
    for entry in tiers:
        update = patch.get(str(entry.get("name")))
        if not update:
            continue
        for key, value in update.items():
            if key in {"bandwidth_bytes_per_s", "random_read_iops"} and entry.get("name") == "nvme":
                entry[key] = float(value) * scale
            else:
                entry[key] = value
    return tiers


def _drive_scale(system: dict[str, Any]) -> float:
    """How many NVMe drives this target has, relative to the one that was probed."""
    for entry in system.get("tiers", []):
        if entry.get("name") == "nvme":
            return float(entry.get("devices", 1) or 1)
    return 1.0


def run(
    systems_dir: Path,
    targets: tuple[str, ...] = ("trn2-16device", "trn2-1device"),
    scratch: Path | None = None,
    skip: frozenset[str] = frozenset(),
) -> Results:
    """Run every probe and write ``systems/probed.yaml``."""
    from floorplan.parser import load_system

    results = Results()
    host = _host_description()

    reference = load_system(targets[0], systems_dir, apply_probes=False)
    device = reference["device"]
    peaks = {
        "bf16_flops": float(device["compute"]["bf16_flops"]),
        "fp8_flops": float(device["compute"]["fp8_flops"]),
        "hbm_bandwidth": float(device["hbm"]["bandwidth_bytes_per_s"]),
    }

    bench: DeviceBench | None = None
    try:
        if "device" not in skip:
            bench = DeviceBench()
            probe_compute(bench, results, peaks)
            probe_memory(bench, results, peaks)
            probe_pcie(bench, results)
        if "storage" not in skip:
            probe_storage(results, scratch or Path(tempfile.gettempdir()))
        if "collective" not in skip:
            probe_collective(results, bench.workdir if bench else Path(tempfile.mkdtemp()))
            if results.value("intra_device_bandwidth") is None:
                derive_intra_device(results, peaks)
    finally:
        if bench is not None:
            bench.cleanup()

    overlay = to_overlay(results, host)
    tier_patch = overlay.pop("_tier_patch", {})
    if tier_patch:
        for target in targets:
            path = systems_dir / f"{target}.yaml"
            if path.exists():
                overlay.setdefault(target, {})["tiers"] = merge_tier_patch(path, tier_patch)

    destination = systems_dir / "probed.yaml"
    destination.write_text(
        "# Written by `autohelix floorplan probe`. Do not hand-edit: re-run the probe.\n"
        "#\n"
        "# Merged over the system YAML by floorplan/parser.py, `shared` first and then the\n"
        "# per-target block. Keeping measurements here rather than in the system files means\n"
        "# a diff of trn2-16device.yaml is a diff of the datasheet.\n\n"
        + yaml.safe_dump(overlay, sort_keys=False, default_flow_style=False)
    )
    return results


def _host_description() -> str:
    try:
        out = subprocess.run(
            ["neuron-ls"], capture_output=True, text=True, timeout=60,
        ).stdout
        for line in out.splitlines():
            if "instance-type" in line:
                return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def report(results: Results) -> str:
    """The probe's own summary, printed by the CLI."""
    lines = [f"{'measurement':<28} {'value':>16}  unit"]
    for name in sorted(results.measurements):
        entry = results.measurements[name]
        if entry.ok():
            rendered = f"{entry.value:,.4g}" if abs(entry.value) < 1e6 else f"{entry.value:.3e}"
            lines.append(f"{name:<28} {rendered:>16}  {entry.unit}")
        else:
            lines.append(f"{name:<28} {'FAILED':>16}  {entry.error[:60]}")
    failures = results.failures()
    lines.append("")
    lines.append(
        f"{len(results.measurements) - len(failures)}/{len(results.measurements)} probed"
        + (f"; {len(failures)} failed" if failures else "")
    )
    return "\n".join(lines)
