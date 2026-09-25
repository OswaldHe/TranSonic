# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure the intra-device NeuronLink, under ``torch.distributed.run --nproc_per_node=4``.

This is the one link on the 16-device target that is both unpublished and measurable on a
one-device host, and it decides a question the floorplan keeps asking: is tensor parallelism
better spent across the four logical NeuronCores of one device, or across devices? Without a
number here the simulator cannot tell those apart, so it refuses to run rather than guess.

Run as a module by ``probe/suite.py``, never imported. It writes a small JSON file and exits;
rank 0 is the only rank that writes, so there is nothing to reconcile.

An all-reduce is timed at several sizes and the cost model ``seconds = latency + bytes /
bandwidth`` is fitted to them, because the two terms matter in different places: decode moves
small tensors and is latency-bound, prefill moves large ones and is bandwidth-bound, and a
single-size measurement would mis-price whichever half it did not sample.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

#: Payload sizes, in bf16 elements per rank. Spans three orders of magnitude so the fit has
#: something to separate: the smallest is latency-dominated, the largest bandwidth-dominated.
SIZES = (1 << 12, 1 << 15, 1 << 18, 1 << 21, 1 << 23)

#: Timed all-reduces per size, after warmup. The median is used.
ITERS = 20

#: Discarded before timing: the first few calls pay compilation and rendezvous.
WARMUP = 5


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="floorplan.probe.collective")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    os.environ.setdefault("NEURON_RT_NUM_CORES", "4")

    import torch
    import torch.distributed as dist
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_backend  # noqa: F401 — registers the 'xla' backend
    import torch_xla.runtime as xr

    if not dist.is_initialized():
        dist.init_process_group("xla")
    # torch_xla 2.9 moved rank/world off `xla_model`; `runtime` is where they live now, and
    # `torch_xla.sync()` replaced `mark_step()`.
    device = torch_xla.device()
    rank = xr.global_ordinal()
    world = xr.world_size()

    samples: list[tuple[int, float]] = []
    for elements in SIZES:
        tensor = torch.ones(elements, dtype=torch.bfloat16, device=device)
        for _ in range(WARMUP):
            xm.all_reduce(xm.REDUCE_SUM, [tensor])
            torch_xla.sync()
        xm.wait_device_ops()

        timings: list[float] = []
        for _ in range(ITERS):
            start = time.perf_counter()
            xm.all_reduce(xm.REDUCE_SUM, [tensor])
            torch_xla.sync()
            xm.wait_device_ops()
            timings.append(time.perf_counter() - start)
        samples.append((elements * 2, statistics.median(timings)))

    if rank != 0:
        return 0

    # A ring all-reduce puts 2(N-1)/N of each rank's tensor on the wire, which is the same
    # factor `sim/collectives.py` charges. Fitting against wire bytes rather than payload
    # bytes keeps the probe and the cost model measuring the same quantity.
    share = 2.0 * (world - 1) / world
    points = [(int(nbytes * share), seconds) for nbytes, seconds in samples]

    # Least squares over every sample, not the two endpoints. One noisy endpoint on a sweep
    # spanning three orders of magnitude should not set the slope for the whole link, and the
    # endpoint fit produced an implausible bandwidth once already. `suite.py` additionally
    # range-checks whatever comes out of here before believing it.
    count = len(points)
    mean_x = sum(x for x, _ in points) / count
    mean_y = sum(y for _, y in points) / count
    variance = sum((x - mean_x) ** 2 for x, _ in points)
    if variance == 0:
        raise RuntimeError("every sample is the same size; cannot fit a slope")
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / variance
    if slope <= 0:
        raise RuntimeError(
            f"the all-reduce sweep is not monotonic in size: {points}. Cannot fit a bandwidth"
        )
    latency_us = max((mean_y - slope * mean_x) * 1e6, 0.0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "world_size": world,
        "bandwidth_bytes_per_s": 1.0 / slope,
        "latency_us": latency_us,
        "sizes_probed": len(SIZES),
        "samples": [
            {"wire_bytes": nbytes, "seconds": seconds} for nbytes, seconds in points
        ],
        "note": (
            "all-reduce across the 4 logical NeuronCores of one Trainium2 device; "
            "fitted seconds = latency + wire_bytes / bandwidth"
        ),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
