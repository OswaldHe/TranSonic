# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only storage benchmark against a raw block device.

Separated from ``suite.py`` and runnable on its own because reading a raw device usually
needs privilege, and the suite would rather re-invoke this one file under ``sudo -n`` than
run every probe as root and leave root-owned files behind.

**Read-only, always.** It opens the device ``O_RDONLY | O_DIRECT`` and never writes, so
there is nothing to undo and no filesystem is created. Whatever bytes are on the device are
what get read, which is fine — a read benchmark does not care what the bytes say.

Why raw rather than a file on a mounted filesystem: on this dev host the mounted root is an
**EBS** volume presented over NVMe, while the target's four 1.92 TB drives are **instance
store**. Benchmarking a file on the root volume measures the wrong device by a factor of
roughly 24 in IOPS, and page cache makes the sequential figure meaningless on top of that —
the first version of this probe reported 8.0 GB/s sequential and 1,376 IOPS, against the raw
instance store's 2.45 GB/s and 33,238 IOPS. Both numbers wrong, in opposite directions.
``O_DIRECT`` is what removes the cache from the sequential measurement.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
from pathlib import Path

#: Random-read block size. 4 KiB is the smallest transfer the device will do, and an Engram
#: row at ~256 B is far below it — which is exactly why the tier is IOPS-bound.
BLOCK = 4096

#: Sequential reads of this size, and how many.
SEQ_BLOCK = 1 << 20
SEQ_COUNT = 512

#: Random reads at queue depth 1. Depth 1 because a per-token gather on the critical path
#: cannot pipeline: the model needs the row before it can continue.
RANDOM_COUNT = 4000


def find_instance_store() -> Path | None:
    """The largest NVMe block device that is not the root volume, or None.

    Instance store is identified by exclusion rather than by model string: anything holding a
    mounted partition is not scratch. ``lsblk`` is asked for mountpoints so a device with any
    mounted child is skipped, which is what keeps this from selecting the boot disk.
    """
    try:
        out = subprocess.run(
            ["lsblk", "-J", "-b", "-o", "NAME,SIZE,TYPE,MOUNTPOINTS"],
            capture_output=True, text=True, timeout=60, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    try:
        tree = json.loads(out)
    except json.JSONDecodeError:
        return None

    def mounted(node: dict) -> bool:
        points = [p for p in (node.get("mountpoints") or []) if p]
        if points:
            return True
        return any(mounted(child) for child in node.get("children") or [])

    candidates = [
        (int(node.get("size") or 0), node["name"])
        for node in tree.get("blockdevices", [])
        if node.get("type") == "disk" and node["name"].startswith("nvme") and not mounted(node)
    ]
    if not candidates:
        return None
    return Path("/dev") / max(candidates)[1]


def benchmark(device: Path, seed: int = 0) -> dict:
    """Sequential bandwidth, random-read IOPS and latency for one device."""
    handle = os.open(str(device), os.O_RDONLY | os.O_DIRECT)
    try:
        size = os.lseek(handle, 0, os.SEEK_END)
        if size < SEQ_BLOCK * SEQ_COUNT:
            raise RuntimeError(f"{device} is only {size} bytes; too small to benchmark")

        import time

        buffer = bytearray(SEQ_BLOCK)
        start = time.perf_counter()
        for index in range(SEQ_COUNT):
            os.preadv(handle, [buffer], index * SEQ_BLOCK)
        sequential = SEQ_BLOCK * SEQ_COUNT / (time.perf_counter() - start)

        generator = random.Random(seed)
        offsets = [
            generator.randrange(0, size - BLOCK) // BLOCK * BLOCK
            for _ in range(RANDOM_COUNT)
        ]
        small = bytearray(BLOCK)
        latencies: list[float] = []
        start = time.perf_counter()
        for offset in offsets:
            each = time.perf_counter()
            os.preadv(handle, [small], offset)
            latencies.append(time.perf_counter() - each)
        elapsed = time.perf_counter() - start
    finally:
        os.close(handle)

    return {
        "device": str(device),
        "size_bytes": size,
        "sequential_bandwidth_bytes_per_s": sequential,
        "random_read_iops": RANDOM_COUNT / elapsed,
        "latency_us": statistics.median(latencies) * 1e6,
        "latency_p99_us": sorted(latencies)[int(RANDOM_COUNT * 0.99)] * 1e6,
        "block_bytes": BLOCK,
        "queue_depth": 1,
        "note": (
            "raw O_DIRECT reads, no filesystem and no page cache; queue depth 1 because a "
            "gather on the critical path cannot pipeline"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m floorplan.probe.storage")
    parser.add_argument("--device", type=Path, default=None,
                        help="block device to read (default: the largest unmounted NVMe)")
    parser.add_argument("--out", type=Path, default=None, help="write JSON here")
    args = parser.parse_args(argv)

    device = args.device or find_instance_store()
    if device is None:
        print("no unmounted NVMe device found", file=sys.stderr)
        return 2
    try:
        payload = benchmark(device)
    except PermissionError:
        print(
            f"cannot read {device}: permission denied. Re-run this module under sudo, or "
            f"pass --device pointing at something readable",
            file=sys.stderr,
        )
        return 3
    except (OSError, RuntimeError) as exc:
        print(f"benchmark failed on {device}: {exc}", file=sys.stderr)
        return 4

    text = json.dumps(payload, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
