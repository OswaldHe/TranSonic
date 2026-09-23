"""Run every published module's ``inference.py`` and report what it scored.

Each one is launched the way somebody who downloaded the artifact would launch it, and
with this harness taken off ``sys.path`` — so it passes only if the artifact's own copies
under ``runtime/`` and ``vendor/`` are enough. Inputs and reference outputs come from the
dumped feature maps; weights come from the dumps where the run kept them and from the
fetched checkpoint shards where it did not.

    python scripts/check_all_modules.py <run-dir> [--group NAME] [--timeout SECONDS]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

METRIC = re.compile(r"##autohelix\[(\w+)=([^\]]+)\]")
#: Harness root to hide from the child, so an import cannot resolve to this checkout.
HARNESS = str(Path(__file__).resolve().parent.parent)

PROGRAM = "\n".join([
    "import pathlib, runpy, sys",
    f"harness = pathlib.Path({HARNESS!r})",
    "sys.path = [p for p in sys.path if pathlib.Path(p or '.').resolve() != harness]",
    "sys.argv = ['inference.py', '--repeat', '1', '--warmup', '0']",
    "runpy.run_path('inference.py', run_name='__main__')",
])


def run_group(directory: Path, timeout: int) -> dict:
    started = time.time()
    try:
        finished = subprocess.run([sys.executable, "-c", PROGRAM], cwd=directory,
                                  capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"group": directory.name, "status": "timeout",
                "seconds": round(time.time() - started, 1)}
    out = finished.stdout + finished.stderr
    metrics = {k: v for k, v in METRIC.findall(out)}
    weights = next((line.split(":", 1)[1].strip() for line in out.splitlines()
                    if line.startswith("weights  :")), "")
    tail = [line for line in out.strip().splitlines() if line.strip()]
    return {
        "group": directory.name,
        "status": "pass" if finished.returncode == 0 else "fail",
        "returncode": finished.returncode,
        "metrics": metrics,
        "weights": weights,
        "seconds": round(time.time() - started, 1),
        "last": tail[-1] if tail else "",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--group", action="append", default=[])
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    directories = sorted(p for p in (args.run_dir / "modules").glob("*/")
                         if (p / "inference.py").is_file()
                         and (not args.group or p.name in args.group))
    results = []
    for directory in directories:
        result = run_group(directory, args.timeout)
        results.append(result)
        mark = {"pass": "ok  ", "fail": "FAIL", "timeout": "TIME"}[result["status"]]
        print(f"{mark} {result['group']:26} {result['seconds']:7.1f}s  "
              f"cos={result['metrics'].get('cosine', '-'):>10} "
              f"lat={result['metrics'].get('latency_ms', '-'):>10}  {result['weights']}",
              flush=True)
        if result["status"] != "pass":
            print(f"       {result['last'][:200]}", flush=True)

    passed = sum(1 for r in results if r["status"] == "pass")
    print(f"\n{passed}/{len(results)} module group(s) pass")
    for result in results:
        if result["status"] != "pass":
            print(f"  {result['status']}: {result['group']} — {result['last'][:160]}")
    if args.out:
        args.out.write_text(json.dumps(results, indent=1) + "\n")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
