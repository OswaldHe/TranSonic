# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reporting the latency the gate already measured, as AutoHelix's metric command.

AutoHelix runs constraints, then — if they pass — runs the metric commands. The obvious metric
command here would be the validator itself, and it would be wrong twice over: it would double every
iteration's device time (four minutes for the submodule, longer for four ranks under `torchrun`), and
the two runs could disagree, leaving an iteration accepted on one measurement and ranked on another.

So the gate is the only thing that runs the candidate, and this reads the number back out of the
verdict it wrote. Cheap, and there is exactly one measurement per iteration by construction.

The verdict is a file inside the worktree, which raises the obvious question: could a candidate write
its own? It could — and it would have to get `passed: true` out of the gate first, which means
actually running, matching the golden and leaving a fresh profile behind. The file is the gate's
output, not the candidate's input, and it is overwritten every iteration before it is read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: The metric name every bound in this pipeline is stated in.
METRIC = "latency_ms"


def read_latency(path: Path) -> tuple[float | None, str]:
    """The latency out of a gate verdict, and a note on where it came from."""
    if not path.is_file():
        return None, f"{path} does not exist — did the gate run?"
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return None, f"{path} is not valid JSON: {exc}"

    # The whole-module gate publishes it at the top level, having also checked it against the
    # per-rank numbers. The submodule gate leaves it in the run's output.
    if isinstance(payload.get(METRIC), (int, float)):
        return float(payload[METRIC]), "from the gate's verdict"

    output = str((payload.get("run") or {}).get("output") or "")
    if not output:
        output = str(payload.get("report") or "")
    import re

    found = re.findall(rf"##autohelix\[{METRIC}=([^\]]+)\]", output)
    for raw in reversed(found):
        try:
            return float(raw.strip()), "from the validator's marker line"
        except ValueError:
            continue
    return None, f"no {METRIC} in {path}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", required=True, help="the gate's verdict file")
    args = parser.parse_args(argv)

    latency, note = read_latency(Path(args.json))
    if latency is None:
        print(f"could not read {METRIC}: {note}", file=sys.stderr)
        return 1
    print(f"{note}")
    print(f"##autohelix[{METRIC}={latency}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
