# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reporting the constraint count the gate already measured, as AutoHelix's metric command.

Same reasoning as `optimization/readback.py`: the loop runs constraints, then runs the metric
commands, and the obvious metric command here would be the gate itself. That would be wrong
twice over — it would double every iteration's cost, which for a config with agent constraints
means paying for every judge twice, and the two runs could disagree, leaving an iteration
accepted on one count and ranked on another.

So the gate is the only thing that evaluates the constraints, and this reads its verdict back.
One evaluation per iteration, by construction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from integrate.config import METRIC


def read_counts(path: Path) -> tuple[int | None, int | None, str]:
    """`(passing, total, note)` out of a gate verdict."""
    if not path.is_file():
        return None, None, f"{path} does not exist — did the gate run?"
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return None, None, f"{path} is not valid JSON: {exc}"
    passing = payload.get(METRIC)
    total = payload.get("constraints_total")
    if not isinstance(passing, int):
        return None, None, f"no {METRIC} in {path}"
    return passing, (total if isinstance(total, int) else None), "from the gate's verdict"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", required=True, help="the gate's verdict file")
    args = parser.parse_args(argv)

    path = Path(args.json)
    passing, total, note = read_counts(path)
    if passing is None:
        print(f"could not read {METRIC}: {note}", file=sys.stderr)
        return 1
    print(note if total is None else f"{note}: {passing}/{total}")
    print(f"##autohelix[{METRIC}={passing}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
