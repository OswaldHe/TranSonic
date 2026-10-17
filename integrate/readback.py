# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reporting the constraint counts the gate already measured, as AutoHelix's metric command.

Same reasoning as `optimization/readback.py`: the loop runs constraints, then runs the metric
commands, and the obvious metric command here would be the gate itself. That would be wrong
twice over — it would double every iteration's cost, which for a config with agent constraints
means paying for every judge twice, and the two runs could disagree, leaving an iteration
accepted on one count and ranked on another.

So the gate is the only thing that evaluates the constraints, and this reads its verdict back.
One evaluation per iteration, by construction.

**Two counts, and only one of them is gated.** `constraints_passing` is the script constraints
that hold; `all_constraints_passing` is every constraint. Acceptance gates the first, because a
script constraint's verdict is a property of the tree, so a drop in it is a regression the
iteration caused. The second moves with agent judges, which can reach opposite verdicts over an
unchanged tree — so it is recorded, reported and ranked on, but a drop in it never rejects an
iteration. `integrate.config.METRIC` has the case that forced this. Both counts come from the
verdict's per-constraint entries rather than its summary keys, so a verdict written before the
split — where `constraints_passing` meant every constraint — reads correctly instead of
reporting its whole count as the gated one.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from integrate.config import METRIC, METRIC_ALL


@dataclass
class Counts:
    """What the gate found, split by what acceptance may act on."""

    script_passing: int
    script_total: int
    passing: int
    total: int


def read_counts(path: Path) -> tuple[Counts | None, str]:
    """The counts out of a gate verdict, or `None` and why not."""
    if not path.is_file():
        return None, f"{path} does not exist — did the gate run?"
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return None, f"{path} is not valid JSON: {exc}"

    entries = payload.get("constraints")
    entries = entries if isinstance(entries, list) else []
    if entries:
        # From the per-constraint entries, not the summary keys, because they are unambiguous:
        # they carry `kind` and `passed`, and they are in every verdict the gate has ever
        # written. A verdict from before the script/all split holds the *whole* count under
        # `constraints_passing`, so reading that key would report it as the script count and
        # inflate the gated metric — which is the one number acceptance acts on.
        script = [e for e in entries if e.get("kind") == "script"]
        return Counts(
            script_passing=sum(1 for e in script if e.get("passed")),
            script_total=len(script),
            passing=sum(1 for e in entries if e.get("passed")),
            total=len(entries),
        ), "from the gate's verdict"

    script_passing, passing = payload.get(METRIC), payload.get(METRIC_ALL)
    if not isinstance(script_passing, int) or not isinstance(passing, int):
        return None, f"no constraint entries and no {METRIC}/{METRIC_ALL} in {path}"
    return Counts(
        script_passing=script_passing,
        script_total=_int(payload.get("script_constraints_total"), script_passing),
        passing=passing,
        total=_int(payload.get("constraints_total"), passing),
    ), "from the gate's verdict"


def _int(value: object, fallback: int) -> int:
    return value if isinstance(value, int) else fallback


def read_metrics(path: Path, names: list[str]) -> tuple[dict[str, float], list[str]]:
    """The named measurements out of a gate verdict, and the names it did not carry.

    The gate parses every `##autohelix[...]` its constraints print and records them in the
    verdict, so a measurement a constraint already produces becomes the loop's metric without a
    second command. `integrate run` uses this for exactly that: the verifier that decides a
    constraint also prints the latencies, and running it twice an iteration would double the
    device time and let the two runs disagree.

    A missing name is returned rather than defaulted. `Harness._capture_baseline` requires every
    declared metric at iteration 0 and aborts the run when one is absent, which is the right
    failure: a metric nothing produces is a gate that silently never fires.
    """
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}, list(names)
    found = payload.get("metrics")
    found = found if isinstance(found, dict) else {}
    out: dict[str, float] = {}
    missing: list[str] = []
    for name in names:
        value = found.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            missing.append(name)
        else:
            out[name] = float(value)
    return out, missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", required=True, help="the gate's verdict file")
    parser.add_argument(
        "--metrics", default="",
        help="comma-separated metric names to read back out of the verdict's `metrics`, where "
             "the gate recorded everything the constraints printed as `##autohelix[...]`. "
             "Used by `integrate run`, whose metrics come from the constraints themselves.",
    )
    args = parser.parse_args(argv)

    path = Path(args.json)
    counts, note = read_counts(path)
    if counts is None:
        print(f"could not read {METRIC}: {note}", file=sys.stderr)
        return 1
    print(
        f"{note}: {counts.script_passing}/{counts.script_total} script constraint(s) hold "
        f"— the gated count — and {counts.passing}/{counts.total} overall"
    )
    print(f"##autohelix[{METRIC}={counts.script_passing}]")
    print(f"##autohelix[{METRIC_ALL}={counts.passing}]")

    names = [n.strip() for n in args.metrics.split(",") if n.strip()]
    if names:
        values, missing = read_metrics(path, names)
        for name, value in values.items():
            print(f"##autohelix[{name}={value:.10g}]")
        if missing:
            # Reported, not invented. The run will stop at the baseline capture, and this line is
            # the reason it did.
            print(
                "the constraints printed no ##autohelix[...] for: " + ", ".join(missing)
                + ". A metric this loop gates on has to come from a constraint's own output.",
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
