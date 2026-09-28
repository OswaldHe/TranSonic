# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The run's report: what each stage achieved, and the two things a reader has to be told.

Every number here is uncalibrated in one specific way and honest about it. The projection diverges
from the ranked floorplan, so the latencies describe a plan the search ranked second; and the
whole-module metric is the *fastest* rank, which understates what the module costs in a pipeline
where the slowest rank gates the next stage. Both are stated at the top rather than in a footnote,
because a reader who takes these numbers as evidence about the 16-device deployment will be wrong in
a way the numbers themselves cannot reveal.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from rich.console import Console

from optimization.config import PipelineConfig
from optimization.loop import METRIC
from optimization.projection import Projection, project_module

REPORT_NAME = "REPORT.md"


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def _stage_table(summary: dict[str, Any]) -> str:
    rows = summary.get("iterations") or []
    if not rows:
        return "_no iterations recorded_"
    has_ranks = any(row.get("rank_spread_ms") is not None for row in rows)
    header = f"| iteration | {METRIC} (fastest rank) | accepted | slot | followed |"
    rule = "|---|---|---|---|---|"
    if has_ranks:
        header = (f"| iteration | {METRIC} (fastest) | slowest rank | spread | accepted | slot "
                  f"| followed |")
        rule = "|---|---|---|---|---|---|---|"
    lines = [f"{header} note |", f"{rule}---|"]
    for row in rows:
        value = row.get(METRIC)
        followed = row.get("slot_followed")
        cells = [str(row.get("iteration")), _fmt(value)]
        if has_ranks:
            cells += [_fmt(row.get("slowest_rank_ms")), _fmt(row.get("rank_spread_ms"))]
        cells += [
            "yes" if row.get("accepted") else "no",
            str(row.get("slot") or "—"),
            "—" if followed is None else ("yes" if followed else "**no**"),
            str(row.get("reason") or ""),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _speedup(before: Any, after: Any) -> str:
    if not isinstance(before, (int, float)) or not isinstance(after, (int, float)) or not after:
        return "—"
    return f"{before / after:.2f}x"


#: How much of a slot's prose fits in a table cell before it has to be cut. Wide enough to clear the
#: bolded lead and reach the sentence after it, which is what tells two slots with the same lead
#: apart: 4-6 and 9-10 both open "NKI and torch-xla are both allowed" and differ only in what
#: follows.
_GIST_WIDTH = 120


def _embed(markdown: str, demote: int = 2) -> str:
    """Demote a standalone document's headings so it reads as a section of this one.

    `Projection.describe()` writes `FLOORPLAN.md`, which has its own `#` title because it is a
    document in its own right. Splicing it in verbatim put an H1 in the middle of the report and
    made its subsections siblings of the report's own, so "What the projection gave up" appeared in
    an outline view as a top-level finding of the run rather than as part of the projection.

    Only a heading at column 0 is demoted, which is where `describe()` puts them. That leaves an
    indented code block containing a `#` comment alone.
    """
    out = []
    for line in markdown.splitlines():
        match = re.match(r"(#{1,6}) ", line)
        out.append(f"{'#' * min(len(match.group(1)) + demote, 6)}{line[match.end(1):]}"
                   if match else line)
    return "\n".join(out)


def _slot_gist(text: str) -> str:
    """One table cell's worth of a slot's prose, cut where a reader can see the cut.

    The full constraint runs to paragraphs. Taking its first physical line ended the cell
    mid-sentence with no ellipsis — "may import and reference `nki` and the Python" — which reads as
    a rendering fault rather than as a summary. So: collapse to one line, cut on a word boundary,
    and mark the cut.

    Two details the obvious version gets wrong. A cut landing inside a slot's bolded lead leaves an
    unpaired `**` that bleeds emphasis across the rest of the table, so an odd count is closed. And
    a constraint that mentions a pipe would split the row, so pipes are escaped.
    """
    flat = " ".join(text.split()).replace("|", "\\|")
    if not flat:
        return "—"
    if len(flat) <= _GIST_WIDTH:
        return flat
    cut = flat[:_GIST_WIDTH].rsplit(" ", 1)[0]
    if cut.count("**") % 2:
        cut += "**"
    return f"{cut}…"


def write_report(config: PipelineConfig, console: Console | None = None) -> Path:
    """Assemble the report from what each stage recorded, and write it to the workspace."""
    console = console or Console()
    state = config.workspace_root / ".optimization"
    projection_record = _read(state / "projection.json")
    baselines = _read(state / "baselines.json")
    submodule_summary = _read(
        config.submodule_repo / ".autohelix" / "optimization" / "submodule-summary.json"
    )
    full_summary = _read(
        config.full_repo / ".autohelix" / "optimization" / "full-summary.json"
    )

    # The recorded projection is the one the run was built against. Recomputing from the scheme
    # would let a scheme edited after the run label finished measurements with a split and a rank
    # count that were never executed — the one thing this report's own caveats exist to prevent.
    # Recomputation is the fallback for a workspace with no record, not the default.
    projection: Projection | None = None
    recorded = (projection_record or {}).get("projection") or {}
    if recorded:
        try:
            projection = Projection.from_dict(recorded)
        except Exception:
            projection = None
    if projection is None:
        try:
            projection = project_module(config.scheme, config.module_id, config.target_units)
        except Exception:
            projection = None

    bootstrap_ms = baselines.get("bootstrap_latency_ms")
    submodule_best = submodule_summary.get("best_ms")
    submodule_baseline = submodule_summary.get("baseline_ms")
    full_best = full_summary.get("best_ms")
    full_baseline = full_summary.get("baseline_ms")

    parts: list[str] = [
        f"# `{config.module_id}` on one device",
        "",
        "## Two things to read the numbers with",
        "",
    ]
    if projection is not None and projection.diverges:
        planned = " * ".join(f"{f.dim}x{f.factor}" for f in projection.planned)
        got = " * ".join(f"{f.dim}x{f.factor}" for f in projection.projected)
        parts += [
            f"**These latencies describe a plan the floorplan ranked second.** The scheme places "
            f"this module on {projection.planned_units} logical NeuronCores across "
            f"{projection.planned_devices} devices (`{planned}`); this run projected that onto the "
            f"one device available (`{got}`), so a rank holds "
            f"{projection.weight_residency_ratio():g}x the planned weight bytes. For `.ffn` that is "
            f"exactly the trade the floorplan's own report priced and rejected — it chose the wider "
            f"split because it lowers per-bank weight residency and decode is bank-bandwidth bound. "
            f"A fast kernel here is a fast single-device kernel, not evidence about the 16-device "
            f"deployment.",
            "",
        ]
    parts += [
        "**The whole-module metric is the fastest rank.** That isolates compute from load imbalance, "
        "which is what makes iteration-to-iteration comparison meaningful — and it understates what "
        "the module costs in a pipeline, where the slowest rank gates the next stage. The per-rank "
        "spread is in the tables below; read it before quoting the headline.",
        "",
        "## What the run achieved",
        "",
        "| | latency (ms) | vs the bootstrapped module |",
        "|---|---|---|",
        f"| bootstrapped module, 1 core | {_fmt(bootstrap_ms)} | — |",
        f"| submodule baseline, 1 core | {_fmt(submodule_baseline)} | — |",
        f"| submodule best, 1 core | {_fmt(submodule_best)} | "
        f"{_speedup(submodule_baseline, submodule_best)} over its own baseline |",
        f"| assembled baseline, {projection.projected_units if projection else '?'} ranks "
        f"| {_fmt(full_baseline)} | {_speedup(bootstrap_ms, full_baseline)} |",
        f"| **assembled best, {projection.projected_units if projection else '?'} ranks** "
        f"| **{_fmt(full_best)}** | **{_speedup(bootstrap_ms, full_best)}** |",
        "",
    ]

    # Two numbers, not one: the bound is tight at assembly and slack afterwards, because the
    # submodule is frozen there while the whole module goes on being optimized past it. Reporting
    # only the second printed `-55.7% against the 10% the gate allows`, which reads as a blown bound.
    # The divisor is the submodule latency remeasured at assembly time, which is what the gate holds
    # the ratio to; stage 3's own best differs from it by the run-to-run variance.
    bound_reference = baselines.get("submodule_latency_ms")
    if not isinstance(bound_reference, (int, float)) or not bound_reference:
        bound_reference = submodule_best
    if isinstance(bound_reference, (int, float)) and bound_reference:
        note = [f"The module gate holds every iteration to two bounds, re-checked each time: faster "
                f"than the bootstrapped module ({_fmt(bootstrap_ms)} ms), and no slower than 1.1x "
                f"the submodule as remeasured at assembly time ({bound_reference:g} ms)."]
        if isinstance(full_baseline, (int, float)):
            note.append(f" Rejoining the ranks cost "
                        f"**{(full_baseline / bound_reference - 1.0) * 100:+.1f}%** of that "
                        f"submodule measurement — that is where the second bound is tight.")
        if isinstance(full_best, (int, float)):
            delta = (full_best / bound_reference - 1.0) * 100
            note.append(f" Stage 5 then finished at **{delta:+.1f}%** of it"
                        + (": faster than the single rank it was cut from, because the submodule "
                           "was frozen at assembly while the whole module kept being optimized."
                           if delta < 0 else "."))
        parts += ["".join(note), ""]

    parts += [
        "## The projection",
        "",
        (_embed(projection.describe()) if projection else "_could not be recomputed_"),
        "",
        "## Stage 3 — one rank",
        "",
        _stage_table(submodule_summary),
        "",
        "### The constraint schedule, and whether it was followed",
        "",
    ]
    schedule = submodule_summary.get("schedule") or []
    if schedule:
        parts += ["| iterations | enforced | constraint |", "|---|---|---|"]
        for slot in schedule:
            text = (slot.get("text") or "").strip()
            enforced = "yes" if (slot.get("enforce") and text) else "no"
            parts.append(f"| {slot.get('label')} | {enforced} | {_slot_gist(text)} |")
    else:
        parts.append("_no per-iteration constraints_")

    parts += [
        "",
        "## Stage 5 — the whole module",
        "",
        _stage_table(full_summary),
        "",
        "## Where everything is",
        "",
        f"- pipeline config: `{config.source or 'unknown'}`",
        f"- floorplan scheme: `{config.scheme}`",
        f"- bootstrapped module: `{config.bootstrap_repo}`",
        f"- submodule repo: `{config.submodule_repo}`",
        f"- assembled repo: `{config.full_repo}`",
        f"- per-stage records: `{state}`",
        "- archived candidates, including rejected ones: "
        f"`{config.submodule_repo / '.autohelix' / 'optimization' / 'candidates'}` and "
        f"`{config.full_repo / '.autohelix' / 'optimization' / 'candidates'}`",
        "",
    ]
    if projection_record.get("projection", {}).get("diverges"):
        parts += [
            "## If this is carried back to the floorplan",
            "",
            "The kernel this run produced is for the projected split, not the planned one. Taking it "
            "to the 16-device target means re-cutting it at the planned width and re-measuring — the "
            "per-rank shapes change, and a kernel tuned for twice the weight per core is not "
            "automatically the right kernel for half of it.",
            "",
        ]

    path = config.workspace_root / REPORT_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts))
    return path


def _fmt(value: Any) -> str:
    return f"{value:g}" if isinstance(value, (int, float)) else "—"
