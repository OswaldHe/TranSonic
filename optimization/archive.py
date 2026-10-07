# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every finished run's `FEEDBACK.md`, in one place, read by the runs that come after.

The feedback stage writes what stopped one module and the document is then read once, by whoever
asked for it. That is a waste twice over.

Forward, because the obstacles are **the toolchain's, not the module's**. `nisa.nc_matmul_mx` is
NeuronCore-v4 on every module; a `tensor_reduce` never reaches a performance mode in any kernel;
the Scalar engine has no 2x mode whatever is pointed at it. A module that meets one of those
spends an iteration — an hour of agent time and a device slot — rediscovering a thing another
module already proved, and the loop has five.

Backward, because a finding filed twice is a finding triaged once. Eight runs on this host have
each written a 100 KB report, and the same four or five obstacles are in all of them, worded
differently enough that nobody downstream can tell. The ninth report should say "already filed,
and here is what this run adds" rather than re-deriving row 3.

So an entry is deposited per finished run and the whole store is seeded into later runs, read-only,
the same way `memory.py` seeds an operator's directory — and for the same reasons. What is new here
is the **index**: 800 KB of reports is not something an agent reads, so each entry contributes one
row per finding, keyed by a digest of its scenario, and that table is what gets scanned.

One rule holds the whole thing up, and it is the opposite of what a cache usually wants. **An entry
is evidence, not settled fact.** `mtp.0.ffn` declined its largest available optimization — 9.62 ms
to 1.05 ms, bit-identical — across twelve iterations of two loops, because a sibling run had filed
the mechanism as a device defect and the claim was wrong. A store of inherited conclusions makes
that failure cheaper to repeat and wider in blast radius, so rulings travel with findings: a later
run that tests a filed claim and disagrees marks it **disputed**, and the index says so next to the
row. Propagating a refutation is the only thing here more valuable than propagating a finding.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from optimization import feedback as fb

#: Where the seeded copy appears inside an iteration worktree or a preparation repo. Beside
#: `memory.SEEDED_REL` and never inside it: one is the operator's, this one is the pipeline's.
SEEDED_REL = Path(".autohelix") / "feedback-archive"

#: The index, named so `memory.entry_names` lists it first — it is the only file here meant to be
#: read whole, and every prompt block tells the agent to open it before anything else.
INDEX_NAME = "README.md"

#: One record per entry, for this module rather than for an agent.
MANIFEST_NAME = "manifest.json"

#: Inside an entry: what the run was and what it found, so the index can be rebuilt from the
#: entries alone and a hand-copied directory is still a legal entry.
ENTRY_NAME = "entry.json"

#: What a later run may say about a filed finding. Deliberately short: a ruling that needs a
#: paragraph belongs in the evidence cell, and a vocabulary nobody can remember goes unused.
RULINGS: dict[str, str] = {
    "agrees": "reproduced here, independently",
    "disagrees": "tested here and it does not hold, so the filed finding is wrong or has been fixed",
    "extends": "holds, and this run adds evidence that changes the scope or the suggested fix",
    "not-applicable": "could not arise in this module, so this run is no evidence either way",
}

#: The table a report uses to rule on already-filed findings. Parsed out of `FEEDBACK.md` the same
#: way `feedback.find_table` reads the findings table.
PRIOR_COLUMNS: tuple[str, ...] = ("Filed", "Ruling", "Evidence")

#: How much of a scenario the index shows. Long enough to recognize a finding, short enough that
#: 50 rows stay scannable.
SCENARIO_WIDTH = 150

#: Markdown only. The whole store is copied into every iteration worktree, so what belongs in it
#: is what an agent reads: prose. A reproduction directory also holds the scripts the feedback
#: agent wrote, the NEFFs and profiles it compiled, and the compiler's logs — and those stay in
#: the run's own workspace, which `Entry.origin` records so they can still be found.
REPRO_SUFFIXES = frozenset({".md"})

#: A per-file ceiling, so one generated file cannot undo the extension filter.
REPRO_MAX_BYTES = 256 * 1024

_ID = re.compile(r"\b([0-9a-f]{12})\b")


def read_ruling(cell: str) -> str:
    """The ruling a table cell states, or empty when it states none.

    Whole-word, and the longest name first, because **`disagrees` contains `agrees`**. A substring
    test read every refutation as a confirmation — which is the one error here worse than recording
    nothing, since it would mark a wrong finding as independently reproduced.
    """
    text = f" {fb._normalize(cell)} "
    for name in sorted(RULINGS, key=len, reverse=True):
        if f" {fb._normalize(name)} " in text:
            return name
    return ""


class ArchiveError(RuntimeError):
    """The archive cannot be read or written as asked."""


@dataclass
class Finding:
    """One row of one run's findings table, as the archive carries it."""

    id: str
    level: str
    scenario: str
    #: Set when a later run ruled on this finding: `(ruling, module, evidence)` each.
    rulings: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def disputed(self) -> bool:
        return any(ruling == "disagrees" for ruling, _, _ in self.rulings)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "level": self.level, "scenario": self.scenario}


@dataclass
class Entry:
    """One finished run's deposit."""

    name: str
    module: str
    recorded: str
    findings: list[Finding] = field(default_factory=list)
    latencies: dict[str, float | None] = field(default_factory=dict)
    #: Rulings this run made on findings filed by *earlier* entries, keyed by finding id.
    rulings: dict[str, tuple[str, str]] = field(default_factory=dict)
    #: Reproduction files left behind, so a filter that is too tight is visible as a number
    #: rather than as a missing file nobody looks for.
    skipped: int = 0
    #: The workspace this came from. The archive carries prose only, so this is how a reader gets
    #: from a finding to the script that demonstrates it.
    origin: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "module": self.module, "recorded": self.recorded,
            "latencies": self.latencies,
            "findings": [f.to_dict() for f in self.findings],
            "rulings": {k: list(v) for k, v in self.rulings.items()},
            "skipped": self.skipped,
            "origin": self.origin,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Entry:
        findings = [
            Finding(id=str(f.get("id") or ""), level=str(f.get("level") or ""),
                    scenario=str(f.get("scenario") or ""))
            for f in (data.get("findings") or [])
        ]
        rulings = {
            str(k): (str(v[0]), str(v[1]) if len(v) > 1 else "")
            for k, v in (data.get("rulings") or {}).items() if v
        }
        return cls(
            name=str(data.get("name") or ""), module=str(data.get("module") or ""),
            recorded=str(data.get("recorded") or ""),
            findings=findings, latencies=data.get("latencies") or {}, rulings=rulings,
            skipped=int(data.get("skipped") or 0),
            origin=str(data.get("origin") or ""),
        )


def finding_id(scenario: str) -> str:
    """A stable short id for a finding, from its scenario text.

    Keyed on the scenario rather than on a counter, so the *same obstacle* filed by two modules
    gets two different ids only if the two agents described it differently — which is exactly when
    a human has to look anyway. Normalized first, so markdown emphasis and line wrapping do not
    make one finding into two.
    """
    normalized = re.sub(r"[^a-z0-9]+", " ", scenario.lower()).strip()
    return hashlib.sha256(normalized.encode()).hexdigest()[:12]


# --------------------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------------------


def read_entries(archive: Path | None) -> list[Entry]:
    """Every entry in the archive, oldest first. Empty for a missing or unreadable store."""
    if archive is None or not Path(archive).is_dir():
        return []
    entries: list[Entry] = []
    for path in sorted(Path(archive).iterdir()):
        record = path / ENTRY_NAME
        if not record.is_file():
            continue
        try:
            entry = Entry.from_dict(json.loads(record.read_text()))
        except (OSError, json.JSONDecodeError, TypeError):
            continue
        entry.name = entry.name or path.name
        entries.append(entry)
    entries.sort(key=lambda e: (e.recorded, e.name))
    return _apply_rulings(entries)


def _apply_rulings(entries: list[Entry]) -> list[Entry]:
    """Hang each entry's rulings on the findings they are about, wherever those were filed."""
    by_id = {f.id: f for entry in entries for f in entry.findings}
    for entry in entries:
        for target, (ruling, evidence) in entry.rulings.items():
            finding = by_id.get(target)
            if finding is not None:
                finding.rulings.append((ruling, entry.module, evidence))
    return entries


def all_findings(entries: list[Entry]) -> list[tuple[Entry, Finding]]:
    """Every filed finding with the entry that filed it, disputed ones last.

    Disputed last because the list is read top-down under a budget, and a claim another run has
    already refuted is the least useful thing on it.
    """
    pairs = [(e, f) for e in entries for f in e.findings]
    return sorted(pairs, key=lambda p: (p[1].disputed, p[1].level, p[0].recorded))


# --------------------------------------------------------------------------------------
# the index
# --------------------------------------------------------------------------------------


def findings_table(entries: list[Entry]) -> str:
    """The index's own table: one row per filed finding, keyed by id."""
    if not entries:
        return "_No findings filed yet._"
    lines = ["| id | level | module | scenario | status |", "|---|---|---|---|---|"]
    for entry, finding in all_findings(entries):
        status = "filed"
        if finding.rulings:
            status = "; ".join(f"**{r}** ({who})" for r, who, _ in finding.rulings)
        lines.append(
            f"| `{finding.id}` | {finding.level} | `{entry.module}` | {_cell(finding.scenario)} "
            f"| {status} |"
        )
    return "\n".join(lines)


#: Shared-word fraction above which two scenarios are shown as possibly the same finding. Tuned on
#: the first backfill: 0.45 pairs "Four things the profile reports mean something other than what
#: they say" with "Four fields the profile reports mean..." and leaves unrelated L0s alone.
SIMILAR_THRESHOLD = 0.45

#: How much of a lead sentence is compared, for a scenario that runs on without one.
LEAD_WIDTH = 240

#: Words too common to carry meaning in a one-line scenario. Deliberately tiny — a long stop list
#: starts dropping the technical words the comparison depends on.
_STOP = frozenset("a an and are as at be by for from in is it its of on or that the this to with "
                  "was were which while not no than then so".split())


def similar_pairs(entries: list[Entry],
                  threshold: float = SIMILAR_THRESHOLD) -> list[tuple[Finding, str, Finding, str]]:
    """Findings from *different* runs whose scenarios share most of their words.

    The archive stops a run from filing a duplicate, because the agent reads the table first. It
    cannot merge the duplicates already in it: an id is a digest of the scenario text, so the same
    obstacle described in two ways is two ids. The first backfill of seven finished runs had the
    profile-fields finding filed twice, worded differently enough that neither agent could have
    known. So the index points at the pairs and leaves the judgement to a reader.
    """
    pairs: list[tuple[Finding, str, Finding, str]] = []
    bags = [(e, f, _words(f.scenario)) for e, f in all_findings(entries)]
    for i, (entry_a, finding_a, words_a) in enumerate(bags):
        for entry_b, finding_b, words_b in bags[i + 1:]:
            if entry_a.name == entry_b.name or len(words_a) < 4 or len(words_b) < 4:
                continue
            overlap = len(words_a & words_b) / len(words_a | words_b)
            if overlap >= threshold:
                pairs.append((finding_a, entry_a.module, finding_b, entry_b.module))
    return pairs


def _words(scenario: str) -> set[str]:
    """The words of a scenario's **claim**, which is its lead sentence.

    Not the whole cell. A scenario runs to a thousand characters because it enumerates the
    specifics, and two reports of one obstacle enumerate different ones — the pair this was built
    for shares a lead sentence to within a single word and scores 0.24 over the full text. Split on
    period-space so a backticked `nisa.topk` does not read as a sentence end.
    """
    lead = re.split(r"\.\s", scenario, maxsplit=1)[0][:LEAD_WIDTH]
    return {w for w in re.sub(r"[^a-z0-9_]+", " ", lead.lower()).split()
            if w not in _STOP and len(w) > 1}


def _cell(text: str) -> str:
    """One table cell: no pipes, no newlines, and short enough to scan."""
    flat = " ".join(text.replace("|", "/").split())
    return flat if len(flat) <= SCENARIO_WIDTH else flat[:SCENARIO_WIDTH - 1] + "…"


def write_index(archive: Path, entries: list[Entry] | None = None) -> Path:
    """Regenerate `README.md` and `manifest.json` from the entries on disk.

    Generated rather than appended, so an entry copied in or deleted by hand is picked up and the
    index cannot drift from what is actually there.
    """
    archive = Path(archive)
    archive.mkdir(parents=True, exist_ok=True)
    entries = read_entries(archive) if entries is None else entries
    disputed = [f for _, f in all_findings(entries) if f.disputed]
    levels = "\n".join(f"- **{k}** — {v}" for k, v in fb.LEVELS.items())
    body = f"""# Feedback archive

{len(entries)} finished optimization run(s) found
**{sum(len(e.findings) for e in entries)} blocking finding(s)** on this toolchain, {len(disputed)}
of them since disputed by a later run.

The obstacles here are the toolchain's, not any one module's, so a finding filed against one
module often applies to the next. This is a lookup rather than something to read through:
it is worth a look when you are about to conclude that something is impossible on this part, when
you are about to spend an iteration establishing that it is, or when you are about to file a
finding of your own.

{levels}

## Filed findings

{findings_table(entries)}

Open `<entry>/{fb.REPORT_NAME}` for the full row — root cause, documentation links, suggested
fix. This archive carries prose only; the runnable reproduction each row names is still in the run
it came from, under the `origin` path in the entries table below. A reproduction you can run in two
minutes settles a question that a sentence only raises, so it is worth the walk.

## A filed finding is evidence, not settled fact

Marked **disagrees** means a later run tested the claim and it did not hold. Treat any row you are
about to build a decision on the same way: if re-testing is cheap, re-test it.

This is not caution for its own sake. One run declined its largest available optimization — 9.62 ms
to 1.05 ms, bit-identical, measured afterwards — across twelve iterations, because a sibling run
had filed the mechanism as a device defect and that claim was wrong. The cost of inheriting a wrong
finding is larger than the cost of re-deriving a right one.

## Possibly the same finding, filed twice

{_duplicates_table(entries)}

## Entries

{_entries_table(entries)}
"""
    (archive / INDEX_NAME).write_text(body)
    (archive / MANIFEST_NAME).write_text(json.dumps(
        {"entries": [e.to_dict() for e in entries]}, indent=2) + "\n")
    return archive / INDEX_NAME


def _duplicates_table(entries: list[Entry]) -> str:
    """Pairs worth a reader's judgement, with the lead sentence that makes them look alike."""
    pairs = similar_pairs(entries)
    if not pairs:
        return "_None found._"
    lines = ["Two runs appear to have filed one obstacle. Nothing is merged automatically — the "
             "wording differs for a reason sometimes — but if you are about to file a third, read "
             "these first.", "",
             "| | id | module | scenario |", "|---|---|---|---|"]
    for a, module_a, b, module_b in pairs:
        lines.append(f"| ⟋ | `{a.id}` | `{module_a}` | {_cell(a.scenario)} |")
        lines.append(f"| ⟍ | `{b.id}` | `{module_b}` | {_cell(b.scenario)} |")
    return "\n".join(lines)


def _entries_table(entries: list[Entry]) -> str:
    if not entries:
        return "_Empty._"
    lines = ["| entry | module | recorded | findings | bootstrap → submodule → module "
             "| reproductions |", "|---|---|---|---|---|---|"]
    for entry in entries:
        lat = entry.latencies or {}
        chain = " → ".join(
            f"{lat[k]:g} ms" if isinstance(lat.get(k), (int, float)) else "—"
            for k in ("bootstrap_ms", "submodule_ms", "module_ms")
        )
        origin = f"`{entry.origin}/{fb.REPRO_DIR}/`" if entry.origin else "—"
        lines.append(f"| `{entry.name}` | `{entry.module}` | {entry.recorded} "
                     f"| {len(entry.findings)} | {chain} | {origin} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# depositing
# --------------------------------------------------------------------------------------


def findings_of(workspace_root: Path) -> list[Finding]:
    """The findings in a workspace's `FEEDBACK.md`, as archive records."""
    report = fb.report_path(Path(workspace_root))
    if not report.is_file():
        return []
    found: list[Finding] = []
    for row in fb.parse_rows(report.read_text(errors="replace")):
        scenario = fb._column(row, "Scenario")
        if not scenario.strip():
            continue
        level = fb._column(row, "Level").strip().upper()
        found.append(Finding(
            id=finding_id(scenario),
            level=next((k for k in fb.LEVELS if k in level), level or "?"),
            scenario=scenario,
        ))
    return found


def rulings_of(workspace_root: Path) -> dict[str, tuple[str, str]]:
    """What a report says about findings earlier entries filed, by finding id."""
    report = fb.report_path(Path(workspace_root))
    if not report.is_file():
        return {}
    out: dict[str, tuple[str, str]] = {}
    for row in _prior_rows(report.read_text(errors="replace")):
        target = _ID.search(fb._column(row, "Filed"))
        ruling = read_ruling(fb._column(row, "Ruling"))
        if target and ruling:
            out[target.group(1)] = (ruling, _cell(fb._column(row, "Evidence")))
    return out


def _prior_rows(text: str) -> list[dict[str, str]]:
    """Rows of the "previously filed" table, or none when the report has no such table."""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        match = fb._ROW.match(line.strip())
        if not match:
            continue
        header = [c.strip() for c in match.group("cells").split("|")]
        normalized = [fb._normalize(c) for c in header]
        if not all(any(fb._normalize(w) in cell for cell in normalized) for w in PRIOR_COLUMNS):
            continue
        if index + 1 >= len(lines) or not fb._SEPARATOR.match(lines[index + 1].strip()):
            continue
        rows: list[dict[str, str]] = []
        for row_line in lines[index + 2:]:
            row = fb._ROW.match(row_line.strip())
            if not row:
                break
            cells = [c.strip() for c in row.group("cells").split("|")]
            cells += [""] * (len(header) - len(cells))
            rows.append({header[i]: cells[i] for i in range(len(header))})
        return rows
    return []


def validate_rulings(workspace_root: Path, entries: list[Entry]) -> list[str]:
    """Problems with a report's rulings on filed findings: unknown ids, unknown rulings.

    Structural like `feedback.validate_report`, and for the same reason — whether a ruling is
    *right* is exactly what no script can decide. What this catches is a ruling that cannot be
    acted on: an id that names nothing, so nobody can tell which finding was meant.
    """
    report = fb.report_path(Path(workspace_root))
    if not report.is_file():
        return []
    known = {f.id for entry in entries for f in entry.findings}
    problems: list[str] = []
    for number, row in enumerate(_prior_rows(report.read_text(errors="replace")), start=1):
        cell = fb._column(row, "Filed")
        target = _ID.search(cell)
        if not target:
            problems.append(
                f"previously-filed row {number}: '{cell or '(empty)'}' names no 12-character "
                f"finding id. Take it from the `id` column of the archive's {INDEX_NAME}"
            )
            continue
        if target.group(1) not in known:
            problems.append(
                f"previously-filed row {number}: no finding `{target.group(1)}` is in the archive"
            )
        if not read_ruling(fb._column(row, "Ruling")):
            problems.append(
                f"previously-filed row {number}: '{fb._column(row, 'Verdict') or '(empty)'}' is "
                f"not one of {', '.join(RULINGS)}"
            )
    return problems


def measured_latencies(workspace_root: Path) -> dict[str, float | None]:
    """What the run achieved, read from its own records rather than passed in.

    So that `optimize archive --add` on a run that finished months ago files the same numbers the
    pipeline would have filed for it. Globbed rather than built from the module slug: the slug is
    derived from the module id, and the point here is to work on a directory without being told
    what produced it.
    """
    root = Path(workspace_root)
    found: dict[str, float | None] = {k: None for k in
                                      ("bootstrap_ms", "submodule_ms", "module_ms")}
    try:
        baselines = json.loads((root / ".optimization" / "baselines.json").read_text())
        found["bootstrap_ms"] = baselines.get("bootstrap_latency_ms")
    except (OSError, json.JSONDecodeError, AttributeError):
        pass
    for stage, key in (("submodule", "submodule_ms"), ("full", "module_ms")):
        for path in sorted(root.glob(f"*/.autohelix/optimization/{stage}-summary.json")):
            try:
                found[key] = json.loads(path.read_text()).get("best_ms") or found[key]
            except (OSError, json.JSONDecodeError, AttributeError):
                continue
    return found


def deposit(
    archive: Path,
    workspace_root: Path,
    module: str,
    latencies: dict[str, float | None] | None = None,
    recorded: str | None = None,
) -> Entry:
    """Copy one finished run's report and reproductions into the archive, and reindex.

    Named by module and date, so a second round on the same module is a second entry rather than
    an overwrite: what the first round could not get past is part of why the second one ran.
    """
    archive, workspace_root = Path(archive), Path(workspace_root)
    report = fb.report_path(workspace_root)
    if not report.is_file():
        raise ArchiveError(f"{workspace_root} has no {fb.REPORT_NAME} to deposit")

    recorded = recorded or date.today().isoformat()
    slug = re.sub(r"[^A-Za-z0-9]+", "-", module).strip("-") or "module"
    name = f"{slug}-{recorded}"
    target = archive / name
    suffix = 2
    while target.exists() and (target / ENTRY_NAME).is_file():
        existing = json.loads((target / ENTRY_NAME).read_text())
        if existing.get("module") == module and existing.get("recorded") == recorded:
            break
        target, suffix = archive / f"{name}-{suffix}", suffix + 1

    target.mkdir(parents=True, exist_ok=True)
    (target / fb.REPORT_NAME).write_bytes(report.read_bytes())
    skipped = _copy_repro(fb.repro_dir(workspace_root), target / fb.REPRO_DIR)

    entry = Entry(
        name=target.name, module=module, recorded=recorded,
        findings=findings_of(workspace_root),
        latencies=dict(latencies or measured_latencies(workspace_root)),
        rulings=rulings_of(workspace_root),
        skipped=len(skipped),
        origin=str(workspace_root.resolve()),
    )
    (target / ENTRY_NAME).write_text(json.dumps(entry.to_dict(), indent=2) + "\n")
    write_index(archive)
    return entry


def _copy_repro(source: Path, target: Path) -> list[str]:
    """The reproductions an agent wrote, not the build output they produced.

    A reproduction directory is where the feedback agent compiles and profiles, so what is in it
    afterwards is mostly the toolchain's: the first backfill here copied 48 MB of `.ntff` profiles
    and `.colz` compile caches out of one run, into a tree that is then seeded into every later
    iteration. The scripts are a few hundred kilobytes and they are the whole point, so this keeps
    source and prose and drops the rest — by extension rather than by pattern, because each run's
    scratch directory is named differently (`_work`, `_build`, `_artifacts`, a content hash).
    """
    if not source.is_dir():
        return []
    skipped: list[str] = []
    for path in sorted(source.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(source)
        if (any(p.startswith((".", "_")) for p in rel.parts)
                or path.suffix.lower() not in REPRO_SUFFIXES
                or path.stat().st_size > REPRO_MAX_BYTES):
            skipped.append(str(rel))
            continue
        (target / rel).parent.mkdir(parents=True, exist_ok=True)
        try:
            (target / rel).write_bytes(path.read_bytes())
        except OSError:
            skipped.append(str(rel))
    return skipped


# --------------------------------------------------------------------------------------
# the prompt blocks
# --------------------------------------------------------------------------------------


def describe_for_optimizer(entries: list[Entry], seeded: int) -> str:
    """The block an optimizing or preparation agent gets. Empty when there is nothing to say.

    Written as a reference you may consult, not as reading you have been assigned. An agent told
    to read 95 findings spends its context on 94 that do not apply to its kernel; an agent told
    what is in there and where looks when it has a question the archive can answer. The second is
    both cheaper and what actually gets used.
    """
    if not entries or seeded <= 0:
        return ""
    return "\n".join([
        f"**`{SEEDED_REL}/` holds what blocked earlier runs on this toolchain.** It carries "
        f"{sum(len(e.findings) for e in entries)} read-only finding(s) from {len(entries)} "
        f"finished run(s) on other modules. Its `{INDEX_NAME}` is a single table keyed by finding "
        f"id, and each row's full text and runnable reproduction sit one directory away.",
        "",
        "Treat it as a lookup rather than as reading to do. Consult it before you conclude that "
        "something is impossible on this part, or before you spend an iteration establishing that "
        "it is, because these findings describe the toolchain rather than the modules that met it.",
        "",
        "Two cautions if you do. Each run fitted its tile widths and SBUF budgets to its own "
        "shapes, so sizes do not transfer, though technique usually does. A row marked "
        "**disputed** is one that a later run tested and refuted, so treat none of this as "
        "settled: when a finding carries a decision you are making and re-testing it is cheap, "
        "re-test it and record what you saw.",
    ])


def describe_for_feedback(entries: list[Entry], seeded: int) -> str:
    """The block the feedback agent gets: do not file what is already filed."""
    if not entries or seeded <= 0:
        return ""
    return "\n".join([
        f"**`{SEEDED_REL}/` already holds {sum(len(e.findings) for e in entries)} finding(s) from "
        f"{len(entries)} earlier run(s).** Its `{INDEX_NAME}` is one table keyed by finding id. "
        f"Consult it per row as you write, rather than reading it through first.",
        "",
        "**Do not file a finding that is already there.** Filing one twice means the Neuron team "
        "triages it once: the same few toolchain obstacles appear in every report on this host, "
        "worded differently enough that nobody downstream can tell they are one thing. Where this "
        "run met an obstacle that is already there, rule on it instead, in one table placed "
        "anywhere in your report:",
        "",
        f"| {' | '.join(PRIOR_COLUMNS)} |",
        "|---|---|---|",
        "| `a1b2c3d4e5f6` | agrees | what you measured, and where the reproduction is |",
        "",
        "Those are the only rulings:",
        "",
        *(f"- **{name}** — {text}" for name, text in RULINGS.items()),
        "",
        "**A `disagrees` is the most valuable row you can write.** Every run after this one reads "
        "that archive, so a wrong finding in it will eventually make one of them decline a real "
        "optimization — that has already happened here, and it cost a 9x. File a new row only for "
        "an obstacle the archive does not have, or for one whose scope or fix your evidence "
        "genuinely changes: rule that one `extends` and say what is new.",
    ])
