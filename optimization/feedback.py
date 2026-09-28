# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The last stage: what stopped the optimization, written for the people who own the toolchain.

Fifteen iterations of two loops leave behind every measurement the agents took and every opinion
the reviewers formed, and nobody reads it. The notes are written per iteration by an agent that did
not know how the run would end, so they contradict each other: iteration 2 records a theory that
iteration 7 disproves, a reviewer doubts a claim that a later profile confirms, and a workaround
found early is superseded twice. A reader who takes any one file at face value learns something
false.

So this stage is one agent whose whole job is the corpus: read all of it, reconcile what disagrees,
and keep only what a toolchain owner could act on. The deliverable separates three things that get
run together — a device that does not do what its documentation says (a bug), a thing the compiler
or the programming interface cannot express (a missing software feature), and a thing this silicon
simply does not have (a missing hardware feature) — because the fix for each belongs to different
people, and a report that blurs them is a report that gets triaged into nothing.

There is no gate here. A gate runs a candidate and this stage produces a document, so what it gets
instead is `validate_report`: the structure is checkable, the claims are not.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

#: The stage's deliverable, at the workspace root beside `REPORT.md`.
REPORT_NAME = "FEEDBACK.md"

#: Where the agent puts a runnable reproduction per finding. One file per row, so a reader can run
#: the thing rather than retype a snippet out of a table cell.
REPRO_DIR = "feedback-repro"

#: The three levels, and what separates them. Quoted into the prompt so the definition the agent is
#: held to and the definition it is given cannot drift apart.
LEVELS: dict[str, str] = {
    "L0": "a bug: the documentation or the API says the target device supports this, and it does not",
    "L1": "a missing software feature: the compiler or the programming interface cannot express it",
    "L2": "a missing hardware feature: this device has no component that could do it",
}

#: The report's columns, in order. The header text is matched exactly, so the table a reader sees is
#: the table this validates.
COLUMNS: tuple[str, ...] = (
    "Level",
    "Scenario",
    "Minimum reproduction",
    "Root cause and why",
    "Related documentation",
    "Suggestion",
)

_ROW = re.compile(r"^\|(?P<cells>.+)\|\s*$")
_SEPARATOR = re.compile(r"^\|[\s:|-]+\|\s*$")
_URL = re.compile(r"https?://[^\s)\]|]+")


def report_path(workspace_root: Path) -> Path:
    return workspace_root / REPORT_NAME


def repro_dir(workspace_root: Path) -> Path:
    return workspace_root / REPRO_DIR


def corpus(*repos: Path) -> dict[str, list[Path]]:
    """Every note and review the run left, by repository.

    Both kinds matter and they are not interchangeable. A note is the optimizing agent's own record
    of what it tried and what the profile said; a review is a second agent's opinion of that work,
    written without the power to change it. Where they disagree the disagreement is the finding.
    """
    found: dict[str, list[Path]] = {}
    for repo in repos:
        if not repo.is_dir():
            continue
        files: list[Path] = []
        for kind in ("notes", "reviews"):
            directory = repo / ".autohelix" / kind
            if directory.is_dir():
                files += sorted(directory.glob("*.md"), key=_iteration_order)
        if files:
            found[repo.name] = files
    return found


def _iteration_order(path: Path) -> tuple[int, str]:
    """`iter-10.md` after `iter-9.md`, which a plain sort gets backwards."""
    match = re.search(r"(\d+)", path.stem)
    return (int(match.group(1)) if match else -1, path.name)


#: The files in each finished repository that are the run's deliverable. Both have already passed a
#: gate, and `inference.py`'s hash is itself a gate check, so either changing afterwards invalidates
#: the result the report is about.
DELIVERABLE_FILES = ("source.py", "inference.py")


def snapshot_deliverables(*repos: Path) -> dict[Path, bytes]:
    """The exact bytes of every deliverable file, to put back if the feedback agent edits one.

    Taken rather than trusted because this stage has no worktree and no scope enforcement: it runs an
    agent at the workspace root, both finished repositories are writable under it, and the prompt
    asks that agent to test claims on the device. A gated kernel silently modified afterwards would
    make the run's own numbers describe code nobody measured.
    """
    return {repo / name: (repo / name).read_bytes()
            for repo in repos for name in DELIVERABLE_FILES if (repo / name).is_file()}


def restore_deliverables(snapshot: dict[Path, bytes]) -> list[str]:
    """Put back anything that changed, and name what did. Empty when the agent left them alone."""
    touched: list[str] = []
    for path, original in snapshot.items():
        try:
            if path.is_file() and path.read_bytes() == original:
                continue
        except OSError:
            pass
        touched.append(f"{path.parent.name}/{path.name}")
        try:
            path.write_bytes(original)
        except OSError:
            continue
    return touched


def word_count(files: dict[str, list[Path]]) -> int:
    total = 0
    for paths in files.values():
        for path in paths:
            try:
                total += len(path.read_text(errors="replace").split())
            except OSError:
                continue
    return total


def parse_rows(text: str) -> list[dict[str, str]]:
    """The findings table's rows. Empty both when the table is missing and when it holds no rows."""
    return find_table(text)[1]


def find_table(text: str) -> tuple[bool, list[dict[str, str]]]:
    """Whether the findings table is present, and its rows.

    The two are reported separately because they mean opposite things: no table is a malformed
    report, and a table with no rows is a run that hit no unresolved blocker. Reads the first table
    whose header names every column in `COLUMNS`, so the agent is free to put other tables in the
    document — a summary, a timeline — without them being mistaken for findings.
    """
    lines = text.splitlines()
    for index, line in enumerate(lines):
        match = _ROW.match(line.strip())
        if not match:
            continue
        header = [c.strip() for c in match.group("cells").split("|")]
        normalized = [_normalize(c) for c in header]
        if not all(any(_normalize(want) in cell for cell in normalized) for want in COLUMNS):
            continue
        if index + 1 >= len(lines) or not _SEPARATOR.match(lines[index + 1].strip()):
            continue
        rows: list[dict[str, str]] = []
        for row_line in lines[index + 2:]:
            row = _ROW.match(row_line.strip())
            if not row:
                break
            cells = [c.strip() for c in row.group("cells").split("|")]
            if len(cells) < len(header):
                cells += [""] * (len(header) - len(cells))
            rows.append({header[i]: cells[i] for i in range(len(header))})
        return True, rows
    return False, []


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _column(row: dict[str, str], want: str) -> str:
    """The cell for one logical column, however the agent worded its header."""
    target = _normalize(want)
    for key, value in row.items():
        if target in _normalize(key):
            return value
    return ""


def validate_report(workspace_root: Path) -> list[str]:
    """Structural problems with the deliverable, as a list of findings.

    Deliberately about shape and traceability, never about whether a finding is real — no script can
    decide that, and one pretending to would be worse than this. What it does catch is the report
    that looks complete and is not: a level outside the three, a reproduction naming a file nobody
    wrote, a documentation cell with no link in it.
    """
    findings: list[str] = []
    path = report_path(workspace_root)
    if not path.is_file():
        return [f"{REPORT_NAME} was not written"]

    text = path.read_text(errors="replace")
    found, rows = find_table(text)
    if not found:
        findings.append(
            f"{REPORT_NAME} has no findings table with the columns "
            f"{', '.join(COLUMNS)} — that table is the deliverable"
        )
        return findings
    # A table with a header and no rows is a legitimate answer: every obstacle this run met was
    # worked around, and the prompt says to leave those out. Rejecting it would leave the agent
    # choosing between inventing a row and failing the stage.
    if not rows:
        return findings

    for number, row in enumerate(rows, start=1):
        level = _column(row, "Level").strip().upper()
        level_key = next((k for k in LEVELS if k in level), "")
        if not level_key:
            findings.append(
                f"row {number}: '{_column(row, 'Level') or '(empty)'}' is not one of "
                f"{', '.join(LEVELS)}"
            )

        for column in ("Scenario", "Root cause and why", "Suggestion"):
            if len(_column(row, column)) < 40:
                findings.append(
                    f"row {number}: '{column}' is empty or too short to be read by someone who was "
                    f"not here"
                )

        docs = _column(row, "Related documentation")
        if not _URL.search(docs):
            findings.append(
                f"row {number}: 'Related documentation' names no link. A finding the Neuron team "
                f"cannot trace to its own documentation or issue tracker is harder for them to act "
                f"on"
            )

        repro = _column(row, "Minimum reproduction")
        named = _repro_targets(repro)
        if not named:
            findings.append(
                f"row {number}: 'Minimum reproduction' names no file under {REPRO_DIR}/"
            )
        for target in named:
            if not (workspace_root / target).is_file():
                findings.append(f"row {number}: reproduction '{target}' does not exist")
    return findings


def _repro_targets(cell: str) -> list[str]:
    """Paths under the reproduction directory that a table cell refers to."""
    return sorted(set(re.findall(rf"{REPRO_DIR}/[\w./-]+", cell)))


def summarize(workspace_root: Path) -> dict[str, Any]:
    """What the stage recorded, for the pipeline's own state directory."""
    rows = parse_rows(report_path(workspace_root).read_text(errors="replace")) \
        if report_path(workspace_root).is_file() else []
    by_level: dict[str, int] = {key: 0 for key in LEVELS}
    for row in rows:
        level = _column(row, "Level").strip().upper()
        for key in LEVELS:
            if key in level:
                by_level[key] += 1
                break
    return {"findings": len(rows), "by_level": by_level,
            "report": str(report_path(workspace_root))}


def main(argv: list[str] | None = None) -> int:
    """`python -m optimization.feedback --check --root <workspace>`.

    Exposed so the agent writing the report can check its own work before finishing, which is
    cheaper for everyone than the stage rejecting it afterwards.
    """
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="validate the report's structure")
    parser.add_argument("--root", required=True, help="the workspace root holding FEEDBACK.md")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    findings = validate_report(root)
    if findings:
        print(f"{REPORT_NAME}: {len(findings)} problem(s)")
        for finding in findings:
            print(f"  - {finding}")
        return 1
    counts = summarize(root)
    levels = ", ".join(f"{k}:{v}" for k, v in counts["by_level"].items())
    print(f"{REPORT_NAME}: {counts['findings']} finding(s) ({levels}), structure OK")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
