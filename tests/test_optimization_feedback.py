# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The feedback stage: reading the corpus, and checking the shape of what the agent writes.

Nothing here decides whether a finding is true — no script can, and one pretending to would be worse
than none. What is checkable is the difference between a report that is complete and one that only
looks it: a level outside the three, a reproduction naming a file nobody wrote, a documentation cell
with no link, a cell too short to be read by someone who was not on the run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from optimization import feedback as fb

pytestmark = pytest.mark.optimization


def _corpus(root: Path, repo: str, notes: int, reviews: int) -> Path:
    path = root / repo / ".autohelix"
    (path / "notes").mkdir(parents=True, exist_ok=True)
    (path / "reviews").mkdir(parents=True, exist_ok=True)
    for i in range(1, notes + 1):
        (path / "notes" / f"iter-{i}.md").write_text(f"note {i} with some words in it\n")
    for i in range(1, reviews + 1):
        (path / "reviews" / f"iter-{i}.md").write_text(f"review {i} with some words\n")
    return root / repo


GOOD_ROW = (
    "| L1 | The routed GEMM's dequant chain runs on the Vector engine at 41% utilization while the "
    "PE engine idles, because each k-tile's six stages are serialised by a buffer dependency the "
    "compiler cannot see through. | `feedback-repro/l1_dequant.py` — run with `python "
    "feedback-repro/l1_dequant.py` | The compiler schedules within a basic block and the loop "
    "boundary drains the pipeline, so every trip pays the drain rather than overlapping with the "
    "next one. | https://awsdocs-neuron.readthedocs-hosted.com/en/latest/ — no existing issue "
    "found | Expose a pipelining hint on the loop construct so a hand-skewed schedule survives "
    "the basic-block boundary. |"
)

HEADER = (
    "| Level | Scenario | Minimum reproduction | Root cause and why | Related documentation "
    "| Suggestion |\n|---|---|---|---|---|---|\n"
)


def _report(root: Path, body: str, repro: str | None = "l1_dequant.py") -> Path:
    if repro:
        (root / fb.REPRO_DIR).mkdir(parents=True, exist_ok=True)
        (root / fb.REPRO_DIR / repro).write_text("print('repro')\n")
    path = fb.report_path(root)
    path.write_text(body)
    return path


# -- reading the corpus ----------------------------------------------------------------


def test_both_repos_notes_and_reviews_are_collected(tmp_path):
    sub = _corpus(tmp_path, "rank0", notes=10, reviews=11)
    full = _corpus(tmp_path, "full", notes=5, reviews=6)
    found = fb.corpus(sub, full)
    assert set(found) == {"rank0", "full"}
    assert len(found["rank0"]) == 21
    assert len(found["full"]) == 11


def test_a_repo_that_does_not_exist_is_skipped(tmp_path):
    sub = _corpus(tmp_path, "rank0", notes=1, reviews=1)
    assert set(fb.corpus(sub, tmp_path / "missing")) == {"rank0"}


def test_iterations_are_ordered_numerically(tmp_path):
    """`iter-10.md` comes after `iter-9.md`, which a plain sort gets backwards — and the order is
    how the agent tells which of two contradicting notes is the later one."""
    repo = _corpus(tmp_path, "rank0", notes=11, reviews=0)
    names = [p.name for p in fb.corpus(repo)["rank0"]]
    assert names.index("iter-9.md") < names.index("iter-10.md")


def test_the_word_count_is_reported_so_the_prompt_can_say_how_much_there_is(tmp_path):
    repo = _corpus(tmp_path, "rank0", notes=2, reviews=0)
    assert fb.word_count(fb.corpus(repo)) == 14


def _archived_round(repo: Path, stamp: str, notes: dict[str, str]) -> Path:
    """One archived round, as `_roll_round` leaves it: a `round.json` marker beside the notes."""
    round_dir = repo / ".autohelix" / "archive" / stamp
    (round_dir / "notes").mkdir(parents=True, exist_ok=True)
    (round_dir / "round.json").write_text('{"round": 1}')
    for name, body in notes.items():
        (round_dir / "notes" / name).write_text(body)
    return round_dir


def test_an_archived_rounds_notes_are_part_of_the_corpus(tmp_path):
    """`rerun-full` restarts iteration numbering at 1, so round 2's `iter-3.md` lands on round 1's.

    This agent exists to reconcile the whole optimization history; reading only the live directories
    made it silently blind to every round but the last.
    """
    repo = _corpus(tmp_path, "full", notes=2, reviews=0)
    _archived_round(repo, "20260928-213824", {
        "iter-1.md": "round one iteration one, the lnc2 split\n",
        "iter-3.md": "round one iteration three, the accumulator cut\n",
    })
    names = [p.name for p in fb.corpus(repo)["full"]]
    bodies = "".join(p.read_text() for p in fb.corpus(repo)["full"])
    assert names.count("iter-1.md") == 2, names  # the live one and the archived one
    assert "the accumulator cut" in bodies


def test_an_archive_without_a_round_marker_is_not_read(tmp_path):
    """`autohelix clear` archives here too, and those are not rounds."""
    repo = _corpus(tmp_path, "full", notes=1, reviews=0)
    cleared = repo / ".autohelix" / "archive" / "20260901-000000" / "notes"
    cleared.mkdir(parents=True)
    (cleared / "iter-7.md").write_text("from a cleared run\n")
    assert [p.name for p in fb.corpus(repo)["full"]] == ["iter-1.md"]


def test_a_note_carried_forward_is_not_counted_twice(tmp_path):
    """A carried note lives in both the live directory and its round's archive. The corpus is fed to
    an agent by the word, and two copies invite reading one finding as two."""
    repo = _corpus(tmp_path, "full", notes=0, reviews=0)
    shared = "the very same note, byte for byte\n"
    (repo / ".autohelix" / "notes" / "iter-1-round1.md").write_text(shared)
    _archived_round(repo, "20260928-213824", {"iter-1.md": shared})
    assert len(fb.corpus(repo)["full"]) == 1


# -- the report's shape ----------------------------------------------------------------


def test_a_well_formed_report_passes(tmp_path):
    _report(tmp_path, HEADER + GOOD_ROW + "\n")
    assert fb.validate_report(tmp_path) == []


def test_a_missing_report_is_reported(tmp_path):
    assert fb.validate_report(tmp_path) == [f"{fb.REPORT_NAME} was not written"]


def test_prose_with_no_findings_table_is_refused(tmp_path):
    _report(tmp_path, "# Feedback\n\nWe found some things that were slow.\n")
    assert any("no findings table" in f for f in fb.validate_report(tmp_path))


def test_a_level_outside_the_three_is_refused(tmp_path):
    _report(tmp_path, HEADER + GOOD_ROW.replace("| L1 |", "| L7 |", 1) + "\n")
    assert any("not one of" in f for f in fb.validate_report(tmp_path))


def test_a_reproduction_that_does_not_exist_is_refused(tmp_path):
    _report(tmp_path, HEADER + GOOD_ROW + "\n", repro=None)
    (tmp_path / fb.REPRO_DIR).mkdir(exist_ok=True)
    assert any("does not exist" in f for f in fb.validate_report(tmp_path))


def test_a_row_naming_no_reproduction_is_refused(tmp_path):
    row = GOOD_ROW.replace("`feedback-repro/l1_dequant.py` — run with `python "
                           "feedback-repro/l1_dequant.py`", "see above")
    _report(tmp_path, HEADER + row + "\n")
    assert any("names no file" in f for f in fb.validate_report(tmp_path))


def test_a_documentation_cell_with_no_link_is_refused(tmp_path):
    row = GOOD_ROW.replace("https://awsdocs-neuron.readthedocs-hosted.com/en/latest/ — no "
                           "existing issue found", "the NKI docs")
    _report(tmp_path, HEADER + row + "\n")
    assert any("names no link" in f for f in fb.validate_report(tmp_path))


def test_a_scenario_too_short_to_be_read_is_refused(tmp_path):
    row = GOOD_ROW.replace(GOOD_ROW.split("|")[2], " it was slow ")
    _report(tmp_path, HEADER + row + "\n")
    assert any("'Scenario'" in f for f in fb.validate_report(tmp_path))


def test_other_tables_in_the_document_are_not_mistaken_for_findings(tmp_path):
    """The agent is asked for a summary at the top, and that is a table too."""
    summary = "| level | count |\n|---|---|\n| L1 | 1 |\n\n"
    _report(tmp_path, "# Feedback\n\n" + summary + HEADER + GOOD_ROW + "\n")
    assert fb.validate_report(tmp_path) == []


def test_the_summary_counts_findings_by_level(tmp_path):
    second = GOOD_ROW.replace("| L1 |", "| L0 |", 1)
    _report(tmp_path, HEADER + GOOD_ROW + "\n" + second + "\n")
    counts = fb.summarize(tmp_path)
    assert counts["findings"] == 2
    assert counts["by_level"] == {"L0": 1, "L1": 1, "L2": 0}


# -- the agent's own self-check --------------------------------------------------------


def test_the_check_command_exits_nonzero_on_a_broken_report(tmp_path, capsys):
    _report(tmp_path, "# Feedback\n\nnothing structured here\n")
    assert fb.main(["--check", "--root", str(tmp_path)]) == 1
    assert "problem(s)" in capsys.readouterr().out


def test_the_check_command_exits_zero_on_a_good_report(tmp_path, capsys):
    _report(tmp_path, HEADER + GOOD_ROW + "\n")
    assert fb.main(["--check", "--root", str(tmp_path)]) == 0
    assert "structure OK" in capsys.readouterr().out


# -- the prompt the stage sends ---------------------------------------------------------


def test_the_prompt_renders_with_the_variables_the_driver_supplies():
    from autohelix.prompt_template import render_template

    from optimization import presets

    rendered = render_template(presets.feedback_prompt(), {
        "module": "layers.1.ffn", "corpus": "- `a/.autohelix/` — 21 file(s)", "words": "82,708",
        "submodule_repo": "/w/rank0", "full_repo": "/w/full", "workspace_root": "/w",
        "report_path": "/w/FEEDBACK.md", "repro_dir": fb.REPRO_DIR, "context_md": "/t/CONTEXT.md",
        "levels": "- **L0** — a bug", "ranks": 4,
        "bootstrap_latency": "2793.41 ms", "submodule_latency": "33.4812 ms",
        "full_latency": "14.8425 ms",
    })
    assert "{{" not in rendered
    assert "layers.1.ffn" in rendered and "82,708" in rendered and "14.8425 ms" in rendered


# -- the review of #6, third pass -------------------------------------------------------


def test_a_table_with_no_rows_is_an_honest_answer(tmp_path):
    """Every obstacle worked around is a legitimate outcome, and the prompt says to omit those. The
    alternative is an agent choosing between inventing a row and failing the stage."""
    _report(tmp_path, HEADER)
    assert fb.validate_report(tmp_path) == []
    assert fb.summarize(tmp_path)["findings"] == 0


def test_find_table_separates_absent_from_empty():
    assert fb.find_table("nothing here") == (False, [])
    found, rows = fb.find_table(HEADER)
    assert found and rows == []


def test_the_deliverables_are_restored_if_the_agent_edits_them(tmp_path):
    """The stage has no worktree and no scope enforcement, and its prompt asks the agent to run
    experiments on the device."""
    for name in ("rank0", "full"):
        repo = tmp_path / name
        repo.mkdir()
        (repo / "source.py").write_text(f"# {name} kernel\n")
        (repo / "inference.py").write_text(f"# {name} validator\n")
    snapshot = fb.snapshot_deliverables(tmp_path / "rank0", tmp_path / "full")
    assert len(snapshot) == 4

    (tmp_path / "rank0" / "source.py").write_text("# an experiment overwrote this\n")
    touched = fb.restore_deliverables(snapshot)
    assert touched == ["rank0/source.py"]
    assert (tmp_path / "rank0" / "source.py").read_text() == "# rank0 kernel\n"


def test_untouched_deliverables_report_nothing(tmp_path):
    repo = tmp_path / "full"
    repo.mkdir()
    (repo / "source.py").write_text("# kernel\n")
    snapshot = fb.snapshot_deliverables(repo)
    assert fb.restore_deliverables(snapshot) == []
