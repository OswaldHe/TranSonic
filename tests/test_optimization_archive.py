# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The cross-run feedback archive: what is collected, what is published, and what is disputed.

The tests that matter most are the ones about *disputes*. The archive's value is that a later run
does not re-derive what an earlier one established; its risk is that a later run inherits something
an earlier one got wrong. Those pull in opposite directions, and the only thing holding them apart
is that a refutation propagates as reliably as a finding does.
"""

from __future__ import annotations

import json
import textwrap

import pytest

from optimization import archive
from optimization import feedback as fb
from optimization import memory as mem

pytestmark = pytest.mark.optimization


def _report(*rows: str) -> str:
    head = (
        "| Level | Scenario | Minimum reproduction | Root cause and why "
        "| Related documentation | Suggestion |\n"
        "|---|---|---|---|---|---|\n"
    )
    return "# What stopped it\n\n" + head + "".join(rows)


def _row(level: str, scenario: str, repro: str = "feedback-repro/01.py") -> str:
    return (f"| {level} | {scenario} | `{repro}` | because of a measured thing "
            f"| https://example.com/doc | do the other thing |\n")


@pytest.fixture
def run(tmp_path):
    """A finished run's workspace: a report with two findings and a reproduction."""
    root = tmp_path / "run-a"
    (root / fb.REPRO_DIR).mkdir(parents=True)
    (root / fb.REPRO_DIR / "01.py").write_text("# a reproduction\n")
    (root / fb.REPORT_NAME).write_text(_report(
        _row("L2", "`nisa.nc_matmul_mx` is NeuronCore-v4 and newer, so not this part"),
        _row("L1", "`tensor_reduce` never reaches a Vector performance mode"),
    ))
    return root


# -- depositing ------------------------------------------------------------------------


def test_a_deposit_carries_the_report_and_its_reproductions(tmp_path, run):
    store = tmp_path / "archive"
    entry = archive.deposit(store, run, "mtp.0.ffn", recorded="2026-10-06")

    assert entry.name == "mtp-0-ffn-2026-10-06"
    landed = store / entry.name
    assert (landed / fb.REPORT_NAME).read_text() == (run / fb.REPORT_NAME).read_text()
    assert not (landed / fb.REPRO_DIR / "01.py").exists(), "the archive carries prose only"
    assert entry.origin.endswith("run-a"), "so a reader can still reach the reproduction"
    assert [f.level for f in entry.findings] == ["L2", "L1"]
    assert (store / archive.INDEX_NAME).is_file()
    assert (store / archive.MANIFEST_NAME).is_file()


def test_a_run_with_no_report_is_refused_rather_than_filed_empty(tmp_path):
    with pytest.raises(archive.ArchiveError):
        archive.deposit(tmp_path / "archive", tmp_path, "mtp.0.ffn")


def test_only_markdown_is_carried(tmp_path, run):
    """The store is copied into every iteration worktree, so it holds what an agent reads.

    A reproduction directory is also where the feedback agent compiled and profiled: the first
    backfill of seven finished runs pulled 48 MB of `.ntff` and `.colz` out of one of them. The
    scripts, the NEFFs and the compiler logs all stay in the run's own workspace.
    """
    (run / fb.REPRO_DIR / "NOTES.md").write_text("# what the reproductions show\n")
    (run / fb.REPRO_DIR / "_work" / "02").mkdir(parents=True)
    (run / fb.REPRO_DIR / "_work" / "02" / "graph.neff").write_bytes(b"\x00" * 64)
    (run / fb.REPRO_DIR / "cafe1234").mkdir()
    (run / fb.REPRO_DIR / "cafe1234" / "source.colz").write_bytes(b"cache")
    (run / fb.REPRO_DIR / "log-neuron-cc.txt").write_text("compiler chatter")

    entry = archive.deposit(tmp_path / "archive", run, "mtp.0.ffn", recorded="2026-10-06")

    landed = tmp_path / "archive" / entry.name / fb.REPRO_DIR
    assert (landed / "NOTES.md").is_file()
    assert {p.suffix for p in landed.rglob("*") if p.is_file()} == {".md"}
    assert not (landed / "_work").exists(), "a scratch directory is not evidence"
    assert not (landed / "cafe1234").exists(), "nor is a content-hash compile cache"
    assert entry.skipped == 4, "and what was left behind is counted, not silent"

    # Read-only: the run's own workspace is never modified by being archived.
    assert (run / fb.REPRO_DIR / "01.py").is_file()
    assert (run / fb.REPRO_DIR / "_work" / "02" / "graph.neff").is_file()


def test_a_second_round_on_one_module_is_a_second_entry(tmp_path, run):
    """What the first round could not get past is part of why the second one ran."""
    store = tmp_path / "archive"
    first = archive.deposit(store, run, "mtp.0.ffn", recorded="2026-10-06")
    second = archive.deposit(store, run, "mtp.0.ffn", recorded="2026-10-20")

    assert first.name != second.name
    assert len(archive.read_entries(store)) == 2


def test_redepositing_the_same_run_updates_it_in_place(tmp_path, run):
    """Re-running `feedback` the same day corrects an entry rather than doubling it."""
    store = tmp_path / "archive"
    archive.deposit(store, run, "mtp.0.ffn", recorded="2026-10-06")
    (run / fb.REPORT_NAME).write_text(_report(_row("L0", "a third thing entirely, measured")))

    archive.deposit(store, run, "mtp.0.ffn", recorded="2026-10-06")

    entries = archive.read_entries(store)
    assert len(entries) == 1
    assert [f.level for f in entries[0].findings] == ["L0"]


# -- identity --------------------------------------------------------------------------


def test_a_findings_id_survives_rewording_that_changes_no_words():
    """Markdown emphasis and line wrapping must not make one finding into two."""
    plain = "tensor_reduce never reaches a Vector performance mode"
    assert archive.finding_id(plain) == archive.finding_id(
        "**tensor_reduce** never reaches\na Vector performance mode!")
    assert archive.finding_id(plain) != archive.finding_id("something else entirely")


# -- rulings ---------------------------------------------------------------------------


@pytest.mark.parametrize("cell, want", [
    ("disagrees", "disagrees"),
    ("agrees", "agrees"),
    ("  AGREES  ", "agrees"),
    ("partly disagrees, see the measurement", "disagrees"),
    ("extends", "extends"),
    ("not-applicable", "not-applicable"),
    ("not applicable", "not-applicable"),
    ("", ""),
    ("probably", ""),
])
def test_a_ruling_is_read_whole_word(cell, want):
    """`disagrees` **contains** `agrees`, and a substring test inverted every refutation.

    That is the worst failure this module can have: a finding another run tested and refuted would
    have been published as independently reproduced, which is precisely the inheritance the
    archive exists to prevent.
    """
    assert archive.read_ruling(cell) == want


def test_a_refutation_reaches_the_finding_it_is_about(tmp_path, run):
    store = tmp_path / "archive"
    first = archive.deposit(store, run, "mtp.0.ffn", recorded="2026-10-06")
    target = first.findings[1].id

    later = tmp_path / "run-b"
    (later / fb.REPRO_DIR).mkdir(parents=True)
    (later / fb.REPRO_DIR / "01.py").write_text("# the counter-measurement\n")
    (later / fb.REPORT_NAME).write_text(textwrap.dedent(f"""
        | Filed | Ruling | Evidence |
        |---|---|---|
        | `{target}` | disagrees | measured here at 0.27 ms, not the 2.2 ms a doubled rate implies |

    """) + _report(_row("L1", "a new obstacle nobody has filed before, measured here")))
    archive.deposit(store, later, "layers.1.ffn", recorded="2026-10-07")

    entries = archive.read_entries(store)
    by_id = {f.id: f for e in entries for f in e.findings}
    assert by_id[target].disputed
    assert by_id[target].rulings == [
        ("disagrees", "layers.1.ffn",
         "measured here at 0.27 ms, not the 2.2 ms a doubled rate implies")]
    assert "**disagrees** (layers.1.ffn)" in archive.findings_table(entries)
    # Disputed last: the list is read under a budget and a refuted claim is the least useful row.
    assert archive.all_findings(entries)[-1][1].id == target


def test_rulings_are_checked_for_being_actionable(tmp_path, run):
    store = tmp_path / "archive"
    filed = [archive.deposit(store, run, "mtp.0.ffn", recorded="2026-10-06")]

    known = filed[0].findings[0].id
    later = tmp_path / "run-b"
    later.mkdir()
    (later / fb.REPORT_NAME).write_text(textwrap.dedent(f"""
        | Filed | Ruling | Evidence |
        |---|---|---|
        | the tensor_reduce one | agrees | no id at all |
        | `ffffffffffff` | agrees | an id that is in no entry |
        | `{known}` | probably | not one of the rulings |
    """))

    problems = archive.validate_rulings(later, filed)

    assert len(problems) == 3
    assert "names no 12-character finding id" in problems[0]
    assert "no finding `ffffffffffff`" in problems[1]
    assert "not one of" in problems[2]


def test_a_report_with_no_rulings_table_is_not_a_problem(tmp_path, run):
    """Ruling on filed findings is for runs that met them; most rows are unrelated."""
    assert archive.validate_rulings(run, []) == []


# -- reading ---------------------------------------------------------------------------


def test_an_absent_archive_reads_as_empty_rather_than_failing(tmp_path):
    assert archive.read_entries(None) == []
    assert archive.read_entries(tmp_path / "nope") == []


def test_the_index_is_rebuilt_from_what_is_on_disk(tmp_path, run):
    """Generated, not appended, so an entry removed or copied in by hand is picked up."""
    store = tmp_path / "archive"
    entry = archive.deposit(store, run, "mtp.0.ffn", recorded="2026-10-06")
    assert entry.findings[0].id in (store / archive.INDEX_NAME).read_text()

    mem.rmtree_unlocked(store / entry.name)
    archive.write_index(store)

    assert entry.findings[0].id not in (store / archive.INDEX_NAME).read_text()
    assert json.loads((store / archive.MANIFEST_NAME).read_text())["entries"] == []


def test_an_unreadable_entry_is_skipped_not_fatal(tmp_path, run):
    store = tmp_path / "archive"
    archive.deposit(store, run, "mtp.0.ffn", recorded="2026-10-06")
    (store / "broken").mkdir()
    (store / "broken" / archive.ENTRY_NAME).write_text("{not json")

    assert len(archive.read_entries(store)) == 1


# -- the prompt blocks -----------------------------------------------------------------


def test_nothing_is_said_about_an_empty_archive():
    """An agent told about a directory with nothing in it has been given a distraction."""
    assert archive.describe_for_optimizer([], 3) == ""
    assert archive.describe_for_feedback([], 3) == ""
    assert archive.describe_for_optimizer([archive.Entry("e", "m", "2026-01-01")], 0) == ""


def test_the_feedback_block_names_every_ruling_the_gate_accepts(tmp_path, run):
    """`validate_rulings` refuses a word the block never offered, which would be a trap."""
    entries = [archive.deposit(tmp_path / "archive", run, "mtp.0.ffn", recorded="2026-10-06")]
    block = archive.describe_for_feedback(entries, 4)

    for ruling in archive.RULINGS:
        assert ruling in block, ruling


# -- seeding ---------------------------------------------------------------------------


def test_the_archive_and_the_operator_s_memory_seed_side_by_side(tmp_path, run):
    """Two stores, two destinations, both read-only. Merging them would lose which is which."""
    store = tmp_path / "archive"
    archive.deposit(store, run, "mtp.0.ffn", recorded="2026-10-06")
    operator = tmp_path / "memory"
    operator.mkdir()
    (operator / "NOTES.md").write_text("what the first round learned")

    worktree = tmp_path / "wt"
    worktree.mkdir()
    assert mem.seed(mem.MemorySpec(path=operator, every=True), worktree) == 1
    assert mem.seed(mem.MemorySpec(path=store, every=True, dest=archive.SEEDED_REL),
                    worktree) >= 2

    assert (worktree / mem.SEEDED_REL / "NOTES.md").is_file()
    assert (worktree / archive.SEEDED_REL / archive.INDEX_NAME).is_file()
    for rel in (mem.SEEDED_REL, archive.SEEDED_REL):
        landed = worktree / rel
        for path in [landed, *landed.rglob("*")]:
            assert not path.stat().st_mode & 0o222, f"{path} is writable"

    # And both come back, or the worktree cannot be torn down.
    mem.unlock(worktree, mem.SEEDED_REL, archive.SEEDED_REL)
    mem.rmtree_unlocked(worktree)
    assert not worktree.exists()


def test_seeding_the_archive_does_not_disturb_the_memory_already_there(tmp_path, run):
    """`seed` replaces its own destination, and only that one."""
    store = tmp_path / "archive"
    archive.deposit(store, run, "mtp.0.ffn", recorded="2026-10-06")
    operator = tmp_path / "memory"
    operator.mkdir()
    (operator / "NOTES.md").write_text("kept")

    worktree = tmp_path / "wt"
    worktree.mkdir()
    mem.seed(mem.MemorySpec(path=operator, every=True), worktree)
    mem.seed(mem.MemorySpec(path=store, every=True, dest=archive.SEEDED_REL), worktree)
    mem.seed(mem.MemorySpec(path=store, every=True, dest=archive.SEEDED_REL), worktree)

    assert (worktree / mem.SEEDED_REL / "NOTES.md").read_text() == "kept"


# -- the pipeline's own plumbing -------------------------------------------------------


def _config(tmp_path, archive_path=None):
    """A minimal pipeline config, optionally pointed at an archive."""
    from tests.test_optimization_driver import _filled
    from optimization.config import PipelineConfig

    extra = {"feedback": {"archive": str(archive_path)}} if archive_path is not None else {}
    return PipelineConfig.load(_filled(tmp_path, **extra))


def test_an_unset_archive_leaves_the_mechanism_entirely_off(tmp_path):
    """The one path this pipeline writes to outside its own workspace, so it is opt-in."""
    from optimization.driver import Pipeline

    config = _config(tmp_path)
    assert config.feedback_archive is None
    pipeline = Pipeline(config)
    assert pipeline._seed_archive(tmp_path) == ""
    pipeline._deposit_feedback()  # a no-op, not an error
    assert "feedback_archive" not in config.derive_loop_config("submodule")


def test_a_configured_archive_reaches_the_loop_through_the_derived_config(tmp_path):
    """The loop re-reads the derived file, so the path has to survive that round trip."""
    store = tmp_path / "shared-archive"
    config = _config(tmp_path, store)

    assert config.feedback_archive == store
    for stage in ("submodule", "full"):
        assert config.derive_loop_config(stage)["feedback_archive"] == str(store)


def test_a_relative_archive_resolves_against_the_config_not_the_cwd(tmp_path):
    """Several modules' configs sit together and share one archive, so relative is the common case."""
    config = _config(tmp_path, "../shared")
    assert config.feedback_archive is not None
    assert config.feedback_archive.is_absolute()
    assert config.feedback_archive.name == "shared"


def test_the_pipeline_seeds_and_then_deposits(tmp_path, run):
    from optimization.driver import Pipeline

    store = tmp_path / "shared-archive"
    archive.deposit(store, run, "some.other.module", recorded="2026-10-01")
    config = _config(tmp_path, store)
    pipeline = Pipeline(config)

    # A later run reads what is filed...
    repo = config.workspace_root
    repo.mkdir(parents=True, exist_ok=True)
    block = pipeline._seed_archive(repo)
    assert str(archive.SEEDED_REL) in block
    assert (repo / archive.SEEDED_REL / archive.INDEX_NAME).is_file()

    # ...and the copy goes away with the agent that read it, rather than into the deliverable.
    pipeline._drop_preparation_memory(repo)
    assert not (repo / archive.SEEDED_REL).exists()

    # ...then files its own, once its report exists.
    (repo / fb.REPRO_DIR).mkdir(parents=True, exist_ok=True)
    (repo / fb.REPRO_DIR / "01.py").write_text("# mine\n")
    (repo / fb.REPORT_NAME).write_text(_report(_row("L0", "a thing this run found, measured")))
    pipeline._deposit_feedback()

    modules = {e.module for e in archive.read_entries(store)}
    assert modules == {"some.other.module", config.module_id}


def test_a_backfilled_run_files_the_numbers_it_achieved(tmp_path, run):
    """`--add` on a run that finished months ago should file what the pipeline would have.

    Read from the run's own records rather than passed in, and found by glob: the caller of
    `--add` names a directory, not the module that produced it.
    """
    (run / ".optimization").mkdir()
    (run / ".optimization" / "baselines.json").write_text(json.dumps(
        {"bootstrap_latency_ms": 1105.130113232}))
    for repo, stage, best in (("mod-rank0", "submodule", 9.62531), ("mod-full", "full", 4.56132)):
        summary = run / repo / ".autohelix" / "optimization" / f"{stage}-summary.json"
        summary.parent.mkdir(parents=True)
        summary.write_text(json.dumps({"stage": stage, "best_ms": best}))

    entry = archive.deposit(tmp_path / "archive", run, "mtp.0.ffn", recorded="2026-10-06")

    assert entry.latencies == {"bootstrap_ms": 1105.130113232,
                               "submodule_ms": 9.62531, "module_ms": 4.56132}
    assert "1105.13 ms → 9.62531 ms → 4.56132 ms" in \
        (tmp_path / "archive" / archive.INDEX_NAME).read_text()


def test_the_submodule_number_falls_back_to_the_baselines_record(tmp_path, run):
    """40-DSparkAttention filed `submodule_ms: None` with the number sitting in `baselines.json`.

    Its stage 3 finished under an earlier invocation that never wrote `submodule-summary.json`,
    which is the only place this used to look.
    """
    (run / ".optimization").mkdir()
    (run / ".optimization" / "baselines.json").write_text(json.dumps(
        {"bootstrap_latency_ms": 17.617743, "submodule_latency_ms": 0.445173}))
    summary = run / "mod-full" / ".autohelix" / "optimization" / "full-summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps({"stage": "full", "best_ms": 0.487923748}))

    assert archive.measured_latencies(run) == {
        "bootstrap_ms": 17.617743, "submodule_ms": 0.445173, "module_ms": 0.487923748}


def test_stage_threes_own_summary_wins_over_the_baseline(tmp_path, run):
    """The baseline is what stage 4 measured once; the summary is what stage 3 settled on."""
    (run / ".optimization").mkdir()
    (run / ".optimization" / "baselines.json").write_text(json.dumps(
        {"submodule_latency_ms": 0.445173}))
    summary = run / "mod-rank0" / ".autohelix" / "optimization" / "submodule-summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps({"stage": "submodule", "best_ms": 0.441002}))

    assert archive.measured_latencies(run)["submodule_ms"] == 0.441002


def test_a_run_with_no_records_files_no_numbers(tmp_path, run):
    """A hand-assembled entry is still a legal entry; the chain just reads as unknown."""
    entry = archive.deposit(tmp_path / "archive", run, "mtp.0.ffn", recorded="2026-10-06")
    assert set(entry.latencies.values()) == {None}
