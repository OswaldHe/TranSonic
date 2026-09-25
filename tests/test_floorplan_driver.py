# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The driver: the manifest, the freeze, the candidate archive, and the blind sandbox.

The archive and the sandbox get the most attention here because both exist to hold a property
that is easy to state and easy to break silently. The archive has to keep a scheme the metric
gate rejected, or step 4 can have nothing to rank; the sandbox has to *not* contain the
simulator, and a copy added later would leak the answer without anything failing.
"""

from __future__ import annotations

import json

import pytest

from floorplan import checker, driver, rank as rank_module

pytestmark = pytest.mark.floorplan


# ---------------------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------------------
def test_manifest_round_trips(tmp_path):
    manifest = driver.Manifest(
        artifact="/a", target="trn2-16device", systems_dir="/s", created="now",
    )
    manifest.save(tmp_path)
    again = driver.Manifest.load(tmp_path)
    assert (again.artifact, again.target, again.systems_dir) == ("/a", "trn2-16device", "/s")
    assert again.hashes == {}


def test_loading_a_non_project_says_so(tmp_path):
    with pytest.raises(driver.DriverError, match="not a floorplan project"):
        driver.Manifest.load(tmp_path)


def test_freeze_records_hashes_and_a_timestamp(tmp_path):
    driver.Manifest(
        artifact="/a", target="t", systems_dir="/s", created="now",
    ).save(tmp_path)
    (tmp_path / "sim").mkdir()
    (tmp_path / "sim" / "engine.py").write_text("x = 1\n")
    (tmp_path / "systems").mkdir()
    (tmp_path / "systems" / "probed.yaml").write_text("efficiency: {}\n")
    for path in checker.framework_paths():
        mirror = tmp_path / "sim" / "framework"
        mirror.mkdir(parents=True, exist_ok=True)
        (mirror / path.name).write_bytes(path.read_bytes())

    frozen = driver.freeze(tmp_path, iterations=2)
    assert {"sim/engine.py", "systems/probed.yaml"} <= set(frozen.hashes)
    assert frozen.frozen_at
    assert frozen.build_iterations == 2
    # The framework that actually computes the metrics is hashed too, by absolute path.
    assert frozen.framework_hashes
    assert all(path.endswith(".py") for path in frozen.framework_hashes)
    assert any(path.endswith("sim/runner.py") for path in frozen.framework_hashes)
    # And the gate agrees nothing has changed.
    assert checker.check_frozen(tmp_path, {
        "hashes": frozen.hashes, "framework_hashes": frozen.framework_hashes,
    }).passed


# ---------------------------------------------------------------------------------------
# The candidate archive
# ---------------------------------------------------------------------------------------
def _archive(project, iteration, metrics, accepted=True):
    (project / "floorplan.yaml").write_text(
        f"version: 1\ntarget: trn2-16device\nnotes: iteration {iteration}\n"
    )
    return driver.archive_candidate(project, iteration, metrics, accepted)


def test_a_rejected_but_feasible_scheme_is_still_archived(tmp_path):
    """The metric gate decides what the next iteration builds on, not what is kept.

    Five iterations under a four-metric ratchet can plausibly accept one and reject four; if
    the archive only held accepted ones, step 4 would have nothing to rank.
    """
    metrics = {"prefill_128_b1_ms": 10.0, "decode_128_b1_ms": 2.0}
    path = _archive(tmp_path, 3, metrics, accepted=False)
    assert path.exists()
    payload = json.loads(path.with_suffix(".json").read_text())
    assert payload["accepted"] is False
    assert payload["metrics"] == metrics
    assert "rejected by the metric gate" in path.read_text()


def test_archived_candidates_carry_their_metrics_in_the_header(tmp_path):
    _archive(tmp_path, 1, {"decode_8192_b1_ms": 42.5})
    text = (tmp_path / driver.CANDIDATES_DIR / "iter-1.yaml").read_text()
    assert "decode_8192_b1_ms: 42.5000" in text
    assert "target: trn2-16device" in text


def test_candidates_are_listed_in_iteration_order(tmp_path):
    for iteration in (4, 1, 2):
        _archive(tmp_path, iteration, {"prefill_128_b1_ms": float(iteration)})
    assert [entry["iteration"] for entry in driver.candidates(tmp_path)] == [1, 2, 4]


# ---------------------------------------------------------------------------------------
# Ranking by score
# ---------------------------------------------------------------------------------------
def test_score_rewards_the_frontier_not_the_sum():
    """A sum of milliseconds would be dominated by prefill_8192 and ignore decode."""
    best = {
        "prefill_128_b1_ms": 10.0, "prefill_8192_b1_ms": 1000.0,
        "decode_128_b1_ms": 1.0, "decode_8192_b1_ms": 2.0,
    }
    balanced = dict(best)
    # Wins big on the largest metric, loses badly on both decode metrics.
    lopsided = {
        "prefill_128_b1_ms": 10.0, "prefill_8192_b1_ms": 500.0,
        "decode_128_b1_ms": 4.0, "decode_8192_b1_ms": 8.0,
    }
    assert driver.rank_score(balanced, best) == pytest.approx(1.0)
    assert driver.rank_score(lopsided, best) > driver.rank_score(balanced, best)


def test_score_ignores_missing_metrics_rather_than_crashing():
    best = {"prefill_128_b1_ms": 10.0, "decode_128_b1_ms": 1.0}
    assert driver.rank_score({"prefill_128_b1_ms": 10.0}, best) == pytest.approx(1.0)
    assert driver.rank_score({}, best) == float("inf")


def test_top_schemes_picks_the_three_best(tmp_path):
    for iteration, decode in enumerate([5.0, 1.0, 3.0, 2.0, 9.0], start=1):
        _archive(tmp_path, iteration, {"decode_128_b1_ms": decode,
                                      "prefill_128_b1_ms": 10.0})
    top = driver.top_schemes(tmp_path, count=3)
    assert [entry["iteration"] for entry in top] == [2, 4, 3]
    assert top[0]["score"] < top[1]["score"] < top[2]["score"]


def test_top_schemes_reports_honestly_on_fewer_than_three(tmp_path):
    _archive(tmp_path, 1, {"decode_128_b1_ms": 1.0})
    assert len(driver.top_schemes(tmp_path, count=3)) == 1
    assert driver.top_schemes(tmp_path / "empty", count=3) == []


# ---------------------------------------------------------------------------------------
# The blind sandbox
# ---------------------------------------------------------------------------------------
@pytest.fixture
def project(tmp_path):
    """A project with three archived candidates and a full complement of briefs."""
    driver.Manifest(
        artifact="/artifact", target="trn2-16device", systems_dir=str(tmp_path / "systems"),
        created="now",
    ).save(tmp_path)
    for name in ("PLATFORM.md", "MODEL.md", "README.md"):
        (tmp_path / name).write_text(f"# {name}\n")
    (tmp_path / "systems").mkdir()
    (tmp_path / "systems" / "trn2-16device.yaml").write_text("name: trn2-16device\n")
    (tmp_path / "systems" / "probed.yaml").write_text(
        "_provenance:\n"
        "  probed_on: trn2.3xlarge\n"
        "  probed_at: now\n"
        "  not_probeable: [links.inter_device.bandwidth_bytes_per_s]\n"
        "shared:\n"
        "  efficiency: {matmul_bf16: 0.376}\n"
        "  links:\n"
        "    intra_device:\n"
        "      source: derived_from_probe\n"
        "      derived_from: hbm bandwidth x dma efficiency\n"
        "      note: NOT MEASURED.\n"
    )
    (tmp_path / "sim").mkdir()
    (tmp_path / "sim" / "engine.py").write_text("# the simulator\n")
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "trace.json").write_text('{"secret": true}')
    for iteration, decode in enumerate([3.0, 1.0, 2.0], start=1):
        _archive(tmp_path, iteration, {"decode_128_b1_ms": decode,
                                      "prefill_128_b1_ms": 10.0})
    return tmp_path


def test_the_sandbox_does_not_contain_the_simulator(project, tmp_path):
    staged = rank_module.stage(project, tmp_path / "sandbox", seed=0)
    present = {p.name for p in staged.directory.rglob("*")}
    assert "engine.py" not in present
    assert "trace.json" not in present
    assert "probed.yaml" not in present, "the probe overlay is the cost model's calibration"
    assert "trn2-16device.yaml" in present, "the datasheet is allowed"
    assert {"TASK.md", "PLATFORM.md", "MODEL.md"} <= present


def test_schemes_are_anonymized_and_carry_no_metrics(project, tmp_path):
    staged = rank_module.stage(project, tmp_path / "sandbox", seed=0)
    names = sorted(p.name for p in (staged.directory / "schemes").glob("*.yaml"))
    assert names == ["scheme-a.yaml", "scheme-b.yaml", "scheme-c.yaml"]
    for path in (staged.directory / "schemes").glob("*.yaml"):
        text = path.read_text()
        assert "decode_128_b1_ms" not in text, "the archive header leaks the metrics"
        assert "iteration" not in text.lower() or "arbitrary" in text
        assert "target: trn2-16device" in text


def test_the_label_order_carries_no_signal(project, tmp_path):
    """`scheme-a` must not be the simulator's favourite, or the blinding buys nothing."""
    assignments = set()
    for seed in range(12):
        staged = rank_module.stage(project, tmp_path / f"sandbox{seed}", seed=seed)
        best = min(staged.mapping, key=lambda label: staged.mapping[label]["score"])
        assignments.add(best)
    assert len(assignments) > 1, "the best scheme always landed on the same label"


def test_the_key_is_written_outside_the_sandbox(project, tmp_path):
    staged = rank_module.stage(project, tmp_path / "sandbox", seed=0)
    key = tmp_path / "key.json"
    staged.save_key(key)
    assert key not in set(staged.directory.rglob("*"))
    payload = json.loads(key.read_text())
    assert set(payload) == {"scheme-a", "scheme-b", "scheme-c"}
    assert all("metrics" in entry for entry in payload.values())


def test_staging_without_candidates_says_what_produces_them(tmp_path):
    driver.Manifest(
        artifact="/a", target="t", systems_dir="/s", created="now",
    ).save(tmp_path)
    with pytest.raises(RuntimeError, match="floorplan run.* archives"):
        rank_module.stage(tmp_path, tmp_path / "sandbox")


# ---------------------------------------------------------------------------------------
# Parsing the ranking, and the appendix
# ---------------------------------------------------------------------------------------
def test_ranking_is_read_from_first_mention(tmp_path):
    report = tmp_path / "REPORT.md"
    report.write_text(
        "# Ranking\n\nThe best deployment is **scheme-b**. It wins because...\n\n"
        "Second is scheme-c, which trades...\n\nLast, scheme-a suffers from...\n"
    )
    assert rank_module.parse_ranking(report, ["scheme-a", "scheme-b", "scheme-c"]) == [
        "scheme-b", "scheme-c", "scheme-a",
    ]


def test_an_unmentioned_scheme_goes_last(tmp_path):
    report = tmp_path / "REPORT.md"
    report.write_text("scheme-c is best, then scheme-a.\n")
    assert rank_module.parse_ranking(report, ["scheme-a", "scheme-b", "scheme-c"])[-1] == (
        "scheme-b"
    )


def test_appendix_is_appended_and_flags_disagreement(project, tmp_path):
    staged = rank_module.stage(project, tmp_path / "sandbox", seed=0)
    key = tmp_path / "key.json"
    staged.save_key(key)
    report = staged.directory / "REPORT.md"

    # Rank them in the reverse of the simulator's order, so the two must disagree.
    simulator_order = sorted(staged.mapping, key=lambda l: staged.mapping[l]["score"])
    report.write_text(
        "The best is " + simulator_order[-1] + ", then " + simulator_order[1]
        + ", then " + simulator_order[0] + ".\n"
    )
    appendix = rank_module.attach_measurements(
        report, key, project, list(reversed(simulator_order)),
    )
    assert "Appendix: the simulator's numbers" in appendix
    assert "disagree on the winner" in appendix
    assert "decode_128_b1_ms" in appendix
    assert "uncalibrated" in appendix.lower()
    # And it landed in the file rather than only being returned.
    assert appendix in report.read_text()


def test_appendix_notes_agreement_when_the_winner_matches(project, tmp_path):
    staged = rank_module.stage(project, tmp_path / "sandbox", seed=0)
    key = tmp_path / "key.json"
    staged.save_key(key)
    report = staged.directory / "REPORT.md"
    report.write_text("nothing useful\n")
    simulator_order = sorted(staged.mapping, key=lambda l: staged.mapping[l]["score"])
    appendix = rank_module.attach_measurements(report, key, project, simulator_order)
    assert "same winner" in appendix
    assert "3 of 3 positions agree" in appendix


def test_appendix_surfaces_the_derived_link_as_a_caveat(project, tmp_path):
    staged = rank_module.stage(project, tmp_path / "sandbox", seed=0)
    key = tmp_path / "key.json"
    staged.save_key(key)
    report = staged.directory / "REPORT.md"
    report.write_text("scheme-a is best.\n")
    appendix = rank_module.attach_measurements(report, key, project)
    assert "derived, not measured" in appendix
    assert "first thing to measure on real 16-device hardware" in appendix
    assert "links.inter_device" in appendix


# ---------------------------------------------------------------------------------------
# The reviewer is read-only (PR #5 review, comment 9)
# ---------------------------------------------------------------------------------------
def test_reviewer_edits_to_the_simulator_are_reverted(tmp_path):
    """The reviewer runs after the invariants and shares the writable project.

    Nothing stops it editing `sim/` too, and those edits would land after the suite had passed,
    be committed, and be frozen on the strength of a green result describing different code.
    """
    (tmp_path / "sim" / "modules").mkdir(parents=True)
    model = tmp_path / "sim" / "modules" / "attention.py"
    model.write_text("ORIGINAL = 1\n")
    constraints = tmp_path / "sim" / "constraints.py"
    constraints.write_text("def check(hardware, plan):\n    return []\n")

    snapshot = driver._snapshot_sim(tmp_path)

    # The reviewer edits a cost model, adds a file, and deletes another.
    model.write_text("ORIGINAL = 999\n")
    (tmp_path / "sim" / "modules" / "sneaky.py").write_text("X = 1\n")
    constraints.unlink()

    reverted = driver._restore_sim(tmp_path, snapshot)
    assert model.read_text() == "ORIGINAL = 1\n"
    assert constraints.exists()
    assert not (tmp_path / "sim" / "modules" / "sneaky.py").exists()
    assert len(reverted) == 3


def test_an_untouched_simulator_reports_nothing_reverted(tmp_path):
    (tmp_path / "sim").mkdir()
    (tmp_path / "sim" / "m.py").write_text("A = 1\n")
    snapshot = driver._snapshot_sim(tmp_path)
    assert driver._restore_sim(tmp_path, snapshot) == []


# ---------------------------------------------------------------------------------------
# Verdict parsing, for the freeze condition (comment 10)
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("review,expected", [
    ("all good\n\nVERDICT: clean\n", "clean"),
    ("hmm\nVERDICT: suspicious", "suspicious"),
    ("VERDICT: circumventing\n", "circumventing"),
    ("no verdict here", ""),
])
def test_verdict_is_parsed_from_the_last_verdict_line(review, expected):
    assert driver._parse_verdict(review) == expected


def test_build_refuses_to_freeze_a_circumventing_final_iteration():
    """Exhausting the iteration budget must not promote a rejected cost model.

    `build_loop` moves on when a reviewer says `circumventing`, but the last allowed iteration
    has nowhere to move on to — and checking only the invariant flag would freeze it anyway.
    """
    source = (
        suite_path := __import__("pathlib").Path(__file__).resolve().parents[1]
        / "floorplan" / "cli.py"
    ).read_text()
    assert 'final.verdict == "circumventing"' in source, (
        f"{suite_path} no longer gates the freeze on the reviewer's verdict"
    )
    assert "NOT frozen" in source


# ---------------------------------------------------------------------------------------
# Explicit ranking line (PR #5 review, comment 16)
# ---------------------------------------------------------------------------------------
def test_an_explicit_ranking_line_beats_prose_order(tmp_path):
    """The failure mode: an ordinary report introduces all three, then chooses.

    First-mention order would publish introduction order as the ranking, contradicting the
    report's own conclusion in the files it writes.
    """
    report = tmp_path / "REPORT.md"
    report.write_text(
        "# Analysis\n\n"
        "We consider scheme-a, scheme-b and scheme-c in turn.\n\n"
        "scheme-a pipelines deeply. scheme-b tiers Engram. scheme-c replicates experts.\n\n"
        "## Conclusion\n\nscheme-c is the best deployment.\n\n"
        "RANKING: scheme-c > scheme-b > scheme-a\n"
    )
    labels = ["scheme-a", "scheme-b", "scheme-c"]
    assert rank_module.parse_ranking(report, labels) == ["scheme-c", "scheme-b", "scheme-a"]
    assert rank_module.ranking_was_explicit(report)


@pytest.mark.parametrize("line", [
    "RANKING: scheme-b > scheme-a > scheme-c",
    "ranking: scheme-b, scheme-a, scheme-c",
    "RANKING:  scheme-b -> scheme-a -> scheme-c ",
    "**RANKING**: scheme-b > scheme-a > scheme-c",
])
def test_ranking_line_separators_and_case(tmp_path, line):
    report = tmp_path / "REPORT.md"
    report.write_text(f"prose mentioning scheme-c first\n\n{line}\n")
    order = rank_module.parse_ranking(report, ["scheme-a", "scheme-b", "scheme-c"])
    assert order[:2] == ["scheme-b", "scheme-a"]


def test_a_ranking_line_omitting_a_scheme_appends_it(tmp_path):
    report = tmp_path / "REPORT.md"
    report.write_text("scheme-a is mentioned first.\n\nRANKING: scheme-c > scheme-b\n")
    order = rank_module.parse_ranking(report, ["scheme-a", "scheme-b", "scheme-c"])
    assert order == ["scheme-c", "scheme-b", "scheme-a"]


def test_no_ranking_line_falls_back_and_says_so(tmp_path):
    report = tmp_path / "REPORT.md"
    report.write_text("scheme-b wins, then scheme-a, then scheme-c.\n")
    assert not rank_module.ranking_was_explicit(report)
    assert rank_module.parse_ranking(report, ["scheme-a", "scheme-b", "scheme-c"]) == [
        "scheme-b", "scheme-a", "scheme-c",
    ]


def test_the_brief_demands_the_ranking_line(project, tmp_path):
    staged = rank_module.stage(project, tmp_path / "sandbox", seed=0)
    task = (staged.directory / "TASK.md").read_text()
    assert "RANKING:" in task
    assert "parsed" in task


# ---------------------------------------------------------------------------------------
# --count is honoured (comment 17)
# ---------------------------------------------------------------------------------------
def test_count_limits_how_many_schemes_are_staged(project, tmp_path):
    for count in (1, 2, 3):
        staged = rank_module.stage(project, tmp_path / f"s{count}", seed=0, count=count)
        assert len(staged.mapping) == count
        assert len(list((staged.directory / "schemes").glob("*.yaml"))) == count


def test_an_unsupported_count_is_rejected_rather_than_ignored(project, tmp_path):
    for count in (0, 4, -1):
        with pytest.raises(ValueError, match="count must be between"):
            rank_module.stage(project, tmp_path / "bad", seed=0, count=count)


# ---------------------------------------------------------------------------------------
# Per-configuration rankings
# ---------------------------------------------------------------------------------------
def test_per_configuration_rankings_are_parsed(tmp_path):
    report = tmp_path / "REPORT.md"
    report.write_text(
        "RANKING: scheme-b > scheme-a > scheme-c\n\n"
        "RANKING[prefill_128_b1]: scheme-a > scheme-b > scheme-c\n"
        "**RANKING[decode_8192_b32]**: scheme-c, scheme-b, scheme-a\n"
        "RANKING[ decode_128_b1 ]: scheme-b -> scheme-c -> scheme-a\n"
    )
    labels = ["scheme-a", "scheme-b", "scheme-c"]
    per_config = rank_module.parse_per_config_rankings(report, labels)
    assert per_config["prefill_128_b1"] == ["scheme-a", "scheme-b", "scheme-c"]
    assert per_config["decode_8192_b32"] == ["scheme-c", "scheme-b", "scheme-a"]
    assert per_config["decode_128_b1"] == ["scheme-b", "scheme-c", "scheme-a"]
    # The bracketed lines must not be mistaken for the overall one.
    assert rank_module.parse_ranking(report, labels) == ["scheme-b", "scheme-a", "scheme-c"]


def test_a_report_with_no_per_configuration_lines_yields_nothing(tmp_path):
    """An absent opinion is recorded as absent, not inferred from the overall order."""
    report = tmp_path / "REPORT.md"
    report.write_text("RANKING: scheme-a > scheme-b > scheme-c\n")
    assert rank_module.parse_per_config_rankings(report, ["scheme-a", "scheme-b"]) == {}


def test_simulator_ranking_for_one_metric_orders_by_that_metric():
    key = {
        "scheme-a": {"metrics": {"decode_128_b1_ms": 9.0, "prefill_8192_b32_ms": 900.0}},
        "scheme-b": {"metrics": {"decode_128_b1_ms": 7.0, "prefill_8192_b32_ms": 1200.0}},
        "scheme-c": {"metrics": {"decode_128_b1_ms": 11.0}},
    }
    labels = ["scheme-a", "scheme-b", "scheme-c"]
    assert rank_module.simulator_ranking_for(key, "decode_128_b1_ms", labels) == [
        "scheme-b", "scheme-a", "scheme-c",
    ]
    # A scheme missing the metric sorts last rather than crashing.
    assert rank_module.simulator_ranking_for(key, "prefill_8192_b32_ms", labels)[-1] == "scheme-c"


def test_metrics_are_reported_in_workload_order_not_alphabetical():
    """Alphabetical interleaves b1, b32, b4, b8 — the one order that hides a batch sweep."""
    from floorplan.sim.runner import WORKLOADS

    key = {"scheme-a": {"metrics": {f"{w.name}_ms": 1.0 for w in WORKLOADS}}}
    ordered = rank_module._ordered_metrics(key)
    assert ordered == [f"{w.name}_ms" for w in WORKLOADS]
    assert ordered[:4] == [
        "prefill_128_b1_ms", "prefill_128_b4_ms", "prefill_128_b8_ms", "prefill_128_b32_ms",
    ]


def test_the_appendix_contains_a_per_configuration_table(project, tmp_path):
    staged = rank_module.stage(project, tmp_path / "sandbox", seed=0)
    key = tmp_path / "key.json"
    staged.save_key(key)
    report = staged.directory / "REPORT.md"
    order = sorted(staged.mapping)
    report.write_text(
        f"RANKING: {' > '.join(order)}\n\n"
        f"RANKING[decode_128_b1]: {' > '.join(reversed(order))}\n"
    )
    appendix = rank_module.attach_measurements(report, key, project, order)
    assert "Per-configuration rankings" in appendix
    assert "decode_128_b1" in appendix
    assert "called 1 of" in appendix


def test_the_brief_asks_for_a_ranking_per_configuration(project, tmp_path):
    from floorplan.sim.runner import WORKLOADS

    staged = rank_module.stage(project, tmp_path / "sandbox", seed=0)
    task = (staged.directory / "TASK.md").read_text()
    assert "RANKING[" in task
    for point in WORKLOADS:
        assert point.name in task, f"TASK.md does not list {point.name}"
