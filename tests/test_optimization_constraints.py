# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The per-iteration constraint schedule.

Two things here are worth failing a test over. Overlapping ranges, because one iteration under two
constraints means the loop silently picks one and the operator believes both applied. And the
fail-closed direction on a broken checker: a checker that crashes without writing a verdict must
read as a violation, because the alternative is a schedule that quietly stops constraining anything
while the prompts keep claiming it does.
"""

from __future__ import annotations

import json

import pytest

from optimization import constraints as cons

pytestmark = pytest.mark.optimization


# -- parsing ---------------------------------------------------------------------------


def test_the_shipped_schedule_shape_parses():
    schedule = cons.Schedule.from_config([
        {"from": 1, "to": 3, "text": "NKI only."},
        {"from": 4, "to": 6, "text": "NKI and torch."},
        {"from": 7, "to": 8, "text": ""},
        {"from": 9, "to": 10, "text": "NKI and torch."},
    ], max_iterations=10)
    assert [s.label for s in schedule.slots] == ["1-3", "4-6", "7-8", "9-10"]
    assert schedule.covered() == set(range(1, 11))
    assert [s.label for s in schedule.enforceable()] == ["1-3", "4-6", "9-10"]


def test_explicit_iteration_lists_work_too():
    schedule = cons.Schedule.from_config([{"iterations": [2, 4, 6], "text": "x"}])
    assert schedule.slots[0].label == "2,4,6"
    assert schedule.slot_for(4) is not None
    assert schedule.slot_for(3) is None


def test_an_absent_block_is_a_valid_empty_schedule():
    """Stage 5 ships with no ranges, and must not need any."""
    schedule = cons.Schedule.from_config(None)
    assert schedule.slots == []
    assert schedule.slot_for(1) is None
    assert "No additional constraint" in schedule.describe_for_prompt(1)


def test_overlapping_ranges_are_refused():
    with pytest.raises(cons.ScheduleError, match="already governed"):
        cons.Schedule.from_config([
            {"from": 1, "to": 4, "text": "a"},
            {"from": 3, "to": 6, "text": "b"},
        ])


def test_a_slot_beyond_the_budget_is_refused():
    """A slot that never runs is a constraint the operator will believe was applied."""
    with pytest.raises(cons.ScheduleError, match="budget.iterations"):
        cons.Schedule.from_config([{"from": 1, "to": 12, "text": "a"}], max_iterations=10)


def test_iteration_zero_is_refused():
    with pytest.raises(cons.ScheduleError, match="baseline"):
        cons.Schedule.from_config([{"iterations": [0, 1], "text": "a"}])


def test_a_backwards_range_is_refused():
    with pytest.raises(cons.ScheduleError, match="before"):
        cons.Schedule.from_config([{"from": 6, "to": 3, "text": "a"}])


def test_unknown_keys_are_refused():
    with pytest.raises(cons.ScheduleError, match="unknown key"):
        cons.Schedule.from_config([{"from": 1, "to": 2, "txt": "typo"}])


def test_a_gap_in_coverage_warns_but_is_legal():
    schedule = cons.Schedule.from_config([{"from": 1, "to": 3, "text": "a"}], max_iterations=10)
    warnings = schedule.validate(10)
    assert any("4, 5, 6" in w for w in warnings)


def test_an_empty_slot_does_not_warn_about_enforcement():
    """A textless slot is how "explore freely here" is written; it is not a mistake."""
    schedule = cons.Schedule.from_config([{"from": 7, "to": 8, "text": ""}], max_iterations=8)
    assert not any("enforce" in w for w in schedule.validate(8))


def test_an_explicit_enforce_with_no_text_does_warn():
    schedule = cons.Schedule.from_config(
        [{"from": 7, "to": 8, "text": "", "enforce": True}], max_iterations=8,
    )
    assert any("nothing to check" in w for w in schedule.validate(8))


def test_an_unfilled_hint_left_in_a_slot_is_reported():
    """A `#` inside a `text: |` block is prompt content, not a YAML comment."""
    schedule = cons.Schedule.from_config(
        [{"from": 1, "to": 3, "text": "NKI only.\n\n# <FILL IN: guidance, if any.>"}],
        max_iterations=3,
    )
    assert any("placeholder" in w for w in schedule.validate(3))


def test_the_shipped_template_leaks_no_hint_into_a_prompt():
    """The first template put its fill-in hints inside the blocks, so a config used as delivered
    sent "<FILL IN: module-specific guidance ...>" to the agent as part of its constraint."""
    import yaml

    from optimization import presets

    data = yaml.safe_load(presets.config_template())
    for stage in ("submodule", "full"):
        schedule = cons.Schedule.from_config(
            data[stage].get("iteration_constraints"), max_iterations=10,
        )
        assert schedule.validate(10) == []
        for slot in schedule.slots:
            assert cons.PLACEHOLDER_MARKER not in slot.text


# -- the prompt's view -----------------------------------------------------------------


def test_the_prompt_states_the_consequence_of_violating():
    """A constraint the agent reads as advice is one it will trade for a faster kernel."""
    schedule = cons.Schedule.from_config([{"from": 1, "to": 3, "text": "NKI only."}])
    text = schedule.describe_for_prompt(2)
    assert "NKI only." in text
    assert "rejected" in text and "discarded" in text
    assert "script you cannot see" in text


def test_an_unenforced_slot_does_not_threaten_rejection():
    schedule = cons.Schedule.from_config(
        [{"from": 1, "to": 2, "text": "Guidance only.", "enforce": False}],
    )
    text = schedule.describe_for_prompt(1)
    assert "Guidance only." in text
    assert "rejected" not in text


# -- verdicts --------------------------------------------------------------------------


def test_a_reported_violation_is_read_from_the_json(tmp_path):
    report = tmp_path / "iter-1.json"
    report.write_text(json.dumps({"passed": False, "findings": ["source.py:4 imports torch"]}))
    verdict = cons.read_slot_verdict(1, "1-3", report, "", return_code=1)
    assert verdict.checked and not verdict.passed
    assert "imports torch" in verdict.summary()


def test_a_pass_needs_both_the_json_and_a_zero_exit(tmp_path):
    report = tmp_path / "iter-1.json"
    report.write_text(json.dumps({"passed": True, "findings": []}))
    assert cons.read_slot_verdict(1, "1-3", report, "", return_code=0).passed
    # A checker claiming success while exiting non-zero is not believed.
    assert not cons.read_slot_verdict(1, "1-3", report, "", return_code=1).passed


def test_a_crashed_checker_fails_closed(tmp_path):
    """No verdict written and a non-zero exit reads as a violation, with the traceback attached.

    The safe direction: the alternative is a schedule that silently stops constraining anything.
    """
    verdict = cons.read_slot_verdict(
        1, "1-3", tmp_path / "missing.json", "Traceback...\nValueError: boom", return_code=1,
    )
    assert verdict.checked and not verdict.passed
    assert any("wrote no verdict" in f for f in verdict.findings)


# -- compiled checkers -----------------------------------------------------------------


def _checker(body: str) -> str:
    return (
        "import argparse, json\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('--repo'); p.add_argument('--json')\n"
        "a = p.parse_args()\n"
        f"{body}\n"
    )


def test_a_usable_checker_validates(tmp_path):
    path = tmp_path / "slot-1-3.py"
    path.write_text(_checker(
        "open(a.json, 'w').write(json.dumps({'passed': True, 'findings': []}))"
    ))
    assert cons.validate_checker_source(path) == []


def test_a_checker_that_does_not_parse_is_caught(tmp_path):
    path = tmp_path / "slot-1-3.py"
    path.write_text("def broken(:\n")
    assert any("does not parse" in f for f in cons.validate_checker_source(path))


def test_a_checker_that_ignores_the_arguments_is_caught(tmp_path):
    """The loop drives it with --repo and --json; one that takes neither cannot be driven."""
    path = tmp_path / "slot-1-3.py"
    path.write_text("print('hello')\n")
    findings = cons.validate_checker_source(path)
    assert any("--repo" in f for f in findings)
    assert any("--json" in f for f in findings)


def test_a_checker_that_writes_no_verdict_is_caught(tmp_path):
    path = tmp_path / "slot-1-3.py"
    path.write_text(_checker("print(a.repo, a.json)"))
    assert any("passed" in f and "findings" in f for f in cons.validate_checker_source(path))


def test_a_missing_checker_is_caught(tmp_path):
    assert cons.validate_checker_source(tmp_path / "nope.py") == ["nope.py was not written"]


def test_the_manifest_detects_an_edited_checker(tmp_path):
    schedule = cons.Schedule.from_config([{"from": 1, "to": 3, "text": "NKI only."}])
    slot = schedule.slots[0]
    path = cons.checker_path(tmp_path, slot)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_checker("open(a.json,'w').write(json.dumps({'passed':True,'findings':[]}))"))
    compiled = cons.CompiledSlot(slot.label, slot.iterations, path, cons.sha256_file(path))
    cons.write_manifest(tmp_path, [compiled], schedule)

    assert cons.verify_manifest(tmp_path) == []
    path.write_text(path.read_text() + "\n# loosened\n")
    assert any("was edited after compilation" in f for f in cons.verify_manifest(tmp_path))
    path.unlink()
    assert any("is missing" in f for f in cons.verify_manifest(tmp_path))


def test_the_manifest_records_the_prose_beside_the_hashes(tmp_path):
    """The pair is the claim "these scripts implement this prose"; a report shows both."""
    schedule = cons.Schedule.from_config([{"from": 1, "to": 3, "text": "NKI only, no torch."}])
    cons.write_manifest(tmp_path, [], schedule)
    manifest = cons.read_manifest(tmp_path)
    assert manifest["schedule"][0]["text"] == "NKI only, no torch."


def test_checkers_live_outside_what_the_worktree_is_seeded_with():
    """The compiled checkers must not land in a directory `prepare_worktree` copies.

    `bootstrap` and `floorplan` both hide their gates; this is the same property, and it is one
    string in `sandbox.py` away from silently breaking.
    """
    seeded = {"notes", "observations", "logs", "peer_notes"}
    parts = cons.CONSTRAINTS_REL.parts
    assert parts[0] == ".autohelix"
    assert parts[1] not in seeded


# -- the end-of-slot escape --------------------------------------------------------


def test_the_last_iteration_of_a_slot_is_identified():
    schedule = cons.Schedule.from_config([
        {"from": 1, "to": 3, "text": "a"},
        {"iterations": [5, 7, 9], "text": "b"},
    ])
    assert schedule.slots[0].last_iteration == 3
    assert schedule.slots[1].last_iteration == 9


def test_a_single_iteration_slot_is_its_own_last():
    schedule = cons.Schedule.from_config([{"iterations": [4], "text": "a"}])
    assert schedule.slots[0].last_iteration == 4


def test_the_prompt_threatens_rejection_inside_a_slot():
    schedule = cons.Schedule.from_config([{"from": 1, "to": 3, "text": "NKI only."}])
    for iteration in (1, 2):
        text = schedule.describe_for_prompt(iteration)
        assert "rejected and its work discarded" in text
        assert "strictly faster" not in text


def test_the_prompt_offers_the_escape_on_the_last_iteration():
    """The agent has to know the terms, or it will follow the constraint into a dead end."""
    schedule = cons.Schedule.from_config([{"from": 1, "to": 3, "text": "NKI only."}])
    text = schedule.describe_for_prompt(3)
    assert "no longer fatal" in text
    assert "strictly faster" in text
    assert "5%" in text
    assert "rejected and its work discarded" not in text


def test_the_checker_command_carries_the_advisory_flag():
    enforcing = cons.CHECKER_COMMAND.format(checker="/c.py", report="/r.json", advisory="")
    advisory = cons.CHECKER_COMMAND.format(
        checker="/c.py", report="/r.json", advisory=" --advisory",
    )
    assert "optimization.slotcheck" in enforcing
    assert "--advisory" not in enforcing
    assert advisory.endswith("--advisory")


# -- the review of #6: the checker's read-only contract, checked statically -------------


def _write(tmp_path, body: str):
    """A checker that already satisfies the structural checks, so only `body` is under test."""
    path = tmp_path / "slot-1-3.py"
    path.write_text(
        "import argparse, json\n"
        "FLAGS = ('--repo', '--json')\n"
        "VERDICT = {'passed': True, 'findings': []}\n"
        + body
    )
    return path


def test_a_checker_that_shells_out_is_refused(tmp_path):
    findings = cons.validate_checker_source(_write(tmp_path, "import subprocess\n"))
    assert any("subprocess" in f for f in findings)


def test_a_checker_that_imports_shutil_is_refused(tmp_path):
    findings = cons.validate_checker_source(_write(tmp_path, "import shutil\n"))
    assert any("shutil" in f for f in findings)


def test_a_checker_that_deletes_is_refused(tmp_path):
    findings = cons.validate_checker_source(_write(tmp_path, "import os\nos.remove('x')\n"))
    assert any("remove" in f for f in findings)


def test_a_checker_that_execs_is_refused(tmp_path):
    findings = cons.validate_checker_source(_write(tmp_path, "exec('x = 1')\n"))
    assert any("exec" in f for f in findings)


def test_writing_the_json_report_is_not_refused(tmp_path):
    """A checker's whole output is that report, so banning writes would reject every checker."""
    findings = cons.validate_checker_source(_write(
        tmp_path, "import pathlib\npathlib.Path('r.json').write_text('{}')\n",
    ))
    assert findings == []


def test_an_ordinary_static_checker_is_accepted(tmp_path):
    findings = cons.validate_checker_source(_write(
        tmp_path, "import ast, re, sys\nt = ast.parse(open('source.py').read())\n",
    ))
    assert findings == []
