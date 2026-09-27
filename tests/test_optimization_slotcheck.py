# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The slot-checker wrapper, and the end-of-interval acceptance rule it enables.

The wrapper is small but it decides two things that are easy to get backwards: a broken checker must
read as a violation rather than as a compliant candidate, and `--advisory` must change only the exit
code — the verdict it writes has to be the same either way, or the loop's stricter rule would be
applied to a different fact than the one the operator sees in the report.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from optimization import slotcheck

pytestmark = pytest.mark.optimization


def _checker(tmp_path: Path, body: str, name: str = "slot-1-3.py") -> Path:
    path = tmp_path / name
    path.write_text(textwrap.dedent(f"""
        import argparse, json, sys
        p = argparse.ArgumentParser()
        p.add_argument("--repo"); p.add_argument("--json")
        a = p.parse_args()
        {body}
    """))
    return path


PASSES = (
    'open(a.json, "w").write(json.dumps({"passed": True, "findings": []}))\n'
    "        sys.exit(0)"
)
VIOLATES = (
    'open(a.json, "w").write(json.dumps('
    '{"passed": False, "findings": ["source.py:4 imports torch"]}))\n'
    "        sys.exit(1)"
)
CRASHES = 'raise SystemExit("boom")'


def _run(checker: Path, repo: Path, report: Path, advisory: bool = False):
    argv = ["--checker", str(checker), "--repo", str(repo), "--json", str(report)]
    if advisory:
        argv.append("--advisory")
    return slotcheck.main(argv)


def test_a_compliant_candidate_exits_zero(tmp_path):
    report = tmp_path / "iter-1.json"
    assert _run(_checker(tmp_path, PASSES), tmp_path, report) == 0
    assert json.loads(report.read_text())["passed"] is True


def test_a_violation_exits_non_zero_when_enforcing(tmp_path):
    report = tmp_path / "iter-1.json"
    assert _run(_checker(tmp_path, VIOLATES), tmp_path, report) == 1
    assert json.loads(report.read_text())["passed"] is False


def test_a_violation_exits_zero_when_advisory(tmp_path):
    """So the real constraints and the measurement still run on the interval's last iteration."""
    report = tmp_path / "iter-3.json"
    assert _run(_checker(tmp_path, VIOLATES), tmp_path, report, advisory=True) == 0


def test_advisory_changes_only_the_exit_code(tmp_path):
    """The verdict is what the loop's stricter rule and the report both read."""
    enforcing, advisory = tmp_path / "a.json", tmp_path / "b.json"
    checker = _checker(tmp_path, VIOLATES)
    _run(checker, tmp_path, enforcing)
    _run(checker, tmp_path, advisory, advisory=True)
    assert json.loads(enforcing.read_text()) == json.loads(advisory.read_text())


def test_a_crashed_checker_reads_as_a_violation(tmp_path):
    """Fail closed: the alternative is a schedule that quietly stops constraining anything."""
    report = tmp_path / "iter-1.json"
    assert _run(_checker(tmp_path, CRASHES), tmp_path, report) == 1
    payload = json.loads(report.read_text())
    assert payload["passed"] is False
    assert any("wrote no verdict" in f for f in payload["findings"])


def test_a_crashed_checker_is_still_recorded_under_advisory(tmp_path):
    """Exit 0 so the run continues, but the verdict says it did not pass."""
    report = tmp_path / "iter-3.json"
    assert _run(_checker(tmp_path, CRASHES), tmp_path, report, advisory=True) == 0
    assert json.loads(report.read_text())["passed"] is False


def test_a_checker_writing_invalid_json_reads_as_a_violation(tmp_path):
    report = tmp_path / "iter-1.json"
    checker = _checker(tmp_path, 'open(a.json, "w").write("not json")\n        sys.exit(0)')
    assert _run(checker, tmp_path, report) == 1
    assert "not valid JSON" in json.loads(report.read_text())["findings"][0]


def test_a_checker_claiming_success_while_exiting_non_zero_is_not_believed(tmp_path):
    report = tmp_path / "iter-1.json"
    checker = _checker(
        tmp_path,
        'open(a.json, "w").write(json.dumps({"passed": True, "findings": []}))\n'
        "        sys.exit(3)",
    )
    assert _run(checker, tmp_path, report) == 1


def test_a_stale_verdict_is_removed_before_the_checker_runs(tmp_path):
    """Otherwise a re-run reads the previous iteration's verdict as this one's."""
    report = tmp_path / "iter-1.json"
    report.write_text(json.dumps({"passed": True, "findings": []}))
    checker = _checker(tmp_path, 'sys.exit(0)')   # writes nothing
    slotcheck.run_checker(checker, tmp_path, report)
    payload = json.loads(report.read_text())
    assert payload["passed"] is True  # exit 0, so a no-write checker is taken at its word
    report.write_text(json.dumps({"passed": True, "findings": []}))
    checker = _checker(tmp_path, 'sys.exit(1)', name="slot-4-6.py")
    slotcheck.run_checker(checker, tmp_path, report)
    assert json.loads(report.read_text())["passed"] is False


def test_a_missing_checker_is_broken_not_compliant(tmp_path):
    assert _run(tmp_path / "nope.py", tmp_path, tmp_path / "r.json") == slotcheck.BROKEN


def test_a_missing_checker_does_not_block_an_advisory_iteration(tmp_path):
    assert _run(tmp_path / "nope.py", tmp_path, tmp_path / "r.json", advisory=True) == 0


def test_a_hanging_checker_times_out_as_a_violation(tmp_path):
    report = tmp_path / "iter-1.json"
    checker = _checker(tmp_path, "import time\n        time.sleep(30)")
    passed, _ = slotcheck.run_checker(checker, tmp_path, report, timeout=1)
    assert not passed
    assert "did not finish" in json.loads(report.read_text())["findings"][0]


def test_it_is_runnable_as_a_module(tmp_path):
    """The loop drives it as `python -m optimization.slotcheck`, so that entry point has to work."""
    report = tmp_path / "iter-1.json"
    result = subprocess.run(
        [sys.executable, "-m", "optimization.slotcheck",
         "--checker", str(_checker(tmp_path, VIOLATES)),
         "--repo", str(tmp_path), "--json", str(report)],
        capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "does not follow this iteration's constraint" in result.stdout
