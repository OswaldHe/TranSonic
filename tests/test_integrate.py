# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The `integrate` pass: config, constraints, the gate, and the loop's exit condition.

No device and no network. Agent constraints run against the `mock` backend, which is what the
judge honouring `agent.type` through `AgentConfig.from_dict` buys — the whole agent-constraint
path is exercisable offline.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
import yaml

from integrate import gate as gate_mod
from integrate.config import METRIC, Constraint, IntegrateConfig

pytestmark = pytest.mark.integrate

#: Not bare `python`, which would pass only in a shell with the venv activated.
PY = sys.executable


# ---------------------------------------------------------------------------- helpers


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    """A git repo that fails two of three constraints."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@t")
    _git(root, "config", "user.name", "t")
    (root / ".gitignore").write_text("__pycache__/\n")
    (root / "impl.py").write_text("VALUE = 0\ndef ready(): return False\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "baseline")
    return root


def _base_config(repo=None) -> dict:
    """A native AutoHelix config: no `project:` block, the project path is passed separately."""
    return {
        "goal": "Make VALUE 42 and ready() true.",
        "scope": {"editable": ["impl.py"]},
        "constraints": [
            {"name": "imports", "kind": "script",
             "command": f'{PY} -c "import impl"', "timeout": 60},
            {"name": "value-42", "kind": "script",
             "command": f'{PY} -c "import impl; assert impl.VALUE == 42"', "timeout": 60},
            {"name": "ready-true", "kind": "script",
             "command": f'{PY} -c "import impl; assert impl.ready() is True"', "timeout": 60},
        ],
        "budget": {"iterations": 3},
        "agent": {"type": "mock"},
    }


def _write(repo, data) -> IntegrateConfig:
    """Write the config into the repo, as `init` does, and load it the way the CLI does."""
    path = repo / "integrate.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return IntegrateConfig.load(repo, path)


# ---------------------------------------------------------------------------- config


def test_config_loads_and_validates(repo):
    config = _write(repo, _base_config())
    assert config.validate() == []
    assert config.target_id == repo.name
    assert len(config.constraints) == 3
    assert config.max_iterations == 3


def test_unbounded_when_no_iteration_budget(repo):
    data = _base_config()
    data["budget"].pop("iterations")
    config = _write(repo, data)
    assert config.max_iterations is None
    assert any("unbounded" in w for w in config.warnings())


def test_missing_constraints_is_an_error(repo):
    """Reported by validate(), not at parse time: a native config may legitimately have none."""
    data = _base_config()
    del data["constraints"]
    problems = _write(repo, data).validate()
    assert any("no `constraints:`" in p for p in problems)


@pytest.mark.parametrize("goal", ["", "   ", "Describe what you want the agent to optimize."])
def test_an_unfilled_goal_is_refused(repo, goal):
    """Reported by AutoHelix's own `Config.validate`, not re-checked here."""
    data = _base_config()
    data["goal"] = goal
    problems = _write(repo, data).validate()
    assert any("goal" in p for p in problems), problems


def test_the_shipped_template_is_refused_until_filled_in(repo):
    """The template has to trip the placeholder check, or `init` blesses an unfilled config."""
    import yaml as _yaml

    from integrate import presets

    data = _yaml.safe_load(presets.config_template())
    problems = _write(repo, data).validate()
    assert any("placeholder" in p for p in problems), problems


def test_metrics_and_acceptance_are_refused_as_overrides(repo):
    """Both are the pass's; silently losing them in the derived file would be worse."""
    data = _base_config()
    data["metrics"] = [{"command": "echo hi", "values": {"x": "higher"}}]
    data["acceptance"] = {"metric_gates": [{"metric": "x", "max_regression_pct": 5}]}
    problems = _write(repo, data).validate()
    assert any("metrics" in p for p in problems), problems
    assert any("acceptance" in p for p in problems), problems


def test_a_bare_string_constraint_is_a_script_named_by_its_command(repo):
    """The native spelling: `constraints: [pytest tests/]`."""
    data = _base_config()
    data["constraints"] = ["pytest tests/"]
    config = _write(repo, data)
    assert len(config.constraints) == 1
    assert config.constraints[0].kind == "script"
    assert config.constraints[0].command == "pytest tests/"
    assert config.constraints[0].name == "pytest tests/"


def test_operator_keys_carry_through_to_the_derived_config(repo):
    """Everything but constraints/metrics/acceptance is the operator's and must survive."""
    data = _base_config()
    data["budget"] = {"iterations": 7, "iteration_time": "30m"}
    data["reviewer"] = {"prompt": "check it", "model": "claude-sonnet-5"}
    derived = _write(repo, data).derive_loop_config()
    assert derived["goal"] == data["goal"]
    assert derived["scope"] == data["scope"]
    assert derived["budget"]["iteration_time"] == "30m"
    assert derived["budget"]["iterations"] == 7
    assert derived["reviewer"]["prompt"] == "check it"


def test_empty_editable_scope_is_refused(repo):
    data = _base_config()
    data["scope"]["editable"] = []
    problems = _write(repo, data).validate()
    assert any("editable" in p for p in problems)


def test_duplicate_constraint_names_are_refused(repo):
    data = _base_config()
    data["constraints"][1]["name"] = "imports"
    problems = _write(repo, data).validate()
    assert any("both named" in p for p in problems)


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"kind": "script", "command": ""}, "needs a command"),
        ({"kind": "agent", "command": "x", "prompt": "p", "criteria": "c"}, "takes no `command:`"),
        ({"kind": "agent", "prompt": "p", "criteria": ""}, "needs a `criteria:`"),
        ({"kind": "agent", "prompt": "", "criteria": "c"}, "needs a `prompt:`"),
        ({"kind": "nonsense", "command": "x"}, "is not one of"),
        ({"kind": "script", "command": "x", "prompt": "p"}, "not a `prompt:`"),
    ],
)
def test_constraint_problems_are_named(overrides, expected):
    raw = {"name": "c"}
    raw.update(overrides)
    problems = Constraint.from_entry(raw, 0).problems()
    assert any(expected in p for p in problems), problems


def test_agent_block_passes_through_verbatim(repo):
    """Every key `AgentConfig.from_dict` understands has to survive to the derived config."""
    data = _base_config()
    data["agent"] = {"type": "mock", "extra_args": ["--x"], "settings": {"k": "v"}}
    derived = _write(repo, data).derive_loop_config()
    assert derived["agent"] == {"type": "mock", "extra_args": ["--x"], "settings": {"k": "v"}}


def test_derived_config_has_no_metric_gate_on_latency(repo):
    """The whole point: this pass is gated on constraints held, not on a speed."""
    derived = _write(repo, _base_config()).derive_loop_config()
    gates = derived["acceptance"]["metric_gates"]
    assert [g["metric"] for g in gates] == [METRIC]
    assert gates[0]["max_regression_pct"] == 0.0
    assert list(derived["metrics"][0]["values"]) == [METRIC]


def test_derived_gate_command_is_advisory(repo):
    """A failing constraint must not reject the iteration, or nothing ever converges."""
    derived = _write(repo, _base_config()).derive_loop_config()
    assert "--advisory" in derived["constraints"][0]["command"]


# ---------------------------------------------------------------------------- gate


def test_gate_counts_the_failing_baseline(repo):
    config = _write(repo, _base_config())
    results, payload = gate_mod.evaluate(config, repo)
    assert payload[METRIC] == 1
    assert payload["constraints_total"] == 3
    assert payload["satisfied"] is False
    assert [r.passed for r in results] == [True, False, False]


def test_gate_reports_satisfied_once_the_repo_is_fixed(repo):
    (repo / "impl.py").write_text("VALUE = 42\ndef ready(): return True\n")
    config = _write(repo, _base_config())
    _, payload = gate_mod.evaluate(config, repo)
    assert payload[METRIC] == 3
    assert payload["satisfied"] is True


def test_stop_early_marks_the_rest_skipped(repo):
    config = _write(repo, _base_config())
    results, payload = gate_mod.evaluate(config, repo, stop_early=True)
    assert payload["stopped_early"] is True
    assert [r.skipped for r in results] == [False, False, True]


def test_script_constraint_timeout_is_a_failure_not_a_hang(repo):
    data = _base_config()
    data["constraints"] = [
        {"name": "slow", "kind": "script",
         "command": f'{PY} -c "import time; time.sleep(30)"', "timeout": 1},
    ]
    config = _write(repo, data)
    results, payload = gate_mod.evaluate(config, repo)
    assert payload[METRIC] == 0
    assert not results[0].passed


# ---------------------------------------------------------------------------- agent constraints


def _agent_config(verdict_text: str) -> dict:
    """A config whose single agent constraint gets `verdict_text` written for it by the mock."""
    judge_file = ".autohelix/integrate/judge/judged.json"
    return {
        "goal": "g",
        "scope": {"editable": ["impl.py"]},
        "constraints": [{
            "name": "judged", "kind": "agent",
            "prompt": "Is impl.py real?", "criteria": "PASS if real.", "timeout": 60,
        }],
        "agent": {"type": "mock", "settings": {
            "change_file": judge_file, "change_content": verdict_text,
        }},
    }


@pytest.mark.parametrize(
    "verdict_text, expected_pass, reason_fragment",
    [
        ('{"verdict":"PASS","reason":"r","evidence":["impl.py:1"]}', True, ""),
        ('{"verdict":"PASS","reason":"r","evidence":[]}', False, "cited no evidence"),
        ('{"verdict":"FAIL","reason":"r","evidence":["impl.py:1"]}', False, ""),
        ('{"verdict":"MAYBE","evidence":["impl.py:1"]}', False, "expected PASS or FAIL"),
        ("not json", False, "no JSON object"),
        ("", False, "no JSON object"),
    ],
)
def test_agent_constraint_verdicts(repo, verdict_text, expected_pass, reason_fragment):
    """Every path through the judge, including the ones that must fail closed."""
    config = _write(repo, _agent_config(verdict_text))
    results, payload = gate_mod.evaluate(config, repo)
    assert results[0].passed is expected_pass
    assert payload["satisfied"] is expected_pass
    if reason_fragment:
        assert reason_fragment in results[0].detail


def test_agent_constraint_fails_when_the_judge_writes_nothing(repo):
    """A judge that produced no verdict has not cleared the constraint."""
    data = _agent_config("irrelevant")
    data["agent"]["settings"]["change_file"] = "somewhere_else.txt"
    config = _write(repo, data)
    results, _ = gate_mod.evaluate(config, repo)
    assert not results[0].passed
    assert "no verdict" in results[0].detail


def test_stale_verdict_is_not_reused(repo):
    """A previous iteration's PASS must not read back as this one's answer."""
    stale = repo / ".autohelix" / "integrate" / "judge" / "judged.json"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text('{"verdict":"PASS","reason":"stale","evidence":["x:1"]}')
    data = _agent_config("irrelevant")
    data["agent"]["settings"]["change_file"] = "somewhere_else.txt"
    config = _write(repo, data)
    results, _ = gate_mod.evaluate(config, repo)
    assert not results[0].passed, "a stale verdict was read back as a pass"


def test_all_agent_constraints_warns(repo):
    config = _write(repo, _agent_config('{"verdict":"FAIL"}'))
    assert any("kind: agent" in w for w in config.warnings())


# ---------------------------------------------------------------------------- the loop


def test_loop_stops_as_soon_as_the_constraints_hold(repo):
    """The exit condition, which is the whole reason this pass exists.

    The mock writes the fix on its first iteration, so a loop that ran its full budget would
    mean the termination check is broken — which it was, when `satisfied()` read a verdict file
    the gate had written inside a discarded worktree.
    """
    from integrate.loop import BaselineLoop

    data = _base_config()
    data["budget"]["iterations"] = 5
    data["agent"]["settings"] = {
        "change_file": "impl.py",
        "change_content": "VALUE = 42\ndef ready(): return True",
    }
    config = _write(repo, data)
    loop = BaselineLoop(config)
    loop.run()

    assert loop.satisfied()
    ran = [r for r in loop.history.load() if r.iteration > 0]
    assert len(ran) == 1, f"stopped late: ran {len(ran)} iteration(s) after already passing"
    summary = json.loads((repo / ".autohelix/integrate/baseline-summary.json").read_text())
    assert summary["satisfied"] is True
    assert summary["best_passing"] == 3
    assert summary["still_failing"] == []


def test_loop_records_a_failing_baseline_instead_of_aborting(repo):
    """A baseline that satisfies nothing is the starting condition, not a broken run."""
    from integrate.loop import BaselineLoop

    data = _base_config()
    data["constraints"] = [
        {"name": "impossible", "kind": "script", "command": f'{PY} -c "raise SystemExit(1)"',
         "timeout": 60},
    ]
    data["budget"]["iterations"] = 1
    data["agent"]["settings"] = {"change_file": "impl.py", "change_content": "# touched"}
    config = _write(repo, data)
    loop = BaselineLoop(config)
    loop.run()

    assert not loop.satisfied()
    baseline = [r for r in loop.history.load() if r.iteration == 0]
    assert baseline and baseline[0].metrics[METRIC] == 0
    assert loop.failing_names() == ["impossible"]


def test_failing_constraints_do_not_reject_the_iteration(repo):
    """The advisory gate: progress has to survive a still-failing constraint."""
    from integrate.loop import BaselineLoop

    data = _base_config()
    data["budget"]["iterations"] = 2
    data["agent"]["settings"] = {"change_file": "impl.py", "change_content": "# fixes nothing"}
    config = _write(repo, data)
    loop = BaselineLoop(config)
    loop.run()

    ran = [r for r in loop.history.load() if r.iteration > 0]
    assert len(ran) == 2
    assert all(r.accepted for r in ran), "a failing constraint rejected the iteration"
    assert not loop.satisfied()
