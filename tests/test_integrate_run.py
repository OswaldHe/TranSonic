# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`autohelix integrate run`: the stage that holds the constraints and improves the numbers.

No device and no network. The measurements come from a constraint that prints
`##autohelix[...]`, which is the whole point of the stage and is exercisable with `echo`.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
import yaml

from integrate import gate as gate_mod
from integrate.config import METRIC, METRIC_ALL, ConfigError, IntegrateConfig
from integrate.schedule import Schedule, ScheduleError

pytestmark = pytest.mark.integrate

PY = sys.executable


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    """A git repo whose one constraint passes and prints two measurements."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@t")
    _git(root, "config", "user.name", "t")
    (root / ".gitignore").write_text("__pycache__/\n")
    (root / "impl.py").write_text("LATENCY = 100.0\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "baseline")
    return root


#: A constraint that passes and reports two numbers, the way the deepseek verifier does.
def _bench_command() -> str:
    return (
        f'{PY} -c "import impl; '
        f"print(f'##autohelix[latency_ms={{impl.LATENCY}}]'); "
        f"print('##autohelix[tokens_per_step=3.6]')\""
    )


def _run_config(**overrides) -> dict:
    data = {
        "goal": "Make it faster without breaking it.",
        "scope": {"editable": ["impl.py"]},
        "constraints": [
            {"name": "verifier", "kind": "script", "command": _bench_command(), "timeout": 60},
        ],
        "metrics": [
            {"values": {"latency_ms": "lower"}},
            {"values": {"tokens_per_step": "higher"}, "gate": False},
        ],
        "acceptance": {"max_regression_pct": 5},
        "budget": {"iterations": 13},
        "agent": {"type": "mock"},
    }
    data.update(overrides)
    return data


def _write(repo, data, name="integrate-run.yaml") -> IntegrateConfig:
    path = repo / name
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return IntegrateConfig.load(repo, path, stage="run")


# ---------------------------------------------------------------------------- config


def test_run_config_loads_metrics_and_default_allowance(repo):
    cfg = _write(repo, _run_config())
    assert cfg.validate() == []
    assert cfg.metric_names == ["latency_ms", "tokens_per_step"]
    assert cfg.gated_metric_names == ["latency_ms"]
    assert cfg.max_regression_pct == 5.0
    assert cfg.iterations == 13


def test_default_allowance_is_five_percent_when_unset(repo):
    data = _run_config()
    data.pop("acceptance")
    cfg = _write(repo, data)
    assert cfg.max_regression_pct == 5.0


def test_run_requires_metrics(repo):
    data = _run_config()
    data["metrics"] = []
    cfg = _write(repo, data)
    assert any("no `metrics:`" in p for p in cfg.validate())


def test_run_rejects_metric_named_like_the_pass_own_counts(repo):
    data = _run_config()
    data["metrics"] = [{"values": {METRIC: "higher"}}]
    cfg = _write(repo, data)
    assert any(METRIC in p and "this pass's own" in p for p in cfg.validate())


def test_run_rejects_all_metrics_ungated(repo):
    data = _run_config()
    data["metrics"] = [{"values": {"latency_ms": "lower"}, "gate": False}]
    cfg = _write(repo, data)
    assert any("every metric is `gate: false`" in p for p in cfg.validate())


def test_run_rejects_duplicate_metric_names(repo):
    data = _run_config()
    data["metrics"] = [
        {"values": {"latency_ms": "lower"}},
        {"command": "echo hi", "values": {"latency_ms": "lower"}},
    ]
    cfg = _write(repo, data)
    assert any("declared twice" in p for p in cfg.validate())


def test_unknown_metric_key_is_a_config_error(repo):
    data = _run_config()
    data["metrics"] = [{"values": {"latency_ms": "lower"}, "direction": "lower"}]
    path = repo / "integrate-run.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    with pytest.raises(ConfigError, match="unknown key"):
        IntegrateConfig.load(repo, path, stage="run")


def test_baseline_still_refuses_metrics(repo):
    """The baseline stage supplies its own metrics; a config that sets them is an error."""
    path = repo / "integrate.yaml"
    path.write_text(yaml.safe_dump(_run_config(), sort_keys=False))
    cfg = IntegrateConfig.load(repo, path, stage="baseline")
    assert any("supplies its own" in p for p in cfg.validate())


def test_baseline_refuses_an_iteration_schedule(repo):
    path = repo / "integrate.yaml"
    data = {
        "goal": "g", "scope": {"editable": ["impl.py"]},
        "constraints": ["true"], "agent": {"type": "mock"},
        "iteration_constraints": [{"at": 1, "text": "x"}],
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    cfg = IntegrateConfig.load(repo, path, stage="baseline")
    assert any("iteration_constraints" in p for p in cfg.validate())


# ---------------------------------------------------------------------------- derivation


def test_derived_config_reads_metrics_back_from_the_gate(repo):
    cfg = _write(repo, _run_config())
    derived = cfg.derive_loop_config()

    assert len(derived["constraints"]) == 1
    assert "integrate.gate" in derived["constraints"][0]["command"]
    assert "--advisory" in derived["constraints"][0]["command"]

    # One metric command, which is the readback: the expensive constraint runs once.
    assert len(derived["metrics"]) == 1
    command = derived["metrics"][0]["command"]
    assert "integrate.readback" in command
    assert "--metrics latency_ms,tokens_per_step" in command
    assert derived["metrics"][0]["values"] == {
        METRIC: "higher", METRIC_ALL: "higher",
        "latency_ms": "lower", "tokens_per_step": "higher",
    }


def test_metric_with_its_own_command_runs_as_an_observable(repo):
    data = _run_config()
    data["metrics"] = [
        {"values": {"latency_ms": "lower"}},
        {"command": "echo '##autohelix[extra=1]'", "values": {"extra": "higher"}},
    ]
    derived = _write(repo, data).derive_loop_config()
    commands = [m["command"] for m in derived["metrics"]]
    assert any("integrate.readback" in c for c in commands)
    assert any(c.startswith("echo") for c in commands)


def test_gates_are_the_constraint_count_and_each_gated_metric(repo):
    gates = {
        g["metric"]: g["max_regression_pct"]
        for g in _write(repo, _run_config()).derive_loop_config()["acceptance"]["metric_gates"]
    }
    assert gates == {METRIC: 0.0, "latency_ms": 5.0}


def test_per_metric_gate_overrides_the_allowance(repo):
    data = _run_config()
    data["acceptance"] = {
        "max_regression_pct": 5,
        "metric_gates": [{"metric": "latency_ms", "max_regression_pct": 1}],
    }
    gates = {
        g["metric"]: g["max_regression_pct"]
        for g in _write(repo, data).derive_loop_config()["acceptance"]["metric_gates"]
    }
    assert gates["latency_ms"] == 1.0


def test_derived_config_carries_the_schedule_and_budget(repo):
    data = _run_config()
    data["iteration_constraints"] = [{"from": 1, "to": 4, "text": "measure first"}]
    derived = _write(repo, data).derive_loop_config()
    assert derived["budget"]["iterations"] == 13
    assert derived["iteration_constraints"][0]["text"] == "measure first"


def test_each_stage_derives_its_own_loop_config_path(repo):
    baseline_path = repo / "integrate.yaml"
    baseline_path.write_text(yaml.safe_dump({
        "goal": "g", "scope": {"editable": ["impl.py"]},
        "constraints": ["true"], "agent": {"type": "mock"},
    }, sort_keys=False))
    baseline = IntegrateConfig.load(repo, baseline_path, stage="baseline")
    run = _write(repo, _run_config())
    assert baseline.loop_config_path != run.loop_config_path
    assert run.loop_config_path.name == "run-loop.yaml"


def test_derived_run_config_validates_as_an_autohelix_config(repo):
    """The derived file is what the Harness loads, so it has to pass upstream's own validator."""
    from autohelix.config import Config

    cfg = _write(repo, _run_config())
    derived = cfg.derive_loop_config()
    issues = Config.from_dict(derived).validate(derived)
    assert [i.message for i in issues if i.level == "error"] == []


# ---------------------------------------------------------------------------- the gate


def test_gate_records_metrics_a_constraint_printed(repo):
    cfg = _write(repo, _run_config())
    results, payload = gate_mod.evaluate(cfg, repo)
    assert [r.passed for r in results] == [True]
    assert payload["metrics"]["latency_ms"] == 100.0
    assert payload["metrics"]["tokens_per_step"] == 3.6
    assert payload["metric_sources"]["latency_ms"] == "verifier"


def test_readback_reports_the_metrics_it_was_asked_for(repo, tmp_path, capsys):
    from integrate import readback

    cfg = _write(repo, _run_config())
    _, payload = gate_mod.evaluate(cfg, repo)
    verdict = tmp_path / "gate.json"
    verdict.write_text(json.dumps(payload))

    rc = readback.main(["--json", str(verdict), "--metrics", "latency_ms,tokens_per_step"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "##autohelix[latency_ms=100]" in out
    assert "##autohelix[tokens_per_step=3.6]" in out
    assert f"##autohelix[{METRIC}=1]" in out


def test_readback_fails_when_a_requested_metric_is_absent(repo, tmp_path, capsys):
    from integrate import readback

    verdict = tmp_path / "gate.json"
    verdict.write_text(json.dumps({
        "constraints": [{"name": "c", "kind": "script", "passed": True}],
        "metrics": {},
    }))
    rc = readback.main(["--json", str(verdict), "--metrics", "latency_ms"])
    assert rc == 1
    assert "latency_ms" in capsys.readouterr().err


def test_metrics_survive_a_constraint_that_prints_a_lot_afterwards(repo):
    """The gate parses metrics before truncating the output it stores."""
    data = _run_config()
    data["constraints"] = [{
        "name": "chatty",
        "kind": "script",
        "command": (
            f'{PY} -c "print(\'##autohelix[latency_ms=7]\');'
            f"print('##autohelix[tokens_per_step=1]');"
            f'print(\'x\' * 9000)"'
        ),
        "timeout": 60,
    }]
    cfg = _write(repo, data)
    _, payload = gate_mod.evaluate(cfg, repo)
    assert payload["metrics"]["latency_ms"] == 7.0
    assert len(payload["constraints"][0]["output"]) <= 4000


def test_skipped_constraint_contributes_no_metrics(repo):
    data = _run_config()
    data["constraints"] = [
        {"name": "fails", "kind": "script", "command": f'{PY} -c "raise SystemExit(1)"',
         "timeout": 60},
        {"name": "verifier", "kind": "script", "command": _bench_command(), "timeout": 60},
    ]
    cfg = _write(repo, data)
    _, payload = gate_mod.evaluate(cfg, repo, stop_early=True)
    assert payload["metrics"] == {}


# ---------------------------------------------------------------------------- schedule


def test_schedule_parses_ranges_and_single_iterations():
    schedule = Schedule.from_config([
        {"from": 1, "to": 3, "text": "a"},
        {"at": 4, "text": "b"},
        {"iterations": [5, 6], "text": "c"},
    ], max_iterations=13)
    assert [s.label for s in schedule.slots] == ["1-3", "4", "5-6"]
    assert schedule.slot_for(2).text == "a"
    assert schedule.slot_for(7) is None


def test_schedule_refuses_overlapping_slots():
    with pytest.raises(ScheduleError, match="already governed"):
        Schedule.from_config([{"from": 1, "to": 3, "text": "a"}, {"at": 2, "text": "b"}])


def test_schedule_refuses_a_slot_past_the_budget():
    with pytest.raises(ScheduleError, match="budget.iterations"):
        Schedule.from_config([{"at": 14, "text": "a"}], max_iterations=13)


def test_schedule_warns_that_hard_enforcement_does_nothing():
    schedule = Schedule.from_config([{"at": 1, "text": "a", "enforcement": "hard"}])
    assert any("no effect" in w for w in schedule.validate(1))


def test_schedule_prompt_block_is_empty_without_a_slot():
    assert Schedule.from_config(None).describe_for_prompt(1) == ""


def test_schedule_prompt_block_names_the_iteration():
    text = Schedule.from_config(
        [{"from": 2, "to": 4, "text": "profile first"}]
    ).describe_for_prompt(3)
    assert "profile first" in text
    assert "iteration 3" in text


# ---------------------------------------------------------------------------- the stage


def test_run_refuses_another_run_history(repo, monkeypatch):
    """A `run` started over the baseline stage's history would spend its budget on arrival."""
    from autohelix.history import History, IterationResult
    from integrate.driver import IntegratePass, StageError

    cfg = _write(repo, _run_config())
    history = History(repo)
    history.append(IterationResult(
        iteration=1, accepted=True, metrics={METRIC: 3.0}, reason="",
    ))
    with pytest.raises(StageError, match="another run's history"):
        IntegratePass(cfg).run()


def _loop(repo, content, **overrides):
    """A `RunLoop` whose agent writes `content` into impl.py on every iteration."""
    from integrate.runloop import RunLoop

    data = _run_config(**overrides)
    data["agent"]["settings"] = {"change_file": "impl.py", "change_content": content}
    return RunLoop(_write(repo, data))


def test_an_improvement_is_accepted(repo):
    loop = _loop(repo, "LATENCY = 50.0\n", budget={"iterations": 1})
    loop.run()
    ran = [r for r in loop.history.load() if r.iteration > 0]
    assert [r.accepted for r in ran] == [True]
    assert ran[0].metrics["latency_ms"] == 50.0
    baseline = [r for r in loop.history.load() if r.iteration == 0]
    assert baseline[0].metrics["latency_ms"] == 100.0


def test_a_regression_past_the_allowance_is_rejected(repo):
    loop = _loop(repo, "LATENCY = 200.0\n", budget={"iterations": 1})
    loop.run()
    ran = [r for r in loop.history.load() if r.iteration > 0]
    assert [r.accepted for r in ran] == [False]
    assert "latency_ms" in (ran[0].reason or "")


def test_a_regression_inside_the_allowance_is_accepted(repo):
    """5% is the allowance, so 3% worse is kept — the noise floor is not a rejection."""
    loop = _loop(repo, "LATENCY = 103.0\n", budget={"iterations": 1})
    loop.run()
    ran = [r for r in loop.history.load() if r.iteration > 0]
    assert [r.accepted for r in ran] == [True]


def test_losing_a_script_constraint_rejects_even_with_a_better_metric(repo):
    """Zero tolerance on the constraint count: faster is not an excuse for broken."""
    data = _run_config()
    data["constraints"] = [
        {"name": "verifier", "kind": "script", "command": _bench_command(), "timeout": 60},
        {"name": "invariant", "kind": "script",
         "command": f'{PY} -c "import impl; assert impl.SAFE"', "timeout": 60},
    ]
    data["agent"]["settings"] = {
        "change_file": "impl.py", "change_content": "LATENCY = 1.0\nSAFE = False\n",
    }
    data["budget"] = {"iterations": 1}
    (repo / "impl.py").write_text("LATENCY = 100.0\nSAFE = True\n")
    _git(repo, "commit", "-qam", "safe baseline")

    from integrate.runloop import RunLoop

    loop = RunLoop(_write(repo, data))
    loop.run()
    ran = [r for r in loop.history.load() if r.iteration > 0]
    assert [r.accepted for r in ran] == [False]
    assert METRIC in (ran[0].reason or "")


def test_the_iterations_slot_text_reaches_the_prompt(repo, tmp_path):
    data = _run_config()
    data["iteration_constraints"] = [
        {"from": 1, "to": 2, "text": "profile before changing anything"},
        {"at": 3, "text": "now attack the biggest cost"},
    ]
    data["agent"]["settings"] = {"change_file": "impl.py", "change_content": "LATENCY = 1\n"}

    from integrate.runloop import RunLoop

    loop = RunLoop(_write(repo, data))
    # What governs *this* iteration, as opposed to the schedule table that shows the whole run.
    assert loop.schedule.describe_for_prompt(1).count("profile before changing anything") == 1
    assert "now attack" not in loop.schedule.describe_for_prompt(1)
    assert "now attack the biggest cost" in loop.schedule.describe_for_prompt(3)
    assert loop.schedule.describe_for_prompt(4) == ""

    first = loop.build_prompt(1, tmp_path)
    assert "profile before changing anything" in first
    # The whole schedule travels too, so an iteration can see where it sits in the run.
    assert "| 1-2 |" in first and "| 3 |" in first


def test_the_prompt_states_the_allowance_and_the_best_so_far(repo, tmp_path):
    loop = _loop(repo, "LATENCY = 50.0\n", budget={"iterations": 2})
    loop.run()
    prompt = loop.build_prompt(2, tmp_path)
    assert "5%" in prompt
    assert "latency_ms" in prompt
    assert "50" in prompt
    # The ungated one says so, rather than looking like something that can reject an iteration.
    assert "recorded, never gated" in prompt


def test_summary_records_the_improvement(repo):
    loop = _loop(repo, "LATENCY = 75.0\n", budget={"iterations": 1})
    loop.run()
    summary = json.loads((repo / ".autohelix/integrate/run-summary.json").read_text())
    assert summary["stage"] == "run"
    assert summary["metrics"]["latency_ms"]["baseline"] == 100.0
    assert summary["metrics"]["latency_ms"]["best"] == 75.0
    assert summary["metrics"]["latency_ms"]["improvement_pct"] == pytest.approx(25.0)
    assert summary["max_regression_pct"] == 5.0
    assert summary["still_failing"] == []


def test_run_resumes_its_own_history(repo):
    """Its own records carry its metrics, so it continues rather than refusing."""
    from autohelix.history import History, IterationResult
    from integrate.driver import IntegratePass

    cfg = _write(repo, _run_config())
    History(repo).append(IterationResult(
        iteration=1, accepted=True, metrics={"latency_ms": 90.0}, reason="",
    ))
    assert IntegratePass(cfg)._foreign_history() == ""
