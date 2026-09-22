# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The loop's own behaviour: how a verdict is read, and that the preset matches the gate.

The four ways this loop differs from `autohelix run` are what these cover. Nothing here
needs a device or an agent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autohelix.checks import ConstraintResult
from bootstrap import nki_checker as chk
from bootstrap import preset
from bootstrap.driver import read_verdict

pytestmark = pytest.mark.bootstrap


def _checks(passing: list[str], failing: list[str]) -> dict:
    return {
        "passed": not failing,
        "checks": (
            [{"check": k, "title": chk.CHECK_TITLES[k], "passed": True} for k in passing]
            + [{"check": k, "title": chk.CHECK_TITLES[k], "passed": False} for k in failing]
        ),
        "report": "a report",
    }


def test_a_verdict_is_read_from_the_json(tmp_path: Path) -> None:
    path = tmp_path / "checks.json"
    path.write_text(json.dumps(_checks(["b", "c", "f"], ["a", "d", "e"])))
    verdict = read_verdict([ConstraintResult("gate", False, "out", 1)], path)
    assert not verdict.passed
    assert verdict.passing == ["b", "c", "f"]
    assert verdict.failing == ["a", "d", "e"]
    assert verdict.total == 6
    assert "3/6 passing" in verdict.summary()


def test_a_full_pass_needs_both_the_json_and_the_exit_code(tmp_path: Path) -> None:
    """A json claiming success while the gate exited non-zero is not a pass.

    The two disagreeing means something went wrong after the checks were written, and a
    loop that stops on the json alone would call the module bootstrapped on the strength of
    a file the candidate's own run could have left behind.
    """
    path = tmp_path / "checks.json"
    path.write_text(json.dumps(_checks(list("abcdef"), [])))
    assert read_verdict([ConstraintResult("gate", True, "", 0)], path).passed
    assert not read_verdict([ConstraintResult("gate", False, "", 1)], path).passed


def test_a_missing_json_falls_back_to_the_exit_code(tmp_path: Path) -> None:
    """A gate that timed out writes nothing; the verdict must still be a verdict."""
    verdict = read_verdict(
        [ConstraintResult("gate", False, "Timeout after 1200s", -1)], tmp_path / "absent.json",
    )
    assert not verdict.passed
    assert verdict.failing == ["?"]
    assert "Timeout" in verdict.report


def test_a_corrupt_json_falls_back_to_the_exit_code(tmp_path: Path) -> None:
    path = tmp_path / "checks.json"
    path.write_text("{not json")
    verdict = read_verdict([ConstraintResult("gate", False, "partial output", 1)], path)
    assert not verdict.passed
    assert verdict.report == "partial output"


def test_no_constraint_result_is_not_a_pass(tmp_path: Path) -> None:
    assert not read_verdict([], tmp_path / "absent.json").passed


# -- the preset and the gate have to agree --------------------------------------------


def test_the_goal_states_every_pinned_tolerance() -> None:
    """The agent is told the bar only by the goal, so the goal must carry all four."""
    goal = preset.render_goal()
    for name, value in chk.PINNED_TOLERANCE.items():
        assert f"{name} = {value:g}" in goal, name


def test_the_goal_names_what_the_gate_looks_for() -> None:
    goal = preset.render_goal()
    for token in (
        chk.KERNEL_FUNCTION, "@nki.jit", "torch_neuronx.trace", "neuron-explorer",
        chk.LATENCY_MARKER, chk.PASSED_MARKER, ".bin",
    ):
        assert token in goal, token


def test_the_goal_forbids_what_the_gate_refuses() -> None:
    goal = preset.render_goal().lower()
    for token in ("torch", "numpy", "scipy", "randn", "ones", "arange", "fill_"):
        assert token in goal, token


def test_the_preset_declares_no_metric() -> None:
    """A declared metric makes baseline capture raise at iteration 0, before a kernel exists."""
    config = preset.load_preset()
    assert config["metrics"] == []
    assert config["scope"]["editable"] == ["source.py", "inference.py"]
    assert config["reviewer"]["prompt"]


def test_the_preset_declares_what_the_loop_needs() -> None:
    data = preset.load_preset()
    for key in ("goal", "constraints", "metrics", "scope", "agent", "reviewer", "budget"):
        assert key in data, key


def test_the_preset_is_fixed_with_nothing_left_to_substitute() -> None:
    """The loop reads this file directly, so a leftover placeholder would reach the shell."""
    command = preset.load_preset()["constraints"][0]["command"]
    assert "${" not in command
    assert "{{" not in command
    # An absolute path would pin the preset to one machine, which a fixed file cannot be.
    for token in command.split():
        assert not token.startswith("/"), token
    assert "nki_checker" in command


def test_the_gate_gets_longer_than_the_run_it_supervises() -> None:
    """Otherwise a hung inference.py kills the constraint instead of failing a check."""
    constraint = preset.load_preset()["constraints"][0]
    inner = int(constraint["command"].split("--timeout")[1].split()[0])
    assert constraint["timeout"] > inner


def test_the_preset_satisfies_autohelixs_own_validator() -> None:
    """The loop hands this file straight to `load_config`, with no preprocessing."""
    from autohelix.config import load_config

    config, raw = load_config(preset.PRESET_PATH.parent, config_file=preset.PRESET_PATH)
    errors = [i for i in config.validate(raw_data=raw) if i.level == "error"]
    assert not errors, [e.message for e in errors]
    assert config.editable == ["source.py", "inference.py"]
    assert config.observables == []
    assert config.reviewer is not None
    assert len(config.constraints) == 1


def test_the_reviewer_is_asked_for_both_sections() -> None:
    prompt = preset.REVIEWER_PROMPT.lower()
    assert "what is left" in prompt
    assert "reward-hacking" in prompt
    for verdict in ("clean", "suspicious", "circumventing"):
        assert verdict in prompt


def test_the_agent_prompt_never_renders_the_constraint() -> None:
    """The gate's command line names the checker module, so the prompt must not show it.

    The stock template renders `constraints`; this preset's does not, and that is the only
    thing keeping the checker's location out of the agent's context.
    """
    assert "constraints" not in preset.PROMPT_TEMPLATE
    assert "nki_checker" not in preset.PROMPT_TEMPLATE
