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
from bootstrap.driver import read_review_verdict, read_verdict

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


def test_the_goal_names_every_tolerance_constant() -> None:
    """The agent learns the bar only from the goal and the README the goal points at.

    The goal names the four constants but not their values: those are per-module, derived
    from the reference's dtype, so a fixed preset cannot state them. It has to send the agent
    to the generated README instead, and say so.
    """
    goal = preset.render_goal()
    for name in chk.TOLERANCE_NAMES:
        assert name in goal, name
    assert "README.md" in goal
    assert "numerical bar" in goal


def test_the_goal_names_the_ceiling_and_says_it_is_hard() -> None:
    """The ceiling is the one constant that is not a fraction, so the goal must say so.

    An agent that treats MAX_ABS_ERR as another pass-fraction knob will declare it and not
    enforce it, and the gate would then fail it for a reason the goal never explained.
    """
    goal = preset.render_goal()
    assert chk.CEILING_NAME in goal
    assert chk.MAX_ABS_ERR_MARKER in goal
    assert "no single element" in goal.lower()
    assert "five" in goal.lower()


def test_the_readme_carries_the_values_the_goal_defers_to() -> None:
    """Whatever the goal points at has to actually be there, with this repo's numbers."""
    from bootstrap import templates
    from bootstrap.materialize import Materialized, TensorRecord

    bar = {"RTOL": 0.1, "ATOL": 0.1, "MIN_COSINE": 0.9999, "MIN_PASS_FRACTION": 0.999,
           "MAX_ABS_ERR": 0.853}
    result = Materialized(
        repo=Path("/repo"), group="g", module_id="m", sample_id="s", step=0, call_index=0,
        tensors=[
            TensorRecord("input", "input", "tensors/input.bin", "float8_e4m3fn", [4], 4, "h",
                         required=True),
            TensorRecord("reference", "golden", "tensors/reference.bin", "float8_e4m3fn", [4],
                         4, "h", required=True),
        ],
        submodules=["a"], tolerance=bar, model="hf:some/model",
    )
    readme = templates.render_readme(result)
    assert "## The numerical bar" in readme
    for name, value in bar.items():
        assert f"{name} = {value:g}" in readme, name
    # The summary reports what the artifact said, not a hardcoded model or composition.
    assert "hf:some/model" in readme
    assert "DeepSeek" not in readme


def test_the_bar_follows_the_reference_dtype() -> None:
    """A global bfloat16 tolerance would hold an fp8 boundary to a bar it cannot meet."""
    assert chk.expected_tolerance({}) == chk.PINNED_TOLERANCE
    recorded = {"tolerance": {"RTOL": 0.1, "ATOL": 0.1, "MIN_COSINE": 0.99,
                              "MIN_PASS_FRACTION": 0.9}}
    assert chk.expected_tolerance(recorded)["RTOL"] == 0.1
    # A partial record still fills in the rest rather than dropping a constant.
    assert chk.expected_tolerance({"tolerance": {"RTOL": 0.5}})["ATOL"] == \
        chk.PINNED_TOLERANCE["ATOL"]


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


def test_the_goal_points_at_every_frozen_reference() -> None:
    """A reference `init` writes and the goal never mentions will not be read.

    Iteration 1 of the first real run spent half an hour deriving the fp8 scale rule from
    recorded tensors because `vendor_kernel.py`, which states it, was not carried in. The
    inverse — carried in but unmentioned — costs the same.
    """
    from bootstrap import materialize as mat

    goal = preset.render_goal()
    for _, target, _ in mat.FROZEN_REFERENCES:
        assert target in goal, target
    assert mat.NUMERICS_TARGET in goal
    assert f"{mat.VENDOR_DIR}/kernel.py" in goal
    assert f"{mat.VENDOR_DIR}/model.py" in goal
    assert f"{mat.COMPAT_DIR}/" in goal


def test_the_reviewer_is_given_the_same_reference_hierarchy() -> None:
    """The reviewer's "what is left" should name the file that settles a failing check.

    Told to work it out instead, the next iteration re-derives what a carried-in reference
    states — which is how the first real run lost an iteration.
    """
    from bootstrap import materialize as mat

    prompt = preset.load_preset()["reviewer"]["prompt"]
    for _, target, _ in mat.FROZEN_REFERENCES:
        assert target in prompt, target
    assert mat.NUMERICS_TARGET in prompt
    assert f"{mat.VENDOR_DIR}/kernel.py" in prompt
    assert f"{mat.VENDOR_DIR}/model.py" in prompt
    assert prompt.index(f"{mat.COMPAT_DIR}/") < prompt.index(f"{mat.VENDOR_DIR}/kernel.py")


def test_compat_outranks_vendor_in_the_goal() -> None:
    """A compat patch replaced a vendor kernel *before* tracing, so for any name it rebinds
    it is what the reference did. The goal has to say which to trust, and in which order."""
    goal = preset.render_goal()
    assert goal.index("compat/") < goal.index("vendor/kernel.py")
    assert "before `vendor/kernel.py`" in goal


def test_the_goal_says_the_reference_can_be_run() -> None:
    """Reading it is not the same as running it, and the second is what localizes an error.

    The first real run's agent had to build its own host reference from scratch because
    nothing told it `vendor/` was importable — or that the fp8 primitives in it are not.
    """
    goal = preset.render_goal()
    assert 'sys.path.insert(0, "vendor")' in goal
    assert "apply(" in goal
    assert "No registered target detector" in goal


def test_the_goal_gives_a_reading_order_starting_at_the_module() -> None:
    goal = preset.render_goal()
    for earlier, later in (
        ("reference_torch.py", "reference_inference.py"),
        ("reference_inference.py", "reference_numerics.py"),
        ("reference_numerics.py", "compat/"),
    ):
        assert goal.index(earlier) < goal.index(later), (earlier, later)


def test_the_frozen_references_cannot_be_opened_at_runtime() -> None:
    """They are there to be read by the agent, not loaded by the candidate.

    Importing one already fails (b) on the import allowlist; this covers the other route,
    reading it as a file.
    """
    from bootstrap import materialize as mat

    names = [t for _, t, _ in mat.FROZEN_REFERENCES] + [mat.NUMERICS_TARGET]
    for name in names:
        assert any(m in name for m in chk.FORBIDDEN_PATH_MARKERS), name


# -- the reviewer's verdict gates success ---------------------------------------------


@pytest.mark.parametrize("line,expected", [
    ("VERDICT: clean", "clean"),
    ("VERDICT: suspicious", "suspicious"),
    ("VERDICT: circumventing", "circumventing"),
    ("verdict:   CIRCUMVENTING", "circumventing"),
    ("  VERDICT: clean  ", "clean"),
])
def test_a_stated_verdict_is_read(tmp_path: Path, line: str, expected: str) -> None:
    path = tmp_path / "review.md"
    path.write_text(f"# Review\n\nsome analysis\n\n{line}\n")
    assert read_review_verdict(path) == expected


def test_the_words_in_prose_are_not_the_verdict(tmp_path: Path) -> None:
    """The reviewer's own instructions list all three words; only the line counts."""
    path = tmp_path / "review.md"
    path.write_text(
        "I considered whether this is circumventing the check or merely suspicious.\n"
        "It is neither.\n\nVERDICT: clean\n"
    )
    assert read_review_verdict(path) == "clean"


def test_the_last_verdict_wins(tmp_path: Path) -> None:
    path = tmp_path / "review.md"
    path.write_text("VERDICT: suspicious\n\non reflection:\n\nVERDICT: circumventing\n")
    assert read_review_verdict(path) == "circumventing"


def test_no_verdict_and_no_review_are_both_none(tmp_path: Path) -> None:
    path = tmp_path / "review.md"
    path.write_text("# Review\n\nno verdict here\n")
    assert read_review_verdict(path) is None
    assert read_review_verdict(tmp_path / "absent.md") is None


def test_the_reviewer_is_told_to_state_the_verdict_the_loop_parses() -> None:
    """The loop acts on this line, so the prompt has to ask for exactly this line."""
    prompt = preset.load_preset()["reviewer"]["prompt"]
    assert "VERDICT:" in prompt
    for value in ("clean", "suspicious", "circumventing"):
        assert value in prompt, value
    # The example in the prompt has to be a line the parser accepts.
    assert read_review_verdict_from_text(prompt) is not None


def read_review_verdict_from_text(text: str) -> str | None:
    from bootstrap.driver import REVIEW_VERDICT

    found = REVIEW_VERDICT.findall(text)
    return found[-1].lower() if found else None


def test_the_reviewer_is_asked_for_both_sections() -> None:
    prompt = preset.load_preset()["reviewer"]["prompt"].lower()
    assert "what is left" in prompt
    assert "reward-hacking" in prompt
    for verdict in ("clean", "suspicious", "circumventing"):
        assert verdict in prompt


def test_the_agent_prompt_never_renders_the_constraint() -> None:
    """The gate's command line names the checker module, so the prompt must not show it.

    The stock template renders `constraints`; this preset's does not, and that is the only
    thing keeping the checker's location out of the agent's context.
    """
    template = preset.load_prompt_template()
    assert "constraints" not in template
    assert "nki_checker" not in template
