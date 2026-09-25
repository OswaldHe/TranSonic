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


def test_the_reviewer_knows_the_bar_is_not_the_agents_to_choose() -> None:
    """Otherwise a retune reads to the reviewer as exactly the cheating it hunts for.

    `bootstrap retune` loosened 00-Attention's rtol from 0.02 to 0.1 mid-run, so the next
    iteration had to raise it — and "a looser tolerance" was on the reviewer's circumvention
    list. A `circumventing` verdict blocks the loop declaring success, so an uninformed
    reviewer could refuse to let a finished module finish.
    """
    prompt = preset.load_preset()["reviewer"]["prompt"]
    assert "README.md" in prompt
    assert "not the agent's to choose" in prompt
    # The list must no longer treat a loose constant as a signal in itself.
    assert "a looser tolerance," not in prompt
    # But the comparison's logic is still fair game.
    for still in ("comparing fewer elements", "swallowing a mismatch", "MAX_ABS_ERR"):
        assert still in prompt, still


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


def _git(repo: Path, *args: str) -> None:
    import subprocess
    subprocess.run(("git", *args), cwd=repo, check=True, capture_output=True)


def _repo_with_one_commit(tmp_path: Path) -> Path:
    repo = tmp_path / "wt"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "source.py").write_text("gated\n")
    (repo / "reference_torch.py").write_text("frozen\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "gated state")
    return repo


class _FakeWorktree:
    def __init__(self, path: Path) -> None:
        self.working_dir = path


def test_a_commit_the_reviewer_makes_is_visible_to_the_head_probe(tmp_path: Path) -> None:
    """`merge_worktree` merges the worktree's HEAD, so a moved HEAD has to be detectable."""
    from bootstrap.driver import BootstrapLoop

    repo = _repo_with_one_commit(tmp_path)
    worktree = _FakeWorktree(repo)
    before = BootstrapLoop._worktree_head(worktree)
    assert before

    (repo / "reference_torch.py").write_text("reviewer rewrote the spec\n")
    _git(repo, "commit", "-aqm", "reviewer commit")
    assert BootstrapLoop._worktree_head(worktree) != before


def test_resetting_the_reviewers_commit_exposes_it_to_scope_enforcement(tmp_path: Path) -> None:
    """Committing is what hides a change from `revert_out_of_scope`, which diffs against HEAD.

    So the reset has to come first: after it the edit is an uncommitted change again, which is
    the only form scope enforcement can see and revert.
    """
    import subprocess

    from bootstrap.driver import BootstrapLoop

    repo = _repo_with_one_commit(tmp_path)
    worktree = _FakeWorktree(repo)
    before = BootstrapLoop._worktree_head(worktree)

    (repo / "reference_torch.py").write_text("reviewer rewrote the spec\n")
    _git(repo, "commit", "-aqm", "reviewer commit")

    def dirty() -> set[str]:
        out = subprocess.run(("git", "diff", "--name-only", "HEAD"), cwd=repo,
                             capture_output=True, text=True, check=True).stdout
        return set(out.split())

    assert dirty() == set(), "a committed change is invisible to a diff against HEAD"

    BootstrapLoop._git(worktree, "reset", "--soft", before)
    assert BootstrapLoop._worktree_head(worktree) == before
    assert dirty() == {"reference_torch.py"}
    assert (repo / "reference_torch.py").read_text() == "reviewer rewrote the spec\n", (
        "--soft keeps the file so scope enforcement, not the reset, decides its fate"
    )


def _manifest(tensors: list[dict]) -> dict:
    return {"tensors": tensors}


def test_a_validator_that_reads_no_parameter_fails_provenance(tmp_path: Path) -> None:
    """(f) claims the kernel runs on the recorded parameters, so reading none must fail."""
    import ast as _ast

    manifest = _manifest([
        {"name": "input", "role": "input", "file": "tensors/input.bin", "required": True},
        {"name": "reference", "role": "golden", "file": "tensors/reference.bin", "required": True},
        {"name": "w", "role": "weight", "file": "tensors/w.bin"},
    ])
    src = (
        'x = open("tensors/input.bin", "rb").read()\n'
        'r = open("tensors/reference.bin", "rb").read()\n'
    )
    result = chk.check_provenance(_ast.parse(src), tmp_path, manifest)
    assert not result.passed
    assert any("reads none of the 1 recorded weight/buffer" in p for p in result.findings), (
        result.findings
    )


def test_reading_one_parameter_is_enough_for_provenance(tmp_path: Path) -> None:
    """A recorded parameter the reference itself never read (`gate.bias_vl`, a prefill
    `window_kv_cache`) must not be forced on a correct kernel, so the bar is aggregate."""
    import ast as _ast

    manifest = _manifest([
        {"name": "input", "role": "input", "file": "tensors/input.bin", "required": True},
        {"name": "reference", "role": "golden", "file": "tensors/reference.bin", "required": True},
        {"name": "w", "role": "weight", "file": "tensors/w.bin"},
        {"name": "bias_vl", "role": "weight", "file": "tensors/bias_vl.bin"},
    ])
    src = (
        'x = open("tensors/input.bin", "rb").read()\n'
        'r = open("tensors/reference.bin", "rb").read()\n'
        'w = open("tensors/w.bin", "rb").read()\n'
    )
    result = chk.check_provenance(_ast.parse(src), tmp_path, manifest)
    assert not any("reads none of" in p for p in result.findings), result.findings


def test_cross_module_state_reaches_the_kernel_signature_with_its_caveat() -> None:
    """A kernel has no shared holder, so recorded state has to arrive as an argument.

    And the README has to say what the snapshot is, because an entry can be one the group
    produces rather than reads — finding your own answer in an argument is the trap.
    """
    from bootstrap import templates
    from bootstrap.materialize import Materialized, TensorRecord

    result = Materialized(
        repo=Path("/repo"), group="24-Attention", module_id="layers.24.attention",
        sample_id="long-needle-8192-0", step=0, call_index=0,
        tensors=[
            TensorRecord("input", "input", "tensors/input.bin", "bfloat16",
                         [1, 8192, 5120], 8, "h", required=True),
            TensorRecord("reference", "golden", "tensors/reference.bin", "bfloat16",
                         [1, 8192, 5120], 8, "h", required=True),
            TensorRecord("state.shared_attn.compress_kv", "state",
                         "tensors/state_shared_attn_compress_kv.bin", "bfloat16",
                         [4, 16384, 512], 8, "h"),
        ],
        submodules=["layers.24.attn_norm", "layers.24.attn"],
        tolerance={"RTOL": 0.1, "ATOL": 0.1, "MIN_COSINE": 0.9995,
                   "MIN_PASS_FRACTION": 0.999, "MAX_ABS_ERR": 2.2},
        model="hf:some/model",
    )

    stub = templates.render_source_stub(result)
    assert "state_shared_attn_compress_kv" in stub, "state must be a kernel parameter"

    readme = templates.render_readme(result)
    assert "| `state.shared_attn.compress_kv` | state |" in readme
    assert "cross-module state" in readme
    # The two ways the snapshot misleads both have to be spelled out.
    assert "produces" in readme
    assert "left over from an earlier pass" in readme


def test_state_is_not_required_so_a_produced_entry_can_be_ignored() -> None:
    """Requiring every state entry would force the kernel to consume its own output."""
    from bootstrap.materialize import TensorRecord

    record = TensorRecord("state.shared_attn.topk_idxs", "state",
                          "tensors/state_shared_attn_topk_idxs.bin", "int32",
                          [1, 8192, 512], 8, "h")
    assert not record.required


def test_omitting_state_is_recorded_in_the_manifest(tmp_path: Path) -> None:
    """Whether the kernel was handed cross-module state has to be auditable after the fact.

    A repo with no `state` rows is ambiguous on its own — the group may have had none
    recorded, or the operator may have judged them to be its own output and left them out.
    """
    from bootstrap.materialize import Materialized, TensorRecord, write_manifest

    def built(state_included: bool) -> dict:
        result = Materialized(
            repo=tmp_path, group="20-Attention", module_id="layers.20.attention",
            sample_id="long-needle-8192-0", step=0, call_index=0,
            tensors=[TensorRecord("input", "input", "tensors/input.bin", "bfloat16",
                                  [1, 8192, 5120], 8, "h", required=True)],
            submodules=["layers.20.attn"], model="hf:m", state_included=state_included,
        )
        return json.loads(write_manifest(result).read_text())

    assert built(False)["state_included"] is False
    assert built(True)["state_included"] is True


def test_a_non_finite_worst_error_does_not_clear_the_ceiling() -> None:
    """`nan > ceiling` is false, so a NaN would otherwise pass the ceiling by not comparing.

    Reachable for real: an error reduction over a tensor holding a non-finite value prints
    `nan`, and the run can still exit 0 and print passed=1.
    """
    import ast as _ast

    from bootstrap.nki_checker import RunOutcome

    bar = {"RTOL": 0.1, "ATOL": 0.1, "MIN_COSINE": 0.9995, "MIN_PASS_FRACTION": 0.999,
           "MAX_ABS_ERR": 0.28}
    src = "".join(f"{k} = {v!r}\n" for k, v in bar.items())

    def findings(marker: str) -> list[str]:
        run = RunOutcome(
            ran=True, return_code=0, duration_s=1.0,
            output=f"##autohelix[latency_ms=1.0]\n##autohelix[max_abs_err={marker}]\n"
                   f"##autohelix[passed=1]\n",
        )
        return chk.check_pass_test(_ast.parse(src), run, bar).findings

    assert any("not a number" in f for f in findings("nan")), findings("nan")
    assert any("not a number" in f for f in findings("inf")), findings("inf")
    # A real value inside the ceiling still passes, so the guard is not over-broad.
    assert not any("not a number" in f or "past the" in f for f in findings("0.09"))


def test_a_bin_literal_that_names_no_file_is_not_checked_against_the_manifest() -> None:
    """Building a filename in a loop has to be allowed, or many-tensor modules cannot load.

    `".bin"` is an f-string tail and `"*.bin"` is a glob; neither names a file, so neither can
    be looked up. Rejecting them left one literal per tensor as the only accepted form, which
    is how a 778-expert validator grew to 52 KB and ran 903s against a 900s budget.
    """
    import ast as _ast

    allowed = {"tensors/input.bin", "tensors/w.bin"}
    empty = _ast.parse("x = 1")

    def manifest_findings(code: str) -> list[str]:
        r = chk.check_self_containment(empty, _ast.parse(code), allowed)
        return [f for f in r.findings if "not one of this repo's tensors" in f]

    # Forms that name no single file: allowed.
    assert manifest_findings('p = f"tensors/{n}.bin"') == []
    assert manifest_findings('from pathlib import Path\nPath("tensors").glob("*.bin")') == []
    assert manifest_findings('SUF = ".bin"') == []

    # A literal that does name a file is still held to the manifest.
    assert manifest_findings('p = "tensors/input.bin"') == []
    assert len(manifest_findings('p = "tensors/not_recorded.bin"')) == 1
    # And one reaching outside the repo is caught twice over.
    escaping = chk.check_self_containment(empty, _ast.parse('p = "../../trace/x.bin"'), allowed)
    assert any("outside" in f for f in escaping.findings)


def test_the_repos_own_golden_is_not_mistaken_for_an_escape() -> None:
    """(f) requires reading the golden; (b) must not forbid its name. Both, or the gate deadlocks.

    A module returning a tuple materializes its goldens as `reference_0.bin`, and
    `reference_` is on the forbidden-marker list to catch the frozen `reference_*.py`. Two
    modules spent five iterations each with a correct kernel and (b) as their only failure.
    """
    import ast as _ast

    allowed = {
        "tensors/input.bin",
        "tensors/reference_0.bin",
        "tensors/reference___tuple___0.bin",   # the name older repos already carry
    }
    reads = "\n".join(f'x{i} = open("{n}", "rb").read()' for i, n in enumerate(sorted(allowed)))
    r = chk.check_self_containment(_ast.parse("x = 1"), _ast.parse(reads), allowed)
    assert r.findings == [], r.findings

    # The markers still do their job for anything that is not a recorded tensor.
    escapes = (
        'a = open("reference_torch.py").read()',
        'b = open("../../trace/activations/x.bin", "rb").read()',
        'c = open("vendor/kernel.py").read()',
    )
    for code in escapes:
        bad = chk.check_self_containment(_ast.parse("x = 1"), _ast.parse(code), allowed)
        assert bad.findings, code


def test_a_tuple_output_is_named_by_position_not_by_its_encoding() -> None:
    """`{"__tuple__": [...]}` is how the artifact encodes a returned tuple, not structure."""
    from pathlib import Path

    from bootstrap.materialize import _flatten_tensors

    class _Reader:
        def decode(self, directory, value, device):
            return value["__tensor__"]["bin"]

    encoded = {"__tuple__": [{"__tensor__": {"bin": f"x{i}.bin"}} for i in range(4)]}
    names = [n for n, _, _ in _flatten_tensors(_Reader(), Path("."), encoded, "reference")]
    assert names == ["reference_0", "reference_1", "reference_2", "reference_3"], names

    # A lone tensor still reads unsuffixed, and a dict still names by key.
    single = {"__tensor__": {"bin": "s.bin"}}
    assert [n for n, _, _ in _flatten_tensors(_Reader(), Path("."), single, "reference")] == ["reference"]
