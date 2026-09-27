# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The pipeline config, the repos it materializes, and the drift between prompts and gates.

The drift tests are the reason this file matters. The preparation prompts are the *entire*
specification the preparation agents get, and the two gates are what actually judge them — so
anything a gate enforces and a prompt does not state is a trap rather than a requirement. `bootstrap`
and `floorplan` both keep that pair honest with tests; this does the same.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
import yaml

from optimization import materialize, module_checker, presets, submodule_checker
from optimization.config import ConfigError, PipelineConfig
from optimization.projection import project

pytestmark = pytest.mark.optimization


# ======================================================================================
# the config
# ======================================================================================


def _filled(tmp_path: Path, **overrides) -> Path:
    """A complete config against dummy paths that exist."""
    bootstrap_repo = tmp_path / "bootstrap"
    (bootstrap_repo / "tensors").mkdir(parents=True)
    (bootstrap_repo / "source.py").write_text("def kernel(x): return x\n")
    (bootstrap_repo / "inference.py").write_text("import source\n")
    (bootstrap_repo / "README.md").write_text(_readme())
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    scheme = tmp_path / "rank1.yaml"
    scheme.write_text(textwrap.dedent("""
        version: 1
        target: trn2-16device
        placements:
          - module: layers.1.ffn
            units: [d0.l0, d1.l0, d0.l1, d1.l1, d0.l2, d1.l2, d0.l3, d1.l3]
            splits:
              - {dim: expert, factor: 8, collective: all_to_all}
    """))

    data = yaml.safe_load(presets.config_template())
    data["module"] = {"id": "layers.1.ffn", "bootstrap_repo": str(bootstrap_repo),
                      "artifact": str(artifact)}
    data["floorplan"]["scheme"] = str(scheme)
    data["workspace"]["root"] = str(tmp_path / "runs")
    data["workspace"]["venv"] = str(tmp_path)
    for stage in ("submodule", "full"):
        data[stage]["reviewer"]["prompt"] = "Review source.py.\n"
    for key, value in overrides.items():
        data[key] = value
    path = tmp_path / "optimization.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def _readme() -> str:
    return textwrap.dedent("""
        # Bootstrap repo

        ## Tensors

        | tensor | role | file | dtype | shape | bytes |
        |---|---|---|---|---|---|
        | `input` | input | `tensors/input.bin` | bfloat16 | (1, 128, 5120) | 1310720 |
        | `reference` | golden | `tensors/reference.bin` | bfloat16 | (1, 128, 5120) | 1310720 |
        | `w1` | weight | `tensors/w1.bin` | int8 | (2304, 2560) | 5898240 |

        ## The numerical bar

        ```python
        RTOL = 0.1
        ATOL = 0.1
        MIN_COSINE = 0.9999
        MIN_PASS_FRACTION = 0.999
        MAX_ABS_ERR = 0.280469
        ```
    """)


def test_the_shipped_template_is_valid_yaml_and_parses():
    """A template that does not round-trip is a template nobody can start from."""
    data = yaml.safe_load(presets.config_template())
    assert set(data) >= {"module", "floorplan", "workspace", "submodule", "full"}


def test_the_template_refuses_to_run_unfilled():
    with pytest.raises(ConfigError, match="module.id"):
        PipelineConfig.from_dict(yaml.safe_load(presets.config_template()))


def test_a_filled_config_validates(tmp_path):
    config = PipelineConfig.load(_filled(tmp_path))
    assert config.validate() == []
    assert config.module_id == "layers.1.ffn"
    assert config.submodule.iterations == 10
    assert config.full.iterations == 5


def test_an_unfilled_reviewer_prompt_is_reported(tmp_path):
    path = _filled(tmp_path)
    data = yaml.safe_load(path.read_text())
    data["submodule"]["reviewer"]["prompt"] = "<FILL IN>\n"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    problems = PipelineConfig.load(path).validate()
    assert any("reviewer.prompt is still" in p for p in problems)


def test_a_missing_bootstrap_repo_is_reported(tmp_path):
    path = _filled(tmp_path)
    data = yaml.safe_load(path.read_text())
    data["module"]["bootstrap_repo"] = str(tmp_path / "nope")
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    assert any("bootstrap_repo does not exist" in p for p in PipelineConfig.load(path).validate())


def test_an_incomplete_bootstrap_repo_is_reported(tmp_path):
    """Pointing at a directory that is not a finished bootstrap project."""
    path = _filled(tmp_path)
    (tmp_path / "bootstrap" / "README.md").unlink()
    assert any("has no README.md" in p for p in PipelineConfig.load(path).validate())


def test_unknown_top_level_keys_are_refused(tmp_path):
    with pytest.raises(ConfigError, match="unknown top-level"):
        PipelineConfig.from_dict({"modules": {}})


def test_on_oversized_must_be_one_of_two_values(tmp_path):
    path = _filled(tmp_path)
    data = yaml.safe_load(path.read_text())
    data["floorplan"]["on_oversized"] = "shrug"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    with pytest.raises(ConfigError, match="on_oversized"):
        PipelineConfig.load(path)


def test_the_two_repos_get_distinct_paths(tmp_path):
    config = PipelineConfig.load(_filled(tmp_path))
    assert config.submodule_repo != config.full_repo
    assert config.submodule_repo.name.endswith("-rank0")
    assert config.full_repo.name.endswith("-full")


# ======================================================================================
# the derived loop config
# ======================================================================================


def test_the_derived_config_pins_what_is_not_the_operators(tmp_path):
    """The metric, the editable scope and the gate command are the pipeline's, not the config's."""
    config = PipelineConfig.load(_filled(tmp_path))
    derived = config.derive_loop_config("submodule")
    assert derived["scope"]["editable"] == ["source.py"]
    values = derived["metrics"][0]["values"]
    assert values["latency_ms"] == "lower"
    # The single-rank stage declares only what it can produce; see the iteration-0 tests below.
    assert set(values) == {"latency_ms"}
    assert [g["metric"] for g in derived["acceptance"]["metric_gates"]] == ["latency_ms"]
    assert "optimization.submodule_checker" in derived["constraints"][0]["command"]
    assert derived["acceptance"]["metric_gates"][0]["max_regression_pct"] == 5


def test_the_full_stage_gets_the_module_gate(tmp_path):
    config = PipelineConfig.load(_filled(tmp_path))
    derived = config.derive_loop_config("full")
    assert "optimization.module_checker" in derived["constraints"][0]["command"]


def test_the_derived_config_is_written_outside_the_repos(tmp_path):
    """It names the hidden gate, so it must not sit where the agent works."""
    config = PipelineConfig.load(_filled(tmp_path))
    path = config.write_loop_config("submodule")
    assert config.submodule_repo not in path.parents
    assert config.full_repo not in path.parents
    assert "optimization.submodule_checker" in path.read_text()


def test_the_derived_config_carries_the_schedule(tmp_path):
    config = PipelineConfig.load(_filled(tmp_path))
    derived = config.derive_loop_config("submodule")
    slots = derived["iteration_constraints"]
    assert [s["iterations"] for s in slots] == [[1, 2, 3], [4, 5, 6], [7, 8], [9, 10]]
    assert slots[2]["enforce"] is False


def test_the_derived_config_is_accepted_by_autohelixs_own_parser(tmp_path):
    """The derived document has to be a valid AutoHelix config, not merely YAML.

    Caught here rather than at iteration 0, where an invalid one is a preflight failure after the
    repos have been built and an agent has run.
    """
    from autohelix.config import Config

    config = PipelineConfig.load(_filled(tmp_path))
    for stage in ("submodule", "full"):
        raw = config.derive_loop_config(stage)
        parsed = Config.from_dict(raw)
        errors = [i for i in parsed.validate(raw_data=raw) if i.level == "error"]
        assert errors == [], (stage, [e.message for e in errors])
        # And no warnings about the pipeline-specific key, which would look like a typo.
        warnings = [i.message for i in parsed.validate(raw_data=raw) if i.level == "warning"]
        assert not any("iteration_constraints" in w for w in warnings), warnings


def test_an_unknown_stage_is_refused(tmp_path):
    config = PipelineConfig.load(_filled(tmp_path))
    with pytest.raises(ConfigError, match="unknown stage"):
        config.derive_loop_config("middle")


# ======================================================================================
# materialization
# ======================================================================================


def test_the_numerical_bar_is_read_from_the_readme(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(_readme())
    assert materialize.read_numerical_bar(readme) == {
        "RTOL": 0.1, "ATOL": 0.1, "MIN_COSINE": 0.9999,
        "MIN_PASS_FRACTION": 0.999, "MAX_ABS_ERR": 0.280469,
    }


def test_a_readme_with_no_bar_is_refused(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# nothing here\n")
    with pytest.raises(materialize.MaterializeError, match="MIN_COSINE"):
        materialize.read_numerical_bar(readme)


def test_the_tensor_table_and_golden_are_found(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(_readme())
    records = materialize.read_tensor_table(readme)
    assert [r.name for r in records] == ["input", "reference", "w1"]
    golden = materialize.golden_record(records)
    assert golden.file == "tensors/reference.bin"
    assert golden.dtype == "bfloat16"
    assert golden.shape == [1, 128, 5120]


def test_a_table_with_no_golden_is_refused(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(_readme().replace("| golden |", "| weight |"))
    with pytest.raises(materialize.MaterializeError, match="nothing for the assembly"):
        materialize.golden_record(materialize.read_tensor_table(readme))


def _bootstrap_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "bootstrap"
    (repo / "tensors").mkdir(parents=True, exist_ok=True)
    (repo / "source.py").write_text("def kernel(x): return x\n")
    (repo / "inference.py").write_text("import source\n")
    (repo / "README.md").write_text(_readme())
    for name in ("input.bin", "reference.bin", "w1.bin"):
        path = repo / "tensors" / name
        if path.exists():
            # Materialization makes the recorded tensors read-only on the shared inode, so a
            # fixture reused within one test has to restore write permission first.
            path.chmod(0o644)
        path.write_bytes(b"\x00\x01" * 8)
    (repo / "reference_torch.py").write_text("# what it computes\n")
    (repo / "reference_numerics.py").write_text("# what counts as matching\n")
    return repo


def test_the_submodule_repo_carries_what_the_agent_reads(tmp_path):
    projection = project("layers.1.ffn", [{"dim": "expert", "factor": 8}],
                         [f"d{d}.l{l}" for d in (0, 1) for l in range(4)])
    repo = tmp_path / "rank0"
    manifest = materialize.materialize_submodule(
        repo=repo, bootstrap_repo=_bootstrap_repo(tmp_path), artifact=tmp_path / "artifact",
        projection=projection, module_id="layers.1.ffn",
    )
    assert (repo / "FLOORPLAN.md").is_file()
    assert (repo / "README.md").is_file()
    assert (repo / "module" / "README.md").is_file()
    assert (repo / "module" / "tensors" / "reference.bin").is_file()
    assert (repo / "reference" / "reference_torch.py").is_file()
    # The stubs, so the gate has something to parse and the agent something to replace.
    assert "NotImplementedError" in (repo / "source.py").read_text()
    # What the pipeline knows and the agent does not get to choose.
    assert manifest["module_tolerance"]["MAX_ABS_ERR"] == 0.280469
    assert manifest["module_output"]["file"] == "module/tensors/reference.bin"
    assert manifest["projection"]["projected"]["units"] == 4
    # And what is left for the agent.
    assert manifest["tolerance"] == {} and manifest["tensors"] == {}


def test_the_submodule_repo_does_not_carry_the_bootstrap_runs_notes(tmp_path):
    """Its notes are about getting a kernel to work at all, which is already solved."""
    bootstrap = _bootstrap_repo(tmp_path)
    (bootstrap / ".autohelix" / "notes").mkdir(parents=True)
    (bootstrap / ".autohelix" / "notes" / "iter-1.md").write_text("how i got it compiling")
    projection = project("m", [{"dim": "expert", "factor": 8}],
                         [f"d{d}.l{l}" for d in (0, 1) for l in range(4)])
    repo = tmp_path / "rank0"
    materialize.materialize_submodule(
        repo=repo, bootstrap_repo=bootstrap, artifact=tmp_path / "artifact",
        projection=projection, module_id="m",
    )
    assert not (repo / "module" / ".autohelix").exists()


def test_profile_artifacts_are_gitignored(tmp_path):
    """A committed `.ntff` would be a stale measurement waiting to be believed."""
    repo = tmp_path / "repo"
    repo.mkdir()
    materialize.write_gitignore(repo)
    text = (repo / ".gitignore").read_text()
    assert "*.neff" in text and "*.ntff" in text and ".autohelix/" in text


def test_tensors_are_shared_rather_than_duplicated(tmp_path):
    """The MoE module's tensors are 6.8 GiB; three copies of them is not a design."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.bin").write_bytes(b"x" * 1024)
    dst = tmp_path / "dst"
    materialize._link_or_copy_tensors(src, dst)
    assert (dst / "a.bin").stat().st_ino == (src / "a.bin").stat().st_ino


def test_the_assembled_repo_records_both_bounds(tmp_path):
    projection = project("layers.1.ffn", [{"dim": "expert", "factor": 8}],
                         [f"d{d}.l{l}" for d in (0, 1) for l in range(4)])
    submodule_repo = tmp_path / "rank0"
    materialize.materialize_submodule(
        repo=submodule_repo, bootstrap_repo=_bootstrap_repo(tmp_path),
        artifact=tmp_path / "artifact", projection=projection, module_id="layers.1.ffn",
    )
    (submodule_repo / "submodule.json").write_text(json.dumps({"factor": 4}))
    repo = tmp_path / "full"
    manifest = materialize.materialize_full(
        repo=repo, bootstrap_repo=_bootstrap_repo(tmp_path), artifact=tmp_path / "artifact",
        submodule_repo=submodule_repo, projection=projection, module_id="layers.1.ffn",
        bootstrap_latency_ms=2793.29, submodule_latency_ms=700.0, best_commit=None,
    )
    assert manifest["ranks"] == 4
    assert manifest["baselines"]["bootstrap_latency_ms"] == 2793.29
    assert manifest["baselines"]["overhead_ceiling_ms"] == pytest.approx(770.0)
    assert manifest["tolerance"]["MAX_ABS_ERR"] == 0.280469
    # Its own tensors at the top level, since that is where its validator looks.
    assert (repo / "tensors" / "reference.bin").is_file()
    assert manifest["tensors"]["reference"]["sha256"]
    # The submodule it is built from, for reference.
    assert (repo / "submodule" / "submodule.json").is_file()
    # Left for the agent to fill when it freezes the validator it wrote.
    assert manifest["frozen"] == {}
    assert "2793.29" in (repo / "README.md").read_text()
    assert "770" in (repo / "README.md").read_text()


# ======================================================================================
# drift between the prompts and the gates
# ======================================================================================


def test_the_submodule_prompt_states_every_marker_its_gate_requires():
    """Anything the gate requires and the prompt omits is a trap, not a requirement."""
    prompt = presets.submodule_prompt()
    for marker in (submodule_checker.LATENCY_MARKER, submodule_checker.PASSED_MARKER,
                   submodule_checker.MAX_ABS_ERR_MARKER):
        assert marker in prompt, marker


def test_the_submodule_prompt_states_the_pinned_constants():
    prompt = presets.submodule_prompt()
    for name in (*submodule_checker.TOLERANCE_NAMES, submodule_checker.CEILING_NAME):
        assert name in prompt, name


def test_the_submodule_prompt_states_the_single_core_requirement():
    prompt = presets.submodule_prompt()
    assert "NEURON_RT_NUM_CORES=1" in prompt


def test_the_submodule_prompt_asks_for_every_declaration_field_the_gate_checks():
    prompt = presets.submodule_prompt()
    for field in ("module", "dim", "factor", "shard", "inputs", "outputs", "reassembly"):
        assert f'"{field}"' in prompt, field


def test_the_submodule_prompt_warns_that_the_reassembly_is_checked():
    """The agent has to know all N goldens are needed, not just its own rank's."""
    prompt = presets.submodule_prompt()
    assert "checked" in prompt and "goldens/" in prompt


def test_the_assemble_prompt_states_every_marker_its_gate_requires():
    prompt = presets.assemble_prompt()
    for marker in (module_checker.LATENCY_MARKER, module_checker.PASSED_MARKER,
                   module_checker.MAX_ABS_ERR_MARKER):
        assert marker in prompt, marker
    assert "latency_rank_" in prompt


def test_the_assemble_prompt_states_the_nki_collectives_requirement():
    """Check (c) refuses torch.distributed reductions, so the prompt has to say so."""
    prompt = presets.assemble_prompt()
    assert module_checker.COLLECTIVE_MODULE in prompt
    assert "torch.distributed" in prompt
    for allowed in ("init_process_group", "barrier", "get_rank"):
        assert allowed in prompt, allowed


def test_the_assemble_prompt_carries_the_six_toolchain_requirements():
    """Each one fails with an internal compiler error naming something else.

    Verified empirically on this toolchain before the pipeline was written; an agent that has to
    rediscover them burns the whole stage.
    """
    prompt = presets.assemble_prompt()
    for marker in ("kernel[2]", "name=", "NCC_IBIR440", "NCC_INLA001", "NCC_ILLC059",
                   "NCC_ISMP902", "shared_hbm", "ReplicaGroup"):
        assert marker in prompt, marker


def test_the_assemble_prompt_states_the_capture_flags():
    prompt = presets.assemble_prompt()
    for flag in ("--collectives-worker-count", "--collectives-workers-per-node",
                 "--collectives-profile-id"):
        assert flag in prompt, flag


def test_the_assemble_prompt_states_both_bounds_and_the_allowance():
    prompt = presets.assemble_prompt()
    assert "{{ bootstrap_latency }}" in prompt
    assert "{{ overhead_ceiling }}" in prompt
    assert "10%" in prompt


def test_the_assemble_prompt_says_the_bar_is_not_to_be_loosened():
    prompt = presets.assemble_prompt()
    assert "Unchanged." in prompt


def test_the_loop_prompt_tells_the_agent_the_validator_is_frozen():
    """Otherwise a whole iteration goes into editing a file that is reverted."""
    prompt = presets.loop_prompt()
    assert "reverted" in prompt
    assert "{{ iteration_constraint }}" in prompt


def test_the_loop_prompt_offers_the_reviewer_as_the_channel_for_layout_feedback():
    """The frozen validator pins the I/O format; the reviewer is how that reaches the operator."""
    assert "reviewer will carry it to the operator" in presets.loop_prompt()


def test_the_compiler_prompt_carries_the_checker_contract():
    from optimization.constraints import CHECKER_CONTRACT

    prompt = presets.compiler_prompt()
    assert "{{ contract }}" in prompt
    assert "--repo" in CHECKER_CONTRACT and "--json" in CHECKER_CONTRACT


def test_the_compiler_prompt_tells_it_to_check_neither_more_nor_less():
    """Both failure modes matter: over-checking fails compliant work, under-checking allows drift."""
    prompt = presets.compiler_prompt()
    assert "not more than the prose" in prompt
    assert "not less than the prose" in prompt
    assert "permissive" in prompt


def test_the_template_documents_the_two_editable_regions():
    """The operator has to be able to find where to write; `### EDIT ME` is the marker."""
    template = presets.config_template()
    assert template.count("### EDIT ME") >= 4
    assert template.count("<FILL IN>") >= 5


# ======================================================================================
# the reviewer timeout, and the end-of-interval acceptance rule
# ======================================================================================


def test_both_reviewers_get_the_long_timeout(tmp_path):
    """Reviewing a kernel is not a skim, and a reviewer killed mid-read leaves no verdict."""
    from optimization.config import REVIEWER_TIMEOUT_SECONDS

    config = PipelineConfig.load(_filled(tmp_path))
    for stage in ("submodule", "full"):
        reviewer = config.derive_loop_config(stage)["reviewer"]
        assert reviewer["timeout_seconds"] == REVIEWER_TIMEOUT_SECONDS == 2000


def test_the_reviewer_timeout_can_be_overridden(tmp_path):
    path = _filled(tmp_path)
    data = yaml.safe_load(path.read_text())
    data["submodule"]["reviewer"]["timeout_seconds"] = 600
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    config = PipelineConfig.load(path)
    assert config.derive_loop_config("submodule")["reviewer"]["timeout_seconds"] == 600
    assert config.derive_loop_config("full")["reviewer"]["timeout_seconds"] == 2000


def test_the_template_sets_the_reviewer_timeout_for_both_stages():
    data = yaml.safe_load(presets.config_template())
    assert data["submodule"]["reviewer"]["timeout_seconds"] == 2000
    assert data["full"]["reviewer"]["timeout_seconds"] == 2000


def test_autohelix_applies_the_reviewer_timeout(tmp_path):
    """It has to reach `AgentConfig`, not merely sit in the YAML."""
    from autohelix.config import Config

    config = PipelineConfig.load(_filled(tmp_path))
    parsed = Config.from_dict(config.derive_loop_config("submodule"))
    assert parsed.reviewer is not None
    assert parsed.reviewer.timeout_seconds == 2000


class _FakeHistory:
    def __init__(self, best):
        self._best = best

    def get_best_metrics(self, directions=None):
        return {"latency_ms": self._best} if self._best is not None else {}


class _FakeConfig:
    """Only what `_check_metric_gates` reads off the config."""

    def metric_directions(self):
        return {"latency_ms": "lower"}


def _loop_stub(schedule, verdicts, best, tmp_path):
    """An OptimizationLoop with only the attributes `_check_metric_gates` reaches for.

    Built by `__new__` rather than by constructing a real loop: the real one needs a git repo, a
    config file and a worktree, none of which this rule touches.
    """
    from rich.console import Console

    from optimization.loop import OptimizationLoop

    loop = OptimizationLoop.__new__(OptimizationLoop)
    loop.schedule = schedule
    loop._slot_verdicts = verdicts
    loop._current_iteration = None
    loop.history = _FakeHistory(best)
    loop.config = _FakeConfig()
    loop.console = Console(quiet=True)
    loop.project_path = tmp_path
    return loop


def _violation(iteration, label):
    from optimization.constraints import SlotVerdict

    return SlotVerdict(iteration, label, checked=True, passed=False,
                       findings=["source.py:4 imports torch"])


def test_an_end_of_interval_violation_is_kept_when_it_improves(tmp_path):
    """Correct and strictly faster: the constraint shaped the search, and the search is over."""
    from optimization.constraints import Schedule

    schedule = Schedule.from_config([{"from": 1, "to": 3, "text": "NKI only."}])
    loop = _loop_stub(schedule, {3: _violation(3, "1-3")}, (100.0, 2), tmp_path)
    loop._current_iteration = 3
    assert loop._check_metric_gates({"latency_ms": 90.0}) is None


def test_an_end_of_interval_violation_is_rejected_when_it_does_not_improve(tmp_path):
    """It does not get the 5% of slack a compliant iteration gets — it has to earn the escape."""
    from optimization.constraints import Schedule

    schedule = Schedule.from_config([{"from": 1, "to": 3, "text": "NKI only."}])
    loop = _loop_stub(schedule, {3: _violation(3, "1-3")}, (100.0, 2), tmp_path)
    loop._current_iteration = 3

    # Inside the 5% slack a compliant iteration would be allowed, and still rejected here.
    reason = loop._check_metric_gates({"latency_ms": 103.0})
    assert reason is not None
    assert "did not improve" in reason and "strict improvement" in reason
    # Equal is not an improvement either.
    assert loop._check_metric_gates({"latency_ms": 100.0}) is not None


def test_a_compliant_iteration_keeps_the_ordinary_slack(tmp_path):
    """The escape must not become the rule: a followed constraint still gets its 5%."""
    from optimization.constraints import Schedule, SlotVerdict

    schedule = Schedule.from_config([{"from": 1, "to": 3, "text": "NKI only."}])
    followed = SlotVerdict(3, "1-3", checked=True, passed=True)
    loop = _loop_stub(schedule, {3: followed}, (100.0, 2), tmp_path)
    loop._current_iteration = 3

    from unittest.mock import patch

    with patch("autohelix.harness.Harness._check_metric_gates", return_value=None) as upstream:
        assert loop._check_metric_gates({"latency_ms": 103.0}) is None
        assert upstream.called


def test_a_violation_with_no_metric_is_rejected(tmp_path):
    from optimization.constraints import Schedule

    schedule = Schedule.from_config([{"from": 1, "to": 3, "text": "NKI only."}])
    loop = _loop_stub(schedule, {3: _violation(3, "1-3")}, (100.0, 2), tmp_path)
    loop._current_iteration = 3
    reason = loop._check_metric_gates({"something_else": 1.0})
    assert reason is not None and "not produced" in reason


# ======================================================================================
# PR review: custody, schedule drift, quoting, the gate command
# ======================================================================================


def test_pipeline_owned_manifest_fields_are_restored(tmp_path):
    """The preparation agent writes the file the gate then trusts for the bar and the bounds."""
    from optimization import custody

    manifest_path = tmp_path / "module.json"
    original = {
        "module": "layers.1.ffn",
        "tolerance": {"MAX_ABS_ERR": 0.280469},
        "baselines": {"bootstrap_latency_ms": 2793.29, "submodule_latency_ms": 700.0},
        "ranks": 4,
        "frozen": {},
    }
    manifest_path.write_text(json.dumps(original))
    held = custody.take_custody(original, custody.MODULE_OWNED, tmp_path / "held.json")

    # The agent loosens the bar, raises a baseline, and adds a field of its own.
    manifest_path.write_text(json.dumps({
        "module": "layers.1.ffn",
        "tolerance": {"MAX_ABS_ERR": 99.0},
        "baselines": {"bootstrap_latency_ms": 99999.0, "submodule_latency_ms": 700.0},
        "ranks": 1,
        "frozen": {"inference.py": "abc"},
    }))
    changed = custody.restore(manifest_path, held)
    restored = json.loads(manifest_path.read_text())

    assert restored["tolerance"]["MAX_ABS_ERR"] == 0.280469
    assert restored["baselines"]["bootstrap_latency_ms"] == 2793.29
    assert restored["ranks"] == 4
    # And the agent's own field survives.
    assert restored["frozen"] == {"inference.py": "abc"}
    assert any("tolerance" in c for c in changed)
    assert any("baselines" in c for c in changed)


def test_a_dropped_pipeline_field_is_restored_and_reported(tmp_path):
    """A manifest rewritten from scratch loses what the agent did not think to copy."""
    from optimization import custody

    manifest_path = tmp_path / "submodule.json"
    original = {"module": "m", "module_tolerance": {"RTOL": 0.1}, "tensors": {}}
    manifest_path.write_text(json.dumps(original))
    held = custody.take_custody(original, custody.SUBMODULE_OWNED, tmp_path / "held.json")

    manifest_path.write_text(json.dumps({"tensors": {"a.bin": {}}}))
    changed = custody.restore(manifest_path, held)
    restored = json.loads(manifest_path.read_text())
    assert restored["module_tolerance"] == {"RTOL": 0.1}
    assert any("was dropped" in c for c in changed)


def test_an_untouched_manifest_reports_nothing(tmp_path):
    from optimization import custody

    manifest_path = tmp_path / "m.json"
    original = {"module": "m", "ranks": 4}
    manifest_path.write_text(json.dumps(original))
    held = custody.take_custody(original, custody.MODULE_OWNED, tmp_path / "held.json")
    assert custody.restore(manifest_path, held) == []


def test_edited_constraint_prose_forces_a_recompile(tmp_path):
    """Same range, different text: the old checker would judge a prompt that says something else."""
    from optimization import constraints as cons

    schedule = cons.Schedule.from_config([{"from": 1, "to": 3, "text": "NKI only."}])
    slot = schedule.slots[0]
    path = cons.checker_path(tmp_path, slot)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# --repo --json 'passed' 'findings'\n")
    cons.write_manifest(
        tmp_path,
        [cons.CompiledSlot(slot.label, slot.iterations, path, cons.sha256_file(path))],
        schedule,
    )
    assert cons.schedule_drift(tmp_path, schedule) == []

    edited = cons.Schedule.from_config([{"from": 1, "to": 3, "text": "NKI only, and no numpy."}])
    assert any("text has changed" in f for f in cons.schedule_drift(tmp_path, edited))


def test_a_changed_range_forces_a_recompile(tmp_path):
    from optimization import constraints as cons

    schedule = cons.Schedule.from_config([{"from": 1, "to": 3, "text": "NKI only."}])
    slot = schedule.slots[0]
    path = cons.checker_path(tmp_path, slot)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# --repo --json 'passed' 'findings'\n")
    cons.write_manifest(
        tmp_path,
        [cons.CompiledSlot(slot.label, slot.iterations, path, cons.sha256_file(path))],
        schedule,
    )
    widened = cons.Schedule.from_config([{"from": 1, "to": 4, "text": "NKI only."}])
    findings = cons.schedule_drift(tmp_path, widened)
    assert findings, "a new range leaves the manifest non-empty and its checker absent"


def test_a_toggled_enforcement_forces_a_recompile(tmp_path):
    from optimization import constraints as cons

    schedule = cons.Schedule.from_config([{"from": 1, "to": 3, "text": "NKI only."}])
    slot = schedule.slots[0]
    path = cons.checker_path(tmp_path, slot)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# --repo --json 'passed' 'findings'\n")
    cons.write_manifest(
        tmp_path,
        [cons.CompiledSlot(slot.label, slot.iterations, path, cons.sha256_file(path))],
        schedule,
    )
    off = cons.Schedule.from_config(
        [{"from": 1, "to": 3, "text": "NKI only.", "enforce": False}],
    )
    # With enforcement off there is nothing to compile, so drift is moot rather than reported.
    assert cons.schedule_drift(tmp_path, off) == []


def test_an_empty_schedule_never_reports_drift(tmp_path):
    from optimization import constraints as cons

    assert cons.schedule_drift(tmp_path, cons.Schedule()) == []


def test_the_checker_command_quotes_both_paths():
    """The command runs under `shell=True`, and the workspace root is the operator's to choose."""
    from optimization import constraints as cons

    command = cons.checker_command(
        Path("/work space/slot-1-3.py"), Path("/work space/iter-1.json"),
    )
    assert "'/work space/slot-1-3.py'" in command
    assert "'/work space/iter-1.json'" in command
    assert cons.checker_command(Path("/a/b.py"), Path("/a/r.json"), advisory=True).endswith(
        "--advisory"
    )


def test_tensors_are_carried_in_read_only(tmp_path):
    """A hard link shares the inode, and `open(path, "wb")` truncates it — the bootstrap repo's
    recorded golden is irreplaceable, and this filesystem has no reflink to fall back on."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.bin").write_bytes(b"x" * 64)
    dst = tmp_path / "dst"
    note = materialize._link_or_copy_tensors(src, dst)
    assert "read-only" in note
    with pytest.raises(PermissionError):
        (dst / "a.bin").open("wb")


def test_superseded_attempts_are_pruned(tmp_path):
    """Each set-aside attempt holds the tensor slices its agent cut, so the attic needs a ceiling."""
    import time as _time

    from rich.console import Console

    from optimization.driver import Pipeline

    pipeline = Pipeline.__new__(Pipeline)
    pipeline.state_dir = tmp_path / ".optimization"
    pipeline.console = Console(quiet=True)

    attic = pipeline.state_dir / "attempts"
    attic.mkdir(parents=True)
    for index in range(5):
        (attic / f"submodule-{index}").mkdir()
        _time.sleep(0.01)
    (attic / "assemble-0").mkdir()

    pipeline._prune_attempts("submodule", keep=2)
    kept = sorted(p.name for p in attic.iterdir())
    assert kept == ["assemble-0", "submodule-3", "submodule-4"], kept


# ======================================================================================
# iteration 0: the baseline has to produce every metric the stage declares
# ======================================================================================


def test_each_stage_declares_only_metrics_it_can_produce(tmp_path):
    """`Harness._capture_baseline` aborts the run when any declared metric is missing at iter 0.

    Declaring the per-rank numbers for the single-rank stage killed the first real attempt at
    stage 3 before its first iteration: a submodule has no ranks, so nothing emits them.
    """
    config = PipelineConfig.load(_filled(tmp_path))
    assert set(config.derive_loop_config("submodule")["metrics"][0]["values"]) == {"latency_ms"}
    assert set(config.derive_loop_config("full")["metrics"][0]["values"]) == {
        "latency_ms", "slowest_rank_ms", "rank_spread_ms",
    }


def _seeded(repo: Path, stage: str, verdict: dict) -> Path:
    from optimization.driver import Pipeline
    from rich.console import Console

    name = "submodule-gate.json" if stage == "submodule" else "module-gate.json"
    d = repo / ".autohelix" / "optimization"
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps(verdict))

    pipeline = Pipeline.__new__(Pipeline)
    pipeline.console = Console(quiet=True)
    pipeline._seed_baseline_verdict(repo, stage)
    return d / "gate.json"


def test_the_baseline_reads_the_acceptance_gates_measurement(tmp_path):
    """Iteration 0 runs the metric commands without the constraints, so there is no fresh verdict.

    The gate ran the validator on exactly this code minutes earlier; seeding its verdict is what
    gives iteration 0 a real measurement without a second device run.
    """
    from autohelix.checks import run_observables
    from autohelix.config import Config

    repo = tmp_path / "rank0"
    repo.mkdir()
    seeded = _seeded(repo, "submodule", {"passed": True, "latency_ms": 1363.48})
    assert seeded.is_file()

    config = PipelineConfig.load(_filled(tmp_path))
    parsed = Config.from_dict(config.derive_loop_config("submodule"))
    metrics: dict[str, float] = {}
    for result in run_observables(parsed, repo):
        metrics.update(result.values)
    assert metrics == {"latency_ms": 1363.48}
    assert not set(parsed.metric_directions()) - set(metrics)


def test_the_assembly_baseline_carries_the_per_rank_spread(tmp_path):
    from autohelix.checks import run_observables
    from autohelix.config import Config

    repo = tmp_path / "full"
    repo.mkdir()
    _seeded(repo, "full", {
        "passed": True, "latency_ms": 700.0,
        "rank_latency_ms": {"0": 740.0, "1": 700.0, "2": 715.0, "3": 760.0},
    })
    config = PipelineConfig.load(_filled(tmp_path))
    parsed = Config.from_dict(config.derive_loop_config("full"))
    metrics: dict[str, float] = {}
    for result in run_observables(parsed, repo):
        metrics.update(result.values)
    assert metrics["latency_ms"] == 700.0
    assert metrics["slowest_rank_ms"] == 760.0
    assert metrics["rank_spread_ms"] == 60.0
    assert not set(parsed.metric_directions()) - set(metrics)


def test_seeding_says_so_when_there_is_no_verdict_to_seed(tmp_path):
    """Silently seeding nothing would surface as an abort inside the loop instead."""
    from optimization.driver import Pipeline
    from rich.console import Console

    repo = tmp_path / "rank0"
    repo.mkdir()
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.console = Console(quiet=True)
    pipeline._seed_baseline_verdict(repo, "submodule")
    assert not (repo / ".autohelix" / "optimization" / "gate.json").exists()


def test_an_unconstrained_slot_is_not_reported_as_drift(tmp_path):
    """The manifest records every slot; only the enforceable ones are compiled.

    Comparing the recorded schedule against `enforceable()` reported each unconstrained slot as
    removed, which recompiled every checker on each `run_loop` invocation — two wasted agent runs on
    the first real pass. The shipped schedule has exactly this shape: 7-8 is deliberately empty.
    """
    from optimization import constraints as cons

    schedule = cons.Schedule.from_config([
        {"from": 1, "to": 3, "text": "NKI only."},
        {"from": 4, "to": 6, "text": "Both allowed."},
        {"from": 7, "to": 8, "text": ""},
        {"from": 9, "to": 10, "text": "Both allowed."},
    ], max_iterations=10)

    compiled = []
    for slot in schedule.enforceable():
        path = cons.checker_path(tmp_path, slot)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# --repo --json 'passed' 'findings'\n")
        compiled.append(
            cons.CompiledSlot(slot.label, slot.iterations, path, cons.sha256_file(path))
        )
    cons.write_manifest(tmp_path, compiled, schedule)

    assert [c.label for c in compiled] == ["1-3", "4-6", "9-10"]
    assert cons.schedule_drift(tmp_path, schedule) == []


def test_a_slot_that_stopped_being_enforceable_is_reported(tmp_path):
    """The real version of that finding: it was compiled, and now it would not be."""
    from optimization import constraints as cons

    schedule = cons.Schedule.from_config([{"from": 1, "to": 3, "text": "NKI only."}])
    slot = schedule.slots[0]
    path = cons.checker_path(tmp_path, slot)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# --repo --json 'passed' 'findings'\n")
    cons.write_manifest(
        tmp_path,
        [cons.CompiledSlot(slot.label, slot.iterations, path, cons.sha256_file(path))],
        schedule,
    )
    narrowed = cons.Schedule.from_config([
        {"from": 1, "to": 3, "text": "NKI only.", "enforce": False},
        {"from": 4, "to": 5, "text": "Both allowed."},
    ])
    findings = cons.schedule_drift(tmp_path, narrowed)
    assert any("no longer enforceable" in f for f in findings)
