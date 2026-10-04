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

from optimization import constraints as cons
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


def test_unknown_top_level_keys_are_refused():
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
    assert slots[2]["enforcement"] == "off"


def test_the_derived_config_survives_the_round_trip_into_the_loop(tmp_path):
    """The operator's schedule, through the derived config, judged the same by the loop.

    The gap this closes cost two iterations of a real round. The derived config carried only the
    legacy `enforce` boolean, which is true for `hard` and `soft` alike; the loop re-parsed it and
    `_ENFORCE_ALIAS` turned every `True` back into `hard`. So a slot the operator wrote as `soft`
    reached the loop as `hard`, ran its checker without `--advisory`, and rejected a candidate that
    only owed a strict improvement. `soften_last: false` was lost the same way.

    Both halves were covered on their own. Only the round trip was not.
    """
    from optimization import constraints as cons

    schedule = [
        {"at": 1, "enforcement": "hard", "text": "one"},
        {"at": 2, "enforcement": "soft", "text": "two"},
        {"from": 3, "to": 4, "enforcement": "hard", "soften_last": False, "text": "three"},
        {"at": 5, "enforcement": "off", "text": "four"},
    ]
    raw = yaml.safe_load(_filled(tmp_path).read_text())
    raw["full"]["budget"]["iterations"] = 5
    raw["full"]["iteration_constraints"] = schedule
    path = tmp_path / "round-trip.yaml"
    path.write_text(yaml.safe_dump(raw))

    config = PipelineConfig.load(path)
    operator = cons.Schedule.from_config(schedule, max_iterations=5)
    through = cons.Schedule.from_config(
        config.derive_loop_config("full")["iteration_constraints"], max_iterations=5,
    )
    modes = {i: through.slot_for(i).enforcement_for(i) for i in range(1, 6)}
    assert modes == {i: operator.slot_for(i).enforcement_for(i) for i in range(1, 6)}
    # Spelled out, so a future change that makes both sides equally wrong still fails.
    assert modes == {1: "hard", 2: "soft", 3: "hard", 4: "hard", 5: "off"}


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
# the reviewer timeout, and the end-of-slot acceptance rule
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


def test_the_prompt_quotes_the_configured_allowance(tmp_path):
    """The template stated a literal 5%, which is wrong the moment an operator changes it."""
    from autohelix.config import Config
    from autohelix.prompt_template import render_template

    path = _filled(tmp_path)
    data = yaml.safe_load(path.read_text())
    data["submodule"]["acceptance"] = {"max_regression_pct": 12}
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    parsed = Config.from_dict(PipelineConfig.load(path).derive_loop_config("submodule"))
    allowance = next(g.max_regression_pct for g in parsed.acceptance.metric_gates
                     if g.metric == "latency_ms")
    assert allowance == 12

    rendered = render_template(presets.loop_prompt(), {
        "best_so_far": "33.48", "metric": "latency_ms",
        "regression_allowance": f"{allowance:g}",
    })
    assert "12% above it is rejected" in rendered
    assert "5%" not in rendered


def test_the_prompt_omits_the_allowance_when_no_gate_names_the_metric():
    """Better silent than stating a bound nothing enforces."""
    from autohelix.prompt_template import render_template

    rendered = render_template(presets.loop_prompt(), {
        "best_so_far": "33.48", "metric": "latency_ms", "regression_allowance": "",
    })
    assert "Best latency_ms so far" in rendered
    assert "above it is rejected" not in rendered


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


def _reviewer_loop(tmp_path, editable, reviewer_does):
    """An OptimizationLoop wired just enough to call `run_reviewer`, with upstream's stubbed out.

    The bug this covers is not in the restore logic, which was right — it is that the override read
    `config.scope.editable`, a path `Config` does not have. The YAML nests it under `scope:` and the
    dataclass flattens it, so the attribute error fired at the *baseline* review and killed the stage
    before iteration 1. Nothing tested it because nothing called `run_reviewer`.
    """
    from rich.console import Console

    from optimization.loop import OptimizationLoop

    class _Loop(OptimizationLoop):
        def __init__(self):
            pass

        def run_reviewer(self, worktree, iteration):  # the override under test
            return OptimizationLoop.run_reviewer(self, worktree, iteration)

    # Upstream's `run_reviewer` is what the override wraps; this stands in for the agent.
    from autohelix.harness import Harness
    original = Harness.run_reviewer
    Harness.run_reviewer = lambda self, worktree, iteration: (reviewer_does(), True)[1]

    loop = _Loop()
    loop.config = _FakeConfig()
    loop.config.editable = editable
    # `resolve_editable(editable, frozen, ...)` reads both. Set here rather than caught: an
    # `AttributeError` from a config path that does not exist is the exact bug that killed stage 3
    # at the baseline review, so the guard must not swallow that class of mistake.
    loop.config.frozen = []
    loop.console = Console(quiet=True)
    # The guard now also reopens the reviewer's commits and reverts what it touched out of scope,
    # which means it talks to the sandbox. These tests are about the byte restore, so the sandbox
    # records the calls and does nothing; `_reviewer_git_loop` exercises the real one.
    loop.sandbox = _RecordingSandbox()
    worktree = type("W", (), {"working_dir": tmp_path})()
    return loop, worktree, (Harness, original)


class _RecordingSandbox:
    def __init__(self):
        self.uncommitted = 0
        self.reverted: list[str] = []

    def uncommit_agent_changes(self, worktree):
        self.uncommitted += 1
        return 0

    def resolve_editable(self, editable, frozen, cwd=None):
        return list(editable) or None

    def revert_out_of_scope(self, worktree, effective_editable):
        return list(self.reverted)


def test_the_reviewer_cannot_replace_the_validated_candidate(tmp_path):
    gated = "# the kernel the gate measured\n"
    (tmp_path / "source.py").write_text(gated)

    def edit():
        (tmp_path / "source.py").write_text("# what the reviewer scribbled\n")

    loop, worktree, (cls, original) = _reviewer_loop(tmp_path, ["source.py"], edit)
    try:
        assert loop.run_reviewer(worktree, 3) is True
    finally:
        cls.run_reviewer = original
    assert (tmp_path / "source.py").read_text() == gated


def test_a_reviewer_that_touches_nothing_leaves_the_candidate_alone(tmp_path):
    gated = "# untouched\n"
    (tmp_path / "source.py").write_text(gated)
    loop, worktree, (cls, original) = _reviewer_loop(tmp_path, ["source.py"], lambda: None)
    try:
        assert loop.run_reviewer(worktree, 1) is True
    finally:
        cls.run_reviewer = original
    assert (tmp_path / "source.py").read_text() == gated


def test_an_empty_editable_scope_still_guards_the_deliverables(tmp_path):
    """`editable` unset means "everything but `frozen`", where guarding nothing would leave the
    reviewer free to rewrite the files whose hashes are themselves gate checks."""
    (tmp_path / "source.py").write_text("a\n")
    (tmp_path / "inference.py").write_text("b\n")

    def edit_both():
        (tmp_path / "source.py").write_text("x\n")
        (tmp_path / "inference.py").write_text("y\n")

    loop, worktree, (cls, original) = _reviewer_loop(tmp_path, [], edit_both)
    try:
        loop.run_reviewer(worktree, 1)
    finally:
        cls.run_reviewer = original
    assert (tmp_path / "source.py").read_text() == "a\n"
    assert (tmp_path / "inference.py").read_text() == "b\n"


def test_an_end_of_slot_violation_is_kept_when_it_improves(tmp_path):
    """Correct and strictly faster: the constraint shaped the search, and the search is over."""
    from optimization.constraints import Schedule

    schedule = Schedule.from_config([{"from": 1, "to": 3, "text": "NKI only."}])
    loop = _loop_stub(schedule, {3: _violation(3, "1-3")}, (100.0, 2), tmp_path)
    loop._current_iteration = 3
    assert loop._check_metric_gates({"latency_ms": 90.0}) is None


def test_an_end_of_slot_violation_is_rejected_when_it_does_not_improve(tmp_path):
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


def test_a_changed_slot_forces_a_recompile(tmp_path):
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


def test_hard_to_soft_is_drift(tmp_path):
    """The change `enforce`'s boolean could not see.

    A round configured `at: 4, enforcement: soft` enforced iteration 4 as hard and rejected a
    candidate that only owed a strict improvement. Drift compared two booleans, so `hard` and
    `soft` were the same value, and nothing anywhere recorded the disagreement.
    """
    from optimization import constraints as cons

    compiled_as = cons.Schedule.from_config(
        [{"at": 4, "text": "Use tensor_scalar.", "enforcement": "hard"}],
    )
    slot = compiled_as.slots[0]
    path = cons.checker_path(tmp_path, slot)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# --repo --json 'passed' 'findings'\n")
    cons.write_manifest(
        tmp_path,
        [cons.CompiledSlot(slot.label, slot.iterations, path, cons.sha256_file(path))],
        compiled_as,
    )
    assert cons.schedule_drift(tmp_path, compiled_as) == []

    softened = cons.Schedule.from_config(
        [{"at": 4, "text": "Use tensor_scalar.", "enforcement": "soft"}],
    )
    findings = cons.schedule_drift(tmp_path, softened)
    assert any("compiled as hard" in f for f in findings), findings


def test_soften_last_changing_one_iteration_of_a_range_is_drift(tmp_path):
    """Same three iterations, same text, same `enforcement`: only the last one's mode moves."""
    from optimization import constraints as cons

    strict = cons.Schedule.from_config(
        [{"from": 1, "to": 3, "text": "NKI only.", "soften_last": False}],
    )
    slot = strict.slots[0]
    path = cons.checker_path(tmp_path, slot)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# --repo --json 'passed' 'findings'\n")
    cons.write_manifest(
        tmp_path,
        [cons.CompiledSlot(slot.label, slot.iterations, path, cons.sha256_file(path))],
        strict,
    )
    assert cons.schedule_drift(tmp_path, strict) == []

    relaxed = cons.Schedule.from_config([{"from": 1, "to": 3, "text": "NKI only."}])
    findings = cons.schedule_drift(tmp_path, relaxed)
    assert any("iteration(s) 3" in f for f in findings), findings


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


# ======================================================================================
# the review of #6, second pass
# ======================================================================================


def test_a_resumed_summary_reads_verdicts_off_disk(tmp_path):
    """`_slot_verdicts` only holds what this instance judged, so before the fix every governed
    iteration from before a resume showed `—` while its `iter-N.json` sat on disk."""
    from optimization import constraints as cons

    project = tmp_path / "repo"
    (project / cons.CONSTRAINTS_REL).mkdir(parents=True)
    cons.report_path(project, 3).write_text(json.dumps({"passed": False, "findings": ["torch"]}))

    verdict = cons.persisted_verdict(project, 3, "1-3")
    assert verdict is not None and verdict.passed is False and verdict.checked


def test_a_missing_persisted_verdict_reads_as_unknown(tmp_path):
    from optimization import constraints as cons

    (tmp_path / cons.CONSTRAINTS_REL).mkdir(parents=True)
    assert cons.persisted_verdict(tmp_path, 9, "9-10") is None


def test_the_derived_commands_and_the_preflight_name_one_interpreter(tmp_path):
    """The preflight probed `sys.executable` while every command ran bare `python` from PATH, so it
    proved something about an interpreter no iteration ever used."""
    from optimization.config import GATE_PYTHON

    config = PipelineConfig.load(_filled(tmp_path))
    for stage in ("submodule", "full"):
        derived = config.derive_loop_config(stage)
        commands = [c["command"] for c in derived["constraints"]]
        commands += [m["command"] for m in derived["metrics"]]
        assert commands, stage
        for command in commands:
            assert command.startswith(f"{GATE_PYTHON} -m "), command


def test_a_legal_schedule_gap_is_a_warning_not_a_configuration_error(tmp_path):
    """`describe_for_prompt` and `run_iteration` both implement an uncovered iteration as
    unconstrained exploration, so a partial schedule must still pass `optimize check`."""
    path = _filled(tmp_path)
    data = yaml.safe_load(path.read_text())
    data["submodule"]["budget"]["iterations"] = 6
    data["submodule"]["iteration_constraints"] = [
        {"from": 1, "to": 3, "text": "NKI only."},
    ]
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    config = PipelineConfig.load(path)
    assert config.validate() == []
    assert any("no constraint slot" in w for w in config.warnings())


def test_a_multi_dimensional_projection_is_refused_rather_than_mis_described(tmp_path):
    """The declaration carries one `dim`, one `factor` and one reassembly operation, and the prompt
    used to be handed `projected[0].dim` with the product of every factor as its width."""
    from optimization.driver import Pipeline, StageError
    from optimization.projection import Factor, Projection

    pipeline = Pipeline(PipelineConfig.load(_filled(tmp_path)))
    mixed = Projection(
        module="m", planned=[Factor("head", 2), Factor("hidden", 2)], planned_units=4,
        planned_devices=1, projected=[Factor("head", 2), Factor("hidden", 2)], projected_units=4,
    )
    with pytest.raises(StageError, match="more than one dimension"):
        pipeline._require_expressible_cut(mixed)


def test_a_single_dimensional_projection_is_allowed(tmp_path):
    from optimization.driver import Pipeline
    from optimization.projection import Factor, Projection

    pipeline = Pipeline(PipelineConfig.load(_filled(tmp_path)))
    single = Projection(module="m", planned=[Factor("expert", 8)], planned_units=8,
                        planned_devices=2, projected=[Factor("expert", 4)], projected_units=4)
    pipeline._require_expressible_cut(single)  # does not raise


def test_the_glossary_the_feedback_prompt_points_at_exists():
    """In a wheel this used to resolve to a site-packages path with nothing at it."""
    from optimization.driver import _context_md

    assert _context_md().is_file()


# ======================================================================================
# another round of stage 5
# ======================================================================================


def _full_repo_with_a_round(repo: Path, best: str = "", best_ms: float = 14.84) -> Path:
    """A whole-module repo that has finished one round, as `rerun_full` expects to find it."""
    import subprocess

    (repo / ".autohelix" / "optimization").mkdir(parents=True, exist_ok=True)
    (repo / ".autohelix" / "notes").mkdir()
    (repo / ".autohelix" / "reviews").mkdir()
    (repo / "source.py").write_text("# round one kernel\n")
    (repo / ".autohelix" / "notes" / "iter-1.md").write_text("what round one learned\n")
    (repo / ".autohelix" / "reviews" / "iter-1.md").write_text("what the reviewer thought\n")
    (repo / ".autohelix" / "history.jsonl").write_text('{"iteration": 1}\n')
    for name in ("candidates", "constraints"):
        (repo / ".autohelix" / "optimization" / name).mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-q", "-m", "round one"], cwd=repo, check=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                          capture_output=True, text=True).stdout.strip()
    (repo / ".autohelix" / "optimization" / "full-summary.json").write_text(json.dumps({
        "best_commit": best or head, "best_ms": best_ms,
    }))
    return repo


def _round_one(tmp_path, **kwargs):
    """A pipeline whose `full_repo` exists and has one finished round in it."""
    from optimization.driver import Pipeline

    path = _filled(tmp_path)
    data = yaml.safe_load(path.read_text())
    data["workspace"]["root"] = str(tmp_path / "runs")
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    pipeline = Pipeline(PipelineConfig.load(path))
    repo = _full_repo_with_a_round(pipeline.config.full_repo, **kwargs)
    return pipeline, repo


def _archive_of(repo: Path) -> Path:
    """The round archive `_roll_round` wrote, under the same `.autohelix/archive/<timestamp>/` that
    `autohelix clear` uses."""
    archives = [d for d in (repo / ".autohelix" / "archive").iterdir()
                if (d / "round.json").is_file()]
    assert len(archives) == 1, archives
    return archives[0]


def test_rolling_a_round_archives_it_and_carries_the_notes_forward(tmp_path):
    """Round 2's agent should read what round 1 learned — that is why the round exists. The archive
    keeps its own copy, so it stays a complete record.

    Carried forward under the round's own name. Iteration numbering restarts at 1 each round, so an
    unstamped `iter-1.md` is exactly the name round 2's first iteration writes: the real round 2
    overwrote round 1's notes for four iterations and two of them were lost outright.
    """
    pipeline, repo = _round_one(tmp_path)

    number = pipeline._roll_round(repo, note="trying the expert skip", previous={"best_ms": 14.84})
    assert number == 1
    state = repo / ".autohelix"
    assert (state / "notes" / "iter-1-round1.md").is_file()
    assert (state / "reviews" / "iter-1-round1.md").is_file()
    # And the name the next round will write is free.
    assert not (state / "notes" / "iter-1.md").exists()
    assert not (state / "history.jsonl").exists()
    archived = _archive_of(repo)
    assert (archived / "history.jsonl").is_file()
    assert (archived / "notes" / "iter-1.md").is_file()
    assert (archived / "full-summary.json").is_file()
    assert json.loads((archived / "round.json").read_text())["note"] == "trying the expert skip"


def test_the_full_stage_delivers_the_best_commit_not_head(tmp_path):
    """The metric gate allows a regression, so HEAD can be a slower accepted candidate than an
    earlier one. Round 2 of the real stage 5 ended with HEAD at 14.8703 ms while an earlier commit
    held 14.8456, and only the number was reported from the better one."""
    import subprocess

    pipeline, repo = _round_one(tmp_path)
    best = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                          capture_output=True, text=True).stdout.strip()
    (repo / "source.py").write_text("# a later, slower accepted kernel\n")
    pipeline._commit_tree(repo, "iteration 5")

    pipeline._materialize_best(repo, best, 14.8456)
    assert (repo / "source.py").read_text() == "# round one kernel\n"
    assert not subprocess.run(["git", "status", "--porcelain"], cwd=repo,
                              capture_output=True, text=True).stdout.strip()


def test_delivering_the_best_commit_is_a_no_op_when_head_already_is_it(tmp_path):
    import subprocess

    pipeline, repo = _round_one(tmp_path)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                          capture_output=True, text=True).stdout.strip()
    before = subprocess.run(["git", "rev-list", "--count", "HEAD"], cwd=repo,
                            capture_output=True, text=True).stdout.strip()
    pipeline._materialize_best(repo, head, 14.84)
    after = subprocess.run(["git", "rev-list", "--count", "HEAD"], cwd=repo,
                           capture_output=True, text=True).stdout.strip()
    assert before == after, "no empty commit for a tree that is already the best"


def test_an_exhausted_budget_is_not_an_aborted_start(tmp_path):
    """`Harness.run()` walks an empty range when every budgeted iteration is on record, returning
    with the history unchanged -- which is exactly what an abort looks like. Re-running a finished
    stage therefore failed instead of reusing what it had already produced."""
    from optimization.driver import Pipeline

    class _Loop:
        def __init__(self, entries):
            self.history = type("H", (), {"load": staticmethod(lambda: entries)})()

    assert Pipeline._last_iteration(_Loop([])) == 0
    assert Pipeline._last_iteration(_Loop([{"iteration": 1}, {"iteration": 5}])) == 5
    # Malformed rows are skipped rather than crashing the stage on its way out.
    assert Pipeline._last_iteration(_Loop([{"iteration": "x"}, {}, {"iteration": 3}])) == 3


def test_stamping_a_round_twice_does_not_double_the_suffix(tmp_path):
    """A third round must not produce `iter-1-round1-round2.md`."""
    pipeline, repo = _round_one(tmp_path)
    notes = repo / ".autohelix" / "notes"

    assert pipeline._stamp_round(notes, 1) == 1
    assert (notes / "iter-1-round1.md").is_file()
    # Already stamped, so the second roll leaves it alone and finds nothing new to rename.
    assert pipeline._stamp_round(notes, 2) == 0
    assert (notes / "iter-1-round1.md").is_file()
    assert not (notes / "iter-1-round1-round2.md").exists()


def test_rolling_a_round_does_not_carry_observations_or_logs_forward(tmp_path):
    """Those belong to the round that produced them; leaving them mixes two rounds' measurements."""
    pipeline, repo = _round_one(tmp_path)
    for name in ("observations", "logs"):
        (repo / ".autohelix" / name).mkdir(exist_ok=True)
        (repo / ".autohelix" / name / "iter-1").mkdir(exist_ok=True)

    pipeline._roll_round(repo, note="", previous={})
    archived = _archive_of(repo)
    for name in ("observations", "logs"):
        assert (archived / name / "iter-1").exists(), name
        assert not (repo / ".autohelix" / name / "iter-1").exists(), name


def test_a_second_roll_gets_the_next_number(tmp_path):
    pipeline, repo = _round_one(tmp_path)
    # A round already on record, filed under an earlier timestamp as it would be in practice. The
    # count comes from the `round.json` markers, not from the number of archive directories, so an
    # `autohelix clear` archive alongside these does not shift the numbering.
    old = repo / ".autohelix" / "archive" / "20260101-000000"
    old.mkdir(parents=True)
    (old / "round.json").write_text(json.dumps({"round": 1}))
    (repo / ".autohelix" / "archive" / "20260102-000000").mkdir()  # a plain `clear`, not a round

    assert pipeline._roll_round(repo, note="", previous={}) == 2


def test_the_round_starts_from_the_named_kernel_on_a_clean_tree(tmp_path):
    """`Harness.run` refuses a dirty tree, and the restored kernel has to be committed."""
    import subprocess

    pipeline, repo = _round_one(tmp_path)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                          capture_output=True, text=True).stdout.strip()
    (repo / "source.py").write_text("# a later, slower iteration\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-q", "-m", "later"], cwd=repo, check=True)

    pipeline._restore_commit(repo, head, round_number=2)
    assert (repo / "source.py").read_text() == "# round one kernel\n"
    assert subprocess.run(["git", "status", "--porcelain"], cwd=repo,
                          capture_output=True, text=True).stdout.strip() == ""


def test_rerunning_without_a_previous_round_says_so(tmp_path):
    from optimization.driver import StageError

    pipeline, repo = _round_one(tmp_path)
    (repo / ".autohelix" / "optimization" / "full-summary.json").write_text("{}")
    with pytest.raises(StageError, match="no previous round"):
        pipeline.rerun_full()


def test_rerunning_without_an_assembly_says_so(tmp_path):
    from optimization.driver import StageError

    pipeline, repo = _round_one(tmp_path)
    (repo / "source.py").unlink()
    with pytest.raises(StageError, match="no source.py"):
        pipeline.rerun_full()


def test_the_new_round_is_numbered_after_the_one_it_archives(tmp_path):
    """The archive keeps the finished round's number, so the new round is the next one — printing
    the archive's number called round 2 "round 1" in the log."""
    pipeline, repo = _round_one(tmp_path)
    assert pipeline._roll_round(repo, note="", previous={}) == 1
    assert (_archive_of(repo) / "round.json").is_file()


def test_rerunning_after_a_failed_start_does_not_file_an_empty_round(tmp_path):
    """The command can fail after rolling — a startup refusal from the loop — and re-running it then
    must not archive a second, empty round and shift every later number."""
    pipeline, repo = _round_one(tmp_path)
    assert pipeline._roll_round(repo, note="", previous={}) == 1
    assert pipeline._roll_round(repo, note="", previous={}) == 1
    archives = [d for d in (repo / ".autohelix" / "archive").iterdir()
                if (d / "round.json").is_file()]
    assert len(archives) == 1


def test_the_last_rounds_result_is_found_in_the_archive(tmp_path):
    """After a roll there is no live summary, and `rerun-full` has to find the commit it filed."""
    pipeline, repo = _round_one(tmp_path)
    expected = json.loads(
        (repo / ".autohelix" / "optimization" / "full-summary.json").read_text()
    )["best_commit"]
    pipeline._roll_round(repo, note="", previous={})
    assert not (repo / ".autohelix" / "optimization" / "full-summary.json").exists()
    assert pipeline._previous_round(repo).get("best_commit") == expected


def test_committing_the_tree_is_a_no_op_when_it_is_clean(tmp_path):
    import subprocess

    pipeline, repo = _round_one(tmp_path)
    pipeline._commit_tree(repo, "absorb what the fixture left")
    before = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                            capture_output=True, text=True).stdout
    pipeline._commit_tree(repo, "nothing to do")
    after = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                           capture_output=True, text=True).stdout
    assert before == after


def test_committing_the_tree_absorbs_what_the_gates_run_left(tmp_path):
    """The gate runs the validator, which rewrites its tracked outputs under `build/`."""
    import subprocess

    pipeline, repo = _round_one(tmp_path)
    (repo / "build").mkdir(exist_ok=True)
    (repo / "build" / "rank0_output.pt").write_text("new output\n")
    pipeline._commit_tree(repo, "round 2 baseline")
    assert subprocess.run(["git", "status", "--porcelain"], cwd=repo,
                          capture_output=True, text=True).stdout.strip() == ""


# -- PR #7: a reviewer commit must not ride `git merge` into the deliverable --------------


def _reviewer_git_loop(tmp_path, reviewer_does):
    """A real `Sandbox` over a real git repo, so the commit path is exercised, not stubbed.

    The finding this covers: `Harness.run_iteration` runs the reviewer *after*
    `uncommit_agent_changes` and `revert_out_of_scope`, and `Sandbox.merge_worktree` then does
    `git merge <worktree.branch>`. That merge only *stages* the editable files for its own commit,
    but it carries every commit already on the branch — so a reviewer that commits reaches main
    with whatever it touched, and a working-tree byte restore never sees it.
    """
    import subprocess as sp

    from rich.console import Console

    from autohelix.harness import Harness
    from autohelix.sandbox import Sandbox, Worktree
    from optimization.loop import OptimizationLoop

    def git(*args):
        return sp.run(["git", *args], cwd=tmp_path, capture_output=True, text=True, check=True)

    git("init", "-q", "-b", "main")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    (tmp_path / "source.py").write_text("# the kernel the gate measured\n")
    (tmp_path / "inference.py").write_text("# the frozen validator\n")
    git("add", "-A")
    git("commit", "-q", "-m", "baseline")
    base = git("rev-parse", "HEAD").stdout.strip()

    class _Loop(OptimizationLoop):
        def __init__(self):
            pass

        def run_reviewer(self, worktree, iteration):
            return OptimizationLoop.run_reviewer(self, worktree, iteration)

    original = Harness.run_reviewer
    Harness.run_reviewer = lambda self, worktree, iteration: (reviewer_does(), True)[1]

    loop = _Loop()
    loop.config = _FakeConfig()
    loop.config.editable = ["source.py"]
    loop.config.frozen = []
    loop.console = Console(quiet=True)
    loop.sandbox = Sandbox(tmp_path)
    worktree = Worktree(path=tmp_path, branch="main", iteration=3,
                        working_dir=tmp_path, base_commit=base)
    return loop, worktree, (Harness, original), base, git


def test_a_reviewer_commit_outside_scope_is_reopened_and_reverted(tmp_path):
    def commit_the_validator():
        (tmp_path / "inference.py").write_text("# RTOL = 9.0\n")
        for args in (("add", "-A"), ("commit", "-q", "-m", "reviewer tidied the validator")):
            __import__("subprocess").run(["git", *args], cwd=tmp_path, check=True,
                                         capture_output=True)

    loop, worktree, (cls, original), base, git = _reviewer_git_loop(
        tmp_path, commit_the_validator,
    )
    try:
        assert loop.run_reviewer(worktree, 3) is True
    finally:
        cls.run_reviewer = original

    # The commit is gone from the branch, so `git merge` has nothing extra to carry...
    assert git("rev-parse", "HEAD").stdout.strip() == base
    # ...and the validator is the one the gate ran, not the one the reviewer wrote.
    assert (tmp_path / "inference.py").read_text() == "# the frozen validator\n"


def test_a_reviewer_commit_inside_scope_is_reopened_and_the_bytes_restored(tmp_path):
    """In scope, so `revert_out_of_scope` leaves it — the byte snapshot is what catches it."""
    def commit_the_kernel():
        (tmp_path / "source.py").write_text("# what the reviewer scribbled\n")
        for args in (("add", "-A"), ("commit", "-q", "-m", "reviewer edited the kernel")):
            __import__("subprocess").run(["git", *args], cwd=tmp_path, check=True,
                                         capture_output=True)

    loop, worktree, (cls, original), base, git = _reviewer_git_loop(tmp_path, commit_the_kernel)
    try:
        loop.run_reviewer(worktree, 3)
    finally:
        cls.run_reviewer = original

    assert git("rev-parse", "HEAD").stdout.strip() == base
    assert (tmp_path / "source.py").read_text() == "# the kernel the gate measured\n"


def test_a_reviewer_that_commits_nothing_leaves_the_branch_where_it_was(tmp_path):
    loop, worktree, (cls, original), base, git = _reviewer_git_loop(tmp_path, lambda: None)
    try:
        loop.run_reviewer(worktree, 3)
    finally:
        cls.run_reviewer = original
    assert git("rev-parse", "HEAD").stdout.strip() == base
    assert (tmp_path / "source.py").read_text() == "# the kernel the gate measured\n"
    assert (tmp_path / "inference.py").read_text() == "# the frozen validator\n"


# ---------------------------------------------------------------------------------------------
#  The checker contract and the validator that enforces it
# ---------------------------------------------------------------------------------------------

def test_the_contract_quotes_every_module_the_validator_allows():
    """The compiler is told the real rule, so it cannot be rejected for obeying a looser one.

    The contract used to say "import nothing outside the standard library" while the validator
    enforced 23 named modules. A checker that imported `operator` — the ordinary way to dispatch
    `ast.Add` when evaluating a trace-time constant — was therefore correct by the contract it was
    given and rejected by the code that read it, which stopped a whole pipeline after the compiler
    had already been paid for.
    """
    for module in cons.CHECKER_ALLOWED_IMPORTS:
        assert module in cons.CHECKER_CONTRACT, (
            f"'{module}' is allowed but the contract does not mention it, so the compiler cannot "
            f"know it may be used"
        )


def test_the_pure_stdlib_helpers_a_checker_needs_are_allowed():
    """Modules with no I/O and no dynamic import cannot reach what the list keeps out."""
    for module in ("operator", "bisect", "heapq", "statistics", "token"):
        assert module in cons.CHECKER_ALLOWED_IMPORTS


def test_the_modules_that_would_defeat_the_sandbox_stay_out():
    for module in ("subprocess", "shutil", "importlib", "socket", "pickle", "ctypes", "tempfile"):
        assert module not in cons.CHECKER_ALLOWED_IMPORTS


# ---------------------------------------------------------------------------------------------
#  Resuming a pipeline that stopped part-way
# ---------------------------------------------------------------------------------------------

def test_a_preparation_stage_that_already_passed_is_not_rebuilt(tmp_path):
    """`optimize all` has to pick a stopped run up, not start it over.

    `submodule` and `assemble` both open with `_set_aside`, so running either again archives the
    finished repo and builds a new one from the bootstrap. Resuming a run that died in stage 5 that
    way costs the tuned single-rank kernel and every iteration of its history.
    """
    from optimization.driver import Pipeline

    pipeline = Pipeline(PipelineConfig.load(_filled(tmp_path)))
    repo = pipeline.config.submodule_repo
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "source.py").write_text("def kernel(x): return x\n")
    pipeline.state_dir.mkdir(parents=True, exist_ok=True)
    (pipeline.state_dir / "submodule-attempt-1.json").write_text('{"passed": true, "report": ""}')

    outcome = pipeline._prepared("submodule")
    assert outcome is not None and outcome.ok
    assert outcome.stage == "submodule"


def test_a_passing_record_without_its_repo_does_not_count_as_prepared(tmp_path):
    """Half the evidence is not enough: the record says the gate passed, the repo says the thing
    it passed on is still there to run."""
    from optimization.driver import Pipeline

    pipeline = Pipeline(PipelineConfig.load(_filled(tmp_path)))
    pipeline.state_dir.mkdir(parents=True, exist_ok=True)
    (pipeline.state_dir / "submodule-attempt-1.json").write_text('{"passed": true, "report": ""}')
    assert pipeline._prepared("submodule") is None


def test_a_failed_attempt_does_not_count_as_prepared(tmp_path):
    from optimization.driver import Pipeline

    pipeline = Pipeline(PipelineConfig.load(_filled(tmp_path)))
    repo = pipeline.config.submodule_repo
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "source.py").write_text("def kernel(x): return x\n")
    pipeline.state_dir.mkdir(parents=True, exist_ok=True)
    (pipeline.state_dir / "submodule-attempt-1.json").write_text('{"passed": false, "report": ""}')
    assert pipeline._prepared("submodule") is None


def test_an_unreopenable_reviewer_commit_stops_the_merge(tmp_path):
    """The reopen step is the only thing between a reviewer commit and the main branch.

    It fails before reporting what it found, so there is no way to tell "nothing to reopen" from
    "commits I could not reopen". Carrying on lets `merge_worktree` take the branch as it stands —
    reviewer commits and all — into the deliverable that was just gated.
    """
    (tmp_path / "source.py").write_text("# the kernel the gate measured\n")

    def noop():
        return None

    loop, worktree, (cls, original) = _reviewer_loop(tmp_path, ["source.py"], noop)

    class _Failing(_RecordingSandbox):
        def uncommit_agent_changes(self, worktree):
            raise RuntimeError("git is wedged")

    loop.sandbox = _Failing()
    try:
        with pytest.raises(RuntimeError, match="must not be merged"):
            loop.run_reviewer(worktree, 1)
    finally:
        cls.run_reviewer = original


# ---------------------------------------------------------------------------------------------
#  The bar the reassembly is held to
# ---------------------------------------------------------------------------------------------

#: What each module's bootstrapped kernel achieved, and what its optimized four-rank module
#: achieved, as measured. The calibration of `tighten_bar` is only meaningful against real numbers:
#: a rule that rejects a good assembly is worse than no rule at all.
_MEASURED = {
    # module: (bootstrap (max_abs, cosine, pass_fraction), final (same)), expected verdict
    "00-Attention-B": ((0.0859375, 0.9999554805, 1.0),
                       (0.0859375, 0.9999444598, 1.0), True),
    "24-Attention": ((0.2518768311, 0.9998087800, 0.9999944448),
                     (0.1738281250, 0.9997568207, 0.9999969959), True),
    "00-Attention-C-iter2": ((0.1093750000, 0.9999013080, 1.0),
                             (0.1171875000, 0.9998860816, 1.0), True),
    "00-Attention-C-final": ((0.1093750000, 0.9999013080, 1.0),
                             (0.1406250000, 0.9995487502, 0.9999990225), False),
    "02-Attention": ((0.0981445000, 0.9998770000, 1.0),
                     (0.6093750000, 0.9996998017, 0.9998433352), False),
}

_DERIVED = {"RTOL": 0.1, "ATOL": 0.1, "MIN_COSINE": 0.9995,
            "MIN_PASS_FRACTION": 0.999, "MAX_ABS_ERR": 0.7875}


def _clears(bar, result):
    max_abs, cosine, pass_fraction = result
    return (max_abs <= bar["MAX_ABS_ERR"] and cosine >= bar["MIN_COSINE"]
            and pass_fraction >= bar["MIN_PASS_FRACTION"])


def test_the_bar_is_tightened_to_what_the_bootstrapped_kernel_achieved():
    """The derived bar describes the recorded output, not what the module can be computed to.

    `layers.2.attention`'s derived ceiling is 0.7875 and its bootstrapped kernel reaches 0.0981445,
    so the pipeline had 8x of headroom and used it: the assembly landed at 0.2773438 and stage 5 at
    0.609375, all of it passing a gate that was never binding.
    """
    achieved = {"max_abs_err": 0.0981445, "cosine": 0.9998770, "pass_fraction": 1.0}
    bar = materialize.tighten_bar(_DERIVED, achieved)
    assert bar["MAX_ABS_ERR"] == pytest.approx(0.0981445 * materialize.WORST_MARGIN)
    assert bar["MAX_ABS_ERR"] < _DERIVED["MAX_ABS_ERR"]
    assert bar["MIN_COSINE"] > _DERIVED["MIN_COSINE"]
    assert bar["MIN_PASS_FRACTION"] > _DERIVED["MIN_PASS_FRACTION"]


def test_tightening_never_loosens_any_of_the_five():
    """The derived bar stays the ceiling; this is a floor under it, never a relaxation."""
    for bootstrap, _final, _ok in _MEASURED.values():
        achieved = dict(zip(("max_abs_err", "cosine", "pass_fraction"), bootstrap))
        bar = materialize.tighten_bar(_DERIVED, achieved)
        assert bar["MAX_ABS_ERR"] <= _DERIVED["MAX_ABS_ERR"]
        assert bar["MIN_COSINE"] >= _DERIVED["MIN_COSINE"]
        assert bar["MIN_PASS_FRACTION"] >= _DERIVED["MIN_PASS_FRACTION"]


def test_the_elementwise_test_itself_is_never_moved():
    """`RTOL`/`ATOL` define what `MIN_PASS_FRACTION` counts, so moving them changes its meaning."""
    achieved = {"max_abs_err": 0.01, "cosine": 0.99999999, "pass_fraction": 1.0}
    bar = materialize.tighten_bar(_DERIVED, achieved)
    assert bar["RTOL"] == _DERIVED["RTOL"]
    assert bar["ATOL"] == _DERIVED["ATOL"]


def test_the_tightened_bar_admits_the_good_assemblies_and_refuses_the_bad(): 
    """Calibration, against every module measured so far.

    Two margins because the statistics fail differently: `MAX_ABS_ERR` is one element and moves
    when something structural changes, while cosine and the pass fraction aggregate over 42 M
    elements and drift whenever the arithmetic is reordered.
    """
    for name, (bootstrap, final, expected) in _MEASURED.items():
        achieved = dict(zip(("max_abs_err", "cosine", "pass_fraction"), bootstrap))
        bar = materialize.tighten_bar(_DERIVED, achieved)
        assert _clears(bar, final) is expected, (
            f"{name}: expected {'to clear' if expected else 'to be refused by'} the tightened bar, "
            f"bar={bar}, achieved={final}"
        )


def test_a_bootstrap_that_reaches_one_does_not_produce_an_unreachable_bar():
    """`pass_fraction = 1.0` would otherwise demand 1.0 of the reassembly: one stray element fails."""
    achieved = {"max_abs_err": 0.05, "cosine": 1.0, "pass_fraction": 1.0}
    bar = materialize.tighten_bar(_DERIVED, achieved)
    assert bar["MIN_PASS_FRACTION"] < 1.0
    assert bar["MIN_COSINE"] < 1.0


def test_a_baseline_measurement_with_no_numerics_leaves_the_derived_bar_alone():
    """An older run recorded no numerics beside its latency; it gets the behaviour it had."""
    assert materialize.tighten_bar(_DERIVED, None) == _DERIVED
    assert materialize.tighten_bar(_DERIVED, {}) == _DERIVED


def test_a_partially_reported_measurement_tightens_only_what_it_reported():
    bar = materialize.tighten_bar(_DERIVED, {"max_abs_err": 0.05})
    assert bar["MAX_ABS_ERR"] == pytest.approx(0.05 * materialize.WORST_MARGIN)
    assert bar["MIN_COSINE"] == _DERIVED["MIN_COSINE"]
    assert bar["MIN_PASS_FRACTION"] == _DERIVED["MIN_PASS_FRACTION"]


# ---------------------------------------------------------------------------------------------
#  The bar the single-rank loop is held to
# ---------------------------------------------------------------------------------------------

def _cut(tmp_path, measured, bar=None):
    """A submodule repo as stage 2 leaves it: a frozen validator, a manifest, and a gate verdict."""
    from optimization.driver import ACCURACY_MARKER
    bar = bar or {"RTOL": 0.1, "ATOL": 0.1, "MIN_COSINE": 0.9995,
                  "MIN_PASS_FRACTION": 0.999, "MAX_ABS_ERR": 0.7875}
    repo = tmp_path / "rank0"
    (repo / ".autohelix" / "optimization").mkdir(parents=True)
    (repo / "inference.py").write_text(
        "\n".join(f"{k} = {v}" for k, v in bar.items()) + "\n")
    (repo / ".autohelix" / "optimization" / "submodule.json").write_text(
        json.dumps({"tolerance": bar, "frozen": {"inference.py": "stale"}}))
    (repo / ".autohelix" / "optimization" / "gate.json").write_text(
        json.dumps({"passed": True, ACCURACY_MARKER: measured}))
    return repo


def test_the_submodule_bar_is_re_pinned_to_what_the_cut_achieved(tmp_path):
    """The agent-derived bar describes the rank's recorded output, not what the cut reached.

    `layers.2.attention`'s cut reached 0.0388967 against a bar of 0.7875, so iteration 1 could buy
    4x of latency with 2.2x of error and nothing objected — and the assembly and stage 5 then
    inherited that error unchanged.
    """
    from optimization.driver import Pipeline
    repo = _cut(tmp_path, 0.038896650075912476)

    lines = Pipeline._tighten_submodule_bar(Pipeline.__new__(Pipeline), repo)

    assert lines and "re-pinned" in lines[0]
    manifest = json.loads((repo / ".autohelix" / "optimization" / "submodule.json").read_text())
    assert manifest["tolerance"]["MAX_ABS_ERR"] == pytest.approx(0.038896650075912476 * 1.10)
    assert manifest["tolerance_derived"]["MAX_ABS_ERR"] == 0.7875
    assert f"MAX_ABS_ERR = {manifest['tolerance']['MAX_ABS_ERR']}" in (
        repo / "inference.py").read_text()


def test_re_pinning_refreshes_the_recorded_validator_hash(tmp_path):
    """`check_frozen_validator` compares the file against the manifest, so both move together or
    every iteration of the loop fails a check the candidate had no part in."""
    import hashlib

    from optimization.driver import Pipeline
    repo = _cut(tmp_path, 0.02)

    Pipeline._tighten_submodule_bar(Pipeline.__new__(Pipeline), repo)

    manifest = json.loads((repo / ".autohelix" / "optimization" / "submodule.json").read_text())
    actual = hashlib.sha256((repo / "inference.py").read_bytes()).hexdigest()
    assert manifest["frozen"]["inference.py"] == actual


def test_re_pinning_leaves_an_already_tight_bar_alone(tmp_path):
    """Nothing to do when the derived bar is already stricter than the cut's own measurement."""
    from optimization.driver import Pipeline
    repo = _cut(tmp_path, 0.5, bar={"RTOL": 0.1, "ATOL": 0.1, "MIN_COSINE": 0.9995,
                                    "MIN_PASS_FRACTION": 0.999, "MAX_ABS_ERR": 0.2})
    before = (repo / "inference.py").read_text()

    assert Pipeline._tighten_submodule_bar(Pipeline.__new__(Pipeline), repo) == []
    assert (repo / "inference.py").read_text() == before


def test_re_pinning_is_inert_without_a_measurement(tmp_path):
    """An older verdict published no accuracy; the cut keeps the bar its agent derived."""
    from optimization.driver import Pipeline
    repo = _cut(tmp_path, 0.02)
    (repo / ".autohelix" / "optimization" / "gate.json").write_text('{"passed": true}')
    before = (repo / "inference.py").read_text()

    assert Pipeline._tighten_submodule_bar(Pipeline.__new__(Pipeline), repo) == []
    assert (repo / "inference.py").read_text() == before


def test_the_re_pinned_bar_admits_every_loop_that_has_finished(tmp_path):
    """Calibration against the measured trajectories, as a test rather than a claim.

    Each module's stage-3 iteration 0 is its cut; the worst `max_abs_err` any later iteration
    reached is what the re-pinned bar has to admit — or refuse, for the one that went wrong.
    """
    from optimization.driver import Pipeline
    # module: (cut's max_abs_err, the worst any accepted iteration reached, should clear)
    trajectories = {
        "00-Attention-B": (0.02549302577972412, 0.0257452130317688, True),
        "00-Attention-C": (0.03719229996204376, 0.03590035438537598, True),
        "24-Attention": (0.2518768310546875, 0.12760743498802185, True),
        "02-Attention": (0.038896650075912476, 0.08605745434761047, False),
    }
    for name, (cut, worst, expected) in trajectories.items():
        repo = _cut(tmp_path / name, cut)
        Pipeline._tighten_submodule_bar(Pipeline.__new__(Pipeline), repo)
        ceiling = json.loads(
            (repo / ".autohelix" / "optimization" / "submodule.json").read_text()
        )["tolerance"]["MAX_ABS_ERR"]
        assert (worst <= ceiling) is expected, (
            f"{name}: worst iteration {worst} against a re-pinned ceiling of {ceiling}"
        )
