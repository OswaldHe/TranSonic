# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The gate, and its agreement with the preset that specifies it.

Mostly negative tests. A hidden gate is defined by what it refuses, and the half of it that
matters is the half an agent could otherwise satisfy without doing the work — so these check
that a plan leaving a module out, or quietly editing a cost model, or fabricating a duration,
is caught.

The drift tests exist because `preset.yaml` is the entire specification the exploration agent
sees while `checker.py` is what actually judges it. Anything the checker enforces and the
preset does not state is a trap rather than a requirement, and the two have to be changed
together.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from floorplan import baseline, checker, invariants
from floorplan.parser import Hardware, load_system
from floorplan.schema import Floorplan

pytestmark = pytest.mark.floorplan

PACKAGE = Path(__file__).resolve().parents[1] / "floorplan"
RUN_PRESET = yaml.safe_load((PACKAGE / "preset.yaml").read_text())
BUILD_PRESET = yaml.safe_load((PACKAGE / "build_preset.yaml").read_text())


# ---------------------------------------------------------------------------------------
# Manifest discovery
# ---------------------------------------------------------------------------------------
def test_manifest_is_found_from_a_worktree(tmp_path):
    """An iteration worktree lives inside the project, so the manifest is always above it.

    This is what lets the constraint command in `preset.yaml` be a fixed string with no
    per-project path — the property the whole hidden-gate arrangement rests on.
    """
    project = tmp_path / "project"
    worktree = project / ".autohelix" / "worktrees" / "iter-3"
    worktree.mkdir(parents=True)
    manifest = project / checker.MANIFEST_RELATIVE
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("{}")
    assert checker.find_manifest(worktree) == manifest


def test_missing_manifest_says_what_writes_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="floorplan init"):
        checker.find_manifest(tmp_path)


# ---------------------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------------------
def test_hash_tree_covers_sim_and_systems_but_ignores_pycache(tmp_path):
    (tmp_path / "sim" / "modules").mkdir(parents=True)
    (tmp_path / "sim" / "engine.py").write_text("x = 1\n")
    (tmp_path / "sim" / "modules" / "attention.py").write_text("y = 2\n")
    (tmp_path / "sim" / "__pycache__").mkdir()
    (tmp_path / "sim" / "__pycache__" / "engine.pyc").write_bytes(b"\x00")
    (tmp_path / "systems").mkdir()
    (tmp_path / "systems" / "probed.yaml").write_text("efficiency: {matmul_bf16: 0.3}\n")
    (tmp_path / "floorplan.yaml").write_text("target: x\n")

    hashes = checker.hash_tree(tmp_path)
    assert set(hashes) == {
        "sim/engine.py", "sim/modules/attention.py", "systems/probed.yaml",
    }


def test_editing_a_probed_coefficient_is_caught(tmp_path):
    """Raising an efficiency would speed up every plan without touching a cost model.

    `systems/probed.yaml` is the simulator as much as `sim/` is, so check (d) has to cover it.
    """
    (tmp_path / "systems").mkdir()
    probed = tmp_path / "systems" / "probed.yaml"
    probed.write_text("shared:\n  efficiency:\n    matmul_bf16: 0.2959\n")
    manifest = {"hashes": checker.hash_tree(tmp_path)}
    assert checker.check_frozen(tmp_path, manifest).passed

    probed.write_text("shared:\n  efficiency:\n    matmul_bf16: 0.9000\n")
    check = checker.check_frozen(tmp_path, manifest)
    assert not check.passed
    assert "systems/probed.yaml" in check.findings[0]


def test_frozen_detects_edit_addition_and_removal(tmp_path):
    (tmp_path / "sim").mkdir()
    (tmp_path / "sim" / "engine.py").write_text("x = 1\n")
    manifest = {"hashes": checker.hash_tree(tmp_path)}

    assert checker.check_frozen(tmp_path, manifest).passed

    (tmp_path / "sim" / "engine.py").write_text("x = 2\n")
    edited = checker.check_frozen(tmp_path, manifest)
    assert not edited.passed
    assert "has been modified" in edited.findings[0]

    (tmp_path / "sim" / "engine.py").write_text("x = 1\n")
    (tmp_path / "sim" / "extra.py").write_text("z = 3\n")
    added = checker.check_frozen(tmp_path, manifest)
    assert not added.passed
    assert "added after" in added.findings[0]

    (tmp_path / "sim" / "extra.py").unlink()
    (tmp_path / "sim" / "engine.py").unlink()
    removed = checker.check_frozen(tmp_path, manifest)
    assert not removed.passed
    assert "is missing" in removed.findings[0]


def test_unfrozen_manifest_fails_rather_than_passing_vacuously(tmp_path):
    """An empty hash set must not read as "nothing changed"."""
    check = checker.check_frozen(tmp_path, {"hashes": {}})
    assert not check.passed
    assert "never frozen" in check.findings[0]


# ---------------------------------------------------------------------------------------
# Coverage and dependencies
# ---------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def graph() -> dict:
    """A small graph in the real one's shape: a chain plus an off-path vision module."""
    return {
        "embed": {
            "kind": "embed", "inputs": ["tokens"], "outputs": ["h.0"], "param_bytes": 10,
        },
        "layers.0.attention": {
            "kind": "attention", "inputs": ["h.0"], "outputs": ["h.1"],
            "param_bytes": 20, "layer_indices": [0],
        },
        "lm_head": {
            "kind": "lm_head", "inputs": ["h.1"], "outputs": ["logits"], "param_bytes": 10,
        },
        "vision": {
            "kind": "vision", "inputs": ["pixels"], "outputs": ["v"], "param_bytes": 5,
        },
    }


def _plan(*modules: str, **overrides) -> Floorplan:
    placements = [
        {"module": module, "units": ["d0.l0"], "weights": {"tier": "hbm_bank"}}
        for module in modules
    ]
    data = {"version": 1, "target": "trn2-16device", "placements": placements}
    data.update(overrides)
    return Floorplan.from_dict(data)


def test_coverage_passes_on_a_complete_plan(graph):
    plan = _plan("embed", "layers.0.attention", "lm_head")
    assert checker.check_coverage(plan, graph).passed


def test_coverage_names_the_missing_modules(graph):
    plan = _plan("embed", "lm_head")
    check = checker.check_coverage(plan, graph)
    assert not check.passed
    assert "layers.0.attention" in check.findings[0]


def test_coverage_rejects_the_vision_tower(graph):
    plan = _plan("embed", "layers.0.attention", "lm_head", "vision")
    check = checker.check_coverage(plan, graph)
    assert not check.passed
    assert any("off the tokens -> logits path" in f for f in check.findings)


def test_coverage_rejects_an_unknown_module(graph):
    plan = _plan("embed", "layers.0.attention", "lm_head", "layers.99.attention")
    check = checker.check_coverage(plan, graph)
    assert not check.passed
    assert any("not in the graph" in f for f in check.findings)


def test_dependencies_catch_an_unplaced_producer(graph):
    # lm_head consumes h.1, produced by the attention module, which is not placed.
    plan = _plan("embed", "lm_head")
    check = checker.check_dependencies(plan, graph)
    assert not check.passed
    assert "layers.0.attention" in check.findings[0]


def test_dependencies_detect_a_cycle():
    cyclic = {
        "a": {"kind": "other", "inputs": ["y"], "outputs": ["x"], "param_bytes": 1},
        "b": {"kind": "other", "inputs": ["x"], "outputs": ["y"], "param_bytes": 1},
    }
    plan = _plan("a", "b")
    check = checker.check_dependencies(plan, cyclic)
    assert not check.passed
    assert "cycle" in check.findings[-1]


# ---------------------------------------------------------------------------------------
# Capacity reporting
# ---------------------------------------------------------------------------------------
def test_capacity_is_read_out_of_the_simulator_output():
    output = (
        "prefill_8192: the plan does not fit:\n"
        "  hbm_bank at d0.l0: 30.0 GiB exceeds 24.0 GiB by 6.0 GiB (layers.1.engram)\n"
    )
    check = checker.check_capacity(output)
    assert not check.passed
    assert "layers.1.engram" in check.findings[0]


def test_capacity_passes_on_clean_output():
    assert checker.check_capacity("prefill_128   12.3 ms\n").passed


# ---------------------------------------------------------------------------------------
# The cost-model scanner
# ---------------------------------------------------------------------------------------
def _write_model(tmp_path: Path, body: str) -> Path:
    (tmp_path / "sim" / "modules").mkdir(parents=True, exist_ok=True)
    path = tmp_path / "sim" / "modules" / "m.py"
    path.write_text(body)
    return path


def test_clean_cost_model_passes(tmp_path):
    _write_model(tmp_path, """
def emit(ctx):
    seconds = ctx.matmul_seconds(128, 512, 5120, "bf16")
    return [ctx.op("gemm", "tensor", seconds)]
""")
    assert invariants.check_costs_derived(tmp_path).passed


@pytest.mark.parametrize("body,expected", [
    ("import time\ndef emit(ctx):\n    return [ctx.op('a','tensor',time.time())]\n",
     "clock"),
    ("import random\ndef emit(ctx):\n    return [ctx.op('a','tensor',random.random())]\n",
     "randomness"),
    ("def emit(ctx):\n    open('/x/loop-00-Attention-long8192.log')\n    return []\n",
     "recorded kernel latencies"),
    ("def emit(ctx):\n    p='DeepSeek-V4.1-Flash-Trainium/layer-0-attention'\n    return []\n",
     "bootstrapped kernels"),
])
def test_banned_constructs_are_caught(tmp_path, body, expected):
    _write_model(tmp_path, body)
    check = invariants.check_costs_derived(tmp_path)
    assert not check.passed
    assert any(expected in finding for finding in check.findings), check.findings


def test_literal_duration_is_caught(tmp_path):
    """The one way around the costing helpers, so the one thing the scanner must see."""
    _write_model(tmp_path, """
def emit(ctx):
    return [ctx.op("gemm", "tensor", 0.0031)]
""")
    check = invariants.check_costs_derived(tmp_path)
    assert not check.passed
    assert "literal duration" in check.findings[0]


def test_zero_duration_marker_is_allowed(tmp_path):
    _write_model(tmp_path, """
def emit(ctx):
    return [ctx.op("marker", "cc", 0.0)]
""")
    assert invariants.check_costs_derived(tmp_path).passed


# ---------------------------------------------------------------------------------------
# Constraint citation
# ---------------------------------------------------------------------------------------
def test_uncited_constraints_are_reported(tmp_path):
    hardware = Hardware.from_system(load_system("trn2-16device", apply_probes=False))
    (tmp_path / "sim" / "modules").mkdir(parents=True)
    (tmp_path / "sim" / "constraints.py").write_text(
        "# constraint 1: gpsimd and tensor cannot share SBUF\n"
        "def check(hardware, plan):\n    return []\n"
    )
    check = invariants.check_constraints_covered(tmp_path, hardware)
    assert not check.passed
    # Item 1 is cited; the rest are not.
    assert "2" in check.findings[0]


def test_all_cited_passes(tmp_path):
    hardware = Hardware.from_system(load_system("trn2-16device", apply_probes=False))
    (tmp_path / "sim" / "modules").mkdir(parents=True)
    import re

    items = re.findall(r"^\s{0,3}(\d+)\.\s", hardware.constraints_text, re.MULTILINE)
    citations = "\n".join(f"# constraint {number}: handled" for number in items)
    (tmp_path / "sim" / "constraints.py").write_text(
        citations + "\ndef check(hardware, plan):\n    return []\n"
    )
    assert invariants.check_constraints_covered(tmp_path, hardware).passed


# ---------------------------------------------------------------------------------------
# Preset / checker agreement — the pair that must not drift
# ---------------------------------------------------------------------------------------
def test_run_preset_declares_the_four_metrics():
    declared = set()
    for entry in RUN_PRESET["metrics"]:
        declared |= set(entry["values"])
    assert declared == {
        "prefill_128_ms", "prefill_8192_ms", "decode_128_ms", "decode_8192_ms",
    }
    assert all(
        direction == "lower"
        for entry in RUN_PRESET["metrics"] for direction in entry["values"].values()
    )


def test_every_metric_has_a_ten_percent_gate():
    gates = {gate["metric"]: gate["max_regression_pct"]
             for gate in RUN_PRESET["acceptance"]["metric_gates"]}
    assert set(gates) == {
        "prefill_128_ms", "prefill_8192_ms", "decode_128_ms", "decode_8192_ms",
    }
    assert set(gates.values()) == {10}


def test_run_scope_is_only_the_floorplan():
    assert RUN_PRESET["scope"]["editable"] == list(checker.EDITABLE)


def test_build_scope_is_only_the_cost_models():
    assert BUILD_PRESET["scope"]["editable"] == ["sim/modules", "sim/constraints.py"]
    assert BUILD_PRESET["metrics"] == []


def test_budget_is_five_iterations_of_an_hour():
    assert RUN_PRESET["budget"]["iterations"] == 5
    assert RUN_PRESET["budget"]["iteration_time"] == "1h"
    assert BUILD_PRESET["budget"]["iterations"] == 2


def test_the_goal_states_what_the_gate_enforces():
    """Every hidden requirement has to be stated in the prose the agent actually reads."""
    goal = RUN_PRESET["goal"]
    for phrase in (
        "24 GiB",                    # the bank limit, check (e)
        "sum to exactly 1",          # fractions, check (b)
        "vision tower is out of scope",
        "10%",                       # the metric gate
        "271 modules",               # coverage, check (b)
    ):
        assert phrase in goal, f"preset.yaml's goal does not mention {phrase!r}"


def test_the_build_goal_forbids_the_bootstrapped_kernels():
    goal = BUILD_PRESET["goal"]
    assert "may not read" in goal
    assert "unoptimized" in goal


def test_reviewer_verdict_line_is_specified_in_both_presets():
    for preset in (RUN_PRESET, BUILD_PRESET):
        prompt = preset["reviewer"]["prompt"]
        assert "VERDICT: clean" in prompt
        assert "circumventing" in prompt


def test_checker_constraint_command_names_no_project_path():
    """The gate command is a fixed string; it must not embed a per-project path."""
    command = RUN_PRESET["constraints"][0]["command"]
    assert "--repo ." in command
    assert "/home/" not in command


# ---------------------------------------------------------------------------------------
# The generated baseline
# ---------------------------------------------------------------------------------------
def test_baseline_is_deterministic(graph):
    hardware = Hardware.from_system(load_system("trn2-16device", apply_probes=False))
    first = baseline.build(graph, hardware)
    second = baseline.build(graph, hardware)
    assert first.to_dict() == second.to_dict()


def test_baseline_places_everything_but_vision(graph):
    hardware = Hardware.from_system(load_system("trn2-16device", apply_probes=False))
    plan = baseline.build(graph, hardware)
    assert plan.modules() == {"embed", "layers.0.attention", "lm_head"}


def test_baseline_puts_engram_off_hbm():
    """94.56 GiB does not fit a 24 GiB bank, and a quarter of it leaves room for nothing."""
    hardware = Hardware.from_system(load_system("trn2-16device", apply_probes=False))
    graph = {
        "layers.1.engram": {
            "kind": "other", "inputs": ["hs.1"], "outputs": ["hs.1.engram"],
            "param_bytes": 101_533_000_000, "layer_indices": [1],
        },
    }
    plan = baseline.build(graph, hardware)
    assert plan.placements[0].residency.tier == baseline.ENGRAM_TIER == "host_dram"


def test_baseline_splits_along_natural_dimensions():
    hardware = Hardware.from_system(load_system("trn2-16device", apply_probes=False))
    graph = {
        "layers.0.attention": {
            "kind": "attention", "inputs": ["a"], "outputs": ["b"],
            "param_bytes": 1000, "layer_indices": [0],
        },
        "layers.0.ffn": {
            "kind": "mlp", "inputs": ["b"], "outputs": ["c"],
            "param_bytes": 1000, "layer_indices": [0],
        },
    }
    plan = baseline.build(graph, hardware)
    dims = {p.module: (p.splits[0].dim, p.splits[0].collective) for p in plan.placements}
    assert dims["layers.0.attention"] == ("head", "allreduce")
    assert dims["layers.0.ffn"] == ("expert", "all_to_all")


def test_baseline_keeps_tensor_parallel_groups_inside_a_device():
    """The whole point of the baseline's shape: TP collectives stay on the fast link."""
    hardware = Hardware.from_system(load_system("trn2-16device", apply_probes=False))
    graph = {
        "layers.0.attention": {
            "kind": "attention", "inputs": ["a"], "outputs": ["b"],
            "param_bytes": 1000, "layer_indices": [0],
        },
    }
    plan = baseline.build(graph, hardware)
    devices = {unit.device for unit in plan.placements[0].units}
    assert len(devices) == 1
