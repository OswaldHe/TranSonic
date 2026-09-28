# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The two gates, and the reassembly verifier that makes the loose half of the design safe.

Mostly negative tests, for the same reason `test_bootstrap_checker.py` is: a hidden gate is defined
by what it refuses. The refusals that matter most here are the ones an agent could otherwise satisfy
without doing the work — a collective taken from `torch.distributed` instead of the traced graph, a
fastest-rank latency that is not one of the numbers the ranks printed, a cut whose parts do not sum
to the whole.

Every test is static or arithmetic. Nothing here needs a device, which is the point: the parts of
the gates that decide whether a run was honest are the parts that can be tested without one.
"""

from __future__ import annotations

import ast
import json
import textwrap
from pathlib import Path

import numpy as np
import pytest

from optimization import candidate, module_checker, submodule_checker
from optimization.recipe import RecipeError, apply_recipe, compare, verify_recipe

pytestmark = pytest.mark.optimization


def _tree(source: str) -> ast.Module:
    return ast.parse(textwrap.dedent(source))


# ======================================================================================
# the helpers both gates are built on
# ======================================================================================


def test_the_bootstrap_helpers_this_package_reuses_still_exist():
    """`optimization/candidate.py` imports private names from `bootstrap.nki_checker`.

    Deliberate — one definition of "path-like literal" is worth more than two — and the risk is
    that a refactor over there renames one and this package silently stops checking something. This
    test turns that into a failure here.
    """
    import bootstrap.nki_checker as nki

    for name in ("_parse", "_import_roots", "_string_literals", "_path_like",
                 "_decorator_names", "_module_level_numbers", "CheckResult", "CheckerError",
                 "SYNTHETIC_CONSTRUCTORS", "SYNTHETIC_INPLACE", "FORBIDDEN_PATH_MARKERS",
                 "DYNAMIC_IMPORT_NAMES", "FILE_IO_NAMES"):
        assert hasattr(nki, name), name


def test_imports_outside_the_allowlist_are_found():
    findings = candidate.import_findings(
        _tree("import json\nimport nki\nimport tensorflow\n"),
        frozenset({"nki"}), "source.py",
    )
    assert len(findings) == 1
    assert "tensorflow" in findings[0]


def test_dynamic_imports_make_the_allowlist_unenforceable():
    findings = candidate.import_findings(
        _tree("import importlib\nm = importlib.import_module('torch')\n"),
        frozenset({"importlib"}), "source.py",
    )
    assert any("unenforceable" in f for f in findings)


def test_a_path_escaping_the_repo_is_found():
    findings = candidate.path_findings(
        _tree("A = '/home/ubuntu/workspace/bootstrap-runs/01-MoE/tensors/reference.bin'\n"
              "B = '../../checkpoint/model.safetensors'\n"
              "C = 'tensors/input.bin'\n"),
        "inference.py",
    )
    assert len(findings) == 2
    assert all("tensors/input.bin" not in f for f in findings)


def test_prose_mentioning_a_filename_is_not_a_path():
    """Without this the rules fire on error messages, which would reject correct work."""
    assert candidate.path_findings(
        _tree("MSG = 'load the tensors from tensors/*.bin before running'\n"), "inference.py",
    ) == []


def test_fabricated_tensors_are_found():
    findings = candidate.fabrication_findings(
        _tree("import torch\nx = torch.randn(4, 4)\ny = torch.zeros(4, 4)\n"), "inference.py",
    )
    assert len(findings) == 1
    assert "randn" in findings[0]


def test_allocating_an_output_buffer_is_not_fabrication():
    """`zeros`/`empty` are how an output buffer is made, and a zeroed weight cannot pass anyway."""
    assert candidate.fabrication_findings(
        _tree("import torch\nout = torch.zeros(8)\nbuf = torch.empty(8)\n"), "inference.py",
    ) == []


def test_a_moved_tolerance_constant_is_found_in_either_direction():
    bar = {"RTOL": 0.1, "MIN_COSINE": 0.9999}
    loosened = candidate.pinned_constants(_tree("RTOL = 0.2\nMIN_COSINE = 0.9999\n"), bar, "inference.py")
    tightened = candidate.pinned_constants(_tree("RTOL = 0.1\nMIN_COSINE = 0.99999\n"), bar,
                                      "inference.py")
    assert any("0.2" in f for f in loosened)
    # A tightened bar looks virtuous and is still a different experiment from the recorded one.
    assert any("MIN_COSINE" in f for f in tightened)


def test_a_computed_tolerance_reads_as_a_missing_one():
    """The bar has to be legible at a glance, so only bare literals count."""
    findings = candidate.pinned_constants(_tree("RTOL = 2e-2 * 5\n"), {"RTOL": 0.1}, "inference.py")
    assert any("number literal" in f for f in findings)


def test_marker_values_take_the_last_occurrence():
    """A warm-up print must not shadow the real result."""
    output = "##autohelix[latency_ms=99.0]\n...\n##autohelix[latency_ms=12.5]\n"
    assert candidate.marker_value(output, "latency_ms") == 12.5
    assert candidate.marker_values(output, "latency_ms") == [99.0, 12.5]


def test_a_missing_marker_is_none_not_zero():
    assert candidate.marker_value("nothing here", "latency_ms") is None


def test_stale_profile_artifacts_are_not_counted_as_fresh(tmp_path):
    """A candidate that produces nothing must not pass on the previous iteration's profile."""
    import os
    import time

    old = tmp_path / "old.neff"
    old.write_bytes(b"x")
    os.utime(old, (time.time() - 10_000, time.time() - 10_000))
    before = candidate._artifact_mtimes(tmp_path)

    started = time.time()
    fresh = tmp_path / "new.ntff"
    fresh.write_bytes(b"y")
    found = candidate._fresh_artifacts(tmp_path, before, started)
    assert found["ntff"] == ["new.ntff"]
    assert found["neff"] == []


# ======================================================================================
# the whole-module gate
# ======================================================================================


def _module_repo(tmp_path: Path, source: str, inference: str = "") -> Path:
    repo = tmp_path / "full"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "source.py").write_text(textwrap.dedent(source))
    # The default validator launches the kernel the way the real one does — `kernel[2](...)`, the
    # LNC=2 grid launch — because check (a) now requires the frozen validator to actually drive
    # the entry point, not merely import the module that defines it.
    (repo / "inference.py").write_text(textwrap.dedent(inference or """
        import source
        from source import kernel
        RTOL = 0.1

        def run(x, group):
            return kernel[2](x, group)
        """))
    return repo


NKI_SOURCE = """
    import nki
    import nki.collectives as ncc
    import nki.isa as nisa
    import nki.language as nl

    def kernel(x, replica_group):
        src = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm, name="src")
        dst = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm, name="dst")
        out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=src, src=x)
        ncc.all_reduce(srcs=[src], dsts=[dst], replica_group=replica_group, op=nl.add)
        nisa.dma_copy(dst=out, src=dst)
        return out
"""


def test_an_nki_collective_passes(tmp_path):
    repo = _module_repo(tmp_path, NKI_SOURCE)
    result = module_checker.check_nki_collectives(repo)
    assert result.passed, result.findings


def test_a_torch_distributed_reduction_is_refused(tmp_path):
    """It works, and it measures a different machine than the floorplan is about."""
    repo = _module_repo(tmp_path, """
        import torch
        import torch.distributed as dist

        def kernel(x):
            dist.all_reduce(x)
            return x
    """)
    result = module_checker.check_nki_collectives(repo)
    assert not result.passed
    assert any("dist.all_reduce" in f for f in result.findings)


def test_an_xla_reduction_is_refused(tmp_path):
    repo = _module_repo(tmp_path, """
        import torch_xla.core.xla_model as xm

        def kernel(x):
            return xm.all_reduce('sum', x)
    """)
    result = module_checker.check_nki_collectives(repo)
    assert not result.passed
    assert any("xm.all_reduce" in f for f in result.findings)


def test_process_group_setup_is_allowed(tmp_path):
    """torch.distributed may organize the four processes; it may not move tensor data."""
    repo = _module_repo(tmp_path, NKI_SOURCE, inference="""
        import torch.distributed as dist
        import source
        RTOL = 0.1
        dist.init_process_group('xla')
        dist.barrier()
        r = dist.get_rank()
        dist.destroy_process_group()
    """)
    result = module_checker.check_nki_collectives(repo)
    assert result.passed, result.findings


def test_no_collective_at_all_is_refused(tmp_path):
    """Four ranks each holding part of the module cannot produce the whole module without one."""
    repo = _module_repo(tmp_path, """
        import nki.language as nl

        def kernel(x):
            return x
    """)
    result = module_checker.check_nki_collectives(repo)
    assert not result.passed
    assert any("no collective at all" in f for f in result.findings)


def _run(output: str, **kwargs) -> candidate.RunOutcome:
    defaults = dict(ran=True, return_code=0, output=output,
                    artifacts={"neff": ["model.neff"],
                               "ntff": [f"profile_rank_{r}.ntff" for r in range(4)]})
    defaults.update(kwargs)
    return candidate.RunOutcome(**defaults)


def _rank_markers(values: list[float]) -> str:
    return "".join(f"##autohelix[latency_rank_{r}_ms={v}]\n" for r, v in enumerate(values))


def test_the_reported_latency_must_be_the_fastest_rank():
    """Otherwise "the fastest rank" is the candidate's unverifiable claim."""
    output = _rank_markers([40.0, 32.0, 35.0, 44.0]) + "##autohelix[latency_ms=20.0]\n"
    result = module_checker.check_measurement(_run(output))
    assert not result.passed
    assert any("fastest rank reported 32" in f for f in result.findings)


def test_a_consistent_fastest_rank_passes_and_reports_the_spread():
    output = _rank_markers([40.0, 32.0, 35.0, 44.0]) + "##autohelix[latency_ms=32.0]\n"
    result = module_checker.check_measurement(_run(output))
    assert result.passed, result.findings
    assert "spread 12" in result.summary


def test_missing_per_rank_markers_are_refused():
    output = _rank_markers([40.0, 32.0]) + "##autohelix[latency_ms=32.0]\n"
    result = module_checker.check_all_ranks(_run(output))
    assert not result.passed
    assert any("rank(s) [2, 3]" in f for f in result.findings)


def test_too_few_per_rank_profiles_are_refused():
    output = _rank_markers([40.0, 32.0, 35.0, 44.0]) + "##autohelix[latency_ms=32.0]\n"
    result = module_checker.check_measurement(
        _run(output, artifacts={"neff": ["m.neff"], "ntff": ["profile_rank_0.ntff"]}),
    )
    assert not result.passed
    assert any("1 fresh per-rank" in f for f in result.findings)


def test_not_beating_the_bootstrap_is_refused():
    manifest = {"baselines": {"bootstrap_latency_ms": 2793.29, "submodule_latency_ms": 700.0}}
    result = module_checker.check_faster(_run("##autohelix[latency_ms=2800.0]\n"), manifest)
    assert not result.passed
    assert any("not faster" in f for f in result.findings)


def test_beating_the_bootstrap_reports_the_speedup():
    manifest = {"baselines": {"bootstrap_latency_ms": 2793.29, "submodule_latency_ms": 700.0}}
    result = module_checker.check_faster(_run("##autohelix[latency_ms=760.0]\n"), manifest)
    assert result.passed
    assert "3.68x" in result.summary


def test_collective_overhead_past_ten_percent_is_refused():
    """The bound that makes a lazy cut fail: three idle ranks cannot hide inside 10%."""
    manifest = {"baselines": {"bootstrap_latency_ms": 2793.29, "submodule_latency_ms": 700.0}}
    result = module_checker.check_overhead(_run("##autohelix[latency_ms=800.0]\n"), manifest)
    assert not result.passed
    assert any("14.3% over" in f for f in result.findings)


def test_overhead_inside_ten_percent_passes():
    manifest = {"baselines": {"bootstrap_latency_ms": 2793.29, "submodule_latency_ms": 700.0}}
    result = module_checker.check_overhead(_run("##autohelix[latency_ms=760.0]\n"), manifest)
    assert result.passed
    assert "+8.6%" in result.summary


def test_an_edited_validator_is_caught(tmp_path):
    """Scope enforcement reverts it, but that depends on git noticing; the hash does not."""
    repo = _module_repo(tmp_path, NKI_SOURCE)
    manifest = {"entry_point": "kernel",
                "frozen": {"inference.py": "0" * 64}}
    result = module_checker.check_frozen_validator(repo, manifest)
    assert not result.passed
    assert any("has changed since stage 4 froze it" in f for f in result.findings)


def test_an_unedited_validator_passes(tmp_path):
    repo = _module_repo(tmp_path, NKI_SOURCE)
    digest = module_checker._sha256(repo / "inference.py")
    result = module_checker.check_frozen_validator(
        repo, {"entry_point": "kernel", "frozen": {"inference.py": digest}},
    )
    assert result.passed, result.findings


def test_the_module_bar_is_not_re_derived():
    """Stage 5 inherits the bootstrapped module's five constants; there is no default."""
    with pytest.raises(candidate.CheckerError, match="MIN_COSINE"):
        module_checker.expected_tolerance({"tolerance": {"RTOL": 0.1, "ATOL": 0.1}})


# ======================================================================================
# the submodule gate
# ======================================================================================


def test_the_submodule_bar_has_no_default():
    """Unlike bootstrap's, which can fall back to the bfloat16 row from a known dtype.

    A submodule's golden is an intermediate the agent chose, so its bar was derived when the repo
    was built or it does not exist. A default would let a repo pass at a bar nobody chose.
    """
    with pytest.raises(submodule_checker.CheckerError, match="numerical bar"):
        submodule_checker.expected_tolerance({})


def test_torch_is_allowed_in_a_submodule_kernel(tmp_path):
    """From iteration 4 the schedule permits it, so the gate must not contradict the schedule."""
    repo = tmp_path / "sub"
    repo.mkdir()
    (repo / "source.py").write_text("import torch\n\ndef kernel(x):\n    return torch.relu(x)\n")
    (repo / "inference.py").write_text("import source\n")
    result = submodule_checker.check_self_contained(repo)
    assert result.passed, result.findings


def test_a_kernel_that_opens_a_file_is_refused(tmp_path):
    """One that can read `tensors/` can read the golden and hand it back."""
    repo = tmp_path / "sub"
    repo.mkdir()
    (repo / "source.py").write_text(
        "def kernel(x):\n    data = open('tensors/reference.bin', 'rb').read()\n    return data\n"
    )
    (repo / "inference.py").write_text("import source\n")
    result = submodule_checker.check_self_contained(repo)
    assert not result.passed
    assert any("may not read files" in f for f in result.findings)


def test_a_submodule_that_launches_torchrun_is_refused(tmp_path):
    """A multi-process run here would measure something the assembly cannot reproduce."""
    repo = tmp_path / "sub"
    repo.mkdir()
    (repo / "inference.py").write_text("import subprocess\nsubprocess.run(['torchrun', 'x.py'])\n")
    result = submodule_checker.check_single_core(repo)
    assert not result.passed
    assert any("torchrun" in f for f in result.findings)


def test_the_single_core_check_applies_the_shared_core_rule(tmp_path):
    """The rule itself is covered against `candidate.core_allocation_findings`; this is the wiring."""
    repo = tmp_path / "sub"
    repo.mkdir()
    (repo / "inference.py").write_text("import os\nos.environ['NEURON_RT_NUM_CORES'] = '4'\n")
    result = submodule_checker.check_single_core(repo)
    assert not result.passed
    assert any("not the validator's to choose" in f for f in result.findings)


def test_a_missing_declaration_blocks_the_next_stage(tmp_path):
    repo = tmp_path / "sub"
    repo.mkdir()
    result = submodule_checker.check_declaration(repo, {})
    assert not result.passed
    assert any("submodule.json is missing" in f for f in result.findings)


def test_a_declaration_whose_factor_contradicts_the_projection_is_refused(tmp_path):
    """A 1/8 cut cannot be reassembled by an assembly that runs 4 ranks."""
    repo = tmp_path / "sub"
    repo.mkdir()
    (repo / "submodule.json").write_text(json.dumps({
        "module": "layers.1.ffn", "dim": "expert", "factor": 8, "shard": 0,
        "inputs": [], "outputs": [],
        "reassembly": {"op": "sum", "shards": [], "dtype": "float32", "shape": [1]},
    }))
    result = submodule_checker.check_declaration(
        # The nested shape materialization actually writes. Stating the flat key here let this test
        # pass throughout the run in which the check was reading 0 and enforcing nothing.
        repo, {"projection": {"projected": {"units": 4}}},
    )
    assert not result.passed
    assert any("4 ranks" in f and "8-way" in f for f in result.findings)


# ======================================================================================
# the reassembly verifier
# ======================================================================================


def _shards(tmp_path: Path, parts: list[np.ndarray], shared: np.ndarray | None = None):
    for index, part in enumerate(parts):
        (tmp_path / f"r{index}.bin").write_bytes(part.astype("<f4").tobytes())
    if shared is not None:
        (tmp_path / "shared.bin").write_bytes(shared.astype("<f4").tobytes())
    total = sum(parts) + (shared if shared is not None else 0)
    # bfloat16 is the top 16 bits of a float32.
    golden = (np.asarray(total).astype("<f4").view("<u4") >> 16).astype("<u2")
    (tmp_path / "golden.bin").write_bytes(golden.tobytes())


def _declaration(shape, shards, then_add=()):
    return {
        "module": "m", "dim": "expert", "factor": len(shards), "shard": 0,
        "inputs": [], "outputs": [],
        "reassembly": {"op": "sum", "shards": list(shards), "then_add": list(then_add),
                       "dtype": "float32", "shape": list(shape)},
    }


def _manifest(shape):
    return {
        "module_output": {"file": "golden.bin", "dtype": "bfloat16", "shape": list(shape)},
        "module_tolerance": {"RTOL": 0.1, "ATOL": 0.1, "MIN_COSINE": 0.9999,
                             "MIN_PASS_FRACTION": 0.999, "MAX_ABS_ERR": 0.28},
    }


def test_a_correct_cut_reproduces_the_module(tmp_path):
    rng = np.random.default_rng(0)
    shape = (4, 8)
    parts = [rng.standard_normal(shape).astype(np.float32) for _ in range(4)]
    shared = rng.standard_normal(shape).astype(np.float32)
    _shards(tmp_path, parts, shared)
    outcome = verify_recipe(
        tmp_path,
        _declaration(shape, [f"r{i}.bin" for i in range(4)], ["shared.bin"]),
        _manifest(shape),
    )
    assert outcome.reproduces, outcome.detail
    assert outcome.metrics["pass_fraction"] == 1.0


def test_a_dropped_rank_is_caught(tmp_path):
    """The failure that would otherwise survive two stages: a cut that does not close."""
    rng = np.random.default_rng(1)
    shape = (4, 8)
    parts = [rng.standard_normal(shape).astype(np.float32) for _ in range(4)]
    _shards(tmp_path, parts)
    outcome = verify_recipe(
        tmp_path, _declaration(shape, [f"r{i}.bin" for i in range(3)]), _manifest(shape),
    )
    assert not outcome.reproduces
    assert "cosine" in outcome.detail


def test_a_forgotten_post_reduce_term_is_caught(tmp_path):
    """For expert parallelism this is the shared expert, added after the ranks rejoin."""
    rng = np.random.default_rng(2)
    shape = (4, 8)
    parts = [rng.standard_normal(shape).astype(np.float32) for _ in range(4)]
    shared = rng.standard_normal(shape).astype(np.float32)
    _shards(tmp_path, parts, shared)
    outcome = verify_recipe(
        tmp_path, _declaration(shape, [f"r{i}.bin" for i in range(4)]), _manifest(shape),
    )
    assert not outcome.reproduces


def test_a_lazy_cut_still_reproduces_the_golden(tmp_path):
    """One rank doing everything and three returning zeros passes *this* check, by design.

    The verifier proves the cut closes, not that it is good. The "+10% over the submodule" bound in
    the whole-module gate is what makes this cut fail, and the division of labour is deliberate —
    conflating them would make the arithmetic check reject legitimate asymmetric cuts.
    """
    rng = np.random.default_rng(3)
    shape = (4, 8)
    whole = rng.standard_normal(shape).astype(np.float32)
    parts = [whole] + [np.zeros(shape, dtype=np.float32) for _ in range(3)]
    _shards(tmp_path, parts)
    outcome = verify_recipe(
        tmp_path, _declaration(shape, [f"r{i}.bin" for i in range(4)]), _manifest(shape),
    )
    assert outcome.reproduces


def test_concat_rejoins_along_a_dimension(tmp_path):
    rng = np.random.default_rng(4)
    parts = [rng.standard_normal((2, 8)).astype(np.float32) for _ in range(4)]
    for index, part in enumerate(parts):
        (tmp_path / f"r{index}.bin").write_bytes(part.astype("<f4").tobytes())
    combined = apply_recipe(tmp_path, {
        "op": "concat", "dim": 0, "shards": [f"r{i}.bin" for i in range(4)],
        "dtype": "float32", "shape": [2, 8],
    })
    assert combined.shape == (8, 8)
    np.testing.assert_allclose(combined, np.concatenate(parts, axis=0), rtol=1e-6)


def test_an_unknown_operation_is_refused(tmp_path):
    """An op the verifier cannot evaluate is a claim it cannot check."""
    with pytest.raises(RecipeError, match="Known: concat, sum"):
        apply_recipe(tmp_path, {"op": "interleave", "shards": ["r0.bin"],
                                "dtype": "float32", "shape": [4]})


def test_a_wrong_declared_shape_is_caught_as_a_size_mismatch(tmp_path):
    (tmp_path / "r0.bin").write_bytes(np.zeros(8, dtype="<f4").tobytes())
    with pytest.raises(RecipeError, match="declares shape"):
        apply_recipe(tmp_path, {"op": "sum", "shards": ["r0.bin"],
                                "dtype": "float32", "shape": [16]})


def test_a_manifest_with_no_module_output_cannot_be_verified_against(tmp_path):
    with pytest.raises(RecipeError, match="nothing to reassemble towards"):
        verify_recipe(tmp_path, _declaration((4,), ["r0.bin"]), {})


def test_the_declaration_cannot_nominate_its_own_target(tmp_path):
    """The golden and the bar come from the manifest, never from the declaration.

    Otherwise a declaration could name an easier tensor to be judged against.
    """
    declaration = _declaration((4,), ["r0.bin"])
    declaration["module_output"] = {"file": "easy.bin", "dtype": "float32", "shape": [4]}
    with pytest.raises(RecipeError, match="nothing to reassemble towards"):
        verify_recipe(tmp_path, declaration, {})


def test_compare_reports_each_reason_it_failed():
    want = np.ones(1000)
    got = want.copy()
    got[:500] = 5.0
    bar = {"RTOL": 0.1, "ATOL": 0.1, "MIN_COSINE": 0.9999,
           "MIN_PASS_FRACTION": 0.999, "MAX_ABS_ERR": 0.28}
    ok, metrics, detail = compare(got, want, bar)
    assert not ok
    assert "within RTOL/ATOL" in detail and "cosine" in detail and "ceiling" in detail
    assert metrics["max_abs_err"] == pytest.approx(4.0)


def test_compare_rejects_a_shape_mismatch():
    ok, _, detail = compare(np.zeros((2, 2)), np.zeros((4, 4)), {
        "RTOL": 0.1, "ATOL": 0.1, "MIN_COSINE": 0.9, "MIN_PASS_FRACTION": 0.9,
        "MAX_ABS_ERR": 1.0,
    })
    assert not ok and "shape" in detail


# ======================================================================================
# regressions from the first real run and the PR review
# ======================================================================================


def test_a_separator_literal_is_not_an_absolute_path():
    """The first real run failed self-containment for building names with `"/".join(...)`."""
    findings = candidate.path_findings(
        _tree('paths = ["/".join(["tensors", n]) for n in NAMES]\nHERE = "."\nUP = ".."\n'),
        "inference.py",
    )
    assert findings == []


def test_a_real_absolute_path_is_still_caught():
    findings = candidate.path_findings(_tree('P = "/home/ubuntu/checkpoint/model.bin"\n'), "inference.py")
    assert len(findings) == 1


def test_non_finite_markers_are_ignored():
    """`nan` defeats every comparison that guards this pipeline, so it must read as absent."""
    for bad in ("nan", "-nan", "inf", "-inf", "Infinity"):
        assert candidate.marker_value(f"##autohelix[latency_ms={bad}]", "latency_ms") is None
    assert candidate.marker_value("##autohelix[latency_ms=12.5]", "latency_ms") == 12.5


def test_a_nan_latency_fails_the_measurement_check():
    result = module_checker.check_measurement(
        _run(_rank_markers([1.0, 1.0, 1.0, 1.0]) + "##autohelix[latency_ms=nan]\n"), 4,
    )
    assert not result.passed
    assert any("no ##autohelix[latency_ms" in f for f in result.findings)


def test_a_nan_max_abs_err_does_not_clear_the_ceiling(tmp_path):
    bar = {"RTOL": 0.1, "ATOL": 0.1, "MIN_COSINE": 0.9999,
           "MIN_PASS_FRACTION": 0.999, "MAX_ABS_ERR": 0.28}
    repo = _module_repo(tmp_path, NKI_SOURCE, inference="\n".join(
        f"{name} = {value!r}" for name, value in bar.items()
    ) + "\nimport source\n")
    result = module_checker.check_matches(
        _run("##autohelix[passed=1]\n##autohelix[max_abs_err=nan]\n"), bar, repo,
    )
    assert not result.passed
    assert any("max_abs_err" in f for f in result.findings)


def test_the_manifest_tensor_record_is_read_in_either_shape(tmp_path):
    """586 provenance failures in the first real run, for a shape the prompt never specified."""
    (tmp_path / "tensors").mkdir()
    (tmp_path / "tensors" / "input.bin").write_bytes(b"abc")
    digest = candidate._sha256(tmp_path / "tensors" / "input.bin")

    by_path = {"tensors": {"tensors/input.bin": {"sha256": digest, "bytes": 3}}}
    by_name = {"tensors": {"input": {"file": "tensors/input.bin", "sha256": digest, "bytes": 3}}}
    as_list = {"tensors": [{"file": "tensors/input.bin", "sha256": digest, "bytes": 3}]}
    for manifest in (by_path, by_name, as_list):
        findings, checked = candidate.provenance_findings(tmp_path, manifest)
        assert findings == [], manifest
        assert checked == 1


def test_an_edited_tensor_is_still_caught_in_either_shape(tmp_path):
    (tmp_path / "tensors").mkdir()
    (tmp_path / "tensors" / "input.bin").write_bytes(b"abc")
    manifest = {"tensors": {"tensors/input.bin": {"sha256": "0" * 64}}}
    findings, _ = candidate.provenance_findings(tmp_path, manifest)
    assert any("has been edited" in f for f in findings)


def test_an_absent_tensor_names_itself(tmp_path):
    """The first run reported 586 findings that all read ' is missing from the repo'."""
    findings, _ = candidate.provenance_findings(tmp_path, {"tensors": {"tensors/gone.bin": {}}})
    assert findings == ["tensors/gone.bin is missing from the repo"]


def test_import_aliases_are_resolved():
    aliases = candidate.import_aliases(_tree(
        "import torch.distributed as foo\n"
        "import nki.collectives as ncc\n"
        "from nki import collectives as c2\n"
        "import nki\n"
    ))
    assert candidate.resolve_call("foo.all_reduce", aliases) == "torch.distributed.all_reduce"
    assert candidate.resolve_call("ncc.all_reduce", aliases) == "nki.collectives.all_reduce"
    assert candidate.resolve_call("c2.all_reduce", aliases) == "nki.collectives.all_reduce"


def test_an_aliased_torch_collective_is_refused(tmp_path):
    """`import torch.distributed as ncc` used to read as the NKI collective."""
    repo = _module_repo(tmp_path, """
        import torch.distributed as ncc
        import nki

        def kernel(x):
            ncc.all_reduce(x)
            return x
    """)
    result = module_checker.check_nki_collectives(repo, 4)
    assert not result.passed
    assert any("torch.distributed.all_reduce" in f for f in result.findings)


def test_the_nki_collective_passes_under_any_alias(tmp_path):
    repo = _module_repo(tmp_path, """
        import nki
        import nki.collectives as whatever
        import nki.isa as nisa
        import nki.language as nl

        def kernel(x, replica_group):
            src = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm, name="src")
            dst = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm, name="dst")
            out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
            nisa.dma_copy(dst=src, src=x)
            whatever.all_reduce(srcs=[src], dsts=[dst], replica_group=replica_group, op=nl.add)
            nisa.dma_copy(dst=out, src=dst)
            return out
    """)
    result = module_checker.check_nki_collectives(repo, 4)
    assert result.passed, result.findings


def test_the_rank_count_comes_from_the_manifest():
    """A 2-unit placement is valid, and the gate used to launch 4 processes regardless."""
    assert module_checker.rank_count({"ranks": 2}) == 2
    assert module_checker.rank_count({"projection": {"projected": {"units": 1}}}) == 1
    assert module_checker.rank_count({}) == module_checker.DEFAULT_RANKS
    assert module_checker.launch(2)[:3] == ["torchrun", "--nproc_per_node", "2"]


def test_a_two_rank_assembly_is_judged_against_two_ranks():
    output = _rank_markers([40.0, 32.0]) + "##autohelix[latency_ms=32.0]\n"
    assert module_checker.check_all_ranks(_run(output), 2).passed
    assert not module_checker.check_all_ranks(_run(output), 4).passed


def test_a_flattened_shard_shape_is_compared_after_reshaping():
    """The reference flattens `(1, 128, 5120)` to `(128, 5120)`; the first run failed on that."""
    want = np.arange(12.0).reshape(1, 3, 4)
    got = np.arange(12.0).reshape(3, 4)
    bar = {"RTOL": 0.1, "ATOL": 0.1, "MIN_COSINE": 0.9999,
           "MIN_PASS_FRACTION": 0.999, "MAX_ABS_ERR": 0.28}
    ok, _, detail = compare(got, want, bar)
    assert ok, detail


def test_a_genuine_size_mismatch_is_still_refused():
    bar = {"RTOL": 0.1, "ATOL": 0.1, "MIN_COSINE": 0.9, "MIN_PASS_FRACTION": 0.9,
           "MAX_ABS_ERR": 1.0}
    ok, _, detail = compare(np.zeros(6), np.zeros(12), bar)
    assert not ok and "element(s)" in detail


def test_the_module_golden_cannot_be_its_own_shard():
    """Reassembling the answer from the answer reproduces the target and proves nothing."""
    from optimization.recipe import validate_shards

    findings = validate_shards(
        {"shards": ["tensors/reference.bin"]}, 4, "tensors/reference.bin",
    )
    assert any("own recorded output" in f for f in findings)


def test_the_shard_count_must_equal_the_cut_width():
    from optimization.recipe import validate_shards

    assert validate_shards({"shards": ["a", "b"]}, 4, None)
    assert validate_shards({"shards": ["a", "b", "c", "d"]}, 4, None) == []


def test_repeated_shards_are_refused():
    from optimization.recipe import validate_shards

    findings = validate_shards({"shards": ["a", "a", "b", "c"]}, 4, None)
    assert any("repeats" in f for f in findings)


def test_a_verified_recipe_still_needs_a_legal_shard_list(tmp_path):
    """The arithmetic and the shard rules are both required, not either."""
    rng = np.random.default_rng(9)
    shape = (4, 8)
    whole = rng.standard_normal(shape).astype(np.float32)
    (tmp_path / "golden.bin").write_bytes(
        (whole.astype("<f4").view("<u4") >> 16).astype("<u2").tobytes()
    )
    (tmp_path / "r0.bin").write_bytes(whole.astype("<f4").tobytes())
    declaration = {
        "module": "m", "dim": "expert", "factor": 4, "shard": 0, "inputs": [], "outputs": [],
        "reassembly": {"op": "sum", "shards": ["r0.bin"], "dtype": "float32",
                       "shape": list(shape)},
    }
    outcome = verify_recipe(tmp_path, declaration, _manifest(shape))
    assert not outcome.reproduces
    assert "1 golden(s) but the cut is 4-way" in outcome.detail


# ======================================================================================
# the review of #6, second pass: what the two gates read out of a manifest, and whether
# the frozen validator is connected to the kernel at all
# ======================================================================================


def test_the_projected_rank_count_is_read_from_where_materialization_writes_it():
    """`projection.projected.units` is the real shape; a flat `projected_units` never existed, so
    the submodule gate's factor-mismatch check read 0 and disabled itself for a whole run."""
    manifest = {"projection": {"projected": {"units": 4, "devices": 1,
                                            "splits": [{"dim": "expert", "factor": 4}]}}}
    assert candidate.projected_units(manifest) == 4


def test_an_older_flat_projected_units_is_still_read():
    assert candidate.projected_units({"projection": {"projected_units": 2}}) == 2


def test_an_explicit_rank_count_is_read():
    assert candidate.projected_units({"ranks": 8}) == 8


def test_a_manifest_that_says_nothing_reports_zero_rather_than_guessing():
    assert candidate.projected_units({}) == 0
    assert candidate.projected_units({"projection": {"projected": {}}}) == 0


def test_the_grid_launch_counts_as_driving_the_kernel():
    """`kernel[2](...)` is the LNC=2 launch idiom, so the callee is a Subscript and an analyzer
    that only understands `f()` reports the real validator as never calling the kernel."""
    tree = ast.parse("from source import kernel\nout = kernel[2](x, group)\n")
    assert candidate.invokes_entry_point(tree, "kernel")


def test_a_plain_call_counts():
    assert candidate.invokes_entry_point(ast.parse("import source\nsource.kernel(x)\n"), "kernel")


def test_handing_the_kernel_to_a_tracer_counts():
    assert candidate.invokes_entry_point(ast.parse("import nki\nf = nki.trace(kernel)\n"), "kernel")


def test_importing_source_without_running_it_does_not_count():
    """The defect: once frozen by hash, such a validator lets every later source.py pass."""
    tree = ast.parse("import source\nprint(source.INTER_PAD)\nout = my_helper(x)\n")
    assert not candidate.invokes_entry_point(tree, "kernel")


def test_tracing_some_other_function_does_not_count():
    tree = ast.parse("import nki\nf = nki.trace(helper)\nout = f(x)\n")
    assert not candidate.invokes_entry_point(tree, "kernel")


# ======================================================================================
# the review of #6, third pass
# ======================================================================================


def test_a_one_rank_assembly_needs_no_collective(tmp_path):
    """`project()` preserves a placement the floorplan already fits on one unit, so a one-rank
    assembly is legal — and it has nothing to reduce."""
    repo = _module_repo(tmp_path, """
        import nki
        import nki.language as nl

        def kernel(x):
            return nl.copy(x)
    """)
    assert module_checker.check_nki_collectives(repo, ranks=1).passed


def test_more_than_one_rank_still_needs_a_collective(tmp_path):
    repo = _module_repo(tmp_path, """
        import nki
        import nki.language as nl

        def kernel(x):
            return nl.copy(x)
    """)
    result = module_checker.check_nki_collectives(repo, ranks=2)
    assert not result.passed
    assert any("no collective at all" in f for f in result.findings)


def test_one_rank_may_still_not_reach_for_a_host_collective(tmp_path):
    repo = _module_repo(tmp_path, """
        import nki
        import torch.distributed as dist

        def kernel(x):
            dist.all_reduce(x)
            return x
    """)
    assert not module_checker.check_nki_collectives(repo, ranks=1).passed


def test_a_literal_core_count_that_disagrees_is_refused():
    """The bot's scenario: a two-rank validator keeping a hard-coded four."""
    tree = ast.parse("import os\nos.environ['NEURON_RT_NUM_CORES'] = '4'\n")
    assert candidate.core_allocation_findings(tree, "inference.py", "2")


def test_a_literal_core_count_that_agrees_is_allowed():
    tree = ast.parse("import os\nos.environ['NEURON_RT_NUM_CORES'] = '4'\n")
    assert candidate.core_allocation_findings(tree, "inference.py", "4") == []


def test_setdefault_is_not_an_override():
    """It cannot replace what the gate set, which is why the real validator uses it."""
    tree = ast.parse("import os\nos.environ.setdefault('NEURON_RT_NUM_CORES', '8')\n")
    assert candidate.core_allocation_findings(tree, "inference.py", "4") == []


def test_a_count_computed_from_the_launch_is_allowed():
    """`env['NEURON_RT_NUM_CORES'] = str(WORLD_SIZE)` is how the real assembly passes the gate's own
    count to the child process that touches the device."""
    tree = ast.parse("env = {}\nenv['NEURON_RT_NUM_CORES'] = str(WORLD_SIZE)\n")
    assert candidate.core_allocation_findings(tree, "inference.py", "4") == []


def test_visible_cores_is_not_a_widening():
    tree = ast.parse("import os\nos.environ['NEURON_RT_VISIBLE_CORES'] = '2'\n")
    assert candidate.core_allocation_findings(tree, "inference.py", "1") == []
