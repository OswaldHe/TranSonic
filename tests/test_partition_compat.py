# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for porting a model's own code to the GPU in front of us.

A vendor's kernels are written for the vendor's hardware. When this card refuses one,
the model cannot run, there is no trace and nothing to partition — so the loop treats
that as work rather than as a wall: the agent writes a compatibility patch under
``compat/`` and the trace is taken again with it.
"""

import pytest

from model_partition.runtime.compat import apply_patches, is_hardware_limit, patch_paths


def test_a_kernel_this_card_refuses_is_told_apart_from_a_bug():
    """Who gets the failure depends on this: a porter, or whoever owns the arithmetic."""
    assert is_hardware_limit(RuntimeError(
        "Failed to set the allowed dynamic shared memory size to 141312"))
    assert is_hardware_limit(RuntimeError("CUDA error: no kernel image is available"))
    assert is_hardware_limit(RuntimeError("kernel input device_type mismatch, expected cuda"))
    # Not hardware: these belong to whoever wrote the code.
    assert not is_hardware_limit(RuntimeError("shapes cannot be multiplied (8x4 and 5x2)"))
    assert not is_hardware_limit(ValueError("expected a positive integer"))
    assert not is_hardware_limit(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"))


def test_patches_are_discovered_in_order(tmp_path):
    (tmp_path / "compat").mkdir()
    for name in ("20-second.py", "10-first.py", "_helper.py"):
        (tmp_path / "compat" / name).write_text("def apply(vendor, device):\n    return []\n")
    assert [p.name for p in patch_paths(tmp_path)] == ["10-first.py", "20-second.py"]
    assert patch_paths(tmp_path / "absent") == []


def test_a_patch_replaces_what_will_not_run(tmp_path):
    """The point: the model's own module now calls the ported version."""
    import types

    vendor = types.SimpleNamespace(sparse_attn=lambda *a: "hopper kernel", other=1)
    (tmp_path / "compat").mkdir()
    (tmp_path / "compat" / "sparse_attn.py").write_text(
        '"""sparse_attn wants 141312 bytes of shared memory; this card allows 101376."""\n'
        "\n"
        "def _ported(*args):\n"
        "    return 'torch fallback'\n"
        "\n"
        "def apply(vendor, device):\n"
        "    vendor.sparse_attn = _ported\n"
        "    return ['sparse_attn']\n"
    )
    report = apply_patches(vendor, patch_paths(tmp_path), device=None)
    assert report.applied
    assert report.replaced == ["sparse_attn"]
    assert vendor.sparse_attn() == "torch fallback"
    assert vendor.other == 1
    assert "replaced 1" in report.summary()


def test_a_patch_is_told_what_it_is_adapting_to(tmp_path):
    """So it can branch on what the card supports instead of on its name."""
    import types

    from model_partition.hardware import GPUInfo

    vendor = types.SimpleNamespace()
    (tmp_path / "compat").mkdir()
    (tmp_path / "compat" / "p.py").write_text(
        "def apply(vendor, device):\n"
        "    vendor.shared = device.shared_memory_per_block\n"
        "    vendor.fp8 = device.supports_fp8()\n"
        "    return ['shared', 'fp8']\n"
    )
    card = GPUInfo(index=0, name="NVIDIA L40S", total_bytes=1, free_bytes=1,
                   capability=(8, 9), shared_memory_per_block=101376)
    apply_patches(vendor, patch_paths(tmp_path), device=card)
    assert vendor.shared == 101376 and vendor.fp8 is True


def test_a_broken_patch_is_reported_not_ignored(tmp_path):
    import types

    (tmp_path / "compat").mkdir()
    (tmp_path / "compat" / "bad.py").write_text("raise RuntimeError('boom')\n")
    (tmp_path / "compat" / "empty.py").write_text("VALUE = 1\n")
    report = apply_patches(types.SimpleNamespace(), patch_paths(tmp_path), None)
    assert not report.applied
    assert set(report.errors) == {"bad.py", "empty.py"}
    assert "failed" in report.summary()


def test_the_loader_applies_a_patch_and_refuses_a_broken_one(tmp_path):
    """The patch runs at import, before anything is constructed or traced."""
    from model_partition.ingest import ingest
    from model_partition.loaders import LoaderError, build_loader
    from model_partition.spec import parse_spec
    from tests.fixtures.tiny_llm import TinyConfig, write_tiny_repo

    pytest.importorskip("torch")
    repo = write_tiny_repo(tmp_path / "repo", TinyConfig())
    spec = parse_spec({"source": str(repo), "name": "tiny", "dtype": "float32",
                       "trust_remote_code": True})
    result = ingest(spec)

    directory = tmp_path / "compat"
    directory.mkdir()
    (directory / "marker.py").write_text(
        "def apply(vendor, device):\n"
        "    vendor.PORTED_HERE = True\n"
        "    return ['PORTED_HERE']\n"
    )
    loader = build_loader(result, compat_paths=patch_paths(tmp_path))
    loaded = loader.build_meta()
    assert loaded.model is not None
    assert loader.compat_report.replaced == ["PORTED_HERE"]

    (directory / "marker.py").write_text("def apply(vendor, device):\n    raise KeyError('x')\n")
    with pytest.raises(LoaderError, match="compatibility patch"):
        build_loader(result, compat_paths=patch_paths(tmp_path)).build_meta()


def _context(tiny_run, tmp_path):
    """A real loop context for the toy model, up to the point of tracing."""
    from model_partition.loop.driver import PartitionLoop
    from model_partition.loop.stages import LoopOptions, stage_ingest, stage_plan

    options = LoopOptions(artifact_root=str(tmp_path / "artifacts"), device="cpu",
                          trace_device="cpu", judge_kind="stub", max_iterations=1,
                          use_agent_planner=False, gpu_memory_gib=1.0)
    ctx = PartitionLoop(spec=tiny_run.spec, options=options,
                        report=lambda _m: None).build_context()
    stage_ingest(ctx)
    stage_plan(ctx)
    return ctx


def test_a_hardware_failure_in_the_trace_goes_to_the_porting_surface(tiny_run, tmp_path,
                                                                    monkeypatch):
    """Not the plan and not the arithmetic: this one is a porting job."""
    from model_partition.loop.stages import stage_trace
    from model_partition.trace import Tracer

    ctx = _context(tiny_run, tmp_path)

    def refuse(self, *args, **kwargs):
        raise RuntimeError("Failed to set the allowed dynamic shared memory size to 141312")

    monkeypatch.setattr(Tracer, "trace_sample", refuse)
    result = stage_trace(ctx)
    assert not result.ok and result.repairable
    assert result.repair_surface == "compat"
    assert "does not run on this GPU" in result.detail


def test_an_ordinary_trace_failure_is_not_a_porting_job(tiny_run, tmp_path, monkeypatch):
    """A wrong shape is a bug in the code, and porting it would fix nothing."""
    from model_partition.loop.stages import stage_trace
    from model_partition.trace import Tracer

    ctx = _context(tiny_run, tmp_path)

    def broken(self, *args, **kwargs):
        raise RuntimeError("shapes cannot be multiplied (8x4 and 5x2)")

    monkeypatch.setattr(Tracer, "trace_sample", broken)
    with pytest.raises(RuntimeError, match="shapes cannot be multiplied"):
        stage_trace(ctx)


def test_importing_a_patch_leaves_no_bytecode_beside_it(tmp_path):
    """A run directory is a deliverable someone reads and copies."""
    import types

    (tmp_path / "compat").mkdir()
    (tmp_path / "compat" / "p.py").write_text("def apply(vendor, device):\n    return []\n")
    apply_patches(types.SimpleNamespace(), patch_paths(tmp_path), None)
    assert not (tmp_path / "compat" / "__pycache__").exists()
