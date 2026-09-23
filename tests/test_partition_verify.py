# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for numeric comparison and module verification."""

import pytest

from model_partition.verify.modules import (
    plan_owned_parameters,
    poison_parameters,
    verify_modules,
)
from model_partition.verify.numerics import (
    TOLERANCES,
    Tolerance,
    compare,
    top1_agreement,
)

torch = pytest.importorskip("torch")


# -- tolerances --------------------------------------------------------------


def test_tolerance_per_dtype():
    assert Tolerance.for_dtype("bfloat16").rtol == TOLERANCES["bfloat16"][0]
    assert Tolerance.for_dtype("float32").rtol < Tolerance.for_dtype("bfloat16").rtol
    # An unknown dtype gets the conservative default rather than crashing.
    assert Tolerance.for_dtype("mystery").rtol == 2e-2


def test_identical_tensors_pass():
    x = torch.randn(64, 32)
    result = compare(x, x.clone(), "x")
    assert result.passed
    assert result.max_abs_err == 0.0
    assert result.cosine == pytest.approx(1.0, abs=1e-6)
    assert result.pass_fraction == 1.0


def test_small_bf16_noise_still_passes():
    reference = torch.randn(256, 64, dtype=torch.bfloat16)
    actual = (reference.float() + torch.randn(256, 64) * 1e-4).to(torch.bfloat16)
    assert compare(actual, reference, "noisy").passed


def test_large_deviation_fails_with_reason_and_location():
    reference = torch.ones(8, 8)
    actual = reference.clone()
    actual[3, 5] = 100.0
    result = compare(actual, reference, "spike")
    assert not result.passed
    assert result.worst_index == [3, 5]
    assert result.max_abs_err == pytest.approx(99.0)
    assert "pass fraction" in result.reason


def test_a_few_outliers_in_a_big_tensor_still_fail_via_pass_fraction():
    """Elementwise closeness alone would let a systematic error through."""
    reference = torch.ones(1000)
    actual = reference * 1.5
    result = compare(actual, reference, "scaled")
    assert not result.passed
    assert result.pass_fraction == 0.0


def test_sign_flip_is_caught_by_cosine():
    reference = torch.randn(512)
    result = compare(-reference, reference, "flipped")
    assert not result.passed
    assert result.cosine < 0
    assert "cosine" in result.reason


def test_shape_mismatch_is_reported_not_crashed():
    result = compare(torch.zeros(4, 8), torch.zeros(4, 9), "mismatched")
    assert not result.passed
    assert "shape (4, 8) != (4, 9)" in result.reason


def test_nan_is_reported_as_non_finite():
    reference = torch.ones(16)
    actual = reference.clone()
    actual[2] = float("nan")
    result = compare(actual, reference, "nan")
    assert not result.passed
    assert "non-finite" in result.reason


def test_inf_is_reported_as_non_finite():
    result = compare(torch.full((4,), float("inf")), torch.ones(4), "inf")
    assert not result.passed and "non-finite" in result.reason


def test_missing_tensors_are_reported():
    assert "missing tensor" in compare(None, torch.ones(2), "x").reason
    assert "missing reference" in compare(torch.ones(2), None, "x").reason


def test_empty_tensors_compare_equal():
    assert compare(torch.zeros(0), torch.zeros(0), "empty").passed


def test_custom_tolerance_can_be_loosened():
    reference = torch.ones(100)
    actual = reference * 1.05
    assert not compare(actual, reference, "x").passed
    loose = Tolerance(rtol=0.1, atol=0.1, min_pass_fraction=0.9, min_cosine=0.99)
    assert compare(actual, reference, "x", loose).passed


def test_comparison_summary_and_dict():
    result = compare(torch.ones(4), torch.ones(4), "ok")
    assert "ok: ok" in result.summary()
    assert result.to_dict()["passed"] is True
    bad = compare(torch.zeros(4), torch.ones(4), "bad")
    assert "FAIL" in bad.summary()


def test_top1_agreement():
    logits = torch.tensor([[[0.1, 0.9], [0.8, 0.2]]])
    assert top1_agreement(logits, logits.clone()) == 1.0
    assert top1_agreement(logits, -logits) == 0.0
    assert top1_agreement(None, logits) == 0.0


# -- poisoning ---------------------------------------------------------------


def test_poison_fills_everything_by_default(tiny_run):
    model = tiny_run.build_model()
    count = poison_parameters(model)
    assert count > 0
    assert all(torch.isnan(p).all() for p in model.parameters())


def test_poison_can_be_restricted_to_named_tensors(tiny_run):
    """Regression: poisoning every buffer NaNs values no dump can restore."""
    model = tiny_run.build_model()
    owned = plan_owned_parameters(model, tiny_run.graph)
    poison_parameters(model, only=owned)
    for name, tensor in model.named_parameters():
        if name in owned:
            assert torch.isnan(tensor).all(), name
        else:
            assert not torch.isnan(tensor).any(), name


def test_plan_owned_parameters_covers_the_plan_and_no_more(tiny_run):
    model = tiny_run.build_model()
    owned = plan_owned_parameters(model, tiny_run.graph)
    all_names = {name for name, _ in model.named_parameters()}
    assert owned and owned <= all_names | {n for n, _ in model.named_buffers()}
    # The dense toy model is fully covered by its plan.
    assert all_names <= owned


def test_plan_owned_parameters_excludes_unplanned_modules(tiny_run):
    graph = tiny_run.graph
    graph.modules = [m for m in graph.modules if m.kind != "lm_head"]
    owned = plan_owned_parameters(tiny_run.build_model(), graph)
    assert not any(name.startswith("lm_head") for name in owned)


# -- verify_modules ----------------------------------------------------------


def test_all_modules_verify_from_their_dumps(tiny_run):
    report = verify_modules(tiny_run.build_model, tiny_run.bundle, tiny_run.graph)
    assert report.passed, report.render()
    assert len(report.results) == len(tiny_run.bundle.records)
    assert report.worst_cosine() > 0.999
    assert report.failures == []


def test_moe_modules_verify(tiny_moe_run):
    report = verify_modules(tiny_moe_run.build_model, tiny_moe_run.bundle, tiny_moe_run.graph)
    assert report.passed, report.render()


def test_weights_are_applied_from_the_dump_not_the_live_model(tiny_run):
    report = verify_modules(tiny_run.build_model, tiny_run.bundle, tiny_run.graph)
    assert all(r.weights_applied > 0 for r in report.results)


def test_truncated_weight_dump_fails_verification(tiny_run):
    """A module whose dump is incomplete must not pass on poisoned memory."""
    target = next(m.id for m in tiny_run.graph.partitioned_modules
                  if m.kind == "decoder_layers")
    tiny_run.bundle.weights[target] = tiny_run.bundle.weights[target][:1]
    report = verify_modules(tiny_run.build_model, tiny_run.bundle, tiny_run.graph)
    assert not report.passed
    assert target in {r.module_id for r in report.failures}


def test_corrupted_output_dump_fails_verification(tiny_run):
    store = tiny_run.bundle.store
    outputs = store.find(role="output")
    assert outputs
    path = store.blob_path(outputs[0])
    path.write_bytes(b"\x00" * outputs[0].nbytes)
    report = verify_modules(tiny_run.build_model, tiny_run.bundle, tiny_run.graph)
    assert not report.passed


def test_module_absent_from_graph_is_reported(tiny_run):
    graph = tiny_run.graph
    dropped = next(m for m in graph.partitioned_modules if m.kind == "decoder_layers")
    graph.modules = [m for m in graph.modules if m.id != dropped.id]
    report = verify_modules(tiny_run.build_model, tiny_run.bundle, graph,
                            module_ids=[dropped.id])
    assert not report.passed
    assert "not in the partition graph" in report.failures[0].error


def test_report_can_be_filtered_to_one_sample(tiny_run):
    report = verify_modules(tiny_run.build_model, tiny_run.bundle, tiny_run.graph,
                            sample_ids=["s0"])
    assert report.passed
    assert {r.sample_id for r in report.results} == {"s0"}


def test_report_serializes_metrics(tiny_run):
    report = verify_modules(tiny_run.build_model, tiny_run.bundle, tiny_run.graph)
    payload = report.to_dict()
    assert payload["passed"] is True
    assert payload["n_failed"] == 0
    assert payload["n_checks"] == len(report.results)
    assert "results" in payload


def test_empty_report_does_not_count_as_passed():
    from model_partition.verify.modules import VerifyReport

    assert VerifyReport().passed is False


# -- per-module GPU residency ------------------------------------------------


def test_module_device_moves_only_the_module_under_test(tiny_run):
    """The plan sizes each module to fit the GPU, so each is verified there even
    when the whole model is resident on the host."""
    from model_partition.verify.modules import move_submodules

    model = tiny_run.build_model()
    target = next(m for m in tiny_run.graph.partitioned_modules
                  if m.kind == "decoder_layers")
    other = next(m for m in tiny_run.graph.partitioned_modules if m.kind == "lm_head")

    moved = move_submodules(model, target, "cpu")
    assert moved == len(target.submodules)
    # Nothing outside the module is touched.
    assert all(p.device.type == "cpu" for p in model.parameters())
    assert move_submodules(model, other, "cpu") == len(other.submodules)


def test_move_submodules_ignores_absent_names(tiny_run):
    from model_partition.planner.graph import ModuleNode
    from model_partition.verify.modules import move_submodules

    node = ModuleNode(id="x", kind="other", submodules=["nope.at.all"])
    assert move_submodules(tiny_run.build_model(), node, "cpu") == 0


def test_verification_records_where_each_module_ran(tiny_run):
    report = verify_modules(tiny_run.build_model, tiny_run.bundle, tiny_run.graph,
                            model_device="cpu", module_device="cpu")
    assert report.passed
    assert {r.device for r in report.results} == {"cpu"}
    assert report.to_dict()["results"][0]["device"] == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_modules_verify_on_the_gpu_with_the_model_on_the_host(tiny_run):
    """The 27B case in miniature: host-resident model, GPU-resident module."""
    report = verify_modules(tiny_run.build_model, tiny_run.bundle, tiny_run.graph,
                            model_device="cpu", module_device="cuda")
    assert report.passed, report.render()
    assert {r.device for r in report.results} == {"cuda"}
    assert "on cuda" in report.render()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_modules_are_returned_to_the_host_after_verification(tiny_run):
    """Otherwise every verified module would accumulate on the GPU."""
    model = tiny_run.build_model().to("cpu")
    verify_modules(lambda: model, tiny_run.bundle, tiny_run.graph,
                   model_device="cpu", module_device="cuda")
    assert all(p.device.type == "cpu" for p in model.parameters())


def test_a_model_of_placeholders_is_verified_without_moving_it(tiny_run):
    """DeepSeek V4.1's case: the model is built with placeholders for weights it cannot
    hold, so there is nothing to move to the host and nothing to poison. Each module
    still gets its real weights from the recording."""
    from model_partition.verify.modules import _has_placeholders

    model = tiny_run.build_model()
    for module in model.modules():
        for leaf, parameter in list(module._parameters.items()):
            if parameter is not None and parameter.dim() > 1:
                module._parameters[leaf] = torch.nn.Parameter(
                    parameter.to("meta"), requires_grad=False)
    assert _has_placeholders(model)

    report = verify_modules(lambda: model, tiny_run.bundle, tiny_run.graph,
                            impl_dirs=None, model_device="cpu", module_device="cpu")
    # Without an implementation the model's own submodule is what replays, and its
    # weights are placeholders, so every check reports rather than passing on a
    # placeholder's contents.
    assert not report.passed
    assert any("placeholders" in note for note in report.notes)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_a_module_whose_weights_the_card_cannot_hold_still_runs_on_it(tiny_run, monkeypatch):
    """An engram table is 94.4 GiB and one piece of work: no partitioning makes it fit.
    The check stays on the accelerator anyway — the launcher leaves the oversized tensor
    on the host — because the fp8 GEMM in the same module runs nowhere else."""
    from model_partition.verify import modules as verify_module

    monkeypatch.setattr(verify_module, "weights_bytes", lambda _weights: 1 << 50)
    report = verify_modules(tiny_run.build_model, tiny_run.bundle, tiny_run.graph,
                            model_device="cpu", module_device="cuda")
    assert report.passed, report.render()
    assert {r.device for r in report.results} == {"cuda"}
    assert any("stay on cpu" in note for note in report.notes)
    assert not report.oversized, "an indivisible module is not a partition failure"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_a_weight_too_large_for_the_card_stays_on_the_host(monkeypatch):
    """What lets an engram module run at all: the table on the host, the rest on the
    card, and the lookup's indices and rows moved across the boundary."""
    import torch
    from torch import nn

    from model_partition.runtime import launcher

    class Lookup(nn.Module):
        def __init__(self):
            super().__init__()
            self.table = nn.Embedding(8, 4)
            self.proj = nn.Linear(4, 4)

        def forward(self, ids):
            return self.proj(self.table(ids))

    module = Lookup()
    table = module.table.weight.numel() * 4
    projection = (module.proj.weight.numel() + module.proj.bias.numel()) * 4
    assert projection < table
    # Free memory that leaves room for the projection and not for the table.
    free = int((projection + table) / 2 / 0.8)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a, **k: (free, free))
    left = launcher.place(module, "cuda")

    assert left == ["table"]
    assert module.table.weight.device.type == "cpu"
    assert module.proj.weight.device.type == "cuda"
    out = module(torch.zeros(2, dtype=torch.long, device="cuda"))
    assert out.device.type == "cuda"


def test_accumulated_tolerance_is_looser_than_single_step():
    """A boundary many layers deep accumulates bf16 drift by construction."""
    single = Tolerance.for_dtype("bfloat16")
    deep = Tolerance.accumulated("bfloat16")
    assert deep.min_pass_fraction < single.min_pass_fraction
    assert deep.min_cosine < single.min_cosine
    assert deep.rtol > single.rtol


def test_accumulated_tolerance_accepts_realistic_depth_drift():
    """The 27B case: 0.996 of elements within tolerance at layer 31."""
    reference = torch.randn(4096, dtype=torch.float32)
    actual = reference.clone()
    drifted = torch.randperm(reference.numel())[:16]
    actual[drifted] += 0.5
    assert not compare(actual, reference, "deep", Tolerance.for_dtype("bfloat16")).passed
    assert compare(actual, reference, "deep", Tolerance.accumulated()).passed


def test_accumulated_tolerance_still_rejects_a_wrong_tensor():
    reference = torch.randn(4096)
    assert not compare(reference * 1.5, reference, "wrong", Tolerance.accumulated()).passed
    assert not compare(-reference, reference, "flipped", Tolerance.accumulated()).passed


# -- extracted implementations ------------------------------------------------


def _extract_impls(run, tmp_path):
    from model_partition.extract import extract
    from model_partition.runtime.module_impl import find_impl_dirs

    extract(run.graph, run.build_model(), tmp_path / "modules",
            run_root=run.layout.root, sample_ids=run.sample_ids,
            weight_tensors=run.bundle.weights)
    return find_impl_dirs(tmp_path / "modules")


def test_verification_runs_the_extracted_implementation(tiny_run, tmp_path):
    """The check must exercise the code the loop owns, not the original model."""
    impl_dirs = _extract_impls(tiny_run, tmp_path)
    assert impl_dirs
    report = verify_modules(tiny_run.build_model, tiny_run.bundle, tiny_run.graph,
                            impl_dirs=impl_dirs)
    assert report.passed, report.render()


def test_a_wrong_implementation_is_reported(tiny_run, tmp_path):
    impl_dirs = _extract_impls(tiny_run, tmp_path)
    target = next(m.id for m in tiny_run.graph.partitioned_modules
                  if m.kind == "decoder_layers")
    (impl_dirs[target] / "inference.py").write_text(
        "def build_module(config, weights, device='cpu'):\n"
        "    return lambda *a, **k: a[0] * 2.0\n"
    )
    report = verify_modules(tiny_run.build_model, tiny_run.bundle, tiny_run.graph,
                            impl_dirs=impl_dirs)
    assert not report.passed
    assert target in {r.module_id for r in report.failures}


def test_an_unimportable_implementation_is_reported_not_skipped(tiny_run, tmp_path):
    impl_dirs = _extract_impls(tiny_run, tmp_path)
    target = next(iter(impl_dirs))
    (impl_dirs[target] / "inference.py").write_text("raise RuntimeError('boom')\n")
    report = verify_modules(tiny_run.build_model, tiny_run.bundle, tiny_run.graph,
                            impl_dirs=impl_dirs)
    assert not report.passed
    assert any("unusable" in r.error for r in report.failures)


def test_an_implementation_returning_a_non_callable_is_reported(tiny_run, tmp_path):
    impl_dirs = _extract_impls(tiny_run, tmp_path)
    target = next(iter(impl_dirs))
    (impl_dirs[target] / "inference.py").write_text(
        "def build_module(config, weights, device='cpu'):\n    return 42\n"
    )
    report = verify_modules(tiny_run.build_model, tiny_run.bundle, tiny_run.graph,
                            impl_dirs=impl_dirs)
    assert not report.passed


def test_a_module_with_no_implementation_is_a_failure_not_a_free_pass(tiny_run, tmp_path):
    """Falling back to the model's own submodule would verify it against itself."""
    impl_dirs = _extract_impls(tiny_run, tmp_path)
    dropped = next(m.id for m in tiny_run.graph.partitioned_modules
                   if m.kind == "decoder_layers")
    del impl_dirs[dropped]
    report = verify_modules(tiny_run.build_model, tiny_run.bundle, tiny_run.graph,
                            impl_dirs=impl_dirs)
    assert not report.passed
    failure = next(r for r in report.failures if r.module_id == dropped)
    assert "no implementation" in failure.error


def test_a_derived_buffer_keeps_the_precision_it_was_recorded_at(tmp_path):
    """Casting a rotary inv_freq down to the weights' dtype drifts long positions."""
    import torch
    from torch import nn

    from model_partition.runtime.launcher import load_weights

    class Rotary(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
            # Non-persistent: computed here, not read from a checkpoint.
            self.register_buffer("inv_freq", torch.ones(4), persistent=False)

    module = Rotary()
    recorded = torch.tensor([1.0, 0.6042963862, 0.365, 0.22], dtype=torch.float32)
    loaded, missing = load_weights(module, {"m.weight": torch.zeros(4, dtype=torch.bfloat16),
                                            "m.inv_freq": recorded})
    assert not missing and loaded == 2
    assert module.inv_freq.dtype is torch.float32
    assert module.inv_freq[1].item() == pytest.approx(0.6042963862, abs=1e-9)


def test_a_derived_buffer_the_recording_lacks_is_not_missing(tmp_path):
    """The class computed it at construction; that is where it comes from."""
    import torch
    from torch import nn

    from model_partition.runtime.launcher import load_weights

    class Rotary(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("inv_freq", torch.ones(4), persistent=False)
            self.register_buffer("scale", torch.ones(4))

    loaded, missing = load_weights(Rotary(), {})
    assert loaded == 0
    assert missing == ["scale"]


def test_a_class_gets_the_config_value_under_the_name_the_config_uses():
    """A class names its argument for what it does with it and a config names it for what
    it configures. DeepSeek's config says `norm_eps: 1e-20` and `RMSNorm.__init__`
    defaults `eps` to 1e-6, so taking the default built a normalization that ran, looked
    built, and was wrong by 0.4% — enough to fail every module downstream of one."""
    from torch import nn

    from model_partition.runtime.launcher import _config_kwargs

    class Norm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.dim, self.eps = dim, eps

    for key in ("norm_eps", "rms_norm_eps", "layer_norm_eps"):
        kwargs = _config_kwargs(Norm, {"hidden_size": 5120, key: 1e-20}, None)
        assert kwargs == {"dim": 5120, "eps": 1e-20}, key
    # The class's own name still wins where the config uses it.
    assert _config_kwargs(Norm, {"dim": 8, "eps": 0.5}, None) == {"dim": 8, "eps": 0.5}


def test_a_recorded_tensor_that_already_fits_becomes_the_parameter():
    """Copying into a freshly allocated parameter holds the module twice, which a 94.4
    GiB n-gram table does not allow — and fp4-packed expert weights have no `copy_` at
    all. A tensor of the right dtype and shape is installed as it is."""
    import torch
    from torch import nn

    from model_partition.runtime.launcher import load_weights

    with torch.device("meta"):
        module = nn.Linear(4, 4, bias=False)
    assert module.weight.is_meta

    recorded = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    loaded, missing = load_weights(module, {"m.weight": recorded})
    assert loaded == 1 and not missing
    assert module.weight.data_ptr() == recorded.data_ptr(), "copied rather than assigned"


def test_a_parameter_no_recording_covers_stays_missing_rather_than_meta():
    """A module built on meta has no values of its own to fall back on, so a gap is a
    gap — reported, not handed back as something that raises on first read."""
    import torch
    from torch import nn

    from model_partition.runtime.launcher import load_weights

    with torch.device("meta"):
        module = nn.Linear(4, 4)
    loaded, missing = load_weights(module, {"m.weight": torch.zeros(4, 4)})
    assert loaded == 1 and missing == ["bias"]


def test_the_launcher_identifies_the_module_from_its_weights(tiny_deep_run, tmp_path):
    """One implementation serves a whole group, so the weights say which instance."""
    from model_partition.runtime.launcher import _module_paths

    graph = tiny_deep_run.graph
    submodules = {m.id: list(m.submodules) for m in graph.partitioned_modules}
    target = [m for m in graph.partitioned_modules if m.kind == "decoder_layers"][-1]
    weights = {f"{target.submodules[0]}.self_attn.q_proj.weight": None}

    found = _module_paths(submodules, weights)
    assert found is not None
    assert found[0] == target.id

    assert _module_paths(submodules, {"nothing.recognisable": None}) is None


def test_every_module_of_a_shared_group_verifies(tiny_deep_run, tmp_path):
    """Regression: one implementation served a whole group but built only its first."""
    impl_dirs = _extract_impls(tiny_deep_run, tmp_path)
    report = verify_modules(tiny_deep_run.build_model, tiny_deep_run.bundle,
                            tiny_deep_run.graph, impl_dirs=impl_dirs)
    assert report.passed, report.render()
    decoders = [m.id for m in tiny_deep_run.graph.partitioned_modules
                if m.kind == "decoder_layers"]
    checked = {r.module_id for r in report.results}
    assert set(decoders) <= checked
def test_the_source_is_imported_once_per_group(tiny_deep_run, tmp_path):
    """Executing a modeling file per module made verification far slower than it is."""
    from model_partition.runtime import launcher

    launcher.clear_source_cache()
    try:
        impl_dirs = _extract_impls(tiny_deep_run, tmp_path)
        report = verify_modules(tiny_deep_run.build_model, tiny_deep_run.bundle,
                                tiny_deep_run.graph, impl_dirs=impl_dirs)
        imported = len(launcher._SOURCE_CACHE)
    finally:
        launcher.clear_source_cache()

    assert report.passed, report.render()
    assert len(report.results) > 5
    # One executed source per group directory, however many modules share it.
    assert imported == len(set(impl_dirs.values())), imported


def test_reuse_does_not_let_one_module_cover_another_s_missing_dump(tiny_deep_run, tmp_path):
    """The shared structure is poisoned per module, so a gap still shows."""
    from model_partition.runtime import launcher

    launcher.clear_source_cache()
    try:
        impl_dirs = _extract_impls(tiny_deep_run, tmp_path)
        decoders = [m.id for m in tiny_deep_run.graph.partitioned_modules
                    if m.kind == "decoder_layers"]
        victim = decoders[-1]
        tiny_deep_run.bundle.weights[victim] = tiny_deep_run.bundle.weights[victim][:1]
        report = verify_modules(tiny_deep_run.build_model, tiny_deep_run.bundle,
                                tiny_deep_run.graph, impl_dirs=impl_dirs)
    finally:
        launcher.clear_source_cache()
    assert not report.passed
    assert victim in {r.module_id for r in report.failures}


# -- parallel groups, functional nodes, windowed records ----------------------
#
# Three kinds of module the single-sequential-group assumption got wrong.


def test_a_parallel_group_is_checked_call_by_call(tiny_split_run, tmp_path):
    """Experts are alternatives: each recorded call is its own computation.

    Running the group as a pipeline — first expert's output into the second — would
    reproduce no reference at all, and comparing one call's input against the last
    call's output compares two different experts.
    """
    from model_partition.runtime import launcher

    graph = tiny_split_run.graph
    parallel = [m for m in graph.partitioned_modules if m.is_parallel]
    if not parallel:
        pytest.skip("this plan produced no parallel expert group")

    launcher.clear_source_cache()
    try:
        impl_dirs = _extract_impls(tiny_split_run, tmp_path)
        report = verify_modules(tiny_split_run.build_model, tiny_split_run.bundle,
                                graph, impl_dirs=impl_dirs)
    finally:
        launcher.clear_source_cache()

    target = parallel[0]
    checks = [r for r in report.checked if r.module_id == target.id]
    records = tiny_split_run.bundle.select(module_id=target.id,
                                          sample_id=tiny_split_run.sample_ids[0])
    assert len(checks) == len(records), report.render()
    assert all(r.passed for r in checks), report.render()


def test_a_functional_module_is_skipped_not_passed(tiny_run):
    """No submodule means no reference, which must not read as a pass."""
    from model_partition.planner.graph import ModuleNode

    graph = tiny_run.graph
    graph.tensors["h.extra"] = graph.tensors[graph.by_id("final_norm").inputs[0]]
    graph.modules.append(ModuleNode(
        id="layers.0.combine", kind="mlp", inputs=["h.0"], outputs=[],
        layer_indices=[0], code_signature="combine",
    ))
    report = verify_modules(tiny_run.build_model, tiny_run.bundle, graph,
                            module_ids=["layers.0.combine"])
    assert report.skipped and not report.checked
    assert report.passed is False  # nothing was checked, so nothing passed
    assert "no submodule" in report.skipped[0].skipped
    assert "skipped" in report.render()


def test_a_windowed_record_is_skipped_not_compared(tiny_run, tmp_path):
    """A windowed output is not a function of the windowed input."""
    impl_dirs = _extract_impls(tiny_run, tmp_path)
    target = next(m.id for m in tiny_run.graph.partitioned_modules
                  if m.kind == "decoder_layers")
    for record in tiny_run.bundle.records:
        if record.module_id == target:
            record.sliced = True

    report = verify_modules(tiny_run.build_model, tiny_run.bundle, tiny_run.graph,
                            impl_dirs=impl_dirs)
    skipped = {r.module_id for r in report.skipped}
    assert target in skipped
    assert "windowed" in next(r.skipped for r in report.skipped if r.module_id == target)
    assert report.passed, report.render()


def test_every_returned_tensor_is_compared_not_just_the_first():
    """A module returning a tuple must be checked element by element.

    Reducing both sides to their first tensor would pass a module whose primary
    output is right and whose auxiliary output is corrupt — attention modules
    commonly return several values.
    """
    import torch

    from model_partition.verify.numerics import compare_outputs

    reference = (torch.ones(2, 2), torch.ones(2, 2))
    actual = (torch.ones(2, 2), torch.ones(2, 2) * 3.0)
    results = compare_outputs(actual, reference, "attn")
    assert [c.name for c in results] == ["attn[0]", "attn[1]"]
    assert results[0].passed and not results[1].passed


def test_nested_outputs_are_walked_by_path():
    import torch

    from model_partition.verify.numerics import compare_outputs

    reference = {"hidden": torch.zeros(2), "aux": [torch.ones(2), torch.ones(2)]}
    actual = {"hidden": torch.zeros(2), "aux": [torch.ones(2), torch.zeros(2)]}
    results = compare_outputs(actual, reference, "out")
    names = {c.name: c.passed for c in results}
    assert names == {"out.hidden": True, "out.aux[0]": True, "out.aux[1]": False}


def test_a_missing_element_is_a_failure_not_an_absence():
    import torch

    from model_partition.verify.numerics import compare_outputs

    results = compare_outputs((torch.ones(2),), (torch.ones(2), torch.ones(2)), "out")
    assert len(results) == 2
    assert not results[1].passed and "missing" in results[1].reason


def test_a_structure_with_no_tensors_falls_back_to_one_comparison():
    from model_partition.verify.numerics import compare_outputs

    results = compare_outputs({"cache": None}, {"cache": None}, "out")
    assert len(results) == 1 and not results[0].passed


def test_identical_zero_tensors_match():
    """Cosine is undefined for zero vectors; a tensor still matches itself."""
    import torch

    from model_partition.verify.numerics import compare

    assert compare(torch.zeros(4), torch.zeros(4)).passed


# -- heterogeneous groups ----------------------------------------------------


def test_a_group_is_handed_every_argument_its_submodules_received(tiny_run):
    """A norm takes only the hidden state; the attention after it needs more.

    Handing the group only its first submodule's recorded call leaves the attention
    without its rotary embeddings and mask, which fails at the call rather than in
    the numbers.
    """
    from model_partition.runtime.module_runner import decode_call, decode_group_call

    module_id = next(m.id for m in tiny_run.graph.partitioned_modules
                     if len(m.submodules) > 1)
    records = tiny_run.bundle.select(module_id=module_id,
                                     sample_id=tiny_run.sample_ids[0])
    assert len(records) > 1

    _, first = decode_call(records[0], tiny_run.bundle.store)
    _, merged = decode_group_call(records, tiny_run.bundle.store)
    every = set()
    for record in records:
        every |= set(decode_call(record, tiny_run.bundle.store)[1])

    assert set(merged) == every
    assert set(first) <= set(merged)


def test_the_group_call_keeps_the_entry_point_s_value_for_a_shared_keyword():
    """First occurrence wins, so a keyword means what the group's input saw."""
    from model_partition.runtime.module_runner import decode_group_call
    from model_partition.trace import CallRecord

    records = [
        CallRecord(module_id="m", submodule="a", sample_id="s", order=0,
                   args=[], kwargs={"depth": 0, "shared": "first"}),
        CallRecord(module_id="m", submodule="b", sample_id="s", order=1,
                   args=[], kwargs={"shared": "second", "extra": 7}),
    ]
    from model_partition.tensorstore import TensorStore

    _, kwargs = decode_group_call(records, TensorStore("/nonexistent"))
    assert kwargs == {"depth": 0, "shared": "first", "extra": 7}
def test_each_submodule_gets_only_the_keywords_it_declares():
    """Passing a norm the attention's mask is a TypeError, not a wrong number."""
    from model_partition.runtime.launcher import _accepted

    class Norm:
        def forward(self, hidden_states):
            return hidden_states

    class Attention:
        def forward(self, hidden_states, position_embeddings=None, attention_mask=None):
            return hidden_states

    class Layer:
        def forward(self, hidden_states, **kwargs):
            return hidden_states

    offered = {"position_embeddings": 1, "attention_mask": 2, "hidden_states": 3}
    assert _accepted(Norm(), offered) == {}
    assert _accepted(Attention(), offered) == {
        "position_embeddings": 1, "attention_mask": 2}
    # A decoder layer taking **kwargs wants everything except the flowing tensor,
    # which is passed positionally — the trace records it as `hidden_states=`.
    assert _accepted(Layer(), offered) == {
        "position_embeddings": 1, "attention_mask": 2}


def test_an_out_of_memory_module_is_a_partition_failure(tiny_run, tmp_path, monkeypatch):
    """A module the plan said would fit and does not is the plan's problem.

    Finishing the check somewhere slower would report the promise as kept. The loop's
    answer to a module that will not fit is a smaller module.
    """
    from model_partition.verify import modules as modules_module

    impl_dirs = _extract_impls(tiny_run, tmp_path)
    target = next(m.id for m in tiny_run.graph.partitioned_modules
                  if m.kind == "decoder_layers")

    def out_of_memory(model, record, bundle, builder, device, branch=False, group=None):
        if record.module_id == target:
            raise RuntimeError("CUDA out of memory. Tried to allocate 16.00 GiB")
        return modules_module.replay_record(model, record, bundle.store, device=device)

    monkeypatch.setattr(modules_module, "_run_once", out_of_memory)
    report = verify_modules(tiny_run.build_model, tiny_run.bundle, tiny_run.graph,
                            impl_dirs=impl_dirs)

    oversized = report.oversized
    assert oversized and all(r.module_id == target for r in oversized)
    assert "Partition it further" in oversized[0].error
    assert not report.passed
    # Not a skip: the check did not pass and the report says why.
    assert target not in {r.module_id for r in report.skipped}


def test_an_oversized_module_sends_the_repair_to_the_plan(tiny_run, tmp_path, monkeypatch):
    """Arithmetic cannot make a module fit, so the plan is the surface to edit."""
    from model_partition.extract import extract
    from model_partition.loop.stages import stage_verify_modules
    from model_partition.verify import modules as modules_module

    def out_of_memory(*args, **kwargs):
        raise RuntimeError("CUDA out of memory. Tried to allocate 16.00 GiB")

    # The stage reads the implementations from the run, and every module needs one
    # before an allocation failure is the reason a check did not pass.
    extract(tiny_run.graph, tiny_run.build_model(), tiny_run.layout.modules_dir,
            run_root=tiny_run.layout.root, sample_ids=tiny_run.sample_ids,
            weight_tensors=tiny_run.bundle.weights)
    monkeypatch.setattr(modules_module, "_run_once", out_of_memory)

    class Ctx:
        graph = tiny_run.graph
        bundle = tiny_run.bundle
        build_model = staticmethod(tiny_run.build_model)
        build_meta_model = None
        inventory = tiny_run.inventory
        layout = tiny_run.layout
        options = type("O", (), {"device": "cpu"})()
        result = tiny_run.result
        budget = type("B", (), {"gpu": None, "total_bytes": 0})()
        notes: list = []
        verify_report = None

    result = stage_verify_modules(Ctx())
    assert not result.ok and result.repairable
    assert result.repair_surface == "plan"
    assert "Partition it further" in result.detail or "need partitioning" in result.detail


def test_an_allocation_failure_is_told_apart_from_a_real_error():
    from model_partition.verify.modules import _is_out_of_memory

    assert _is_out_of_memory(RuntimeError("CUDA out of memory. Tried to allocate 16 GiB"))
    assert not _is_out_of_memory(RuntimeError("shape mismatch"))


# -- accumulated error through the chained implementations --------------------


#: Appended to a generated implementation to make it wrong in a way cosine can see.
#: A uniform scale would not do: cosine ignores it and the next norm removes it.
BREAK_THE_OUTPUT = '''

def build_module(config, weights, device="cpu", submodule=None):
    inner = _original(config, weights, device, submodule)

    def run(*args, **kwargs):
        out = inner(*args, **kwargs)
        first = out[0] if isinstance(out, tuple) else out
        broken = first.flip(-1)
        return (broken,) + tuple(out[1:]) if isinstance(out, tuple) else broken

    return run
'''


def _chain(run, tmp_path, **kwargs):
    from model_partition.runtime import launcher
    from model_partition.verify.chain import verify_chain

    launcher.clear_source_cache()
    try:
        impl_dirs = _extract_impls(run, tmp_path)
        return verify_chain(run.bundle, run.graph, impl_dirs, **kwargs)
    finally:
        launcher.clear_source_cache()


def test_the_chain_carries_each_output_into_the_next_module(tiny_run, tmp_path):
    """Per-module checks restart from the recording, so nothing accumulates there."""
    suite = _chain(tiny_run, tmp_path)
    assert suite.reports
    report = suite.reports[0]
    assert report.chained_steps, report.render()
    assert suite.passed, suite.render()
    assert suite.worst_cosine() > 0.99


def test_the_chain_reports_drift_in_dependency_order(tiny_run, tmp_path):
    suite = _chain(tiny_run, tmp_path)
    curve = suite.reports[0].drift_curve()
    order = tiny_run.graph.topological_order()
    positions = [order.index(module_id) for module_id, _ in curve]
    assert positions == sorted(positions)


def test_an_unbroken_chain_predicts_the_same_tokens(tiny_run, tmp_path):
    report = _chain(tiny_run, tmp_path).reports[0]
    assert report.unbroken, report.render()
    assert report.tokens and report.tokens == report.reference_tokens
    assert report.top1 == 1.0


def test_a_broken_implementation_shows_up_as_drift_downstream(tiny_run, tmp_path):
    """The point of chaining: an error in one module reaches the ones after it."""
    from model_partition.runtime import launcher
    from model_partition.verify.chain import verify_chain

    launcher.clear_source_cache()
    try:
        impl_dirs = _extract_impls(tiny_run, tmp_path)
        target = next(m.id for m in tiny_run.graph.partitioned_modules
                      if m.kind == "decoder_layers")
        # Reversing the hidden dimension is wrong in a way cosine can see. A uniform
        # scale would not do: cosine ignores it, and the next norm removes it.
        # Edited the way a porter edits it: keep the launcher, wrap what it returns.
        path = impl_dirs[target] / "inference.py"
        path.write_text(path.read_text().replace("def build_module(", "def _original(")
                        + BREAK_THE_OUTPUT)
        suite = verify_chain(tiny_run.bundle, tiny_run.graph, impl_dirs)
    finally:
        launcher.clear_source_cache()

    assert not suite.passed, suite.render()
    report = suite.reports[0]
    diverged = report.first_divergence()
    assert diverged is not None
    # The broken module is where it starts, and later modules carry it.
    assert diverged.module_id == target
    assert target in suite.diverging_modules()
    downstream = [s for s in report.steps
                  if s.chained and s.module_id != target and not s.passed]
    assert downstream, "the error should reach the modules after it"
    assert not report.tokens_agree


def test_a_split_layers_residual_add_is_reconstructed_from_the_recording(tiny_split_run,
                                                                        tmp_path):
    """Splitting attention from the FFN leaves the residual add in neither half.

    The recording identifies it — the gap between one module's output and the next
    one's input is exactly a tensor the chain already holds — so the arithmetic is
    known and drift keeps travelling. Refusing the edge instead would measure each
    half against a fresh recording and see no propagation at all, which is the thing
    the chain exists to measure.
    """
    suite = _chain(tiny_split_run, tmp_path)
    report = suite.reports[0]
    assert suite.passed, suite.render()
    assert report.unbroken, report.render()
    assert not report.unchained, report.render()
    residual = [s for s in report.steps if "residual add" in s.via]
    assert len(residual) >= 2, report.render()
    assert report.tokens_agree


def test_an_edge_the_recording_contradicts_is_not_carried(tiny_run, tmp_path):
    """A plan can claim dataflow that does not exist, and that is not drift.

    Carrying such an edge would feed a module a tensor the model never gave it and
    report the plan's mistake as an implementation's.
    """
    graph = tiny_run.graph
    last = graph.partitioned_modules[-1]
    first_hidden = graph.partitioned_modules[0].outputs[0]
    # Claim the head reads the embedding: nothing between them is a single tensor.
    last.inputs = [first_hidden]

    suite = _chain(tiny_run, tmp_path)
    report = suite.reports[0]
    broken = next(s for s in report.steps if s.module_id == last.id)
    assert not broken.chained
    assert "belongs to no module" in broken.unchained_reason
    assert not report.unbroken


def test_tokens_are_only_credited_when_the_chain_is_unbroken(tiny_run, tmp_path):
    """Otherwise a broken chain would claim credit for the recording's tokens."""
    graph = tiny_run.graph
    last = graph.partitioned_modules[-1]
    last.inputs = [graph.partitioned_modules[0].outputs[0]]

    suite = _chain(tiny_run, tmp_path)
    report = suite.reports[0]
    assert not report.unbroken
    # The head restarted from the recording, so its tokens are the recording's own.
    assert report.tokens_agree and not report.credited_tokens
    assert suite.kept_tokens() == 0
    assert suite.mean_top1() == 0.0
    assert report.passed  # held, because nothing carried drifted


# -- one submodule, several invocations --------------------------------------


class _Rec:
    """Just enough of a CallRecord for the run-splitting logic."""

    def __init__(self, submodule: str) -> None:
        self.submodule = submodule


class _Node:
    def __init__(self, is_parallel: bool = False) -> None:
        self.is_parallel = is_parallel


def test_a_submodule_called_twice_is_two_checks_not_one_long_chain():
    """DeepSeek's draft head calls `markov_head` once per drafted position.

    Those calls are not a chain — the module is pure, and each one runs on the token the
    previous call sampled. Pairing the first call's input with the last call's output
    checks neither, and it only looked like a pass because the dumps used to share tensor
    paths, so every record read the last invocation's numbers.
    """
    from model_partition.verify.modules import _record_runs

    records = [_Rec("mtp.2.markov_head") for _ in range(5)]
    runs = _record_runs(_Node(), records, with_impl=True)
    assert len(runs) == 5
    assert all(len(run) == 1 for run in runs)
    assert [run[0] for run in runs] == records


def test_a_heterogeneous_group_is_still_one_computation():
    """A norm feeding attention is input-from-the-first, reference-from-the-last."""
    from model_partition.verify.modules import _record_runs

    records = [_Rec("layers.0.attn_norm"), _Rec("layers.0.attn")]
    runs = _record_runs(_Node(), records, with_impl=True)
    assert runs == [records]


def test_a_group_invoked_twice_splits_into_two_computations():
    """Both things at once: distinct submodules chained, and the chain run again."""
    from model_partition.verify.modules import _record_runs

    norm_a, attn_a = _Rec("layers.0.attn_norm"), _Rec("layers.0.attn")
    norm_b, attn_b = _Rec("layers.0.attn_norm"), _Rec("layers.0.attn")
    runs = _record_runs(_Node(), [norm_a, attn_a, norm_b, attn_b], with_impl=True)
    assert runs == [[norm_a, attn_a], [norm_b, attn_b]]


def test_a_parallel_group_checks_every_call_against_its_own_output():
    """Each expert sees its own routed tokens, so there is no chain to collapse."""
    from model_partition.verify.modules import _record_runs

    records = [_Rec("experts.0"), _Rec("experts.1")]
    assert _record_runs(_Node(is_parallel=True), records, with_impl=True) == [
        [records[0]], [records[1]],
    ]
    # And without an implementation each submodule is replayed against its own record.
    assert _record_runs(_Node(), records, with_impl=False) == [[records[0]], [records[1]]]
