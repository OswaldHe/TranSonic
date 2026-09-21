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
    Comparison,
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


def test_comparison_defaults_are_passing():
    assert Comparison(name="x", passed=True).pass_fraction == 1.0


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


def test_the_baseline_identifies_the_module_from_its_weights(tiny_deep_run, tmp_path):
    """One implementation serves many modules; the weights say which instance."""
    from model_partition.runtime.baseline import _module_for_weights
    from model_partition.runtime.module_runner import load_named_weights

    graph = tiny_deep_run.graph
    group = [m.id for m in graph.partitioned_modules if m.kind == "decoder_layers"]
    assert len(group) > 1
    for module_id in group[:3]:
        weights = load_named_weights(tiny_deep_run.bundle, module_id)
        assert _module_for_weights(graph, group, weights) == module_id


def test_every_module_of_a_shared_group_verifies(tiny_deep_run, tmp_path):
    """Regression: the baseline built the group's first module for all of them."""
    impl_dirs = _extract_impls(tiny_deep_run, tmp_path)
    report = verify_modules(tiny_deep_run.build_model, tiny_deep_run.bundle,
                            tiny_deep_run.graph, impl_dirs=impl_dirs)
    assert report.passed, report.render()
    decoders = [m.id for m in tiny_deep_run.graph.partitioned_modules
                if m.kind == "decoder_layers"]
    checked = {r.module_id for r in report.results}
    assert set(decoders) <= checked


def test_structure_is_reused_across_modules(tiny_deep_run, tmp_path):
    """Instantiating the model per module made verification 80x slower."""
    from model_partition.runtime import baseline

    baseline.clear_structure_cache()
    built = {"n": 0}
    original = baseline._structure

    def counting(run, device):
        if (str(run.layout.root), device) not in baseline._STRUCTURE_CACHE:
            built["n"] += 1
        return original(run, device)

    baseline._structure = counting
    try:
        impl_dirs = _extract_impls(tiny_deep_run, tmp_path)
        report = verify_modules(tiny_deep_run.build_model, tiny_deep_run.bundle,
                                tiny_deep_run.graph, impl_dirs=impl_dirs)
    finally:
        baseline._structure = original
        baseline.clear_structure_cache()
    assert report.passed, report.render()
    assert len(report.results) > 5
    assert built["n"] == 1, f"structure built {built['n']} times"


def test_reuse_does_not_let_one_module_cover_another_s_missing_dump(tiny_deep_run, tmp_path):
    """The shared structure is poisoned per module, so a gap still shows."""
    from model_partition.runtime import baseline

    baseline.clear_structure_cache()
    try:
        impl_dirs = _extract_impls(tiny_deep_run, tmp_path)
        decoders = [m.id for m in tiny_deep_run.graph.partitioned_modules
                    if m.kind == "decoder_layers"]
        victim = decoders[-1]
        tiny_deep_run.bundle.weights[victim] = tiny_deep_run.bundle.weights[victim][:1]
        report = verify_modules(tiny_deep_run.build_model, tiny_deep_run.bundle,
                                tiny_deep_run.graph, impl_dirs=impl_dirs)
    finally:
        baseline.clear_structure_cache()
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
    from model_partition.runtime import baseline

    graph = tiny_split_run.graph
    parallel = [m for m in graph.partitioned_modules if m.is_parallel]
    if not parallel:
        pytest.skip("this plan produced no parallel expert group")

    baseline.clear_structure_cache()
    try:
        impl_dirs = _extract_impls(tiny_split_run, tmp_path)
        report = verify_modules(tiny_split_run.build_model, tiny_split_run.bundle,
                                graph, impl_dirs=impl_dirs)
    finally:
        baseline.clear_structure_cache()

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
    from model_partition.runtime.baseline import _accepted_kwargs

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
    assert _accepted_kwargs(Norm(), offered) == {}
    assert _accepted_kwargs(Attention(), offered) == {
        "position_embeddings": 1, "attention_mask": 2}
    # A decoder layer taking **kwargs wants everything except the flowing tensor,
    # which is passed positionally — the trace records it as `hidden_states=`.
    assert _accepted_kwargs(Layer(), offered) == {
        "position_embeddings": 1, "attention_mask": 2}


# -- running out of accelerator memory ---------------------------------------


def test_an_out_of_memory_check_moves_to_the_host_and_still_counts(tiny_run, tmp_path,
                                                                  monkeypatch):
    """Full attention at long context is quadratic; that is capacity, not error."""
    from model_partition.verify import modules as modules_module

    impl_dirs = _extract_impls(tiny_run, tmp_path)
    target = next(m.id for m in tiny_run.graph.partitioned_modules
                  if m.kind == "decoder_layers")
    real_run_once = modules_module._run_once
    calls = {"n": 0}

    def flaky(model, record, bundle, builder, device, branch=False, group=None):
        if record.module_id == target and device == "cuda" and calls["n"] == 0:
            calls["n"] += 1
            raise RuntimeError("CUDA out of memory. Tried to allocate 16.00 GiB")
        return real_run_once(model, record, bundle, builder, device,
                             branch=branch, group=group)

    monkeypatch.setattr(modules_module, "_run_once", flaky)
    monkeypatch.setattr(modules_module, "move_submodules", lambda *a, **k: 0)

    report = verify_modules(tiny_run.build_model, tiny_run.bundle, tiny_run.graph,
                            model_device="cpu", module_device="cuda",
                            impl_dirs=impl_dirs)
    assert calls["n"] == 1, "the failure should have been injected once"
    moved = [r for r in report.checked if r.note]
    assert moved and "not enough cuda memory" in moved[0].note
    assert all(r.passed for r in report.checked), report.render()


def test_an_allocation_failure_is_told_apart_from_a_real_error():
    from model_partition.verify.modules import _is_out_of_memory

    assert _is_out_of_memory(RuntimeError("CUDA out of memory. Tried to allocate 16 GiB"))
    assert not _is_out_of_memory(RuntimeError("shape mismatch"))
