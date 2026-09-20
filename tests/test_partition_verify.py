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
