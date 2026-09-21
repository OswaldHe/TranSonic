# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for emulated end-to-end inference from partition artifacts."""

import pytest

from model_partition.runtime.emulation import (
    EmulationError,
    capture_boundaries,
    fill_from_dumps,
    generate,
    greedy_step,
)
from model_partition.verify.emulate import EmulationInput, EmulationReport, emulate
from model_partition.verify.judge import StubJudge, Verdict

torch = pytest.importorskip("torch")


def inputs_for(run, n_tokens: int = 8):
    from tests.fixtures.tiny_llm import sample_inputs

    ids = sample_inputs(len(run.sample_ids), n_tokens)
    return [EmulationInput(sample_id, ids[i:i + 1], prompt=f"prompt {i}")
            for i, sample_id in enumerate(run.sample_ids)]


class AcceptAll:
    def judge(self, prompt, continuation, sample_id=""):
        return Verdict(fluent=True, score=5, reason="accepted", sample_id=sample_id)


class RejectAll:
    def judge(self, prompt, continuation, sample_id=""):
        return Verdict(fluent=False, score=1, reason="rejected", sample_id=sample_id)


# -- assembling from dumps ---------------------------------------------------


def test_model_assembles_entirely_from_dumps(tiny_run):
    model = tiny_run.build_model()
    report = fill_from_dumps(model, tiny_run.bundle, tiny_run.graph)
    assert report.complete
    assert report.applied > 0
    assert report.missing == []
    assert sum(report.by_module.values()) == report.applied


def test_incomplete_dumps_are_reported_not_tolerated(tiny_run):
    """Regression: a missing parameter must fail loudly, not silently pass."""
    target = next(m.id for m in tiny_run.graph.partitioned_modules
                  if m.kind == "decoder_layers")
    tiny_run.bundle.weights[target] = tiny_run.bundle.weights[target][:2]
    with pytest.raises(EmulationError, match="no dumped value"):
        fill_from_dumps(tiny_run.build_model(), tiny_run.bundle, tiny_run.graph)


def test_non_strict_fill_returns_the_gap_instead_of_raising(tiny_run):
    target = next(m.id for m in tiny_run.graph.partitioned_modules
                  if m.kind == "decoder_layers")
    tiny_run.bundle.weights[target] = []
    report = fill_from_dumps(tiny_run.build_model(), tiny_run.bundle, tiny_run.graph, strict=False)
    assert not report.complete
    assert report.missing
    assert "missing" in report.summary()


# -- boundary capture --------------------------------------------------------


def test_boundaries_capture_only_the_first_forward(tiny_run):
    """Regression: generation re-runs the forward, and later steps have a
    different sequence length than the trace."""
    model = tiny_run.build_model()
    handles, sink = capture_boundaries(model, tiny_run.graph)
    try:
        first = torch.zeros(1, 8, dtype=torch.long)
        model(first)
        model(torch.zeros(1, 13, dtype=torch.long))
    finally:
        for handle in handles:
            handle.remove()
    for module_id, observed in sink.items():
        tensor = observed[0] if isinstance(observed, tuple) else observed
        assert tensor.shape[1] == 8, module_id


def test_boundaries_are_hooked_on_each_group_s_last_submodule(tiny_run):
    model = tiny_run.build_model()
    handles, sink = capture_boundaries(model, tiny_run.graph)
    try:
        model(torch.zeros(1, 8, dtype=torch.long))
    finally:
        for handle in handles:
            handle.remove()
    assert set(sink) == {m.id for m in tiny_run.graph.partitioned_modules if m.submodules}


# -- generation --------------------------------------------------------------


def test_greedy_step_is_argmax():
    logits = torch.tensor([[[0.0, 5.0, 1.0]]])
    assert greedy_step(logits) == 1


def test_sampling_with_a_seed_is_reproducible():
    logits = torch.randn(1, 4, 100)
    first = greedy_step(logits, temperature=1.0, seed=7)
    second = greedy_step(logits, temperature=1.0, seed=7)
    assert first == second


def test_generate_produces_the_requested_number_of_tokens(tiny_run):
    model = tiny_run.build_model()
    tokens, logits = generate(model, torch.zeros(1, 4, dtype=torch.long), max_new_tokens=5)
    assert len(tokens) == 5
    assert logits is not None


def test_generate_stops_at_eos(tiny_run):
    model = tiny_run.build_model()
    tokens, _ = generate(model, torch.zeros(1, 4, dtype=torch.long), max_new_tokens=10)
    stop = tokens[0]
    stopped, _ = generate(model, torch.zeros(1, 4, dtype=torch.long),
                          max_new_tokens=10, eos_token_id=stop)
    assert stopped == [stop]


def test_generate_is_deterministic_at_temperature_zero(tiny_run):
    model = tiny_run.build_model()
    prompt = torch.zeros(1, 4, dtype=torch.long)
    assert generate(model, prompt, max_new_tokens=4)[0] == generate(model, prompt, max_new_tokens=4)[0]


# -- emulate -----------------------------------------------------------------


def test_emulation_passes_with_intact_artifacts(tiny_run):
    report = emulate(tiny_run.build_model, tiny_run.bundle, tiny_run.graph,
                     inputs=inputs_for(tiny_run), judge=AcceptAll(), max_new_tokens=4)
    assert report.passed, report.render()
    assert all(o.boundaries_passed for o in report.outcomes)
    assert all(len(o.token_ids) == 4 for o in report.outcomes)
    assert report.mean_score() == 5.0


def test_boundary_checks_compare_against_the_right_record(tiny_run):
    """Regression: a multi-layer group's boundary was compared to its first
    submodule's output instead of its last."""
    report = emulate(tiny_run.build_model, tiny_run.bundle, tiny_run.graph,
                     inputs=inputs_for(tiny_run), judge=AcceptAll(), max_new_tokens=2)
    multi = [m for m in tiny_run.graph.partitioned_modules if len(m.submodules) > 1]
    assert multi, "expected at least one grouped module in this plan"
    checked = {c.name.removeprefix("boundary:") for o in report.outcomes
               for c in o.boundary_checks}
    assert {m.id for m in multi} <= checked
    assert all(c.passed for o in report.outcomes for c in o.boundary_checks)


def test_emulated_tokens_match_the_pristine_model_exactly(tiny_run):
    """The point of the whole pipeline: a model assembled from dumps must
    generate the same tokens as the original."""
    pristine = tiny_run.build_model()
    items = inputs_for(tiny_run)
    expected = {
        item.sample_id: generate(pristine, item.input_ids, max_new_tokens=6)[0]
        for item in items
    }
    report = emulate(tiny_run.build_model, tiny_run.bundle, tiny_run.graph,
                     inputs=items, judge=AcceptAll(), max_new_tokens=6)
    assert report.passed, report.render()
    for outcome in report.outcomes:
        assert outcome.token_ids == expected[outcome.sample_id], outcome.sample_id


def test_a_rejecting_judge_fails_the_report(tiny_run):
    report = emulate(tiny_run.build_model, tiny_run.bundle, tiny_run.graph,
                     inputs=inputs_for(tiny_run), judge=RejectAll(), max_new_tokens=2)
    assert not report.passed
    assert len(report.failures) == len(report.outcomes)


def test_boundary_mismatch_fails_even_when_the_judge_is_happy(tiny_run):
    """A plausible-looking continuation must not mask a wrong module boundary."""
    store = tiny_run.bundle.store
    target = next(m for m in tiny_run.graph.partitioned_modules if m.kind == "decoder_layers")
    outputs = [e for e in store.find(role="output", module_id=target.id)]
    assert outputs
    blob = store.blob_path(outputs[-1])
    raw = bytearray(blob.read_bytes())
    for i in range(0, min(len(raw), 4096)):
        raw[i] = 0x7F
    blob.write_bytes(bytes(raw))
    report = emulate(tiny_run.build_model, tiny_run.bundle, tiny_run.graph,
                     inputs=inputs_for(tiny_run), judge=AcceptAll(), max_new_tokens=2)
    assert not report.passed
    assert any(o.failed_boundaries() for o in report.outcomes)


def test_decode_is_used_for_the_text(tiny_run):
    report = emulate(tiny_run.build_model, tiny_run.bundle, tiny_run.graph,
                     inputs=inputs_for(tiny_run), judge=AcceptAll(), max_new_tokens=3,
                     decode=lambda ids: "|".join(str(i) for i in ids))
    assert report.outcomes[0].text.count("|") == 2


def test_boundary_checks_can_be_skipped(tiny_run):
    report = emulate(tiny_run.build_model, tiny_run.bundle, tiny_run.graph,
                     inputs=inputs_for(tiny_run), judge=AcceptAll(),
                     max_new_tokens=2, check_boundaries=False)
    assert report.passed
    assert all(o.boundary_checks == [] for o in report.outcomes)


def test_report_serializes(tiny_run):
    report = emulate(tiny_run.build_model, tiny_run.bundle, tiny_run.graph,
                     inputs=inputs_for(tiny_run), judge=AcceptAll(), max_new_tokens=2)
    payload = report.to_dict()
    assert payload["passed"] is True
    assert payload["fill"]["complete"] is True
    assert len(payload["outcomes"]) == len(report.outcomes)


def test_moe_model_emulates(tiny_moe_run):
    report = emulate(tiny_moe_run.build_model, tiny_moe_run.bundle, tiny_moe_run.graph,
                     inputs=inputs_for(tiny_moe_run), judge=AcceptAll(), max_new_tokens=3)
    assert report.passed, report.render()


def test_empty_report_is_not_passed():
    assert EmulationReport().passed is False


def test_stub_judge_flags_degenerate_repetition():
    judge = StubJudge()
    assert not judge.judge("p", "the the the the the the the the").passed()
    assert judge.judge("p", "a clearly varied continuation of text").passed()


# -- generation runs the loop's own code -------------------------------------
#
# The implementations are the deliverable. Generating through the model's own
# modules would check the reference against itself and print tokens the shipped
# code never produced.


def _impls(run, tmp_path):
    from model_partition.extract import extract
    from model_partition.runtime.module_impl import find_impl_dirs

    extract(run.graph, run.build_model(), tmp_path / "modules",
            run_root=run.layout.root, sample_ids=run.sample_ids,
            weight_tensors=run.bundle.weights, param_names=run.bundle.weight_params)
    return find_impl_dirs(tmp_path / "modules")


def test_installing_the_implementations_replaces_every_submodule(tiny_run, tmp_path):
    from model_partition.runtime import launcher
    from model_partition.runtime.emulation import install_implementations
    from model_partition.trace import _lookup

    launcher.clear_source_cache()
    try:
        impl_dirs = _impls(tiny_run, tmp_path)
        model = tiny_run.build_model()
        report = install_implementations(model, tiny_run.graph, tiny_run.bundle,
                                         impl_dirs)
        assert report.installed, report.summary()
        for name in report.installed:
            assert type(_lookup(model, name)).__name__ == "Implemented"
    finally:
        launcher.clear_source_cache()
def test_installation_does_not_instantiate_another_model(tiny_run, tmp_path):
    """The wrappers call source.py, so nothing else gets built to serve them."""
    from model_partition.runtime import launcher
    from model_partition.runtime.emulation import install_implementations
    from model_partition.trace import _lookup

    launcher.clear_source_cache()
    built = {"n": 0}
    original = launcher.load_source

    def counting(directory, source_module=None):
        built["n"] += 1
        return original(directory, source_module)

    launcher.load_source = counting
    try:
        impl_dirs = _impls(tiny_run, tmp_path)
        model = tiny_run.build_model()
        report = install_implementations(model, tiny_run.graph, tiny_run.bundle, impl_dirs)
    finally:
        launcher.load_source = original
        launcher.clear_source_cache()

    assert report.installed
    # Every replacement runs code loaded from a module directory, not the model's.
    for name in report.installed:
        wrapper = _lookup(model, name)
        inner = type(wrapper._built)
        # The copy is registered under its original package so its own relative
        # imports resolve, so the private leaf is the last component.
        leaf = inner.__module__.rsplit(".", 1)[-1]
        assert leaf.startswith(launcher.SOURCE_MODULE_PREFIX), inner.__module__
    assert built["n"] >= 1


def test_emulation_generates_through_the_extracted_implementations(tiny_run, tmp_path):
    from model_partition.runtime import launcher

    launcher.clear_source_cache()
    try:
        impl_dirs = _impls(tiny_run, tmp_path)
        report = emulate(
            tiny_run.build_model, tiny_run.bundle, tiny_run.graph,
            inputs=inputs_for(tiny_run), judge=AcceptAll(), max_new_tokens=3,
            impl_dirs=impl_dirs,
        )
    finally:
        launcher.clear_source_cache()

    assert report.install is not None and report.install.installed
    assert report.mechanically_passed, report.render()
    assert "extracted implementation" in report.render()


def test_emulation_says_when_a_submodule_kept_the_model_s_own_code(tiny_run, tmp_path):
    """Partial installation has to be visible, not assumed away."""
    from model_partition.runtime import launcher
    from model_partition.runtime.emulation import install_implementations

    launcher.clear_source_cache()
    try:
        impl_dirs = _impls(tiny_run, tmp_path)
        dropped = sorted(impl_dirs)[0]
        del impl_dirs[dropped]
        report = install_implementations(tiny_run.build_model(), tiny_run.graph,
                                         tiny_run.bundle, impl_dirs)
    finally:
        launcher.clear_source_cache()

    assert dropped in report.skipped
    assert not report.complete
    assert "left as the model's own" in report.summary()


# -- a judge that failed is not a judge that said no -------------------------


class JudgeErrors:
    def judge(self, prompt, continuation, sample_id=""):
        from model_partition.verify.judge import Verdict

        return Verdict(fluent=False, score=0, sample_id=sample_id,
                       error="claude exited 1: refusal")


def test_an_errored_judge_is_not_counted_as_a_decline(tiny_run):
    report = emulate(tiny_run.build_model, tiny_run.bundle, tiny_run.graph,
                     inputs=inputs_for(tiny_run), judge=JudgeErrors(), max_new_tokens=2)
    assert report.mechanically_passed
    assert not report.judge_declined
    assert len(report.judge_errored) == len(report.outcomes)
    # Nothing was assessed, so there is no mean to report.
    assert report.mean_score() == 0.0


def test_a_low_score_is_a_decline_not_an_error(tiny_run):
    report = emulate(tiny_run.build_model, tiny_run.bundle, tiny_run.graph,
                     inputs=inputs_for(tiny_run), judge=RejectAll(), max_new_tokens=2)
    assert report.judge_declined and not report.judge_errored


def test_the_mean_score_ignores_samples_the_judge_could_not_assess(tiny_run):
    from model_partition.verify.judge import Verdict

    class Mixed:
        def __init__(self):
            self.n = 0

        def judge(self, prompt, continuation, sample_id=""):
            self.n += 1
            if self.n == 1:
                return Verdict(fluent=False, score=0, error="timed out")
            return Verdict(fluent=True, score=5)

    report = emulate(tiny_run.build_model, tiny_run.bundle, tiny_run.graph,
                     inputs=inputs_for(tiny_run), judge=Mixed(), max_new_tokens=2)
    assert report.mean_score() == 5.0


# -- memory during generation ------------------------------------------------
#
# A 1.6 GiB model reached tens of gigabytes of cached allocator blocks because a
# vocabulary-wide logits tensor was held per step, at a different size each step.


def test_generate_keeps_only_the_last_position(tiny_run):
    """The returned logits must not be the whole [1, seq, vocab] tensor."""
    ids = _first_input_ids(tiny_run)
    tokens, logits = generate(tiny_run.build_model(), ids, max_new_tokens=3)
    assert len(tokens) == 3
    assert logits.shape[-2] == 1, f"kept {logits.shape[-2]} positions, expected 1"


def test_boundary_hooks_are_gone_before_generation_starts(tiny_run, monkeypatch):
    """Otherwise every module's output is held for the whole generation."""
    from model_partition.runtime import emulation

    hooked_during_generate = {"any": False}
    real_generate = emulation.generate

    def checking(model, *args, **kwargs):
        for module in model.modules():
            if getattr(module, "_forward_hooks", None):
                hooked_during_generate["any"] = True
        return real_generate(model, *args, **kwargs)

    monkeypatch.setattr("model_partition.verify.emulate.generate", checking)
    emulate(tiny_run.build_model, tiny_run.bundle, tiny_run.graph,
            inputs=inputs_for(tiny_run), judge=AcceptAll(), max_new_tokens=2)
    assert not hooked_during_generate["any"]


def test_boundaries_are_still_checked_on_their_own_forward(tiny_run):
    report = emulate(tiny_run.build_model, tiny_run.bundle, tiny_run.graph,
                     inputs=inputs_for(tiny_run), judge=AcceptAll(), max_new_tokens=2)
    assert all(o.boundary_checks for o in report.outcomes)
    assert all(o.boundaries_passed for o in report.outcomes)


def _first_input_ids(run):
    return inputs_for(run)[0].input_ids


def test_generation_asks_the_model_for_only_the_last_row(tiny_run):
    """A full-sequence logits tensor is 8 GB at long context, and unused but for one row."""
    from model_partition.runtime.emulation import _last_logits_argument

    model = tiny_run.build_model()
    keyword = _last_logits_argument(model)
    if keyword is None:
        pytest.skip("this model's forward takes no such argument")

    seen = {}
    original = type(model).forward

    def recording(self, *args, **kwargs):
        seen.update(kwargs)
        return original(self, *args, **kwargs)

    type(model).forward = recording
    try:
        generate(model, _first_input_ids(tiny_run), max_new_tokens=2)
    finally:
        type(model).forward = original
    assert seen.get(keyword) == 1


def test_generation_still_works_when_the_model_takes_no_such_argument(tiny_run):
    from model_partition.runtime import emulation

    model = tiny_run.build_model()
    ids = _first_input_ids(tiny_run)
    with_flag, _ = emulation.generate(model, ids, max_new_tokens=3)

    real = emulation._last_logits_argument
    emulation._last_logits_argument = lambda _model: None
    try:
        without, _ = emulation.generate(model, ids, max_new_tokens=3)
    finally:
        emulation._last_logits_argument = real
    assert with_flag == without
