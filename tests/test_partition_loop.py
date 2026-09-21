# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for loop state, stage caching, and the loop end to end."""

import json

import pytest

from model_partition.layout import RunLayout
from model_partition.loop.state import (
    FAILED,
    OK,
    PENDING,
    STAGES,
    LoopState,
    content_hash,
)

pytest.importorskip("torch")


# -- state -------------------------------------------------------------------


def test_content_hash_is_stable_and_order_independent_within_dicts():
    assert content_hash({"a": 1, "b": 2}) == content_hash({"b": 2, "a": 1})
    assert content_hash([1, 2]) != content_hash([2, 1])
    assert len(content_hash("x")) == 16


def test_fresh_only_when_status_and_hash_both_match():
    state = LoopState()
    state.mark("plan", OK, "hash-1")
    assert state.is_fresh("plan", "hash-1")
    assert not state.is_fresh("plan", "hash-2")
    state.mark("plan", FAILED, "hash-1")
    assert not state.is_fresh("plan", "hash-1")


def test_invalidation_clears_the_stage_and_everything_after_it():
    """Editing the plan must not invalidate ingest, which is expensive."""
    state = LoopState()
    for stage in STAGES:
        state.mark(stage, OK, f"h-{stage}")
    cleared = state.invalidate_from("plan")
    assert cleared == ["plan", "trace", "extract", "verify_modules", "verify_chain",
                       "emulate", "retain"]
    assert state.record("ingest").status == OK
    assert state.record("ingest").input_hash == "h-ingest"
    assert state.record("trace").status == PENDING
    assert state.record("trace").input_hash == ""


def test_invalidating_the_last_stage_touches_only_it():
    state = LoopState()
    for stage in STAGES:
        state.mark(stage, OK, "h")
    assert state.invalidate_from("retain") == ["retain"]
    assert state.record("emulate").status == OK


def test_invalidate_unknown_stage_raises():
    with pytest.raises(ValueError, match="Unknown stage"):
        LoopState().invalidate_from("nonsense")


def test_first_incomplete_walks_in_order():
    state = LoopState()
    assert state.first_incomplete() == "ingest"
    state.mark("ingest", OK)
    assert state.first_incomplete() == "plan"


def test_all_passed_requires_every_stage():
    state = LoopState()
    for stage in STAGES[:-1]:
        state.mark(stage, OK)
    assert not state.all_passed()
    state.mark(STAGES[-1], OK)
    assert state.all_passed()


def test_state_round_trips_through_disk(tmp_path):
    state = LoopState(slug="m", iteration=3)
    state.mark("ingest", OK, "h", "detail", {"n": 1}, 1.5)
    state.log_iteration({"result": "failed"})
    path = state.save(tmp_path / "state.json")
    reloaded = LoopState.load(path)
    assert reloaded.slug == "m" and reloaded.iteration == 3
    assert reloaded.record("ingest").status == OK
    assert reloaded.record("ingest").metrics == {"n": 1}
    assert reloaded.record("ingest").duration_s == 1.5
    assert reloaded.history[0]["result"] == "failed"


def test_missing_or_corrupt_state_file_yields_a_fresh_state(tmp_path):
    assert LoopState.load(tmp_path / "absent.json").iteration == 0
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert LoopState.load(bad).iteration == 0


def test_render_lists_every_stage():
    text = LoopState().render()
    for stage in STAGES:
        assert stage in text


# -- layout ------------------------------------------------------------------


def test_layout_creates_the_expected_tree(tmp_path):
    layout = RunLayout.create("my-model", tmp_path).ensure()
    assert layout.root.name == "my-model"
    for path in (layout.plan_dir, layout.trace_dir, layout.modules_dir, layout.reports_dir):
        assert path.is_dir()
    assert layout.graph_path.parent == layout.plan_dir
    assert layout.tokens_file.parent == layout.reports_dir


def test_layout_run_manifest_round_trip(tmp_path):
    layout = RunLayout.create("m", tmp_path)
    layout.write_run({"spec": {"source": "hf:a/b"}})
    assert layout.read_run()["spec"]["source"] == "hf:a/b"


def test_reading_absent_manifest_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No run manifest"):
        RunLayout.create("m", tmp_path).read_run()


# -- the loop ----------------------------------------------------------------


class AcceptingJudge:
    """Deterministic judge for plumbing tests.

    The toy model has random weights, so real judging would reject its output —
    correctly. Judge behaviour itself is covered in test_partition_judge.py.
    """

    def judge(self, prompt, continuation, sample_id=""):
        from model_partition.verify.judge import Verdict

        return Verdict(fluent=True, score=5, reason="accepted", sample_id=sample_id)


def loop_for(run, tmp_path, judge=None, **overrides):
    from model_partition.loop.driver import PartitionLoop
    from model_partition.loop.stages import LoopOptions

    settings = dict(
        artifact_root=str(tmp_path / "loop-artifacts"),
        device="cpu", trace_device="cpu", judge_kind="stub",
        max_new_tokens=3, max_iterations=1, use_agent_planner=False,
        gpu_memory_gib=1.0,
    )
    settings.update(overrides)
    options = LoopOptions(**settings)
    return PartitionLoop(spec=run.spec, options=options, report=lambda _msg: None,
                         judge=judge or AcceptingJudge())


def test_loop_passes_every_stage_on_the_toy_model(tiny_run, tmp_path):
    result = loop_for(tiny_run, tmp_path).run()
    assert result.passed, result.error
    assert result.state.all_passed()
    assert result.iterations == 1
    assert result.summary_path and result.summary_path.is_file()


def test_loop_writes_the_expected_artifacts(tiny_run, tmp_path):
    result = loop_for(tiny_run, tmp_path).run()
    layout = result.context.layout
    assert layout.graph_path.is_file()
    assert (layout.trace_dir / "manifest.yaml").is_file()
    assert (layout.trace_dir / "records.yaml").is_file()
    assert (layout.modules_dir / "index.yaml").is_file()
    assert (layout.reports_dir / "verify.json").is_file()
    assert (layout.reports_dir / "emulate.json").is_file()
    assert (layout.reports_dir / "storage_estimate.txt").is_file()
    assert layout.tokens_file.is_file()
    assert layout.summary_file.is_file()


def test_loop_prints_sampled_tokens_for_human_review(tiny_run, tmp_path):
    result = loop_for(tiny_run, tmp_path).run()
    text = result.tokens_text()
    assert "sample:" in text and "tokens:" in text and "judge :" in text


def test_summary_records_the_plan_and_reports(tiny_run, tmp_path):
    result = loop_for(tiny_run, tmp_path).run()
    summary = result.summary_path.read_text()
    assert "## Plan" in summary
    assert "## Module verification" in summary
    assert "## Emulated inference" in summary
    assert "deduplicated implementation groups" in summary


def test_verify_report_json_is_machine_readable(tiny_run, tmp_path):
    result = loop_for(tiny_run, tmp_path).run()
    payload = json.loads((result.context.layout.reports_dir / "verify.json").read_text())
    assert payload["passed"] is True
    assert payload["n_failed"] == 0


def test_rerunning_the_loop_reuses_cached_stages(tiny_run, tmp_path):
    """The whole point of content-hash caching: no re-trace on a second run.

    Retention is off: a passed-and-pruned run short-circuits instead, which
    test_completed_pruned_run_short_circuits covers.
    """
    messages: list[str] = []
    first = loop_for(tiny_run, tmp_path, retain=False)
    assert first.run().passed

    second = loop_for(tiny_run, tmp_path, retain=False)
    second.report = messages.append
    assert second.run().passed
    cached = [m for m in messages if "cached" in m]
    # Tracing is the expensive stage and must not repeat. Ingest is metadata-only
    # and re-runs so a fresh process has the context later stages need.
    assert any("trace" in m for m in cached)
    assert any("extract" in m for m in cached)
    assert not any("ingest" in m for m in cached)


def test_editing_the_plan_invalidates_downstream_stages(tiny_run, tmp_path):
    from model_partition.planner.graph import PartitionGraph

    first = loop_for(tiny_run, tmp_path, retain=False)
    result = first.run()
    assert result.passed
    layout = result.context.layout

    graph = PartitionGraph.load(layout.graph_path)
    # The partition itself: one module now owns a submodule it did not own before.
    moved = next(m for m in graph.partitioned_modules if len(m.submodules) > 1)
    moved.submodules = moved.submodules[:-1]
    graph.save(layout.graph_path)

    messages: list[str] = []
    second = loop_for(tiny_run, tmp_path, retain=False)
    second.report = messages.append
    second.run()
    # Changing the plan re-runs everything downstream of it, tracing included.
    assert not any("trace" in m and "cached" in m for m in messages)
    assert not any("emulate" in m and "cached" in m for m in messages)


def test_rewriting_the_plan_without_changing_it_keeps_the_trace(tiny_run, tmp_path):
    """A re-plan that lands on the same partition must not re-trace.

    Re-tracing a large model costs hours and hundreds of gigabytes, so the question
    the cache asks is whether the partition changed — not whether the file did.
    """
    from model_partition.planner.graph import PartitionGraph

    first = loop_for(tiny_run, tmp_path, retain=False)
    layout = first.run().context.layout

    graph = PartitionGraph.load(layout.graph_path)
    graph.metadata["rationale"] = "reworded, same partition"
    graph.modules.reverse()
    graph.save(layout.graph_path)

    messages: list[str] = []
    second = loop_for(tiny_run, tmp_path, retain=False)
    second.report = messages.append
    assert second.run().passed
    assert any("trace" in m and "cached" in m for m in messages), messages


def test_loop_reports_a_failure_without_an_agent(tiny_run, tmp_path):
    """A broken plan fails the loop and names the stage, rather than hanging."""
    from model_partition.planner.graph import PartitionGraph

    loop = loop_for(tiny_run, tmp_path)
    ctx = loop.build_context()
    from model_partition.loop.stages import stage_ingest, stage_plan

    stage_ingest(ctx)
    stage_plan(ctx)
    graph = PartitionGraph.load(ctx.layout.graph_path)
    for module in graph.partitioned_modules:
        module.submodules = ["model.does_not_exist"]
    graph.save(ctx.layout.graph_path)

    result = loop_for(tiny_run, tmp_path).run()
    assert not result.passed
    assert "trace" in result.error
    assert "do not exist" in result.error


def test_retention_runs_as_the_final_stage(tiny_run, tmp_path):
    result = loop_for(tiny_run, tmp_path).run()
    retention = result.state.record("retain")
    assert retention.status == OK
    assert "kept_layers" in retention.metrics


def test_retention_can_be_disabled(tiny_run, tmp_path):
    result = loop_for(tiny_run, tmp_path, retain=False).run()
    assert result.passed
    assert result.state.record("retain").detail == "retention disabled"


def test_completed_pruned_run_short_circuits(tiny_deep_run, tmp_path):
    """Retention deletes non-representative dumps, so re-verifying a pruned run
    would fail for a reason unrelated to correctness."""
    first = loop_for(tiny_deep_run, tmp_path)
    assert first.run().passed

    messages: list[str] = []
    second = loop_for(tiny_deep_run, tmp_path)
    second.report = messages.append
    result = second.run()
    assert result.passed
    text = "\n".join(messages)
    assert "already passed" in text and "--force" in text
    assert not any("trace" in m and "ok" in m for m in messages)


def test_force_reruns_a_completed_run(tiny_run, tmp_path):
    first = loop_for(tiny_run, tmp_path, retain=False)
    assert first.run().passed

    messages: list[str] = []
    second = loop_for(tiny_run, tmp_path, retain=False, force=True)
    second.report = messages.append
    assert second.run().passed
    assert not any("already passed" in m for m in messages)


class _RecordingPlanner:
    """Stands in for AgentPlanner, recording each call it receives."""

    calls: list[dict] = []

    def __init__(self, **kwargs):
        pass

    def edit_plan(self, layout, graph, context, iteration, tag="plan"):
        from model_partition.planner.agent import AgentOutcome

        type(self).calls.append({"context": context, "iteration": iteration, "kind": tag})
        graph.metadata["refined_by"] = "agent"
        return AgentOutcome(ok=True), graph

    def review(self, layout, context, iteration):
        from model_partition.planner.agent import AgentOutcome

        type(self).calls.append({"context": context, "iteration": iteration,
                                 "kind": "review"})
        layout.review_file.parent.mkdir(parents=True, exist_ok=True)
        layout.review_file.write_text("diagnosis\n")
        return AgentOutcome(ok=True)

    def repair_modules(self, layout, context, iteration):
        from model_partition.planner.agent import AgentOutcome

        type(self).calls.append({"context": context, "iteration": iteration,
                                 "kind": "modules"})
        return AgentOutcome(ok=True)


def _patch_planner(monkeypatch, planner):
    from model_partition.planner import agent as agent_module

    monkeypatch.setattr(agent_module, "AgentPlanner", planner)


def test_plan_refinement_is_off_by_default(tiny_run, tmp_path, monkeypatch):
    """Refinement costs an agent call, so it must be opt-in."""
    _RecordingPlanner.calls = []
    _patch_planner(monkeypatch, _RecordingPlanner)
    loop = loop_for(tiny_run, tmp_path, retain=False, use_agent_planner=True)
    assert loop.options.refine_plan is False
    assert loop.run().passed
    assert _RecordingPlanner.calls == []


def test_refinement_is_skipped_when_the_agent_is_disabled(tiny_run, tmp_path, monkeypatch):
    _RecordingPlanner.calls = []
    _patch_planner(monkeypatch, _RecordingPlanner)
    loop = loop_for(tiny_run, tmp_path, retain=False, refine_plan=True,
                    use_agent_planner=False)
    assert loop.run().passed
    assert _RecordingPlanner.calls == []


def test_plan_refinement_invokes_the_agent_when_enabled(tiny_run, tmp_path, monkeypatch):
    """The agent gets a chance to improve the seed plan for kernel development."""
    _RecordingPlanner.calls = []
    _patch_planner(monkeypatch, _RecordingPlanner)
    result = loop_for(tiny_run, tmp_path, retain=False, refine_plan=True,
                      use_agent_planner=True).run()
    assert result.passed
    assert len(_RecordingPlanner.calls) == 1
    context = _RecordingPlanner.calls[0]["context"]
    assert context["model"] == tiny_run.spec.source
    assert context["budget_bytes"] > 0
    assert context["module_table"]
    assert result.context.graph.metadata["refined_by"] == "agent"


def test_refinement_runs_once_not_per_iteration(tiny_run, tmp_path, monkeypatch):
    _RecordingPlanner.calls = []
    _patch_planner(monkeypatch, _RecordingPlanner)
    loop = loop_for(tiny_run, tmp_path, retain=False, refine_plan=True,
                    use_agent_planner=True, max_iterations=3)
    assert loop.run().passed
    assert len(_RecordingPlanner.calls) == 1


def test_failed_refinement_keeps_the_seed_plan(tiny_run, tmp_path, monkeypatch):
    from model_partition.planner.agent import AgentOutcome

    class FailingPlanner:
        def __init__(self, **kwargs):
            pass

        def edit_plan(self, layout, graph, context, iteration, tag="plan"):
            return AgentOutcome(ok=False, error="rolled back"), graph

    _patch_planner(monkeypatch, FailingPlanner)
    result = loop_for(tiny_run, tmp_path, retain=False, refine_plan=True,
                      use_agent_planner=True).run()
    assert result.passed
    assert any("refinement failed" in note for note in result.context.notes)


def test_state_is_persisted_after_each_stage(tiny_run, tmp_path, monkeypatch):
    """An interrupted run must not have to redo tracing."""
    from model_partition.loop import stages as stages_module
    from model_partition.loop.state import LoopState

    loop = loop_for(tiny_run, tmp_path, retain=False)
    ctx = loop.build_context()

    original = stages_module.stage_verify_modules
    seen: dict[str, str] = {}

    def failing(inner_ctx):
        # By the time a later stage runs, the earlier ones are already on disk.
        saved = LoopState.load(inner_ctx.layout.state_file)
        seen["trace"] = saved.record("trace").status
        seen["ingest"] = saved.record("ingest").status
        raise RuntimeError("interrupted")

    monkeypatch.setattr(stages_module, "stage_verify_modules", failing)
    monkeypatch.setitem(stages_module.STAGE_FUNCTIONS, "verify_modules",
                        (failing, stages_module.hash_verify))
    try:
        result = loop_for(tiny_run, tmp_path, retain=False).run()
    finally:
        monkeypatch.setitem(stages_module.STAGE_FUNCTIONS, "verify_modules",
                            (original, stages_module.hash_verify))
    assert not result.passed
    assert seen == {"trace": "ok", "ingest": "ok"}
    del ctx


def test_an_interrupted_run_reuses_its_trace(tiny_run, tmp_path):
    """The stage after an interruption starts from the persisted trace."""
    from model_partition.loop import stages as stages_module
    from model_partition.loop.state import LoopState

    original = stages_module.STAGE_FUNCTIONS["verify_modules"]
    stages_module.STAGE_FUNCTIONS["verify_modules"] = (
        lambda ctx: (_ for _ in ()).throw(RuntimeError("interrupted")),
        original[1],
    )
    try:
        first = loop_for(tiny_run, tmp_path, retain=False).run()
        assert not first.passed
        saved = LoopState.load(first.context.layout.state_file)
        assert saved.record("trace").status == "ok"
    finally:
        stages_module.STAGE_FUNCTIONS["verify_modules"] = original

    messages: list[str] = []
    second = loop_for(tiny_run, tmp_path, retain=False)
    second.report = messages.append
    assert second.run().passed
    assert any("trace" in m and "cached" in m for m in messages)


class DecliningJudge:
    """Reproduces the model fine, but the judge is never satisfied."""

    def judge(self, prompt, continuation, sample_id=""):
        from model_partition.verify.judge import Verdict

        return Verdict(fluent=False, score=2, reason="unconvincing", sample_id=sample_id)


def test_a_declining_judge_does_not_fail_the_run(tiny_run, tmp_path):
    """The judge is advisory: the partition reproduced the model, so the run passed."""
    result = loop_for(tiny_run, tmp_path, retain=False,
                      judge=DecliningJudge()).run()
    assert result.passed
    assert result.judge_declined == tiny_run.sample_ids
    assert result.error == ""


def test_a_declining_judge_keeps_the_loop_iterating(tiny_run, tmp_path, monkeypatch):
    """With iterations left, a declined judgement continues rather than stopping."""
    _RecordingPlanner.calls = []
    _patch_planner(monkeypatch, _RecordingPlanner)
    messages: list[str] = []
    loop = loop_for(tiny_run, tmp_path, retain=False, judge=DecliningJudge(),
                    max_iterations=3, use_agent_planner=True)
    loop.report = messages.append
    result = loop.run()
    assert result.passed
    text = "\n".join(messages)
    assert "continuing to iteration 2" in text
    assert "iteration 2/3" in text


def test_a_declining_judge_stops_at_the_iteration_cap(tiny_run, tmp_path, monkeypatch):
    """max_iterations bounds the quality loop so it cannot spin forever."""
    _RecordingPlanner.calls = []
    _patch_planner(monkeypatch, _RecordingPlanner)
    messages: list[str] = []
    loop = loop_for(tiny_run, tmp_path, retain=False, judge=DecliningJudge(),
                    max_iterations=2, use_agent_planner=True)
    loop.report = messages.append
    result = loop.run()
    assert result.passed
    assert result.iterations == 2
    assert "iteration 3/" not in "\n".join(messages)


def test_a_declining_judge_with_no_agent_accepts_the_result(tiny_run, tmp_path):
    """Nothing can change without an agent, so the run concludes immediately."""
    messages: list[str] = []
    loop = loop_for(tiny_run, tmp_path, retain=False, judge=DecliningJudge(),
                    max_iterations=5, use_agent_planner=False)
    loop.report = messages.append
    result = loop.run()
    assert result.passed and result.iterations == 1
    text = "\n".join(messages)
    assert "agent disabled" in text
    assert "not satisfied" in text


def test_tokens_are_printed_even_when_the_judge_declines(tiny_run, tmp_path):
    """A human makes the final call, so the sampled tokens are always shown."""
    messages: list[str] = []
    loop = loop_for(tiny_run, tmp_path, retain=False, judge=DecliningJudge())
    loop.report = messages.append
    result = loop.run()
    text = "\n".join(messages)
    assert "sampled tokens (human verification)" in text
    assert "Review the sampled tokens above" in text
    assert result.tokens_text()


def test_a_wrong_module_implementation_fails_the_run(tiny_run, tmp_path):
    """Only the judge is advisory. The implementation is the loop's, and a wrong
    one is a genuine failure the agent is asked to fix."""
    first = loop_for(tiny_run, tmp_path, retain=False, judge=AcceptingJudge()).run()
    assert first.passed, first.error

    # Replace an implementation with one that returns the wrong thing.
    impls = sorted(first.context.layout.modules_dir.glob("*/inference.py"))
    assert impls
    target = next(p for p in impls if "decoder" in p.parent.name)
    target.write_text(
        "MODULE_IDS = " + repr([m.id for m in first.context.graph.partitioned_modules
                               if m.kind == "decoder_layers"]) + "\n"
        "def build_module(config, weights, device='cpu'):\n"
        "    def forward(*args, **kwargs):\n"
        "        return args[0] * 0.5\n"
        "    return forward\n"
    )

    result = loop_for(tiny_run, tmp_path, retain=False, judge=AcceptingJudge(),
                      force=True).run()
    assert not result.passed
    assert "verify_modules" in result.error


def test_an_edited_implementation_reruns_verification(tiny_run, tmp_path):
    """Editing inference.py must invalidate verification without re-tracing."""
    first = loop_for(tiny_run, tmp_path, retain=False, judge=AcceptingJudge()).run()
    assert first.passed

    impl = sorted(first.context.layout.modules_dir.glob("*/inference.py"))[0]
    impl.write_text(impl.read_text() + "\n# touched by the agent\n")

    messages: list[str] = []
    second = loop_for(tiny_run, tmp_path, retain=False, judge=AcceptingJudge(), force=True)
    second.report = messages.append
    assert second.run().passed
    assert any("trace" in m and "cached" in m for m in messages)
    assert not any("verify_modules" in m and "cached" in m for m in messages)


def test_extraction_preserves_an_edited_implementation(tiny_run, tmp_path):
    """The implementation is the loop's editable surface; extract must not clobber it."""
    first = loop_for(tiny_run, tmp_path, retain=False, judge=AcceptingJudge()).run()
    impl = sorted(first.context.layout.modules_dir.glob("*/inference.py"))[0]
    marker = "\n# agent edit that must survive\n"
    impl.write_text(impl.read_text() + marker)

    loop_for(tiny_run, tmp_path, retain=False, judge=AcceptingJudge(), force=True).run()
    assert marker in impl.read_text()


# -- the resolved config travels with the run --------------------------------


def test_the_resolved_config_is_persisted_for_implementations(tiny_run, tmp_path):
    """An implementation is built against config; an empty one is a silent trap."""
    import yaml

    from model_partition.runtime.module_runner import TraceBundle

    result = loop_for(tiny_run, tmp_path).run()
    layout = result.context.layout

    bundle = TraceBundle.load(layout.trace_dir)
    assert bundle.config.get("hidden_size")
    manifest = yaml.safe_load(layout.run_file.read_text())
    assert manifest["config"]["hidden_size"] == bundle.config["hidden_size"]


def test_the_weight_index_is_recorded_even_when_weights_are_cached(tiny_run, tmp_path):
    from model_partition.runtime.module_runner import TraceBundle

    result = loop_for(tiny_run, tmp_path).run()
    bundle = TraceBundle.load(result.context.layout.trace_dir)
    assert bundle.weight_params
    for module_id, names in bundle.weight_params.items():
        assert len(names) == len(bundle.weights[module_id])
        assert all("." in name for name in names)


# -- trace invalidation and cleanup ------------------------------------------


def test_a_changed_prompt_of_the_same_length_retraces(tiny_run, tmp_path):
    """Sample ids and token counts are not the input; the tokens are."""
    from model_partition.loop.stages import hash_trace
    from model_partition.loop.state import content_hash

    loop = loop_for(tiny_run, tmp_path)
    ctx = loop.build_context()
    from model_partition.loop.stages import stage_ingest

    stage_ingest(ctx)
    ctx.graph = tiny_run.graph
    before = content_hash(*hash_trace(ctx))

    ctx.samples[0].token_ids = list(reversed(ctx.samples[0].token_ids))
    assert content_hash(*hash_trace(ctx)) != before


def test_retracing_does_not_leave_the_previous_trace_behind(tiny_run, tmp_path):
    """Superseded blobs are outside the new manifest, so retention never sees them."""
    from model_partition.loop.stages import clear_trace

    result = loop_for(tiny_run, tmp_path, retain=False).run()
    trace_dir = result.context.layout.trace_dir
    orphan = trace_dir / "activations" / "stale" / "orphan.bin"
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_bytes(b"\x00" * 4096)

    freed = clear_trace(trace_dir)
    assert freed >= 4096
    assert not orphan.exists()
    assert list(trace_dir.iterdir()) == []


def test_a_second_run_retraces_into_a_clean_directory(tiny_run, tmp_path):
    loop_for(tiny_run, tmp_path, retain=False).run()
    layout = loop_for(tiny_run, tmp_path, retain=False).build_context().layout
    orphan = layout.trace_dir / "activations" / "gone" / "orphan.bin"
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_bytes(b"\x00" * 1024)

    # decode_steps is part of the trace hash, so changing it re-runs that stage.
    assert loop_for(tiny_run, tmp_path, retain=False, decode_steps=8).run().passed
    assert not orphan.exists()


# -- the partition instruction ----------------------------------------------


def test_a_spec_instruction_reaches_the_agent_and_turns_refinement_on(tiny_run, tmp_path,
                                                                     monkeypatch):
    """A prompt nothing acts on would be a prompt that does nothing."""
    _RecordingPlanner.calls = []
    _patch_planner(monkeypatch, _RecordingPlanner)
    tiny_run.spec.partition.prompt = "Split attention from the FFN."

    loop = loop_for(tiny_run, tmp_path, retain=False, use_agent_planner=True)
    assert loop.options.refine_plan is False
    assert loop.run().passed

    assert len(_RecordingPlanner.calls) == 1
    assert _RecordingPlanner.calls[0]["context"]["partition_prompt"] == \
        "Split attention from the FFN."


def test_an_explicit_instruction_wins_over_the_spec(tiny_run, tmp_path, monkeypatch):
    _RecordingPlanner.calls = []
    _patch_planner(monkeypatch, _RecordingPlanner)
    tiny_run.spec.partition.prompt = "from the spec"
    loop = loop_for(tiny_run, tmp_path, retain=False, use_agent_planner=True,
                    partition_prompt="from the command line")
    assert loop.run().passed
    assert _RecordingPlanner.calls[0]["context"]["partition_prompt"] == "from the command line"


def test_a_spec_asking_for_split_attention_ffn_gets_it(tiny_run, tmp_path):
    """The deterministic half of the instruction, so it needs no agent call."""
    tiny_run.spec.partition.split_attention_ffn = True
    result = loop_for(tiny_run, tmp_path, retain=False).run()
    assert result.passed, result.error

    kinds = {m.kind for m in result.context.graph.partitioned_modules}
    assert {"attention", "mlp"} <= kinds
    assert "decoder_layers" not in kinds
    for module in result.context.graph.partitioned_modules:
        if module.kind == "attention":
            assert all("attn" in s or "norm" in s for s in module.submodules)


# -- review before repair ---------------------------------------------------


def test_a_failure_is_reviewed_before_anything_is_changed(tiny_run, tmp_path, monkeypatch):
    """The agent that has to produce a fix is the wrong one to decide what broke."""
    _RecordingPlanner.calls = []
    _patch_planner(monkeypatch, _RecordingPlanner)

    layout = _broken_plan(tiny_run, tmp_path)
    loop_for(tiny_run, tmp_path, use_agent_planner=True, max_iterations=2).run()
    kinds = [call["kind"] for call in _RecordingPlanner.calls]
    ctx_layout = layout
    assert kinds[0] == "review"
    assert "repair" in kinds
    assert ctx_layout.review_file.is_file()


def test_the_review_is_handed_to_the_agent_that_fixes_things(tiny_run, tmp_path,
                                                             monkeypatch):
    _RecordingPlanner.calls = []
    _patch_planner(monkeypatch, _RecordingPlanner)

    _broken_plan(tiny_run, tmp_path)
    loop_for(tiny_run, tmp_path, use_agent_planner=True, max_iterations=2).run()
    repair = next(c for c in _RecordingPlanner.calls if c["kind"] == "repair")
    assert repair["context"]["review"] == "diagnosis"


def _broken_plan(run, tmp_path):
    """Plan the run, then point a module at a submodule the model does not have."""
    from model_partition.loop.stages import stage_ingest, stage_plan

    loop = loop_for(run, tmp_path)
    ctx = loop.build_context()
    stage_ingest(ctx)
    stage_plan(ctx)
    module = next(m for m in ctx.graph.partitioned_modules if m.kind == "decoder_layers")
    module.submodules = ["model.layers.nope"]
    ctx.graph.save(ctx.layout.graph_path)
    return ctx.layout


def test_the_reviewer_archives_each_review_and_requires_one(tiny_run, tmp_path):
    """The trail survives the next review, and a reviewer that wrote nothing fails."""
    from model_partition.planner.agent import AgentOutcome, AgentPlanner

    layout = loop_for(tiny_run, tmp_path).build_context().layout
    planner = AgentPlanner()
    context = {
        "model": "tiny", "num_layers": 4, "hidden_size": 8, "n_signatures": 1,
        "budget_h": "1 GiB", "gpu_name": "none", "run_root": str(layout.root),
        "failed_stage": "verify_modules", "failure_detail": "mismatch",
        "failing_modules": ["layers.0"], "n_modules": 4, "n_groups": 1,
        "module_table": "|m|", "partition_prompt": "", "review_path": str(layout.review_file),
    }

    def writes_a_review(**kwargs):
        layout.review_file.write_text("root cause\n")
        return AgentOutcome(ok=True)

    planner.invoke = writes_a_review
    assert planner.review(layout, context, 2).ok
    assert (layout.reports_dir / "reviews" / "iter-2.md").read_text() == "root cause\n"

    planner.invoke = lambda **kwargs: AgentOutcome(ok=True)
    outcome = planner.review(layout, context, 3)
    assert not outcome.ok and "wrote no review.md" in outcome.error


# -- prompt rendering --------------------------------------------------------
#
# Prompts render with StrictUndefined, so a key the context does not supply is an
# error at the moment the agent would have been invoked — after tracing.


@pytest.mark.parametrize("prompt", ["planner.md", "reviewer.md", "repair_module.md"])
def test_every_prompt_renders_from_the_real_context(tiny_run, tmp_path, prompt):
    from model_partition.loop.stages import (
        StageResult,
        dump_agent_context,
        stage_ingest,
        stage_plan,
    )
    from model_partition.planner.agent import render_prompt

    ctx = loop_for(tiny_run, tmp_path).build_context()
    stage_ingest(ctx)
    stage_plan(ctx)
    context = dump_agent_context(ctx, "verify_modules", StageResult(
        ok=False, detail="cosine 0.7", failing_modules=["layers.0"]))

    extra = {"graph": ctx.graph} if prompt == "planner.md" else {}
    text = render_prompt(prompt, **context, **extra)
    assert "layers.0" in text and tiny_run.spec.source in text


def test_the_planner_prompt_carries_the_instruction_and_the_review(tiny_run, tmp_path):
    from model_partition.loop.stages import StageResult, dump_agent_context, stage_ingest
    from model_partition.planner.agent import render_prompt

    ctx = loop_for(tiny_run, tmp_path, partition_prompt="split the FFN off").build_context()
    stage_ingest(ctx)
    ctx.layout.review_file.parent.mkdir(parents=True, exist_ok=True)
    ctx.layout.review_file.write_text("the gate is transposed\n")

    context = dump_agent_context(ctx, "verify_modules", StageResult(ok=False, detail="d"))
    text = render_prompt("planner.md", graph=ctx.graph, **context)
    assert "split the FFN off" in text
    assert "the gate is transposed" in text


def test_the_planner_prompt_states_the_granularity_trade_off():
    from model_partition.planner.agent import PROMPTS_DIR

    text = (PROMPTS_DIR / "planner.md").read_text()
    assert "Too coarse" in text and "Too fine" in text
    assert "fusion" in text or "fused" in text
    assert "Trainium" not in text


# -- retention runs after the loop, not inside it ----------------------------


def test_retention_is_not_an_iteration_stage():
    """Pruning the reference tensors mid-loop would make a later pass verify less."""
    from model_partition.loop.state import ITERATION_STAGES, STAGES

    assert "retain" in STAGES
    assert "retain" not in ITERATION_STAGES
    assert list(ITERATION_STAGES) == [s for s in STAGES if s != "retain"]


def test_a_declining_judge_does_not_prune_before_the_next_iteration(tiny_run, tmp_path,
                                                                   monkeypatch):
    """The loop keeps iterating on quality, so the trace has to survive."""
    _RecordingPlanner.calls = []
    _patch_planner(monkeypatch, _RecordingPlanner)
    from model_partition.loop import stages as stages_module

    real_retain, hasher = stages_module.STAGE_FUNCTIONS["retain"]
    pruned: list[str] = []

    def counting_retain(ctx):
        pruned.append(ctx.spec.name)
        return real_retain(ctx)

    monkeypatch.setitem(stages_module.STAGE_FUNCTIONS, "retain",
                        (counting_retain, hasher))

    result = loop_for(tiny_run, tmp_path, judge=DecliningJudge(), max_iterations=2,
                      use_agent_planner=True).run()
    assert result.passed
    # Once, at the end — not once per iteration.
    assert len(pruned) == 1


def test_retention_still_runs_when_the_run_passes(tiny_run, tmp_path):
    result = loop_for(tiny_run, tmp_path).run()
    assert result.passed
    assert result.state.record("retain").status == "ok"
    assert result.state.record("retain").metrics.get("kept_layers")


def test_a_cached_stage_whose_output_is_gone_runs_again(tiny_run, tmp_path):
    """Otherwise deleting an artifact yields a run that reports success with nothing."""
    result = loop_for(tiny_run, tmp_path, retain=False).run()
    assert result.passed
    layout = result.context.layout
    index = layout.modules_dir / "index.yaml"
    assert index.is_file()

    import shutil

    shutil.rmtree(layout.modules_dir)
    assert loop_for(tiny_run, tmp_path, retain=False).run().passed
    assert index.is_file(), "extract should have run again"


def test_a_model_too_large_for_the_machine_is_refused_before_allocating(tiny_run, tmp_path):
    """Discovering it by allocating means the kernel kills the run's terminal."""
    from model_partition.loop.stages import _memory_shortfall

    class Ctx:
        layout = tiny_run.layout
        result = tiny_run.result
        budget = None

        class inventory:
            dequant_bytes = 0

            @staticmethod
            def total_param_bytes(include_excluded=True):
                return 1 << 50  # a petabyte

    detail = _memory_shortfall(Ctx())
    assert "resident for one forward" in detail
    assert "larger machine" in detail

    Ctx.inventory.total_param_bytes = staticmethod(lambda include_excluded=True: 1024)
    assert _memory_shortfall(Ctx()) == ""


def test_a_forced_rerun_after_retention_retraces(tiny_deep_run, tmp_path):
    """Retention keeps a few layers, so the trace it leaves is not a cached stage.

    Reusing it verified the layers that survived and then could not assemble the model
    at all, which is how a re-run reported 251 parameters with no dumped value.
    """
    # One module per layer, so retention has something to drop.
    first = loop_for(tiny_deep_run, tmp_path, retain=True, one_layer_per_module=True)
    result = first.run()
    assert result.passed, result.error
    assert result.context.retention.removed_files, result.context.retention.summary()

    messages: list[str] = []
    second = loop_for(tiny_deep_run, tmp_path, retain=False, force=True,
                      one_layer_per_module=True)
    second.report = messages.append
    assert second.run().passed, messages
    assert not any("trace" in m and "cached" in m for m in messages), messages
