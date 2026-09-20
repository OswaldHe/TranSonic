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
    assert cleared == ["plan", "extract", "trace", "verify_modules", "emulate", "retain"]
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
    graph.metadata["nudge"] = "changed"
    graph.save(layout.graph_path)

    messages: list[str] = []
    second = loop_for(tiny_run, tmp_path, retain=False)
    second.report = messages.append
    assert second.run().passed
    # Changing the plan re-runs everything downstream of it, tracing included.
    assert not any("trace" in m and "cached" in m for m in messages)
    assert not any("emulate" in m and "cached" in m for m in messages)


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
    """Stands in for AgentPlanner, recording each refine_plan call."""

    calls: list[dict] = []

    def __init__(self, **kwargs):
        pass

    def refine_plan(self, layout, graph, context, iteration):
        from model_partition.planner.agent import AgentOutcome

        type(self).calls.append({"context": context, "iteration": iteration})
        graph.metadata["refined_by"] = "agent"
        return AgentOutcome(ok=True), graph


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

        def refine_plan(self, layout, graph, context, iteration):
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
