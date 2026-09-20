# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The partition loop.

Specialized rather than reusing AutoHelix's harness, because the verification and
invalidation model differ: stages are content-hash cached so a failure re-runs
only what it must, and the expensive artifacts live outside any worktree so a
rejected iteration never forces a re-trace. The agent's editable surface is the
partition plan alone.

The loop exits as soon as every module verifies and every sample's emulated
continuation is judged sound, printing the sampled tokens for human review.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from model_partition.hardware import format_bytes, resolve_budget
from model_partition.layout import RunLayout
from model_partition.loop.stages import (
    STAGE_FUNCTIONS,
    LoopContext,
    LoopOptions,
    StageResult,
    dump_agent_context,
    write_summary,
)
from model_partition.loop.state import FAILED, OK, STAGES, LoopState, content_hash
from model_partition.spec import ModelSpec


@dataclass
class LoopResult:
    """Outcome of a whole run."""

    passed: bool
    state: LoopState
    context: LoopContext
    iterations: int = 0
    summary_path: Path | None = None
    error: str = ""

    def tokens_text(self) -> str:
        path = self.context.layout.tokens_file
        return path.read_text() if path.is_file() else ""


Reporter = Callable[[str], None]


def _print(message: str) -> None:
    print(message, flush=True)


@dataclass
class PartitionLoop:
    """Drives stages, invalidation, and agent repair."""

    spec: ModelSpec
    options: LoopOptions = field(default_factory=LoopOptions)
    report: Reporter = _print

    def build_context(self) -> LoopContext:
        layout = RunLayout.create(self.spec.slug, self.options.artifact_root).ensure()
        budget = resolve_budget(
            headroom=self.options.headroom,
            fallback_bytes=int(self.options.gpu_memory_gib * 1024 ** 3)
            if self.options.gpu_memory_gib else None,
        )
        ctx = LoopContext(spec=self.spec, layout=layout, options=self.options, budget=budget)
        ctx.judge = self._build_judge()
        ctx.build_model = self._model_builder(ctx)
        return ctx

    def _build_judge(self):
        from model_partition.verify.judge import build_judge

        if self.options.judge_kind == "claude":
            return build_judge("claude", model=self.options.judge_model)
        return build_judge(self.options.judge_kind)

    def _model_builder(self, ctx: LoopContext) -> Callable[[], Any]:
        """Lazily build the model, reusing the resolved ingest result."""

        def build():
            from model_partition.ingest import ingest
            from model_partition.loaders import build_loader

            result = ctx.result or ingest(ctx.spec)
            ctx.result = result
            return build_loader(result).build().model

        return build

    # -- execution -------------------------------------------------------------

    def run(self) -> LoopResult:
        ctx = self.build_context()
        state = LoopState.load(ctx.layout.state_file)
        state.slug = self.spec.slug

        self.report(f"model     : {self.spec.source}")
        self.report(f"artifacts : {ctx.layout.root}")
        self.report(f"gpu       : {ctx.budget.gpu.name if ctx.budget.gpu else 'none'}"
                    f"  per-module budget {format_bytes(ctx.budget.usable_bytes)}")

        last_failure: tuple[str, StageResult] | None = None
        for iteration in range(1, self.options.max_iterations + 1):
            state.iteration = iteration
            self.report(f"\n=== iteration {iteration}/{self.options.max_iterations} ===")
            failure = self._run_stages(ctx, state)
            state.save(ctx.layout.state_file)

            if failure is None:
                state.passed = True
                state.finished = True
                state.log_iteration({"result": "passed"})
                state.save(ctx.layout.state_file)
                summary = write_summary(ctx, state)
                self._print_tokens(ctx)
                self.report(f"\nAll stages passed. Summary: {summary}")
                return LoopResult(passed=True, state=state, context=ctx,
                                  iterations=iteration, summary_path=summary)

            stage_name, result = failure
            last_failure = failure
            state.log_iteration({"result": "failed", "stage": stage_name, "detail": result.detail})
            self.report(f"\nstage '{stage_name}' failed: {result.detail}")

            if iteration == self.options.max_iterations:
                break
            if not self._repair(ctx, state, stage_name, result, iteration):
                break

        write_summary(ctx, state)
        state.finished = True
        state.save(ctx.layout.state_file)
        detail = last_failure[1].detail if last_failure else "unknown failure"
        stage = last_failure[0] if last_failure else "?"
        self.report(f"\nLoop finished without passing. Last failure in '{stage}': {detail}")
        return LoopResult(passed=False, state=state, context=ctx,
                          iterations=state.iteration, error=f"{stage}: {detail}",
                          summary_path=ctx.layout.summary_file)

    def _run_stages(self, ctx: LoopContext, state: LoopState) -> tuple[str, StageResult] | None:
        """Run every stage in order; return the first failure, or None."""
        for name in STAGES:
            runner, hasher = STAGE_FUNCTIONS[name]
            try:
                stage_hash = content_hash(*hasher(ctx))
            except Exception as exc:
                state.mark(name, FAILED, detail=f"could not hash inputs: {exc}")
                return name, StageResult(ok=False, detail=f"could not hash inputs: {exc}")

            if state.is_fresh(name, stage_hash) and self._rehydrate(ctx, name):
                self.report(f"  {name:<15} cached")
                continue

            started = time.time()
            try:
                result = runner(ctx)
            except Exception as exc:
                elapsed = time.time() - started
                detail = f"{type(exc).__name__}: {exc}"
                state.mark(name, FAILED, stage_hash, detail, duration_s=elapsed)
                self.report(f"  {name:<15} FAIL  {detail}")
                return name, StageResult(ok=False, detail=detail, repairable=(name != "ingest"))

            elapsed = time.time() - started
            state.mark(name, OK if result.ok else FAILED, stage_hash,
                       result.detail, result.metrics, elapsed)
            marker = "ok" if result.ok else "FAIL"
            self.report(f"  {name:<15} {marker:<5} {result.detail}  ({elapsed:.1f}s)")
            if not result.ok:
                return name, result
        return None

    def _rehydrate(self, ctx: LoopContext, name: str) -> bool:
        """Reload what a cached stage would have produced. False forces a re-run."""
        from model_partition.planner.graph import PartitionGraph
        from model_partition.runtime.module_runner import TraceBundle

        try:
            if name in ("plan", "extract") and ctx.graph is None:
                ctx.graph = PartitionGraph.load(ctx.layout.graph_path)
            if name in ("trace", "verify_modules", "emulate", "retain") and ctx.bundle is None:
                ctx.bundle = TraceBundle.load(ctx.layout.trace_dir)
        except Exception:
            return False
        return True

    # -- repair ----------------------------------------------------------------

    def _repair(
        self, ctx: LoopContext, state: LoopState,
        stage_name: str, result: StageResult, iteration: int,
    ) -> bool:
        """Let the agent revise the plan. False when no repair is possible."""
        if not result.repairable:
            self.report("  failure is not plan-repairable; stopping")
            return False
        if not self.options.use_agent_planner:
            self.report("  agent planner disabled; stopping")
            return False

        from model_partition.planner.agent import AgentPlanner

        self.report("  asking the agent to revise the plan...")
        planner = AgentPlanner(model=self.options.agent_model,
                               timeout_seconds=self.options.agent_timeout_seconds)
        context = dump_agent_context(ctx, stage_name, result)
        outcome, revised = planner.repair_plan(ctx.layout, ctx.graph, context, iteration)
        if not outcome.ok:
            self.report(f"  agent repair failed: {outcome.error}")
            ctx.notes.append(f"iteration {iteration}: agent repair failed: {outcome.error}")
            return False

        changed = revised.to_dict() != (ctx.graph.to_dict() if ctx.graph else None)
        if not changed:
            self.report("  agent left the plan unchanged; stopping")
            return False

        ctx.graph = revised
        cleared = state.invalidate_from("plan")
        ctx.bundle = None
        self.report(f"  plan revised; invalidated: {', '.join(cleared) or 'nothing'}")
        return True

    def _print_tokens(self, ctx: LoopContext) -> None:
        """Print the sampled continuations for human verification."""
        if not ctx.emulate_report:
            return
        self.report("\n--- sampled tokens (human verification) ---")
        for outcome in ctx.emulate_report.outcomes:
            score = outcome.verdict.score if outcome.verdict else 0
            self.report(f"\n[{outcome.sample_id}] judge {score}/5")
            self.report(f"  prompt : {outcome.prompt[:200].replace(chr(10), ' ')}")
            self.report(f"  tokens : {outcome.token_ids}")
            self.report(f"  text   : {outcome.text[:600]}")
        self.report(f"\nFull transcript: {ctx.layout.tokens_file}")


def run_partition_loop(spec: ModelSpec, options: LoopOptions | None = None,
                       report: Reporter = _print) -> LoopResult:
    """Convenience entry point."""
    return PartitionLoop(spec=spec, options=options or LoopOptions(), report=report).run()
