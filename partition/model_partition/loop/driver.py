# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The partition loop.

Specialized rather than reusing AutoHelix's harness, because the verification and
invalidation model differ: stages are content-hash cached so a failure re-runs
only what it must, and the expensive artifacts live outside any worktree so a
rejected iteration never forces a re-trace.

The agent's editable surface is the partition plan and the extracted module
implementations — where boundaries fall, and whether the arithmetic is right.
Everything that decides *whether* a module is correct stays with the harness.

The loop exits as soon as every module verifies and every sample's emulated
continuation is judged sound, printing the sampled tokens for human review. A
judge that stays unconvinced is advisory: it keeps the loop iterating while
iterations remain, and is reported rather than failing the run.
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
from model_partition.loop.state import (
    FAILED,
    ITERATION_STAGES,
    OK,
    LoopState,
    content_hash,
)
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
    #: Samples the partition reproduced but whose text the judge rejected.
    judge_declined: list[str] = field(default_factory=list)

    def tokens_text(self) -> str:
        path = self.context.layout.tokens_file
        return path.read_text() if path.is_file() else ""


Reporter = Callable[[str], None]


def _print(message: str) -> None:
    print(message, flush=True)


#: What a cached stage must have left behind. A stage whose inputs are unchanged but
#: whose output is gone has to run again — otherwise deleting an artifact yields a run
#: that reports success with nothing to show for it.
STAGE_OUTPUTS: dict[str, Callable[[RunLayout], bool]] = {
    "plan": lambda layout: layout.graph_path.is_file(),
    "extract": lambda layout: (layout.modules_dir / "index.yaml").is_file(),
    "trace": lambda layout: _trace_is_whole(layout),
    "verify_modules": lambda layout: (layout.reports_dir / "verify.json").is_file(),
    "verify_chain": lambda layout: (layout.reports_dir / "chain.json").is_file(),
    "emulate": lambda layout: (layout.reports_dir / "emulate.json").is_file(),
}


def _trace_is_whole(layout: RunLayout) -> bool:
    """Whether the trace on disk still covers every module.

    Retention keeps a few representative layers and drops the rest, so a pruned trace
    is a record of a finished run rather than a cached stage: reusing it verifies the
    layers that survived and cannot assemble the model at all. Re-running after
    retention re-traces.
    """
    if not (layout.trace_dir / "records.yaml").is_file():
        return False
    manifest = layout.trace_dir / "manifest.yaml"
    if not manifest.is_file():
        return False
    from model_partition import yamlio

    payload = yamlio.load_path(manifest) or {}
    return not (payload.get("metadata") or {}).get("retention")


def _release_accelerator() -> None:
    """Return cached blocks to the driver so the next stage sees a full card."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


@dataclass
class PartitionLoop:
    """Drives stages, invalidation, and agent repair."""

    spec: ModelSpec
    options: LoopOptions = field(default_factory=LoopOptions)
    report: Reporter = _print
    #: Overrides ``options.judge_kind``. Lets callers and tests supply their own.
    judge: Any = None
    #: Plan refinement happens once per run, not once per iteration.
    _refined: bool = field(default=False, repr=False)

    def build_context(self) -> LoopContext:
        self._adopt_spec_partition()
        layout = RunLayout.create(self.spec.slug, self.options.artifact_root).ensure()
        budget = resolve_budget(
            headroom=self.options.headroom,
            fallback_bytes=int(self.options.gpu_memory_gib * 1024 ** 3)
            if self.options.gpu_memory_gib else None,
        )
        ctx = LoopContext(spec=self.spec, layout=layout, options=self.options, budget=budget)
        ctx.judge = self._build_judge()
        ctx.build_model = self._model_builder(ctx)
        ctx.build_meta_model = self._meta_builder(ctx)
        return ctx

    def _adopt_spec_partition(self) -> None:
        """Let the spec supply the partition instruction the command line did not.

        A model's spec is the natural place to say how it wants to be cut up, so it
        travels with the model; an explicit command-line value still wins.
        """
        wanted = self.spec.partition
        if wanted.prompt and not self.options.partition_prompt:
            self.options.partition_prompt = wanted.prompt
        if wanted.split_attention_ffn:
            self.options.split_attention_ffn = True

    def _meta_builder(self, ctx: LoopContext) -> Callable[[], Any]:
        """Structure-only builder: no weights, so it works for any model size."""

        def build():
            from model_partition.ingest import ingest
            from model_partition.loaders import build_loader

            result = ctx.result or ingest(ctx.spec)
            ctx.result = result
            return build_loader(result).build_meta().model

        return build

    def _build_judge(self):
        from model_partition.verify.judge import build_judge

        if self.judge is not None:
            return self.judge
        if self.options.judge_kind == "claude":
            return build_judge("claude", model=self.options.judge_model)
        return build_judge(self.options.judge_kind)

    def _model_builder(self, ctx: LoopContext) -> Callable[..., Any]:
        """Lazily build the model, fetching weights the first time it is needed.

        With ``placed=True`` the model is spread across GPU and host so a
        checkpoint larger than the GPU still runs most of its compute there;
        otherwise it lands on one device for the caller to move.
        """
        fetched = {"done": False}

        def build(placed: bool = False):
            from model_partition.ingest import ensure_weights, ingest
            from model_partition.loaders import build_loader

            result = ctx.result or ingest(ctx.spec)
            ctx.result = result
            if not fetched["done"]:
                missing = result.missing_shards()
                if missing:
                    self.report(f"  fetching {len(missing)} weight shard(s)"
                                f" ({format_bytes(sum(result.index.shard_bytes[s] for s in missing))})")
                ensure_weights(result)
                fetched["done"] = True

            device_map, max_memory = ctx.placement() if placed else (None, None)
            loaded = build_loader(result, device_map=device_map, max_memory=max_memory).build()
            ctx.last_placement = loaded.placement
            return loaded.model

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

        if (short_circuit := self._already_complete(ctx, state)) is not None:
            return short_circuit

        last_failure: tuple[str, StageResult] | None = None
        for iteration in range(1, self.options.max_iterations + 1):
            state.iteration = iteration
            self.report(f"\n=== iteration {iteration}/{self.options.max_iterations} ===")
            failure = self._run_stages(ctx, state)
            state.save(ctx.layout.state_file)

            if failure is None:
                declined = state.record("emulate").metrics.get("judge_declined") or []
                if declined and iteration < self.options.max_iterations:
                    # The partition reproduced the model, so nothing failed — the
                    # judge is just not satisfied with the text. Keep iterating on
                    # quality rather than stopping or declaring a failure.
                    state.log_iteration({"result": "judge_declined", "samples": list(declined)})
                    state.save(ctx.layout.state_file)
                    self.report(f"\njudge declined {len(declined)} sample(s); "
                                f"continuing to iteration {iteration + 1}")
                    if not self._improve_quality(ctx, state, iteration):
                        return self._finish_passed(ctx, state, iteration, declined)
                    continue

                return self._finish_passed(ctx, state, iteration, declined)

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

    def _finish_passed(self, ctx: LoopContext, state: LoopState, iteration: int,
                       declined: list) -> LoopResult:
        """Conclude a run whose partition reproduced the model.

        A judge that stayed unconvinced is reported, not treated as a failure: the
        sampled tokens are printed either way so a human makes the final call.
        """
        state.passed = True
        self._retain(ctx, state)
        state.finished = True
        state.log_iteration({"result": "passed", "judge_declined": list(declined)})
        state.save(ctx.layout.state_file)
        summary = write_summary(ctx, state)
        self._print_tokens(ctx)
        if declined:
            self.report(f"\nEvery module verified and every boundary matched, but the judge "
                        f"was not satisfied with {len(declined)} sample(s): {', '.join(declined)}.")
            self.report("Review the sampled tokens above and decide.")
        else:
            self.report("\nAll stages passed.")
        self.report(f"Summary: {summary}")
        return LoopResult(passed=True, state=state, context=ctx,
                          iterations=iteration, summary_path=summary,
                          judge_declined=list(declined))

    def _retain(self, ctx: LoopContext, state: LoopState) -> None:
        """Prune the artifacts, now that no further iteration will read them."""
        from model_partition.loop.stages import STAGE_FUNCTIONS

        runner, hasher = STAGE_FUNCTIONS["retain"]
        started = time.time()
        try:
            result = runner(ctx)
            stage_hash = content_hash(*hasher(ctx))
        except Exception as exc:
            state.mark("retain", FAILED, detail=f"{type(exc).__name__}: {exc}")
            self.report(f"  {'retain':<15} FAIL  {exc}")
            return
        state.mark("retain", OK if result.ok else FAILED, stage_hash, result.detail,
                   result.metrics, time.time() - started)
        self.report(f"  {'retain':<15} {'ok' if result.ok else 'FAIL':<5} {result.detail}")

    def _improve_quality(self, ctx: LoopContext, state: LoopState, iteration: int) -> bool:
        """Give the agent a chance to improve output the judge rejected.

        The partition is numerically sound, so the plausible remaining cause is
        arithmetic too subtle for the boundary check. False when no further
        progress is possible, which ends the run as passed-with-reservation.
        """
        if not self.options.use_agent_planner:
            self.report("  agent disabled; accepting the result as is")
            return False

        from model_partition.planner.agent import AgentPlanner

        planner = AgentPlanner(model=self.options.agent_model,
                               timeout_seconds=self.options.agent_timeout_seconds)
        result = StageResult(
            ok=False,
            detail=("Every module verified and every boundary matched, but the judge "
                    "found the generated continuation unconvincing. Look for arithmetic "
                    "that is close enough to pass a per-module check yet degrades the "
                    "output over the whole stack."),
        )
        self._review(ctx, planner, "emulate", result, iteration)
        self.report("  asking the agent to improve the module implementations...")
        outcome = planner.repair_modules(ctx.layout, dump_agent_context(ctx, "emulate", result),
                                        iteration)
        if not outcome.ok:
            self.report(f"  no improvement made: {outcome.error}")
            return False
        cleared = state.invalidate_from("verify_modules")
        self.report(f"  implementations revised; invalidated: {', '.join(cleared) or 'nothing'}")
        return True

    def _already_complete(self, ctx: LoopContext, state: LoopState) -> LoopResult | None:
        """Short-circuit a run that already passed and had its artifacts pruned.

        Retention deletes the dumps for non-representative layers, so re-running
        verification against the pruned trace would fail for a reason that has
        nothing to do with correctness. ``--force`` re-runs anyway.
        """
        if self.options.force or not state.passed:
            return None
        retention = state.record("retain").metrics
        if not retention.get("kept_layers"):
            return None
        self.report(
            f"\nThis run already passed (iteration {state.iteration}) and its artifacts "
            f"were pruned to layers {retention['kept_layers']}.\n"
            f"Nothing to do. Pass --force to re-run, or --no-retain on a fresh run "
            f"to keep every layer's artifacts."
        )
        self._print_tokens_from_disk(ctx)
        return LoopResult(passed=True, state=state, context=ctx,
                          iterations=state.iteration, summary_path=ctx.layout.summary_file)

    def _print_tokens_from_disk(self, ctx: LoopContext) -> None:
        if ctx.layout.tokens_file.is_file():
            self.report(f"\nSampled tokens: {ctx.layout.tokens_file}")

    def _run_stages(self, ctx: LoopContext, state: LoopState) -> tuple[str, StageResult] | None:
        """Run every stage in order; return the first failure, or None.

        State is persisted after each stage, not just at the end: tracing is
        expensive enough that an interrupted run must not have to redo it.
        """
        for name in ITERATION_STAGES:
            runner, hasher = STAGE_FUNCTIONS[name]
            try:
                stage_hash = content_hash(*hasher(ctx))
            except Exception as exc:
                state.mark(name, FAILED, detail=f"could not hash inputs: {exc}")
                state.save(ctx.layout.state_file)
                return name, StageResult(ok=False, detail=f"could not hash inputs: {exc}")

            if state.is_fresh(name, stage_hash) and self._rehydrate(ctx, name):
                self.report(f"  {name:<15} cached")
                if name == "plan":
                    self._maybe_refine(ctx, state)
                continue

            started = time.time()
            # Hand back whatever the previous stage left reserved. The allocator
            # holds on to freed blocks, and a stage that traced a 16k-token forward
            # can leave tens of GB reserved — enough to make the next stage fall off
            # the GPU for no reason.
            _release_accelerator()
            try:
                result = runner(ctx)
            except Exception as exc:
                elapsed = time.time() - started
                detail = f"{type(exc).__name__}: {exc}"
                state.mark(name, FAILED, stage_hash, detail, duration_s=elapsed)
                state.save(ctx.layout.state_file)
                self.report(f"  {name:<15} FAIL  {detail}")
                return name, StageResult(ok=False, detail=detail, repairable=(name != "ingest"))

            elapsed = time.time() - started
            state.mark(name, OK if result.ok else FAILED, stage_hash,
                       result.detail, result.metrics, elapsed)
            state.save(ctx.layout.state_file)
            marker = "ok" if result.ok else "FAIL"
            self.report(f"  {name:<15} {marker:<5} {result.detail}  ({elapsed:.1f}s)")
            if not result.ok:
                return name, result
            if name == "plan":
                self._maybe_refine(ctx, state)
        return None

    def _maybe_refine(self, ctx: LoopContext, state: LoopState) -> None:
        """Let the agent improve the seed plan for kernel-development convenience.

        Runs once, right after planning. Even a model that fits whole on the GPU
        benefits from boundaries chosen for how a kernel gets written and tested,
        which is a judgement call the deterministic planner cannot make. A spec that
        states how it wants to be partitioned implies this step, since otherwise the
        instruction would never reach anything that can act on it.
        """
        wanted = self.options.refine_plan or bool(self.options.partition_prompt.strip())
        if self._refined or not wanted or not self.options.use_agent_planner:
            return
        if ctx.graph is None or state.refined:
            # Recorded in the run state, so resuming an interrupted run does not pay
            # for refinement again — or worse, refine differently and throw away a
            # trace that is already correct for the plan on disk.
            self._refined = True
            return
        self._refined = True

        from model_partition.planner.agent import AgentPlanner

        self.report("  asking the agent to refine the plan for kernel development...")
        planner = AgentPlanner(model=self.options.agent_model,
                               timeout_seconds=self.options.agent_timeout_seconds)
        before = ctx.graph.to_dict()
        outcome, revised = planner.edit_plan(
            ctx.layout, ctx.graph, dump_agent_context(ctx, "plan", StageResult(ok=True)),
            state.iteration,
        )
        if not outcome.ok:
            self.report(f"  refinement failed, keeping the seed plan: {outcome.error}")
            ctx.notes.append(f"plan refinement failed: {outcome.error}")
            return
        state.refined = True
        state.save(ctx.layout.state_file)
        if revised.to_dict() == before:
            self.report("  agent kept the seed plan unchanged")
            return
        ctx.graph = revised
        cleared = state.invalidate_from("extract")
        ctx.bundle = None
        self.report(f"  plan refined: {len(revised.partitioned_modules)} modules in "
                    f"{len(revised.signature_groups())} groups; "
                    f"invalidated: {', '.join(cleared) or 'nothing'}")

    def _rehydrate(self, ctx: LoopContext, name: str) -> bool:
        """Reload what a cached stage would have produced. False forces a re-run.

        Ingest is metadata-only and cheap, but later stages depend on the context
        it populates, so a fresh process must re-run it rather than skip it.
        """
        from model_partition.loop.stages import load_bundle
        from model_partition.planner.graph import PartitionGraph

        if name == "ingest":
            return ctx.result is not None and ctx.inventory is not None and bool(ctx.samples)
        present = STAGE_OUTPUTS.get(name)
        if present is not None and not present(ctx.layout):
            return False
        try:
            if name in ("plan", "extract") and ctx.graph is None:
                ctx.graph = PartitionGraph.load(ctx.layout.graph_path)
            if name in ("trace", "verify_modules", "verify_chain", "emulate", "retain"):
                load_bundle(ctx)
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

        planner = AgentPlanner(model=self.options.agent_model,
                               timeout_seconds=self.options.agent_timeout_seconds)
        self._review(ctx, planner, stage_name, result, iteration)

        if stage_name == "verify_modules" and result.repair_surface == "modules":
            return self._repair_modules(ctx, state, result, iteration)

        self.report("  asking the agent to revise the plan...")
        context = dump_agent_context(ctx, stage_name, result)
        outcome, revised = planner.edit_plan(ctx.layout, ctx.graph, context, iteration,
                                             tag="repair")
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

    def _review(self, ctx: LoopContext, planner: Any, stage_name: str,
                result: StageResult, iteration: int) -> bool:
        """Have a reviewer diagnose the failure before anything is changed.

        The fix agent then works from a written diagnosis rather than from the raw
        failure. A reviewer that cannot produce one is not fatal: the fix proceeds
        with the failure detail alone, which is what it had before.
        """
        self.report("  asking the reviewer to diagnose the failure...")
        context = dump_agent_context(ctx, stage_name, result)
        outcome = planner.review(ctx.layout, context, iteration)
        if not outcome.ok:
            self.report(f"  no review produced: {outcome.error}")
            return False
        self.report(f"  review written to {ctx.layout.review_file}")
        return True

    def _repair_modules(
        self, ctx: LoopContext, state: LoopState,
        result: StageResult, iteration: int,
    ) -> bool:
        """Numeric failure: the module's inference code is what needs fixing.

        The plan decides *where* boundaries fall; the implementation decides
        whether the arithmetic is right. A verification mismatch is the second
        kind of problem, so the agent is pointed at inference.py rather than at
        the partition.
        """
        from model_partition.planner.agent import AgentPlanner

        self.report("  asking the agent to fix the failing module implementation(s)...")
        planner = AgentPlanner(model=self.options.agent_model,
                               timeout_seconds=self.options.agent_timeout_seconds)
        context = dump_agent_context(ctx, "verify_modules", result)
        outcome = planner.repair_modules(ctx.layout, context, iteration)
        if not outcome.ok:
            self.report(f"  module repair failed: {outcome.error}")
            ctx.notes.append(f"iteration {iteration}: module repair failed: {outcome.error}")
            return False
        cleared = state.invalidate_from("verify_modules")
        self.report(f"  implementations revised; invalidated: {', '.join(cleared) or 'nothing'}")
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
