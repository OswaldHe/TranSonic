# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The partition loop's stages.

Each stage declares a hash of its inputs so the driver can skip it when nothing
relevant changed, and returns a result carrying metrics for the report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from model_partition.hardware import GIB, MemoryBudget, format_bytes, move_to_device
from model_partition.inputs import SampleInput, load_input_set, summarize
from model_partition.ingest import IngestResult, ingest
from model_partition.layout import RunLayout
from model_partition.planner import auto
from model_partition.planner.graph import PartitionGraph
from model_partition.retention import RetentionPolicy, apply_retention, plan_retention, write_retention_report
from model_partition.sizing import CostModel, ModelInventory
from model_partition.spec import ModelSpec
from model_partition.storage import DumpPolicy, TraceShape, estimate_storage, preflight
from model_partition.tensorstore import TensorStore
from model_partition.trace import Tracer
from model_partition.runtime.module_runner import TraceBundle
from model_partition.verify.emulate import EmulationInput, emulate
from model_partition.verify.judge import Judge, StubJudge
from model_partition.verify.modules import verify_modules


#: Fraction of GPU memory offered to a whole-model stage. Lower than 1.0 so
#: activations and workspaces still fit alongside the resident layers.
WHOLE_MODEL_GPU_FRACTION = 0.80


@dataclass
class LoopOptions:
    """Everything the loop needs beyond the spec."""

    artifact_root: str | Path | None = None
    device: str = "cuda"
    headroom: float = 0.60
    gpu_memory_gib: float | None = None
    seq_len: int | None = None
    max_iterations: int = 5
    max_layers_per_module: int | None = None
    one_layer_per_module: bool = False
    experts_per_group: int | None = None
    cache_weights: bool = True
    cache_dequant: bool = True
    full_dumps: bool = False
    decode_steps: int = 4
    max_new_tokens: int = 32
    temperature: float = 0.0
    seed: int = 0
    judge_kind: str = "claude"
    judge_model: str | None = None
    min_judge_score: int = 4
    use_agent_planner: bool = True
    #: Ask the agent to improve the seed plan for kernel-development convenience
    #: before tracing. Off by default: it costs an agent call even when the
    #: deterministic plan is already fine.
    refine_plan: bool = False
    agent_model: str | None = None
    agent_timeout_seconds: int = 1800
    retain: bool = True
    retention_layers: tuple[int, ...] = (1, 5)
    trace_device: str | None = None
    strict_storage: bool = True
    #: Re-run a completed run even though retention already pruned its artifacts.
    force: bool = False

    def dump_policy(self) -> DumpPolicy:
        return DumpPolicy(
            full_dumps=self.full_dumps,
            cache_weights=self.cache_weights,
            cache_dequant=self.cache_dequant,
            decode_steps=self.decode_steps,
        )

    def plan_options(self, seq_len: int) -> auto.PlanOptions:
        return auto.PlanOptions(
            seq_len=seq_len,
            max_layers_per_module=self.max_layers_per_module,
            one_layer_per_module=self.one_layer_per_module,
            experts_per_group=self.experts_per_group,
        )


@dataclass
class StageResult:
    """Outcome of one stage."""

    ok: bool
    detail: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    #: Set when a failure should be handed to the agent for repair.
    repairable: bool = False
    failing_modules: list[str] = field(default_factory=list)


@dataclass
class LoopContext:
    """Mutable state threaded through the stages."""

    spec: ModelSpec
    layout: RunLayout
    options: LoopOptions
    budget: MemoryBudget
    result: IngestResult | None = None
    inventory: ModelInventory | None = None
    graph: PartitionGraph | None = None
    bundle: TraceBundle | None = None
    samples: list[SampleInput] = field(default_factory=list)
    tokenizer: Any = None
    judge: Judge | None = None
    build_model: Callable[..., Any] | None = None
    build_meta_model: Callable[[], Any] | None = None
    verify_report: Any = None
    emulate_report: Any = None
    retention: Any = None
    notes: list[str] = field(default_factory=list)
    #: Placement the last built model actually got: "single" or "auto".
    last_placement: str = "single"

    @property
    def seq_len(self) -> int:
        if self.options.seq_len:
            return self.options.seq_len
        return max((s.n_tokens for s in self.samples), default=2048)

    def placement(self) -> tuple[str | None, dict | None]:
        """Layer placement for whole-model stages.

        Returns ``(device_map, max_memory)``. A checkpoint that fits the GPU needs
        no map — it goes there wholesale. One that does not gets ``"auto"``, which
        fills the GPU with as many layers as possible and leaves the remainder on
        the host, rather than abandoning the GPU entirely.
        """
        gpu = self.budget.gpu
        if gpu is None or not self.options.device.startswith("cuda"):
            return None, None
        needed = self.result.index.total_bytes if self.result else 0
        if needed and needed <= gpu.total_bytes * WHOLE_MODEL_GPU_FRACTION:
            return None, None
        host = _host(self)
        return "auto", {
            gpu.index: f"{int(gpu.total_bytes * WHOLE_MODEL_GPU_FRACTION / GIB)}GiB",
            "cpu": f"{max(int(host.ram_available_bytes * 0.8 / GIB), 1)}GiB",
        }


# -- stage: ingest -----------------------------------------------------------


def stage_ingest(ctx: LoopContext) -> StageResult:
    """Resolve the model, load the tokenizer and inputs, check storage."""
    from model_partition.tokenization import load_tokenizer

    ctx.result = ingest(ctx.spec)
    ctx.inventory = ModelInventory.build(
        ctx.result.index, ctx.result.config, include=_included_subtrees(ctx.spec),
    )
    ctx.tokenizer = load_tokenizer(
        ctx.result.root, trust_remote_code=ctx.spec.trust_remote_code,
        vocab_size=ctx.inventory.vocab_size or None,
    )
    short, long = ctx.spec.inputs.resolve(ctx.spec.base_dir())
    if short or long:
        ctx.samples = load_input_set(
            short, long, ctx.tokenizer,
            max_short=ctx.spec.inputs.max_short, max_long=ctx.spec.inputs.max_long,
        )
    else:
        # Planning only needs a sequence length, so a spec with no inputs can
        # still be inspected and planned; tracing reports the gap instead.
        ctx.samples = []
        ctx.notes.append("spec declares no input set; tracing will have nothing to run")

    estimate = _storage_estimate(ctx)
    warnings = preflight(estimate, strict=ctx.options.strict_storage)
    ctx.notes.extend(warnings)

    ctx.layout.ensure()
    ctx.layout.write_run({
        "spec": ctx.spec.to_dict(),
        "revision": ctx.result.revision,
        "loader": ctx.result.loader,
        "snapshot": str(ctx.result.root),
        "inputs": [s.to_dict() for s in ctx.samples],
        "storage_estimate": {
            "total_bytes": estimate.total_bytes,
            "lines": [{"label": l.label, "bytes": l.bytes_} for l in estimate.lines],
        },
    })
    (ctx.layout.reports_dir / "storage_estimate.txt").write_text(estimate.render() + "\n")

    return StageResult(ok=True, detail=(
        f"{ctx.result.loader} loader, {len(ctx.inventory.layers)} layers, "
        f"{format_bytes(ctx.result.index.total_bytes)} checkpoint; {summarize(ctx.samples)}; "
        f"projected artifacts {format_bytes(estimate.total_bytes)}"
    ), metrics={
        "checkpoint_bytes": ctx.result.index.total_bytes,
        "n_layers": len(ctx.inventory.layers),
        "n_samples": len(ctx.samples),
        "projected_bytes": estimate.total_bytes,
        "revision": ctx.result.revision,
    })


def hash_ingest(ctx: LoopContext) -> list[Any]:
    return [ctx.spec.to_dict(), ctx.spec.revision]


def _included_subtrees(spec: ModelSpec) -> tuple[str, ...]:
    return tuple(name for name, on in (
        ("vision", spec.scope.vision), ("mtp", spec.scope.mtp), ("engram", spec.scope.engram),
    ) if on)


def _storage_estimate(ctx: LoopContext):
    inventory = ctx.inventory
    assert inventory is not None
    hidden = inventory.hidden_size or 1
    dtype_bytes = 2
    n_modules = max(len(inventory.layers), 1) + 3
    return estimate_storage(
        checkpoint_bytes=ctx.result.index.total_bytes if ctx.result else 0,
        module_weight_bytes=inventory.total_param_bytes(include_excluded=False),
        dequant_bytes=inventory.dequant_bytes,
        trace=TraceShape(
            n_modules=n_modules,
            boundary_bytes_per_token=hidden * dtype_bytes,
            short_seq_lens=[s.n_tokens for s in ctx.samples if not s.is_long],
            long_seq_lens=[s.n_tokens for s in ctx.samples if s.is_long],
        ),
        policy=ctx.options.dump_policy(),
        host=_host(ctx),
    )


def _host(ctx: LoopContext):
    from model_partition.hardware import detect_host

    return detect_host(ctx.layout.root)


# -- stage: plan -------------------------------------------------------------


def stage_plan(ctx: LoopContext) -> StageResult:
    """Produce (or reuse) the partition graph, reconciled against the model."""
    assert ctx.inventory is not None and ctx.result is not None
    reused = False
    graph: PartitionGraph | None = None
    if ctx.layout.graph_path.is_file():
        try:
            graph = PartitionGraph.load(ctx.layout.graph_path)
            graph.validate(ctx.budget.usable_bytes)
            reused = True
        except Exception as exc:
            ctx.notes.append(f"existing plan rejected, regenerating: {exc}")
            graph = None

    if graph is None:
        cost = CostModel.from_config(ctx.result.config, dtype_bytes=2, seq_len=ctx.seq_len)
        graph = auto.plan(
            ctx.inventory, ctx.budget.usable_bytes, ctx.options.plan_options(ctx.seq_len),
            cost=cost, model_name=ctx.spec.source, revision=ctx.result.revision,
        )
        ctx.notes.extend(graph.validate(ctx.budget.usable_bytes))

    detail = f"{'reused' if reused else 'planned'} {len(graph.partitioned_modules)} modules"
    reconcile = _reconcile(ctx, graph)
    if reconcile is not None and (reconcile.changed or reconcile.unresolved):
        detail += f"; {reconcile.summary()}"
        if reconcile.unresolved:
            ctx.notes.append(f"plan submodules not found in the model: "
                             f"{', '.join(sorted(set(reconcile.unresolved))[:8])}")

    ctx.graph = graph
    graph.save(ctx.layout.graph_path)
    return StageResult(ok=True, detail=(
        f"{detail}; {len(graph.signature_groups())} implementation groups"
    ), metrics=_plan_metrics(graph))


def _reconcile(ctx: LoopContext, graph: PartitionGraph):
    """Remap plan submodule names onto the real module tree.

    Uses the meta-device structure so no weights are needed.
    """
    from model_partition.planner.reconcile import reconcile_submodules

    if ctx.build_meta_model is None:
        return None
    try:
        model = ctx.build_meta_model()
    except Exception as exc:
        ctx.notes.append(f"could not instantiate structure for reconciliation: {exc}")
        return None
    report = reconcile_submodules(graph, model)
    names = sorted(name for name, _ in model.named_modules() if name)
    ctx.layout.plan_dir.mkdir(parents=True, exist_ok=True)
    (ctx.layout.plan_dir / "valid_submodules.txt").write_text("\n".join(names) + "\n")
    return report


def hash_plan(ctx: LoopContext) -> list[Any]:
    options = ctx.options
    return [
        ctx.budget.usable_bytes, ctx.seq_len, options.max_layers_per_module,
        options.one_layer_per_module, options.experts_per_group,
        _graph_fingerprint(ctx.layout.graph_path),
    ]


def _graph_fingerprint(path: Path) -> str:
    from model_partition.loop.state import content_hash

    return content_hash(path.read_text()) if path.is_file() else ""


def _plan_metrics(graph: PartitionGraph) -> dict[str, Any]:
    return {
        "n_modules": len(graph.partitioned_modules),
        "n_groups": len(graph.signature_groups()),
        "max_module_bytes": max((m.resident_bytes for m in graph.partitioned_modules), default=0),
    }


# -- stage: extract ----------------------------------------------------------


def stage_extract(ctx: LoopContext) -> StageResult:
    """Write one implementation per signature group."""
    from model_partition.extract import extract

    assert ctx.graph is not None and ctx.build_model is not None
    model = ctx.build_model()
    groups = extract(
        ctx.graph, model, ctx.layout.modules_dir, run_root=ctx.layout.root,
        sample_ids=[s.id for s in ctx.samples],
        weight_tensors=ctx.bundle.weights if ctx.bundle else None,
    )
    lines = sum(g.source_lines for g in groups)
    return StageResult(ok=True, detail=(
        f"{len(groups)} implementation group(s), {lines} source lines"
    ), metrics={"n_groups": len(groups), "source_lines": lines})


def hash_extract(ctx: LoopContext) -> list[Any]:
    return [_graph_fingerprint(ctx.layout.graph_path)]


# -- stage: trace ------------------------------------------------------------


def stage_trace(ctx: LoopContext) -> StageResult:
    """Capture real per-module IO and weights for every sample."""
    assert ctx.graph is not None and ctx.build_model is not None
    if not ctx.samples:
        return StageResult(ok=False, detail=(
            "no sample inputs: set inputs.short (and optionally inputs.long) in the spec"
        ))
    store = TensorStore(ctx.layout.trace_dir)
    policy = ctx.options.dump_policy()

    # Weights are read straight off the parameters, and layer placement leaves
    # host-assigned layers on the meta device between forwards, so dump from a
    # single-device copy and release it before placing the model for the forward.
    weights: dict[str, list[str]] = {}
    if policy.cache_weights:
        host_model = ctx.build_model()
        unresolved = Tracer(host_model, ctx.graph, store).unresolved_submodules()
        if unresolved:
            return _unresolved_result(ctx, unresolved)
        weight_tracer = Tracer(host_model, ctx.graph, store, policy=policy)
        weights = weight_tracer.dump_weights()
        del host_model, weight_tracer
        _collect()

    model = ctx.build_model(placed=True)
    if ctx.last_placement == "auto":
        device = input_device(model)
    else:
        device = ctx.options.trace_device or _accelerator(ctx)
        model, device = move_to_device(model, device)

    tracer = Tracer(model, ctx.graph, store, policy=policy)
    unresolved = tracer.unresolved_submodules()
    if unresolved:
        return _unresolved_result(ctx, unresolved)
    for sample in ctx.samples:
        tracer.trace_sample(sample.id, sample.tensor(device))

    bundle = TraceBundle(store=store, records=tracer.records, weights=weights, metadata={
        "model": ctx.spec.source, "revision": ctx.result.revision if ctx.result else None,
        "device": device, "samples": [s.to_dict() for s in ctx.samples],
    })
    bundle.save()
    ctx.bundle = bundle
    traced = {r.module_id for r in bundle.records}
    missing = [m.id for m in ctx.graph.partitioned_modules if m.id not in traced]
    if missing:
        return StageResult(
            ok=False, repairable=True, failing_modules=missing,
            detail=f"{len(missing)} module(s) produced no trace records: {', '.join(missing[:8])}",
        )
    return StageResult(ok=True, detail=(
        f"{len(bundle.records)} record(s) over {len(ctx.samples)} sample(s), "
        f"{format_bytes(sum(e.nbytes for e in store.entries))} dumped"
    ), metrics={
        "n_records": len(bundle.records),
        "bytes_dumped": sum(e.nbytes for e in store.entries),
        "unique_blobs": len({e.sha256 for e in store.entries}),
        "device": device,
    })


def _unresolved_result(ctx: LoopContext, unresolved: list[str]) -> StageResult:
    """A plan naming submodules the model does not have is the agent's to fix."""
    assert ctx.graph is not None
    return StageResult(
        ok=False, repairable=True,
        detail=(f"{len(unresolved)} plan submodule(s) do not exist in the model: "
                f"{', '.join(unresolved[:8])}"),
        failing_modules=[m.id for m in ctx.graph.partitioned_modules
                         if set(m.submodules) & set(unresolved)],
    )


def _collect() -> None:
    """Release a freed model's memory before the next one is built."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def hash_trace(ctx: LoopContext) -> list[Any]:
    options = ctx.options
    return [
        _graph_fingerprint(ctx.layout.graph_path),
        [s.id for s in ctx.samples], [s.n_tokens for s in ctx.samples],
        options.full_dumps, options.cache_weights, options.decode_steps,
    ]


# -- stage: verify_modules ---------------------------------------------------


def stage_verify_modules(ctx: LoopContext) -> StageResult:
    """Replay every module from its dumps and compare against the trace."""
    assert ctx.graph is not None and ctx.build_model is not None
    bundle = ctx.bundle or TraceBundle.load(ctx.layout.trace_dir)
    ctx.bundle = bundle
    if bundle.metadata.get("retention"):
        kept = bundle.metadata["retention"].get("kept_layers")
        ctx.notes.append(
            f"trace was pruned to layers {kept}; verification covers the kept "
            "modules only. Re-trace for full coverage."
        )
    host = _host_device(ctx)
    target = _accelerator(ctx)
    report = verify_modules(
        ctx.build_model, bundle, ctx.graph,
        model_device=host, module_device=target,
    )
    ctx.verify_report = report
    if target != host:
        ctx.notes.append(
            f"modules verified one at a time on {target} with the model resident on {host}"
        )
    (ctx.layout.reports_dir / "verify.json").write_text(json.dumps(report.to_dict(), indent=2))

    if not report.passed:
        failing = sorted({r.module_id for r in report.failures})
        return StageResult(
            ok=False, repairable=True, failing_modules=failing,
            detail=(f"{len(report.failures)}/{len(report.results)} checks failed; "
                    f"modules: {', '.join(failing[:8])}"),
            metrics=_verify_metrics(report),
        )
    return StageResult(ok=True, detail=(
        f"{len(report.results)} checks passed, worst cosine {report.worst_cosine():.6f}"
    ), metrics=_verify_metrics(report))


def _verify_metrics(report) -> dict[str, Any]:
    return {
        "n_checks": len(report.results),
        "n_failed": len(report.failures),
        "worst_cosine": report.worst_cosine(),
        "max_abs_err": report.max_abs_err(),
        "devices": sorted({r.device for r in report.results if r.device}),
    }


def _host_device(ctx: LoopContext) -> str:
    """Where a single-device whole model can be held.

    The GPU when the checkpoint fits, the host otherwise — module verification
    then moves one module at a time onto the accelerator.
    """
    target = _accelerator(ctx)
    if target == "cpu" or not ctx.budget.gpu:
        return "cpu"
    needed = ctx.result.index.total_bytes if ctx.result else 0
    fits = needed and needed <= ctx.budget.gpu.total_bytes * WHOLE_MODEL_GPU_FRACTION
    return target if fits else "cpu"


def _accelerator(ctx: LoopContext) -> str:
    """The device a single module should run on, ignoring whole-model size.

    Per-module verification is the one stage that can always use the GPU: the
    plan sizes every module to fit, which is the guarantee being tested.
    """
    requested = ctx.options.device or "cpu"
    if not requested.startswith("cuda"):
        return requested
    try:
        import torch

        return requested if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def hash_verify(ctx: LoopContext) -> list[Any]:
    return [_graph_fingerprint(ctx.layout.graph_path), _trace_fingerprint(ctx)]


def _trace_fingerprint(ctx: LoopContext) -> str:
    """Content hash of the trace records.

    Content rather than mtime, so the fingerprint survives copying a run
    directory between machines and does not churn on a no-op rewrite.
    """
    from model_partition.loop.state import content_hash

    records = ctx.layout.trace_dir / "records.yaml"
    return content_hash(records.read_text()) if records.is_file() else ""


def input_device(model: Any) -> str:
    """Device a model's inputs should be placed on.

    With layer placement the first parameter's device is where the embedding
    lives; accelerate's hooks move activations onward from there.
    """
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return "cpu"




# -- stage: emulate ----------------------------------------------------------


def stage_emulate(ctx: LoopContext) -> StageResult:
    """Assemble the model from dumps, generate tokens, and judge them."""
    assert ctx.graph is not None and ctx.build_model is not None
    bundle = ctx.bundle or TraceBundle.load(ctx.layout.trace_dir)
    ctx.bundle = bundle
    judge = ctx.judge or StubJudge()
    eos = getattr(ctx.tokenizer, "eos_token_id", None)
    # Emulation overwrites parameters with dumped values, so the model is
    # assembled on a single writable device — layer placement would leave
    # host-assigned layers on meta. Once assembled it is dispatched across GPU and
    # host so generation still uses the accelerator.
    device = _host_device(ctx)
    _, place_max_memory = ctx.placement() if device == "cpu" else (None, None)

    report = emulate(
        ctx.build_model, bundle, ctx.graph,
        inputs=[EmulationInput(s.id, s.tensor(device), prompt=s.prompt) for s in ctx.samples],
        decode=(lambda ids: ctx.tokenizer.decode(ids)) if ctx.tokenizer else None,
        judge=judge, max_new_tokens=ctx.options.max_new_tokens,
        temperature=ctx.options.temperature, seed=ctx.options.seed,
        eos_token_id=eos, device=device, min_score=ctx.options.min_judge_score,
        strict_fill=True, place_max_memory=place_max_memory,
    )
    ctx.emulate_report = report
    (ctx.layout.reports_dir / "emulate.json").write_text(json.dumps(report.to_dict(), indent=2))
    _write_tokens(ctx, report)

    if not report.passed:
        failures = report.failures
        boundary_failures = sorted({
            c.name.removeprefix("boundary:")
            for o in failures for c in o.failed_boundaries()
        })
        return StageResult(
            ok=False, repairable=bool(boundary_failures), failing_modules=boundary_failures,
            detail=(f"{len(failures)}/{len(report.outcomes)} sample(s) failed; "
                    f"mean judge score {report.mean_score():.1f}"),
            metrics=_emulate_metrics(report),
        )
    return StageResult(ok=True, detail=(
        f"{len(report.outcomes)} sample(s) passed, mean judge score {report.mean_score():.1f}"
    ), metrics=_emulate_metrics(report))


def _emulate_metrics(report) -> dict[str, Any]:
    return {
        "n_samples": len(report.outcomes),
        "n_failed": len(report.failures),
        "mean_judge_score": report.mean_score(),
        "weights_filled": report.fill.applied if report.fill else 0,
    }


def _write_tokens(ctx: LoopContext, report) -> None:
    """Write the sampled continuations for human inspection."""
    lines: list[str] = []
    for outcome in report.outcomes:
        lines.append("=" * 72)
        lines.append(f"sample: {outcome.sample_id}")
        if outcome.verdict:
            lines.append(f"judge : {outcome.verdict.score}/5 "
                         f"fluent={outcome.verdict.fluent} — {outcome.verdict.reason}")
        lines.append(f"prompt: {outcome.prompt[:400]}")
        lines.append(f"tokens: {outcome.token_ids}")
        lines.append(f"text  : {outcome.text}")
        if outcome.error:
            lines.append(f"error : {outcome.error}")
        lines.append("")
    ctx.layout.tokens_file.write_text("\n".join(lines))


def hash_emulate(ctx: LoopContext) -> list[Any]:
    options = ctx.options
    return [
        _graph_fingerprint(ctx.layout.graph_path), _trace_fingerprint(ctx),
        options.max_new_tokens, options.temperature, options.seed,
        options.judge_kind, options.min_judge_score,
    ]


# -- stage: retain -----------------------------------------------------------


def stage_retain(ctx: LoopContext) -> StageResult:
    """Prune tensors to a representative layer set, keeping deduplicated code."""
    assert ctx.graph is not None and ctx.inventory is not None
    if not ctx.options.retain:
        return StageResult(ok=True, detail="retention disabled")
    bundle = ctx.bundle or TraceBundle.load(ctx.layout.trace_dir)
    policy = RetentionPolicy(preferred_layers=tuple(ctx.options.retention_layers))
    plan = plan_retention(ctx.graph, ctx.inventory, bundle, policy)
    result = apply_retention(bundle, plan)
    ctx.retention = result
    write_retention_report(ctx.layout.reports_dir / "retention.yaml", result)
    return StageResult(ok=True, detail=result.summary(), metrics={
        "kept_layers": list(plan.kept_layers),
        "bytes_freed": result.bytes_freed,
        "kept_modules": len(plan.kept_modules),
        "dropped_modules": len(plan.dropped_modules),
    })


def hash_retain(ctx: LoopContext) -> list[Any]:
    return [
        _graph_fingerprint(ctx.layout.graph_path), _trace_fingerprint(ctx),
        list(ctx.options.retention_layers), ctx.options.retain,
    ]


#: Stage name -> (runner, input-hash builder).
STAGE_FUNCTIONS: dict[str, tuple[Callable[[LoopContext], StageResult],
                                 Callable[[LoopContext], list[Any]]]] = {
    "ingest": (stage_ingest, hash_ingest),
    "plan": (stage_plan, hash_plan),
    "extract": (stage_extract, hash_extract),
    "trace": (stage_trace, hash_trace),
    "verify_modules": (stage_verify_modules, hash_verify),
    "emulate": (stage_emulate, hash_emulate),
    "retain": (stage_retain, hash_retain),
}


def write_summary(ctx: LoopContext, state: Any) -> Path:
    """Human-readable run summary."""
    lines = [
        f"# Partition run: {ctx.spec.source}", "",
        f"- revision: `{ctx.result.revision if ctx.result else 'local'}`",
        f"- loader: `{ctx.result.loader if ctx.result else '?'}`",
        f"- device budget: {format_bytes(ctx.budget.usable_bytes)}"
        f" on {ctx.budget.gpu.name if ctx.budget.gpu else 'no GPU'}",
        f"- artifacts: `{ctx.layout.root}`", "",
        "## Stages", "", "```", state.render(), "```", "",
    ]
    if ctx.graph:
        lines += [
            "## Plan", "",
            f"- {len(ctx.graph.partitioned_modules)} partitioned modules",
            f"- {len(ctx.graph.signature_groups())} deduplicated implementation groups", "",
            "| module | kind | layers | params |", "|---|---|---|---|",
        ]
        for module in ctx.graph.partitioned_modules:
            layers = ",".join(str(i) for i in module.layer_indices) or "-"
            lines.append(f"| `{module.id}` | {module.kind} | {layers} "
                         f"| {format_bytes(module.param_bytes)} |")
        lines.append("")
    if ctx.verify_report:
        lines += ["## Module verification", "", "```", ctx.verify_report.render(), "```", ""]
    if ctx.emulate_report:
        lines += ["## Emulated inference", "", "```", ctx.emulate_report.render(), "```", ""]
    if ctx.retention:
        lines += ["## Retention", "", "```", ctx.retention.plan.render(), "```", ""]
    if ctx.notes:
        lines += ["## Notes", ""] + [f"- {note}" for note in ctx.notes] + [""]

    ctx.layout.summary_file.parent.mkdir(parents=True, exist_ok=True)
    ctx.layout.summary_file.write_text("\n".join(lines))
    return ctx.layout.summary_file


def dump_agent_context(ctx: LoopContext, failed_stage: str, result: StageResult) -> dict[str, Any]:
    """Context handed to the agent for planning or repair."""
    graph = ctx.graph
    inventory = ctx.inventory
    module_table = ""
    if graph:
        rows = ["| module | kind | layers | params | resident |", "|---|---|---|---|---|"]
        for module in graph.partitioned_modules:
            layers = ",".join(str(i) for i in module.layer_indices) or "-"
            rows.append(f"| `{module.id}` | {module.kind} | {layers} "
                        f"| {format_bytes(module.param_bytes)} "
                        f"| {format_bytes(module.resident_bytes)} |")
        module_table = "\n".join(rows)
    layer_types = inventory.layer_types if inventory else []
    return {
        "model": ctx.spec.source,
        "num_layers": len(inventory.layers) if inventory else 0,
        "hidden_size": inventory.hidden_size if inventory else 0,
        "n_signatures": len(inventory.signature_groups()) if inventory else 0,
        "layer_types": layer_types,
        "layer_types_summary": _summarize_types(layer_types),
        "checkpoint_bytes_h": format_bytes(ctx.result.index.total_bytes) if ctx.result else "?",
        "gpu_name": ctx.budget.gpu.name if ctx.budget.gpu else "no GPU",
        "budget_h": format_bytes(ctx.budget.usable_bytes),
        "budget_bytes": ctx.budget.usable_bytes,
        "n_modules": len(graph.partitioned_modules) if graph else 0,
        "n_groups": len(graph.signature_groups()) if graph else 0,
        "module_table": module_table,
        "failed_stage": failed_stage,
        "failure_detail": result.detail,
        "failing_modules": result.failing_modules,
    }


def _summarize_types(layer_types: list[str]) -> str:
    if not layer_types:
        return ""
    counts: dict[str, int] = {}
    for name in layer_types:
        counts[name] = counts.get(name, 0) + 1
    return ", ".join(f"{name} x{count}" for name, count in counts.items())

