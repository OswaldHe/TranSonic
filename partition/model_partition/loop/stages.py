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
from model_partition.verify.chain import verify_chain
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
    #: Give attention and the FFN/MoE of every layer their own module.
    split_attention_ffn: bool = False
    #: Free-text instruction on how the model should be partitioned. Comes from
    #: the spec's ``partition.prompt`` or the command line, and is handed to the
    #: agent whenever it plans or repairs.
    partition_prompt: str = ""
    cache_weights: bool = True
    cache_dequant: bool = True
    slice_long: bool = False
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
            slice_long=self.slice_long,
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
            split_attention_ffn=self.split_attention_ffn,
        )


@dataclass
class StageResult:
    """Outcome of one stage."""

    ok: bool
    detail: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    #: Set when a failure should be handed to the agent for repair.
    repairable: bool = False
    #: Which editable surface the repair belongs to. A module that does not fit its
    #: device is a partition problem; one that runs but disagrees is an arithmetic
    #: problem, and pointing the agent at the wrong one wastes an iteration.
    repair_surface: str = "plan"
    failing_modules: list[str] = field(default_factory=list)
    #: The stage succeeded mechanically but the judge was not satisfied. Advisory:
    #: it keeps the loop iterating without marking the run failed.
    judge_declined: bool = False


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
    chain_report: Any = None
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
        # Extracted implementations are built against this, so it travels with the
        # run rather than being re-read from a snapshot that may be gone.
        "config": ctx.result.config,
        "partition_prompt": ctx.options.partition_prompt,
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
    # The plan does not exist yet, so the module count is projected from the one
    # knob that changes it structurally: splitting every layer in two.
    per_layer = 2 if ctx.options.split_attention_ffn else 1
    n_modules = max(len(inventory.layers), 1) * per_layer + 3
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

    unassigned = graph.metadata.get("layer_owned_params") or []
    if unassigned:
        ctx.notes.append(
            f"{len(unassigned)} tensor(s) are held by a layer module itself rather than "
            f"by a child, so no partition module owns them "
            f"(e.g. {', '.join(unassigned[:3])}); they show up as unclaimed when the "
            "model is assembled from dumps"
        )

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
        options.split_attention_ffn, options.partition_prompt,
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
        param_names=_param_names(ctx),
    )
    lines = sum(g.source_lines for g in groups)
    preserved = sum(1 for g in groups if g.preserved)
    detail = f"{len(groups)} implementation group(s), {lines} source lines"
    if preserved:
        detail += f"; {preserved} existing implementation(s) preserved"
    return StageResult(ok=True, detail=detail, metrics={
        "n_groups": len(groups), "source_lines": lines, "preserved": preserved,
    })


def _param_names(ctx: LoopContext) -> dict[str, list[str]]:
    """Original parameter names per module, for the generated implementations."""
    if ctx.bundle is None:
        return {}
    if ctx.bundle.weight_params:
        return dict(ctx.bundle.weight_params)
    by_name = {entry.name: entry for entry in ctx.bundle.store.entries}
    result: dict[str, list[str]] = {}
    for module_id, names in ctx.bundle.weights.items():
        originals = []
        for name in names:
            entry = by_name.get(name)
            originals.append((entry.extra or {}).get("param") or name if entry else name)
        result[module_id] = originals
    return result


def hash_extract(ctx: LoopContext) -> list[Any]:
    # Not the implementation's content: an edit must re-run verification, not
    # re-extract over the top of the edit.
    return [_graph_fingerprint(ctx.layout.graph_path), _trace_fingerprint(ctx)]


def _impl_fingerprint(ctx: LoopContext) -> str:
    """Content hash of every extracted implementation.

    The agent edits these, so a change must re-run verification without
    re-extracting over the top of the edit.
    """
    from model_partition.loop.state import content_hash

    paths = sorted(ctx.layout.modules_dir.glob("*/inference.py"))
    return content_hash([p.read_text() for p in paths]) if paths else ""


# -- stage: trace ------------------------------------------------------------


def stage_trace(ctx: LoopContext) -> StageResult:
    """Capture real per-module IO and weights for every sample."""
    assert ctx.graph is not None and ctx.build_model is not None
    if not ctx.samples:
        return StageResult(ok=False, detail=(
            "no sample inputs: set inputs.short (and optionally inputs.long) in the spec"
        ))
    # A re-trace replaces the previous one entirely. Leaving the old blobs behind
    # would grow the run by a whole trace per iteration, and retention only ever
    # sees what the new manifest lists.
    freed = clear_trace(ctx.layout.trace_dir)
    if freed:
        ctx.notes.append(f"cleared {format_bytes(freed)} of superseded trace data")
    store = TensorStore(ctx.layout.trace_dir)
    policy = ctx.options.dump_policy()

    # Which parameters each module owns is a question about structure, so it is
    # answered on the meta device — no weights, no second full load of a checkpoint
    # that may be larger than this machine.
    weight_params, unresolved = _index_weights(ctx, store)
    if unresolved:
        return _unresolved_result(ctx, unresolved)

    # Their values are another matter: they are read straight off the parameters,
    # and layer placement leaves host-assigned layers on the meta device between
    # forwards, so dump from a single-device copy and release it before placing the
    # model for the forward.
    weights: dict[str, list[str]] = {}
    if policy.cache_weights:
        host_model = ctx.build_model()
        weight_tracer = Tracer(host_model, ctx.graph, store, policy=policy)
        unresolved = weight_tracer.unresolved_submodules()
        if unresolved:
            return _unresolved_result(ctx, unresolved)
        weights = weight_tracer.dump_weights()
        weight_params = weight_params or weight_tracer.index_weights()
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

    bundle = TraceBundle(store=store, records=tracer.records, weights=weights,
                         weight_params=weight_params, metadata={
        "model": ctx.spec.source, "revision": ctx.result.revision if ctx.result else None,
        "device": device, "samples": [s.to_dict() for s in ctx.samples],
        # Implementations are handed this: hidden size, head counts, RoPE settings.
        # Without it every extracted module would be built against an empty config.
        "config": ctx.result.config if ctx.result else {},
        "cache_weights": policy.cache_weights,
    })
    bundle.save()
    attach_weight_source(ctx, bundle)
    ctx.bundle = bundle
    traced = {r.module_id for r in bundle.records}
    # Functional nodes own no submodule, so a hook-based trace cannot produce
    # records for them; they are unverified by construction, not untraced.
    missing = [m.id for m in ctx.graph.partitioned_modules
               if m.id not in traced and not m.functional]
    if missing:
        return StageResult(
            ok=False, repairable=True, failing_modules=missing,
            detail=f"{len(missing)} module(s) produced no trace records: {', '.join(missing[:8])}",
        )
    sliced = sum(1 for r in bundle.records if r.sliced)
    if sliced:
        ctx.notes.append(f"{sliced} record(s) were windowed by --slice-long and will "
                         "not be numerically verified")
    detail = (f"{len(bundle.records)} record(s) over {len(ctx.samples)} sample(s), "
              f"{format_bytes(sum(e.nbytes for e in store.entries))} dumped")
    if not policy.cache_weights:
        detail += "; weights indexed, read from the checkpoint on demand"
    return StageResult(ok=True, detail=detail, metrics={
        "n_records": len(bundle.records),
        "bytes_dumped": sum(e.nbytes for e in store.entries),
        "unique_blobs": len({e.sha256 for e in store.entries}),
        "n_sliced_records": sliced,
        "device": device,
    })


def _index_weights(ctx: LoopContext, store: TensorStore) -> tuple[dict[str, list[str]], list[str]]:
    """Enumerate each module's parameter names from the model's structure alone."""
    if ctx.build_meta_model is None:
        return {}, []
    try:
        structure = ctx.build_meta_model()
    except Exception as exc:
        ctx.notes.append(f"could not enumerate parameters on the meta device: {exc}")
        return {}, []
    probe = Tracer(structure, ctx.graph, store)
    unresolved = probe.unresolved_submodules()
    params = {} if unresolved else probe.index_weights()
    del structure, probe
    _collect()
    return params, unresolved


def clear_trace(trace_dir: Path) -> int:
    """Delete a previous trace's contents; returns the bytes reclaimed."""
    import shutil

    if not trace_dir.is_dir():
        return 0
    freed = sum(p.stat().st_size for p in trace_dir.rglob("*") if p.is_file())
    for path in trace_dir.iterdir():
        shutil.rmtree(path) if path.is_dir() else path.unlink()
    return freed


def load_bundle(ctx: LoopContext) -> TraceBundle:
    """The run's trace, from memory or disk, with a weight source attached."""
    bundle = ctx.bundle or TraceBundle.load(ctx.layout.trace_dir)
    attach_weight_source(ctx, bundle)
    ctx.bundle = bundle
    return bundle


def attach_weight_source(ctx: LoopContext, bundle: TraceBundle) -> None:
    """Give a bundle with no dumped weights a way to read them.

    Without this, ``cache_weights: false`` would reach verification with no
    parameter values at all. The checkpoint is the fallback — never the default,
    because a dump is what makes a module replayable away from this machine.
    """
    from model_partition.weights_source import CheckpointWeights

    if bundle.weights or not bundle.weight_params or ctx.result is None:
        return
    bundle.checkpoint = CheckpointWeights.from_ingest(ctx.result)


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
    # The token ids, not just their count: two different prompts can tokenize to
    # the same length, and reusing the old activations against a new prompt would
    # verify a computation nothing asked for.
    return [
        _graph_fingerprint(ctx.layout.graph_path),
        [[s.id, s.token_ids] for s in ctx.samples],
        options.slice_long, options.cache_weights, options.decode_steps,
    ]


# -- stage: verify_modules ---------------------------------------------------


def stage_verify_modules(ctx: LoopContext) -> StageResult:
    """Replay every module from its dumps and compare against the trace."""
    assert ctx.graph is not None and ctx.build_model is not None
    bundle = load_bundle(ctx)
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
        impl_dirs=_impl_dirs(ctx),
    )
    ctx.verify_report = report
    if target != host:
        ctx.notes.append(
            f"modules verified one at a time on {target} with the model resident on {host}"
        )
    (ctx.layout.reports_dir / "verify.json").write_text(json.dumps(report.to_dict(), indent=2))

    if not report.passed:
        failing = sorted({r.module_id for r in report.failures})
        oversized = sorted({r.module_id for r in report.oversized})
        detail = (f"{len(report.failures)}/{len(report.checked)} checks failed; "
                  f"modules: {', '.join(failing[:8])}")
        if oversized:
            detail += (f". {len(oversized)} did not fit {target} and need partitioning "
                       f"further: {', '.join(oversized[:8])}")
        return StageResult(
            ok=False, repairable=True, failing_modules=failing, detail=detail,
            # Capacity first: an arithmetic fix cannot make a module fit.
            repair_surface="plan" if oversized else "modules",
            metrics=_verify_metrics(report),
        )
    detail = (f"{len(report.checked)} checks passed, "
              f"worst cosine {report.worst_cosine():.6f}")
    if report.skipped:
        detail += f"; {len(report.skipped)} skipped"
    return StageResult(ok=True, detail=detail, metrics=_verify_metrics(report))


def _verify_metrics(report) -> dict[str, Any]:
    return {
        "n_checks": len(report.results),
        "n_failed": len(report.failures),
        "n_skipped": len(report.skipped),
        "n_oversized": len(report.oversized),
        "worst_cosine": report.worst_cosine(),
        "max_abs_err": report.max_abs_err(),
        "devices": sorted({r.device for r in report.results if r.device}),
    }


def _impl_dirs(ctx: LoopContext) -> dict[str, Path]:
    """Extracted implementation per module, so verification exercises the loop's code."""
    from model_partition.runtime.module_impl import find_impl_dirs

    return find_impl_dirs(ctx.layout.modules_dir)


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
    return [_graph_fingerprint(ctx.layout.graph_path), _trace_fingerprint(ctx),
            _impl_fingerprint(ctx)]


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




# -- stage: verify_chain -----------------------------------------------------


def stage_verify_chain(ctx: LoopContext) -> StageResult:
    """Chain the implementations together and measure how the error compounds.

    Per-module verification starts each module from its recorded input, so it cannot
    see one module's error reaching the next. This feeds each computed output forward
    and reports where the drift first leaves tolerance, and whether the logits still
    predict the same tokens.
    """
    assert ctx.graph is not None
    bundle = load_bundle(ctx)
    impl_dirs = _impl_dirs(ctx)
    if not impl_dirs:
        return StageResult(ok=True, detail="no extracted implementations to chain")

    suite = verify_chain(bundle, ctx.graph, impl_dirs, device=_accelerator(ctx))
    ctx.chain_report = suite
    (ctx.layout.reports_dir / "chain.json").write_text(json.dumps(suite.to_dict(), indent=2))

    metrics = {
        "n_samples": len(suite.reports),
        "n_failed": len(suite.failures),
        "worst_cosine": suite.worst_cosine(),
        "mean_top1_agreement": suite.mean_top1(),
        "diverging_modules": suite.diverging_modules(),
    }
    if not suite.passed:
        diverging = suite.diverging_modules()
        kept = sum(1 for r in suite.reports if r.tokens_agree)
        return StageResult(
            ok=False, repairable=True, repair_surface="modules",
            failing_modules=diverging,
            detail=(f"{len(suite.failures)}/{len(suite.reports)} chain(s) drifted past "
                    f"tolerance; {kept}/{len(suite.reports)} kept every token"
                    + (f"; first divergence at {', '.join(diverging[:4])}"
                       if diverging else "")),
            metrics=metrics,
        )
    return StageResult(ok=True, detail=(
        f"{len(suite.reports)} chain(s) held, worst cosine {suite.worst_cosine():.6f}, "
        f"top-1 agreement {suite.mean_top1():.4f}"
    ), metrics=metrics)


def hash_verify_chain(ctx: LoopContext) -> list[Any]:
    return [_graph_fingerprint(ctx.layout.graph_path), _trace_fingerprint(ctx),
            _impl_fingerprint(ctx)]


# -- stage: emulate ----------------------------------------------------------


def stage_emulate(ctx: LoopContext) -> StageResult:
    """Assemble the model from dumps, generate tokens, and judge them."""
    assert ctx.graph is not None and ctx.build_model is not None
    bundle = load_bundle(ctx)
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
        # Generation runs the loop's own implementations, not the model's modules:
        # these tokens are the deliverable's tokens or they are worth nothing.
        impl_dirs=_impl_dirs(ctx), run_dir=ctx.layout.root,
    )
    ctx.emulate_report = report
    (ctx.layout.reports_dir / "emulate.json").write_text(json.dumps(report.to_dict(), indent=2))
    _write_tokens(ctx, report)

    if not report.mechanically_passed:
        boundary_failures = sorted({
            c.name.removeprefix("boundary:")
            for o in report.outcomes for c in o.failed_boundaries()
        })
        broken = [o for o in report.outcomes if not o.mechanically_passed]
        return StageResult(
            ok=False, repairable=bool(boundary_failures), failing_modules=boundary_failures,
            detail=(f"{len(broken)}/{len(report.outcomes)} sample(s) did not reproduce "
                    f"the model; {len(boundary_failures)} boundary mismatch(es)"),
            metrics=_emulate_metrics(report),
        )

    declined = report.judge_declined
    detail = (f"{len(report.outcomes)} sample(s) reproduced the model, "
              f"mean judge score {report.mean_score():.1f}")
    if report.judge_errored:
        # Not a verdict, so not something to iterate against.
        names = ", ".join(o.sample_id for o in report.judge_errored[:3])
        detail += f"; judge gave no verdict for {len(report.judge_errored)} sample(s) ({names})"
        ctx.notes.append(
            f"the judge failed to assess {len(report.judge_errored)} sample(s): "
            + "; ".join(f"{o.sample_id}: {(o.verdict.error if o.verdict else 'no verdict')[:120]}"
                        for o in report.judge_errored)
        )
    if declined:
        # The judge is advisory: the partition is sound, so the stage succeeds and
        # the loop keeps iterating on quality rather than reporting a failure.
        detail += f"; judge declined {len(declined)} sample(s)"
        return StageResult(ok=True, detail=detail, judge_declined=True,
                           metrics=_emulate_metrics(report))
    return StageResult(ok=True, detail=detail, metrics=_emulate_metrics(report))


def _emulate_metrics(report) -> dict[str, Any]:
    return {
        "n_samples": len(report.outcomes),
        "n_failed": len(report.failures),
        "mechanically_passed": report.mechanically_passed,
        "judge_declined": [o.sample_id for o in report.judge_declined],
        "judge_errored": [o.sample_id for o in report.judge_errored],
        "mean_judge_score": report.mean_score(),
        "weights_filled": report.fill.applied if report.fill else 0,
        "implementations_installed": len(report.install.installed) if report.install else 0,
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
    bundle = load_bundle(ctx)
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
    "trace": (stage_trace, hash_trace),
    "extract": (stage_extract, hash_extract),
    "verify_modules": (stage_verify_modules, hash_verify),
    "verify_chain": (stage_verify_chain, hash_verify_chain),
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
    if ctx.chain_report:
        lines += ["## Accumulated error through the chained implementations", "",
                  "```", ctx.chain_report.render(), "```", ""]
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
    review_path = ctx.layout.review_file
    return {
        "model": ctx.spec.source,
        "partition_prompt": ctx.options.partition_prompt.strip(),
        "review": review_path.read_text().strip() if review_path.is_file() else "",
        "review_path": str(review_path),
        "run_root": str(ctx.layout.root),
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

