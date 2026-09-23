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
from model_partition.planner.graph import GraphError, PartitionGraph
from model_partition.retention import RetentionPolicy, apply_retention, plan_retention, write_retention_report
from model_partition.runtime.compat import is_hardware_limit
from model_partition.sizing import CostModel, ModelInventory
from model_partition.spec import ModelSpec
from model_partition.storage import (
    DumpPolicy,
    StoragePreflightError,
    TraceShape,
    estimate_storage,
    preflight,
)
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

#: Left free on the filesystem the Hub writes its cache to, when that is not the one the
#: artifacts go to. The same reserve the artifact-side preflight keeps.
CHECKPOINT_RESERVE_BYTES = 8 * 1024 ** 3


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
        )

    def plan_options(self, seq_len: int,
                     dequant_resident: bool = False) -> auto.PlanOptions:
        return auto.PlanOptions(
            seq_len=seq_len,
            max_layers_per_module=self.max_layers_per_module,
            one_layer_per_module=self.one_layer_per_module,
            experts_per_group=self.experts_per_group,
            split_attention_ffn=self.split_attention_ffn,
            dequant_resident=dequant_resident,
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
    #: Compatibility patches in effect, when the model's own code needed porting to
    #: this hardware. Recorded because the trace they produce is the reference.
    compat: Any = None
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
            long_token_budgets=tuple(ctx.spec.inputs.long_token_budgets),
        )
    else:
        # Planning only needs a sequence length, so a spec with no inputs can
        # still be inspected and planned; tracing reports the gap instead.
        ctx.samples = []
        ctx.notes.append("spec declares no input set; tracing will have nothing to run")

    estimate = _storage_estimate(ctx)
    warnings = preflight(estimate, strict=ctx.options.strict_storage)
    # And the download separately, when it lands on another filesystem than the artifacts.
    warnings += _checkpoint_preflight(ctx, ctx.options.strict_storage)
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
        "compat": ctx.compat.to_dict() if ctx.compat else None,
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


def _scope_mismatch(spec: ModelSpec, graph: PartitionGraph) -> str:
    """Why an existing plan no longer answers the spec's scope, or "".

    A plan is kept across iterations on purpose — the agent refines it, and that work
    must survive a re-run. Which means a change to the spec has to be able to reject
    it: a plan that leaves the n-gram memory unpartitioned cannot reproduce a model
    whose forward writes it into the residual stream, however good the rest of it is.
    """
    from model_partition.sizing import SUBTREE_MARKERS

    included = set(_included_subtrees(spec))
    for node in graph.modules:
        if node.kind not in SUBTREE_MARKERS:
            continue
        if node.kind in included and not node.partitioned:
            return (f"{node.kind} is in scope for this run and the plan leaves it "
                    f"unpartitioned")
        if node.kind not in included and node.partitioned:
            return f"{node.kind} is out of scope for this run and the plan partitions it"
    return ""


def _excluded_markers(spec: ModelSpec) -> tuple[str, ...]:
    """Name fragments of the subtrees this run does not partition.

    Their parameters are expected to be claimed by no module, and assembling from dumps
    has to tell that apart from a parameter the plan simply forgot.
    """
    from model_partition.sizing import SUBTREE_MARKERS

    included = set(_included_subtrees(spec))
    return tuple(marker for name, markers in SUBTREE_MARKERS.items()
                 if name not in included for marker in markers)


def _checkpoint_to_fetch(ctx: LoopContext) -> int:
    """Bytes of checkpoint this run still has to write to disk."""
    if not ctx.result:
        return 0
    missing = set(ctx.result.missing_shards())
    if not missing:
        return 0
    shard_bytes = ctx.result.index.shard_bytes
    return sum(size for shard, size in shard_bytes.items() if shard in missing) or \
        ctx.result.index.total_bytes


def _device_of(path: Any) -> int | None:
    """The filesystem backing a path, walking up to the nearest ancestor that exists."""
    probe = Path(path).expanduser()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return probe.stat().st_dev
    except OSError:
        return None


def _checkpoint_shares_artifact_volume(ctx: LoopContext) -> bool:
    """Whether the downloaded shards land on the same filesystem as the artifacts."""
    if not ctx.result:
        return True
    left = _device_of(ctx.result.root)
    right = _device_of(ctx.layout.root)
    return left is not None and left == right


def _checkpoint_preflight(ctx: LoopContext, strict: bool) -> list[str]:
    """Check the shards still to download against the filesystem they are written to.

    Remote shards go under the Hub's own cache root, which need not be the volume
    ``--artifact-root`` names. Measuring them against the artifact volume gets it wrong in
    both directions: a large artifact disk passes preflight while a small home disk fills
    partway through a several-hundred-gigabyte download, and a small artifact disk refuses
    a download that was never going to touch it.
    """
    import shutil

    to_fetch = _checkpoint_to_fetch(ctx)
    if not to_fetch or _checkpoint_shares_artifact_volume(ctx):
        return []
    probe = Path(ctx.result.root).expanduser()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        free = shutil.disk_usage(probe).free
    except OSError:
        return [f"Could not determine free space at {probe}; the checkpoint download was "
                f"not checked against it."]
    budget = max(free - CHECKPOINT_RESERVE_BYTES, 0)
    if to_fetch <= budget:
        return []
    message = (
        f"The {format_bytes(to_fetch)} of checkpoint shards still to fetch are written "
        f"under {probe}, which is a different filesystem from the artifact root and has "
        f"{format_bytes(budget)} usable ({format_bytes(free)} free minus "
        f"{format_bytes(CHECKPOINT_RESERVE_BYTES)} reserve). Point HF_HOME at a larger "
        f"volume, or fetch the checkpoint there first."
    )
    if strict:
        raise StoragePreflightError(message)
    return [message]


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
        # Only what still has to be fetched, and only when it lands here: shards already
        # in the snapshot are on this disk and counting them again asks for room for a
        # second copy of the model, while shards bound for another volume are checked
        # against that one by `_checkpoint_preflight`.
        checkpoint_bytes=(_checkpoint_to_fetch(ctx)
                          if _checkpoint_shares_artifact_volume(ctx) else 0),
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
            stale = _scope_mismatch(ctx.spec, graph)
            if stale:
                raise GraphError(stale)
            reused = True
        except Exception as exc:
            ctx.notes.append(f"existing plan rejected, regenerating: {exc}")
            graph = None

    if graph is None:
        cost = CostModel.from_config(ctx.result.config, dtype_bytes=2, seq_len=ctx.seq_len)
        graph = auto.plan(
            ctx.inventory, ctx.budget.usable_bytes,
            # Sized against what the loader will actually hold. Vendor code casts a
            # quantized checkpoint to the spec's dtype as it loads, so the module that
            # has to fit the card is the bf16 one, not the fp8 bytes on disk.
            ctx.options.plan_options(
                ctx.seq_len,
                dequant_resident=bool(ctx.inventory.dequant_bytes
                                      and ctx.result.loader == "repo_code"),
            ),
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
        # What the run is partitioning at all. Taking a subtree into scope changes
        # which modules exist, so a cached plan from before is a plan of a different
        # model — the n-gram memory being the case that showed it.
        ctx.spec.scope.excluded(),
        _graph_fingerprint(ctx.layout.graph_path),
    ]


def _graph_fingerprint(path: Path, edges: bool = True) -> str:
    """What the later stages depend on in the plan, canonically.

    ``edges`` covers which tensor each module consumes and produces. Tracing does not
    depend on that — it hooks submodules and records what they were called with, whatever
    the plan believes flows where — so correcting an edge should not cost a re-trace of
    hundreds of gigabytes. Chaining and emulation do depend on it.

    The partition itself: which modules there are, what each is, which submodules it
    owns and how they relate. Hashing the file's bytes instead made a re-plan that
    produced the *same* partition look like a new one — a reconcile writing the same
    modules in a different order, a rewritten rationale — and the re-trace that
    follows costs hours and hundreds of gigabytes on a large model.
    """
    from model_partition.loop.state import content_hash

    if not path.is_file():
        return ""
    try:
        graph = PartitionGraph.load(path)
    except Exception:
        # Unreadable, so it is about to fail anyway; hash the bytes rather than
        # claiming two broken plans are the same one.
        return content_hash(path.read_text())
    return content_hash(sorted(
        [m.id, m.kind, m.composition, list(m.submodules), list(m.layer_indices)]
        + ([list(m.inputs), list(m.outputs)] if edges else [])
        for m in graph.partitioned_modules
    ))


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
    # Structure, not values: what extraction reads off the model is each module's
    # class and the config it was constructed with. Building it for real would
    # allocate the weights, and one of DeepSeek V4.1's n-gram tables is 91.5 GiB on
    # its own — more than the card, for a stage that never looks at a number.
    model = (ctx.build_meta_model or ctx.build_model)()
    groups = extract(
        ctx.graph, model, ctx.layout.modules_dir, run_root=ctx.layout.root,
        sample_ids=[s.id for s in ctx.samples],
        weight_tensors=ctx.bundle.weights if ctx.bundle else None,
        param_names=_param_names(ctx),
        config=_extraction_config(ctx),
        bundle=ctx.bundle,
        # So recorded provenance names a place inside the checkpoint rather than a
        # directory on this machine, which is what gets published.
        origin_root=ctx.result.root if ctx.result else None,
        # And so a run that read its weights from the checkpoint ships the means to fetch
        # them again, pinned to the revision these feature maps came from.
        checkpoint=_checkpoint_identity(ctx),
    )
    lines = sum(g.source_lines for g in groups)
    preserved = sum(1 for g in groups if g.preserved)
    detail = f"{len(groups)} implementation group(s), {lines} source lines"
    if preserved:
        detail += f"; {preserved} existing implementation(s) preserved"
    return StageResult(ok=True, detail=detail, metrics={
        "n_groups": len(groups), "source_lines": lines, "preserved": preserved,
    })


def _checkpoint_identity(ctx: LoopContext) -> dict[str, str] | None:
    """The repo and pinned revision a fetch script would download from.

    ``None`` for a local checkpoint, which cannot be fetched by name — and a script that
    named a repo nobody can reach would be worse than its absence.
    """
    if ctx.result is None:
        return None
    # The spec's own reading of its source, so `org/model` and `hf:org/model` are the
    # same checkpoint here as they are to ingestion.
    repo_id = ctx.spec.repo_id or ""
    if "/" not in repo_id:
        return None
    size = format_bytes(ctx.inventory.total_param_bytes()) if ctx.inventory else "large"
    return {"repo_id": repo_id, "revision": str(ctx.result.revision or "main"),
            "checkpoint_size": size}


def _extraction_config(ctx: LoopContext) -> dict[str, Any]:
    """The model's config, plus the dtype its activations flow in.

    A module has to be built the way the model was built, and the weights do not say
    how: DeepSeek's MoE holds two float32 biases and one bfloat16 gate among 2 300 fp4
    and fp8 tensors, so nothing about them points at the bfloat16 its kernels compute in.
    Recorded here so a module directory carries it and can be built without this run.
    """
    from model_partition.extract import COMPUTE_DTYPE_KEY
    from model_partition.loaders.repo_code_loader import compute_dtype_for

    config = dict(ctx.result.config) if ctx.result else {}
    config[COMPUTE_DTYPE_KEY] = compute_dtype_for(ctx.spec.dtype)
    return config


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
    # re-extract over the top of the edit. The templates are an input, though: the
    # harness's files are rendered from them afresh every extraction, and while it stays
    # cached a fixed `verify.py` would never reach the artifact.
    return [_graph_fingerprint(ctx.layout.graph_path), _trace_fingerprint(ctx),
            _templates_fingerprint()]


def _templates_fingerprint() -> str:
    """Content hash of the templates extraction renders module directories from."""
    from model_partition.loop.state import content_hash

    root = Path(__file__).resolve().parent.parent / "templates"
    return content_hash([[p.name, p.read_text()] for p in sorted(root.glob("*.tmpl"))])


def _impl_fingerprint(ctx: LoopContext) -> str:
    """Content hash of every extracted implementation.

    The agent edits these, so a change must re-run verification without
    re-extracting over the top of the edit.

    ``source.py`` as well as ``inference.py``, and it matters more: the module's classes
    live in ``source.py`` and ``inference.py`` is the template that builds them. Hashing
    only the template let an edited class keep the pass recorded for the one before it.
    """
    from model_partition.loop.state import content_hash

    root = ctx.layout.modules_dir
    paths = sorted([*root.glob("*/source.py"), *root.glob("*/inference.py")])
    return (content_hash([[p.relative_to(root).as_posix(), p.read_text()] for p in paths])
            if paths else "")


# -- stage: trace ------------------------------------------------------------


def _can_stream(ctx: LoopContext) -> bool:
    """Whether this model's weights can be read per module instead of held.

    Needs a checkpoint of safetensors shards to read from and a loader that knows how:
    the vendor path builds on meta and materializes each module for its own forward.

    Asked of the tensor index rather than of the filesystem. A fresh remote ingest reads
    metadata and deliberately downloads no shards, so a directory listing answers "cannot
    stream" for exactly the models large enough to need it — and the streamed build only
    ever runs after ``build_model`` has fetched what it reads.
    """
    if not (ctx.result and ctx.result.loader == "repo_code"):
        return False
    return any(str(shard).endswith(".safetensors")
               for shard in ctx.result.index.shard_bytes)


def memory_shortfall(ctx: LoopContext) -> str:
    """Why this machine cannot hold the model for a forward, or "" if it can.

    Unlike every later stage, tracing needs the whole model to *run*: it records what
    each module saw during a real forward, and partitioning cannot make a forward
    smaller. Whether it must be *resident* is a different question — see
    :func:`_can_stream` — but when it must be and does not fit, saying so beats
    allocating until the kernel kills the process and takes the terminal with it.
    """
    if ctx.inventory is None:
        return ""
    from model_partition.hardware import detect_host

    # What the loader will hold: a quantized checkpoint read through vendor code is
    # cast to the spec's dtype, so the bf16 mirror is the resident size, not the fp8
    # bytes on disk.
    on_disk = ctx.inventory.total_param_bytes(include_excluded=False)
    needed, why = on_disk, ""
    if ctx.inventory.dequant_bytes and ctx.result and ctx.result.loader == "repo_code":
        # Vendor code casts the checkpoint to the spec's dtype as it loads, so the
        # bf16 mirror is what has to fit, not the fp8 bytes on disk.
        needed = max(needed, ctx.inventory.dequant_bytes)
        if needed > on_disk:
            why = (f" ({format_bytes(on_disk)} of quantized weights, cast to "
                   f"{ctx.spec.dtype} as the vendor code loads them)")
    host = detect_host(ctx.layout.root)
    gpu_bytes = ctx.budget.gpu.total_bytes if ctx.budget and ctx.budget.gpu else 0
    # Only memory the loader can actually build into. The repo-code loader's non-streamed
    # path merges every shard into a host `state_dict` and constructs the model there; it
    # does not honour a `device_map`, so the card's memory is not capacity for it and a
    # checkpoint that fits RAM-plus-GPU is still killed while the host copy is made. A run
    # asked for `--device cpu` has no card in play either way.
    builds_on_host = bool(ctx.result and ctx.result.loader == "repo_code")
    usable_gpu = 0 if builds_on_host or not ctx.options.device.startswith("cuda") \
        else gpu_bytes
    capacity = host.ram_available_bytes + usable_gpu
    if not capacity or needed <= capacity:
        return ""
    where = f"{format_bytes(host.ram_available_bytes)} host RAM available"
    if usable_gpu:
        where += f" + {format_bytes(usable_gpu)} on {ctx.budget.gpu.name}"
    elif gpu_bytes:
        where += (f"; the {format_bytes(gpu_bytes)} on {ctx.budget.gpu.name} is not "
                  f"capacity for the {ctx.result.loader if ctx.result else '?'} loader, "
                  f"which constructs on the host")
    return (
        f"the model needs about {format_bytes(needed)} resident for one forward{why} "
        f"and this machine has {format_bytes(capacity)} ({where}). Tracing needs the "
        "whole model, which partitioning cannot change: run it on a larger machine, or "
        "narrow the scope in the spec."
    )


def stage_trace(ctx: LoopContext) -> StageResult:
    """Capture real per-module IO and weights for every sample."""
    assert ctx.graph is not None and ctx.build_model is not None
    if not ctx.samples:
        return StageResult(ok=False, detail=(
            "no sample inputs: set inputs.short (and optionally inputs.long) in the spec"
        ))
    shortfall = memory_shortfall(ctx)
    if shortfall and not _can_stream(ctx):
        return StageResult(ok=False, detail=shortfall)
    if shortfall:
        ctx.notes.append(
            f"{shortfall} Streaming the weights instead: one module's at a time, read "
            "from the shards for its forward and released after."
        )
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
        release_memory()

    model = ctx.build_model(placed=True)
    if ctx.last_placement == "auto":
        device = input_device(model)
    elif ctx.last_placement == "streamed":
        # Already placed, and deliberately incomplete: its weights are placeholders until
        # each module reads its own. Moving it would try to copy out of them.
        device = ctx.options.trace_device or _accelerator(ctx)
    else:
        device = ctx.options.trace_device or _accelerator(ctx)
        model, device = move_to_device(model, device)

    tracer = Tracer(model, ctx.graph, store, policy=policy,
                    state_objects=ctx.spec.trace.state)
    unresolved = tracer.unresolved_submodules()
    if unresolved:
        return _unresolved_result(ctx, unresolved)
    try:
        for sample in ctx.samples:
            input_ids = sample.tensor(device)
            main = tracer.trace_sample(sample.id, input_ids)
            # A module the main forward never calls has no reference and cannot be
            # verified. Where the spec says which entry point does call it — a
            # speculative-decoding stack is the case — drive that too, on this sample,
            # while the state its prefill left behind is still current. Asked per
            # sample, from that sample's own forward, so every sample covers the same
            # modules rather than only the first.
            unreached = _unreached_submodules(ctx.graph, {r.module_id for r in main})
            if unreached and ctx.spec.trace.extra_passes:
                extra = tracer.trace_extra_passes(
                    sample.id, input_ids, ctx.spec.trace, only=unreached,
                )
                ctx.notes.append(
                    f"{len(ctx.spec.trace.extra_passes)} extra pass(es) reached "
                    f"{len({r.module_id for r in extra})} module(s) that "
                    f"{type(model).__name__}.forward does not call"
                )
    except Exception as exc:
        if not is_hardware_limit(exc):
            raise
        # The model's own code will not run on this card. That is a porting job, not a
        # partitioning one: the agent writes a replacement under compat/ for the parts
        # this hardware refuses, and the trace is taken again.
        where = ctx.budget.gpu.describe() if ctx.budget.gpu else "no GPU"
        return StageResult(
            ok=False, repairable=True, repair_surface="compat",
            detail=(f"the model's own code does not run on this GPU: {type(exc).__name__}: "
                    f"{exc}. This card is {where}."),
        )

    # The buffers a class computes for itself, taken from the copy that just ran: the
    # host copy above holds its own, and they are not always the same — a rotary
    # inv_freq built in bf16 there and float32 here.
    #
    # Dumped whatever `cache_weights` says. That option is about not writing a second
    # copy of the checkpoint; these are not in the checkpoint, so leaving them out
    # leaves nothing to read them from. Identical ones across layers — a window KV
    # cache is zeros in every layer — land on one blob, the store deduplicating by
    # digest.
    for module_id, names in tracer.dump_derived().items():
        weights.setdefault(module_id, []).extend(names)

    records = tracer.records
    # The traced model and what its last forward returned are the largest things this
    # process holds; the stage after this builds its own. Released here rather than at
    # the end of the function so the allocator gives the pages back before then.
    del model, tracer
    release_memory()

    bundle = TraceBundle(store=store, records=records, weights=weights,
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
        why = (". Nothing this run drives calls them: name the entry point that does "
               "in the spec's trace.extra_passes, or take them out of scope"
               if not ctx.spec.trace.extra_passes else "")
        return StageResult(
            ok=False, repairable=True, failing_modules=missing,
            detail=(f"{len(missing)} module(s) produced no trace records: "
                    f"{', '.join(missing[:8])}{why}"),
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


def _unreached_submodules(graph: Any, reached: set[str]) -> set[str]:
    """Submodules of the partitioned modules this pass did not call.

    Functional nodes own no submodule and are left out: there is nothing to hook.
    """
    return {submodule for module in graph.partitioned_modules
            if module.id not in reached and not module.functional
            for submodule in module.submodules}


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
    release_memory()
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

    Decided by the policy the trace recorded, not by whether anything was dumped: such
    a run still dumps its derived buffers, so a non-empty dump does not mean the
    checkpoint is unnecessary.
    """
    from model_partition.weights_index import CheckpointWeights

    cached = bundle.metadata.get("cache_weights")
    if cached is None:
        cached = bool(bundle.weights)
    if cached or not bundle.weight_params or ctx.result is None:
        return
    # With the spec's rename rules, because the names being asked for are the model's and
    # the names on disk are the checkpoint's.
    bundle.checkpoint = CheckpointWeights.from_ingest(
        ctx.result, rename=ctx.spec.checkpoint.rename)


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


def release_memory() -> None:
    """Return a freed model's cached blocks so the next stage sees a full card."""
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
    #
    # The checkpoint the numbers came out of is not here: it is recorded in the trace's
    # own manifest, and `_trace_is_whole` refuses a trace taken from another revision.
    # Provenance belongs with the artifacts — a spec that names no revision still traced
    # one, and hashing the resolved value would re-take a perfectly good 69 GiB trace
    # every time the published checkpoint moved, including when it moved back.
    return [
        _graph_fingerprint(ctx.layout.graph_path, edges=False),
        [[s.id, s.token_ids] for s in ctx.samples],
        options.slice_long, options.cache_weights,
        # Which entry points get driven decides which modules have a reference at all.
        ctx.spec.trace.to_dict(),
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
    # What gets verified is each module's own implementation, built from its source and
    # the weights the trace recorded for it. The model is only the structure those are
    # checked against — so when it cannot be resident anywhere, the structure is all
    # this stage builds. A resident build would allocate a 94.4 GiB n-gram table that
    # no check here reads.
    build = ctx.build_model
    if memory_shortfall(ctx) and ctx.build_meta_model is not None:
        build = ctx.build_meta_model
    report = verify_modules(
        build, bundle, ctx.graph,
        model_device=host, module_device=target,
        impl_dirs=_impl_dirs(ctx),
    )
    ctx.verify_report = report
    if target != host:
        ctx.notes.append(
            f"modules verified one at a time on {target} with the model resident on {host}"
        )
    ctx.notes.extend(report.notes)
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

    # Edges the recording did not bear out. Not drift: the plan claims a value flows from
    # one module to the next and it does not, which is a statement about the partition
    # rather than about anyone's arithmetic. Reported either way, because a chain that
    # restarts from the recording covers less than one that does not.
    uncarried = sorted({step.module_id for report in suite.reports
                        for step in report.unchained})
    carried = sum(len(r.chained_steps) for r in suite.reports)
    boundaries = sum(len(r.steps) for r in suite.reports)
    metrics = {
        "n_samples": len(suite.reports),
        "n_failed": len(suite.failures),
        "worst_cosine": suite.worst_cosine(),
        "mean_top1_agreement": suite.mean_top1(),
        "diverging_modules": suite.diverging_modules(),
        "carried_boundaries": carried,
        "boundaries": boundaries,
        "uncarried_edges": uncarried,
    }
    coverage = f"{carried}/{boundaries} boundary value(s) carried"
    if uncarried:
        coverage += f"; {len(uncarried)} edge(s) not carried: {', '.join(uncarried[:4])}"
    if not suite.passed:
        diverging = suite.diverging_modules()
        # Repairable with or without a boundary to point at. Tokens can move on drift no
        # single boundary flags, and that is still the implementations' to fix; the
        # modules that drifted furthest are where a repair starts.
        suspects = diverging or suite.most_drifted()
        kept = suite.kept_tokens()
        where = (f"; first divergence at {', '.join(diverging[:4])}" if diverging
                 else f"; every carried boundary held, most drift at {', '.join(suspects)}"
                 if suspects else "")
        return StageResult(
            ok=False, repairable=True, repair_surface="modules",
            failing_modules=suspects,
            detail=(f"{len(suite.failures)}/{len(suite.reports)} chain(s) failed; "
                    f"{kept}/{len(suite.reports)} kept every token; {coverage}{where}"),
            metrics=metrics,
        )
    return StageResult(ok=True, detail=(
        f"{len(suite.reports)} chain(s) held, worst cosine {suite.worst_cosine():.6f}, "
        f"top-1 agreement {suite.mean_top1():.4f}; {coverage}"
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
    # A model whose weights cannot all be resident is generated from the way it was
    # traced: streamed, one module's weights at a time, with the implementations built at
    # the moment each submodule is called and dropped after. There is no instant at which
    # 96 rebuilt modules of a 475 GiB model could all exist, so the alternative to this is
    # no generation at all — and the point of the stage is to read what the loop's own
    # code writes. The report says the weights came from the checkpoint, not from dumps.
    streamed = bool(memory_shortfall(ctx)) and _can_stream(ctx)
    if streamed:
        device = ctx.options.trace_device or _accelerator(ctx)
        place_max_memory = None
        ctx.notes.append(
            "generated from the streamed model with the implementations built per call: "
            "its weights do not fit anywhere at once, so they were read from the "
            "checkpoint rather than filled from dumps"
        )
    impl_dirs = _impl_dirs(ctx)
    if not impl_dirs:
        # Generating through the model's own modules would print tokens the deliverable
        # never produced and compare the reference against itself.
        return StageResult(ok=False, detail=(
            "no extracted implementations to emulate with: extraction produced none, so "
            "generation would be the model's own code rather than the loop's"
        ))

    report = emulate(
        ctx.build_model, bundle, ctx.graph,
        inputs=[EmulationInput(s.id, s.tensor(device), prompt=s.prompt) for s in ctx.samples],
        decode=(lambda ids: ctx.tokenizer.decode(ids)) if ctx.tokenizer else None,
        judge=judge, max_new_tokens=ctx.options.max_new_tokens,
        temperature=ctx.options.temperature, seed=ctx.options.seed,
        eos_token_id=eos, device=device, min_score=ctx.options.min_judge_score,
        strict_fill=not streamed, place_max_memory=place_max_memory, streamed=streamed,
        # Generation runs the loop's own implementations, not the model's modules:
        # these tokens are the deliverable's tokens or they are worth nothing.
        impl_dirs=impl_dirs,
        out_of_scope=_excluded_markers(ctx.spec),
        # Which of the forward's return values is the logits. A model that returns a
        # bare tuple says nothing about that, and the spec is where it is recorded.
        returns=ctx.spec.trace.returns,
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
        gap = report.install_gap()
        detail = (f"{len(broken)}/{len(report.outcomes)} sample(s) did not reproduce "
                  f"the model; {len(boundary_failures)} boundary mismatch(es)")
        if gap:
            detail = gap if not broken else f"{detail}. {gap}"
        return StageResult(
            ok=False, repairable=bool(boundary_failures), failing_modules=boundary_failures,
            detail=detail, metrics=_emulate_metrics(report),
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
        # What emulation actually runs: the implementations installed into the model. A
        # pass recorded before a module's `source.py` was edited says nothing about the
        # tokens the edited one generates.
        _impl_fingerprint(ctx),
        options.max_new_tokens, options.temperature, options.seed,
        # Including which judge read the tokens, and which model it was: a verdict from
        # the stub is not a verdict from an LLM.
        options.judge_kind, options.judge_model, options.min_judge_score,
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
        # What a port has to target, and what it is porting from.
        "device": ctx.budget.gpu.describe() if ctx.budget.gpu else "no GPU",
        "snapshot": str(ctx.result.root) if ctx.result else "",
    }


def _summarize_types(layer_types: list[str]) -> str:
    if not layer_types:
        return ""
    counts: dict[str, int] = {}
    for name in layer_types:
        counts[name] = counts.get(name, 0) + 1
    return ", ".join(f"{name} x{count}" for name, count in counts.items())

