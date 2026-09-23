# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chain the extracted implementations and watch the error compound.

The other two checks each answer half the question. Per-module verification starts
every module from its *recorded* input, so an error in one module cannot appear in
another — the checks are independent by construction, which is what lets them hold
a near-exact bar. Emulation does let error accumulate, but through the model's own
modules filled with dumped weights, not through the code the loop owns.

Neither answers what a kernel author actually needs to know: if I chain my
implementations, how far does the drift travel before it changes a token? So this
walks the graph in dependency order, feeds each module's *computed* output into the
next, and compares at every boundary — reporting the drift as a curve, the first
boundary that falls outside tolerance, and whether the logits still predict the same
tokens.

Only the residual stream is carried. Masks, rotary embeddings and per-architecture
extras come from the recording, because they are inputs to the model rather than
products of it: carrying them would measure the trace's own consistency instead of
the implementations' drift.

A plan edge is a claim about dataflow, and the recording settles it: both ends are
dumps of the same forward, so if this module's input really is that module's output
they are the same numbers. A plan that splits a layer puts the residual add between two
modules and inside neither — the attention module's output is pre-residual while the
FFN's recorded input is post-residual — and there the recording settles something
better than "not an edge": the gap is *exactly* another tensor the chain holds, which
identifies the missing arithmetic. So the add is reconstructed, the drift keeps
flowing, and a split layer measures the same thing an unsplit one does. When the gap
explains nothing, the edge is not dataflow, the module restarts from the recording and
the report says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from model_partition.planner.graph import PartitionGraph
from model_partition.runtime.module_runner import (
    TraceBundle,
    decode_group_call,
    expected_output,
    first_tensor,
)
from model_partition.verify.numerics import Comparison, Tolerance, compare, top1_agreement

#: Positions whose next-token prediction is reported. The logits at position i are
#: the model's answer for the prefix ending there, so the last few are the tokens it
#: would actually emit — which is the thing drift either changes or does not.
DEFAULT_TOKEN_POSITIONS = 8

#: Recorded tensors kept alive beyond the plan's own dataflow, so a residual add can
#: be recognized from the tensors around it. A residual is always a recent one, and a
#: feature map is tens of megabytes where the logits are gigabytes.
RESIDUAL_WINDOW = 4

#: The bar for "these two recordings are the same numbers". Both ends come out of the
#: same store in the same dtype, so a true edge is exact; this leaves room only for a
#: dtype round-trip, not for a residual add, whose magnitude is that of the residual
#: stream itself.
EDGE_TOLERANCE = Tolerance(rtol=1e-3, atol=1e-3, min_pass_fraction=0.999,
                           min_cosine=0.99999)


@dataclass
class ChainStep:
    """One module's output, computed from the previous module's output."""

    module_id: str
    #: The graph tensor this module produced.
    tensor: str
    comparison: Comparison | None = None
    error: str = ""
    #: True when the module's input came from the chain rather than the recording.
    chained: bool = False
    #: Why an edge the plan declares was not carried.
    unchained_reason: str = ""
    #: How the input was assembled when it took more than the upstream output alone —
    #: the residual add a split layer leaves between two modules.
    via: str = ""

    @property
    def cosine(self) -> float:
        return self.comparison.cosine if self.comparison else 0.0

    @property
    def passed(self) -> bool:
        return bool(self.comparison) and self.comparison.passed and not self.error

    def summary(self) -> str:
        if self.error:
            return f"{self.module_id}: ERROR {self.error}"
        source = (f"chained{f' via {self.via}' if self.via else ''}" if self.chained
                  else f"from trace: {self.unchained_reason}")
        return (f"{self.module_id} ({source}): "
                f"{self.comparison.summary() if self.comparison else 'not compared'}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_id": self.module_id, "tensor": self.tensor,
            "chained": self.chained, "unchained_reason": self.unchained_reason,
            "via": self.via, "error": self.error,
            "comparison": self.comparison.to_dict() if self.comparison else None,
        }


@dataclass
class ChainReport:
    """Accumulated error along one sample's chain of implementations."""

    sample_id: str
    steps: list[ChainStep] = field(default_factory=list)
    #: Next-token predictions from the chained logits, and from the recorded ones.
    tokens: list[int] = field(default_factory=list)
    reference_tokens: list[int] = field(default_factory=list)
    #: Fraction of positions where the chained logits pick the same token.
    top1: float = 0.0
    error: str = ""

    @property
    def chained_steps(self) -> list[ChainStep]:
        return [s for s in self.steps if s.chained]

    @property
    def failures(self) -> list[ChainStep]:
        """Steps that failed *while carrying* an upstream value.

        A step that started from the recording is what per-module verification
        already covers, so holding it to the accumulated bar here would double-report
        the same check.
        """
        return [s for s in self.steps if s.chained and not s.passed] + \
               [s for s in self.steps if s.error and not s.chained]

    @property
    def unchained(self) -> list[ChainStep]:
        """Edges the plan declares that the recording did not bear out."""
        return [s for s in self.steps if not s.chained and s.unchained_reason]

    @property
    def unbroken(self) -> bool:
        """True when every edge from the first module to the logits was carried."""
        return bool(self.steps) and not self.unchained

    @property
    def tokens_agree(self) -> bool:
        return bool(self.tokens) and self.tokens == self.reference_tokens

    @property
    def credited_tokens(self) -> bool:
        """Token agreement that the chain actually earned.

        A chain interrupted before the logits produces them from a fresh recording, so
        they say nothing about drift; counting those would report coverage the run does
        not have.
        """
        return self.tokens_agree and self.unbroken

    @property
    def passed(self) -> bool:
        """Sound when every carried boundary held and the predictions did not move.

        Token agreement is the part that matters — drift that never changes a token
        is drift a deployment would not notice — but it only means anything when the
        chain reached the logits unbroken. A chain interrupted by an edge the plan
        got wrong would otherwise claim credit for the recording's tokens.
        """
        if self.error or not self.steps or self.failures:
            return False
        return self.tokens_agree if self.unbroken else True

    def first_divergence(self) -> ChainStep | None:
        """The earliest carried boundary outside tolerance — where to start looking.

        An edge the recording did not bear out is not a divergence: nothing was carried
        into that module, so nothing of anyone's can have drifted. Naming it here sent
        the repair agent after an implementation that reproduces its reference exactly.
        """
        return next((s for s in self.steps if not s.passed and (s.chained or s.error)),
                    None)

    def drift_curve(self) -> list[tuple[str, float]]:
        """Cosine against the trace at each boundary, in dependency order."""
        return [(s.module_id, s.cosine) for s in self.steps if s.comparison]

    def worst_cosine(self) -> float:
        values = [s.cosine for s in self.steps if s.comparison]
        return min(values) if values else 1.0

    def render(self) -> str:
        lines = [f"[{self.sample_id}] chained {len(self.chained_steps)}/{len(self.steps)} "
                 f"module(s), worst cosine {self.worst_cosine():.6f}"]
        if self.error:
            lines.append(f"  error: {self.error}")
        for step in self.unchained[:4]:
            lines.append(f"  edge not carried: {step.module_id} — {step.unchained_reason}")
        diverged = self.first_divergence()
        if diverged:
            lines.append(f"  first divergence: {diverged.summary()}")
        if self.tokens:
            verdict = "same" if self.tokens_agree else "DIFFERENT"
            lines.append(f"  next-token predictions {verdict}: {self.tokens}")
            if not self.tokens_agree:
                lines.append(f"  reference:                      {self.reference_tokens}")
            lines.append(f"  top-1 agreement over all positions: {self.top1:.4f}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "passed": self.passed,
            "unbroken": self.unbroken,
            "worst_cosine": self.worst_cosine(),
            "unchained": [{"module_id": s.module_id, "reason": s.unchained_reason}
                          for s in self.unchained],
            "tokens": list(self.tokens),
            "reference_tokens": list(self.reference_tokens),
            "tokens_agree": self.tokens_agree,
            "top1_agreement": self.top1,
            "error": self.error,
            "first_divergence": (self.first_divergence().module_id
                                 if self.first_divergence() else None),
            "drift_curve": [{"module_id": mid, "cosine": cos}
                            for mid, cos in self.drift_curve()],
            "steps": [s.to_dict() for s in self.steps],
        }


@dataclass
class ChainSuite:
    """One chain report per sample."""

    reports: list[ChainReport] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.reports) and all(r.passed for r in self.reports)

    @property
    def failures(self) -> list[ChainReport]:
        return [r for r in self.reports if not r.passed]

    def diverging_modules(self) -> list[str]:
        """Modules where drift first exceeded tolerance, across samples."""
        seen: list[str] = []
        for report in self.reports:
            step = report.first_divergence()
            if step and step.module_id not in seen:
                seen.append(step.module_id)
        return seen

    def worst_cosine(self) -> float:
        values = [r.worst_cosine() for r in self.reports]
        return min(values) if values else 1.0

    def mean_top1(self) -> float:
        """Mean agreement over the chains that reached the logits unbroken.

        A chain that restarted at the head would otherwise contribute the recording's
        own agreement with itself.
        """
        values = [r.top1 for r in self.reports if r.tokens and r.unbroken]
        return sum(values) / len(values) if values else 0.0

    def kept_tokens(self) -> int:
        """Chains that carried the drift all the way and still predicted the same."""
        return sum(1 for r in self.reports if r.credited_tokens)

    def render(self) -> str:
        lines = [r.render() for r in self.reports]
        agreed = self.kept_tokens()
        lines.append(f"{len(self.reports) - len(self.failures)}/{len(self.reports)} "
                     f"chain(s) held; {agreed} kept every token")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "n_samples": len(self.reports),
            "n_failed": len(self.failures),
            "worst_cosine": self.worst_cosine(),
            "mean_top1_agreement": self.mean_top1(),
            "n_kept_tokens": self.kept_tokens(),
            "diverging_modules": self.diverging_modules(),
            "reports": [r.to_dict() for r in self.reports],
        }


def verify_chain(
    bundle: TraceBundle,
    graph: PartitionGraph,
    impl_dirs: dict[str, Any],
    sample_ids: list[str] | None = None,
    device: str = "cpu",
    tolerance: Tolerance | None = None,
    token_positions: int = DEFAULT_TOKEN_POSITIONS,
) -> ChainSuite:
    """Run each sample through the chained implementations, measuring drift."""
    suite = ChainSuite()
    for sample_id in (sample_ids or bundle.sample_ids()):
        suite.reports.append(_chain_sample(
            bundle, graph, impl_dirs, sample_id, device,
            tolerance or Tolerance.accumulated(), token_positions,
        ))
    return suite


def _chain_sample(
    bundle: TraceBundle,
    graph: PartitionGraph,
    impl_dirs: dict[str, Any],
    sample_id: str,
    device: str,
    tolerance: Tolerance,
    token_positions: int,
) -> ChainReport:
    import torch

    from model_partition.runtime.module_impl import load_impl

    report = ChainReport(sample_id=sample_id)
    try:
        order = graph.topological_order()
    except Exception as exc:
        report.error = f"plan is not orderable: {exc}"
        return report

    by_id = {module.id: module for module in graph.partitioned_modules}
    #: Graph tensor name -> the value this chain computed for it.
    carried: dict[str, Any] = {}
    #: Graph tensor name -> what the trace recorded for it. What makes an edge check a
    #: question about the recording rather than about how far the chain has drifted.
    recorded: dict[str, Any] = {}
    #: The order tensors were produced in, so the recent ones stay available as
    #: candidates for a residual add the plan leaves between two modules.
    recent: list[str] = []
    #: How many modules still ahead of us consume each tensor, so a feature map can
    #: be released the moment nothing wants it. One per module is gigabytes at long
    #: context, and holding them all to the end reserves memory for nothing.
    waiting = _consumer_counts(order, by_id)
    logits_tensor = graph.output_tensors[0] if graph.output_tensors else None
    reference_logits = None

    for module_id in order:
        module = by_id.get(module_id)
        if module is None or module.functional or module.is_parallel:
            # A functional node has no reference, and a parallel group's calls are
            # independent rather than a link in the chain.
            continue
        records = bundle.select(module_id=module_id, sample_id=sample_id, step=0)
        if not records or any(r.sliced for r in records):
            continue
        impl_dir = impl_dirs.get(module_id)
        if impl_dir is None:
            continue

        upstream = next((t for t in module.inputs if t in carried), None)
        produced = module.outputs[0] if module.outputs else module_id
        step = ChainStep(module_id=module_id, tensor=produced)
        try:
            args, kwargs = decode_group_call(records, bundle.store, device)
            # What the trace gave this module, before anything is substituted for it.
            entering = args[0] if args else None
            if upstream is not None and args:
                value, step.via, step.unchained_reason = _carry(
                    upstream, entering, recorded, carried)
                step.chained = value is not None
                if value is not None:
                    args = (value,) + tuple(args[1:])
            if entering is not None:
                # A module's input is itself a residual-stream tensor, and a split layer
                # adds it back after the half that follows: layer n+1 reads
                # `mlp_out + the FFN's own input`, a sum the plan names nowhere. Keeping
                # it under this module's name leaves that arithmetic identifiable.
                name = f"the input of {module_id}"
                recorded[name] = entering
                carried[name] = args[0]
                recent.append(name)
            impl = load_impl(impl_dir)
            weights = _weights_for(bundle, module_id)
            built = impl.build(bundle.config, weights, device)
            # The cross-module state this call read, put back where the module reads it.
            # DeepSeek's attention layers share their compressed KV through a module-level
            # object, so a consumer of it has no argument naming the largest thing it
            # reads — and without it the module subscripts a None rather than computing.
            _restore_state(impl, records, bundle, device)
            with torch.no_grad():
                output = built(*args, **kwargs)
            actual = first_tensor(output)
            reference = first_tensor(expected_output(records[-1], bundle.store, device=device))
            step.comparison = compare(actual, reference, module_id, tolerance)
            carried[produced] = actual
            recorded[produced] = reference
            recent.append(produced)
            if upstream is not None:
                waiting[upstream] = waiting.get(upstream, 1) - 1
            for name in _releasable(recent, waiting, logits_tensor):
                carried.pop(name, None)
                recorded.pop(name, None)
                recent.remove(name)
            if produced == logits_tensor:
                reference_logits = reference
        except Exception as exc:
            step.error = str(exc)
            report.steps.append(step)
            break
        report.steps.append(step)

    if logits_tensor in carried and reference_logits is not None:
        _score_tokens(report, carried[logits_tensor], reference_logits, token_positions)
    elif not report.error and not report.failures:
        report.error = "the chain never reached the logits, so no token was predicted"
    carried.clear()
    _release(device)
    return report


def _release(device: str) -> None:
    """Hand cached blocks back, so one sample's feature maps do not outlive it."""
    if str(device).startswith("cuda"):
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass


def _consumer_counts(order: list[str], by_id: dict[str, Any]) -> dict[str, int]:
    """How many modules consume each tensor, so it can be freed once none remain."""
    counts: dict[str, int] = {}
    for module_id in order:
        module = by_id.get(module_id)
        if module is None:
            continue
        for tensor in module.inputs:
            counts[tensor] = counts.get(tensor, 0) + 1
    return counts


def _carry(upstream: str, consumed: Any, recorded: dict[str, Any],
           carried: dict[str, Any]) -> tuple[Any, str, str]:
    """The value to feed this module — computed, not recorded — or None with a reason.

    Returns ``(value, how, why not)``. The question is about the *recording*: this
    module's recorded input either is the upstream module's recorded output or it is
    not, and both are dumps of the same forward, so the answer is exact. Asking the
    computed value instead conflates a false edge with accumulated drift, and gets it
    wrong in both directions — the same split-layer residual was 12% of the signal on
    one model, where a real edge looked broken, and 0.3% on another, where a false edge
    looked real and the chain carried a tensor the model never produced.

    When the gap is exactly another tensor the chain holds, the missing arithmetic is
    identified rather than merely detected: a split layer's residual add belongs to the
    parent layer and so to neither half, and adding it back is what lets drift travel
    through a plan that splits attention from the FFN.
    """
    import torch

    produced = recorded.get(upstream)
    if produced is None or consumed is None:
        return None, "", "the trace does not record both ends of this edge"
    if tuple(produced.shape) != tuple(consumed.shape):
        return None, "", (f"the upstream output is {tuple(produced.shape)} and this "
                          f"module's recorded input is {tuple(consumed.shape)}")
    if _same(produced, consumed):
        return carried[upstream], "", ""

    for name, value in recorded.items():
        if name == upstream or tuple(value.shape) != tuple(consumed.shape):
            continue
        with torch.no_grad():
            if _same(produced + value, consumed):
                return (carried[upstream] + carried[name],
                        f"a residual add of `{name}`", "")

    gap = compare(produced, consumed, "edge")
    return None, "", (
        f"this module's recorded input is not the upstream output (cosine "
        f"{gap.cosine:.6f}, max_abs {gap.max_abs_err:.3e}) and the difference is none "
        "of the tensors around it: something between them belongs to no module"
    )


def _same(left: Any, right: Any) -> bool:
    """Whether two recordings are the same numbers, allowing a dtype round-trip."""
    return compare(left, right, "edge", EDGE_TOLERANCE).passed


def _releasable(recent: list[str], waiting: dict[str, int], keep: str | None) -> list[str]:
    """Tensors nothing ahead consumes and nothing behind could still explain.

    The plan's own dataflow says when a feature map is finished with; a residual add
    needs the few before it as well, which is why the window exists.
    """
    stale = recent[:-RESIDUAL_WINDOW] if len(recent) > RESIDUAL_WINDOW else []
    return [name for name in stale if waiting.get(name, 0) <= 0 and name != keep]


def _restore_state(impl: Any, records: list, bundle: TraceBundle, device: str) -> int:
    """Put each recorded call's cross-module state back on the implementation's source.

    Per-module verification does the same thing, and for the same reason: the check is
    of this module against its reference, not of this module against another module
    having run first. What the chain carries is the residual stream; a tensor passed
    between modules through a module-level object is state, and it comes from the
    recording either way.
    """
    from model_partition.runtime.module_runner import apply_state

    source = getattr(impl.module, "load_implementation", None)
    if source is None:
        return 0
    applied = 0
    for record in records:
        if getattr(record, "state", None):
            applied += apply_state(source(), record, bundle.store, device)
    return applied


def _weights_for(bundle: TraceBundle, module_id: str) -> dict[str, Any]:
    """One module's weights, read onto the host whatever the chain runs on.

    Reading them straight onto the accelerator asks it for the whole module at once, and
    DeepSeek V4.1's n-gram table is 94.6 GiB — the chain died trying to allocate it on a
    44 GiB card. The launcher places what it builds: a submodule too large for the device
    stays on the host with its inputs moved across the boundary, which is how the table's
    lookup runs beside an fp8 GEMM that runs nowhere but the GPU.
    """
    from model_partition.runtime.module_runner import load_named_weights

    weights = load_named_weights(bundle, module_id, device="cpu")
    if not weights:
        raise RuntimeError(f"no weights available for {module_id!r}")
    return weights


def _score_tokens(report: ChainReport, actual: Any, reference: Any, positions: int) -> None:
    """Compare what the chained logits predict against what the trace predicted."""
    import torch

    if actual is None or reference is None or actual.shape != reference.shape:
        report.error = "chained logits do not match the recorded logits in shape"
        return
    with torch.no_grad():
        report.top1 = top1_agreement(actual, reference)
        flat_actual = actual[0] if actual.dim() == 3 else actual
        flat_reference = reference[0] if reference.dim() == 3 else reference
        take = min(positions, flat_actual.shape[0])
        report.tokens = [int(t) for t in flat_actual[-take:].argmax(-1)]
        report.reference_tokens = [int(t) for t in flat_reference[-take:].argmax(-1)]
