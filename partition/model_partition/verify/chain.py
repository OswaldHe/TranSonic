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

An edge is trusted only after the recording confirms it — but only while the module
that produced it is still reproducing its own reference. A plan that splits a layer
puts the residual add between two modules and inside neither: the attention module's
output is pre-residual while the FFN's recorded input is post-residual, so carrying
that edge would feed the next module a tensor the model never gave it. Once a module
*has* drifted, the same mismatch means the opposite thing and the value is carried
regardless, because watching the error travel is the whole purpose.
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

#: How closely a carried tensor must match the module's recorded input for the edge
#: to be a real dataflow edge. Loose on purpose: this is asking "is this the same
#: tensor", not "is it numerically perfect", and by this point it carries drift.
EDGE_COSINE = 0.99


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

    @property
    def cosine(self) -> float:
        return self.comparison.cosine if self.comparison else 0.0

    @property
    def passed(self) -> bool:
        return bool(self.comparison) and self.comparison.passed and not self.error

    def summary(self) -> str:
        if self.error:
            return f"{self.module_id}: ERROR {self.error}"
        source = "chained" if self.chained else f"from trace: {self.unchained_reason}"
        return (f"{self.module_id} ({source}): "
                f"{self.comparison.summary() if self.comparison else 'not compared'}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_id": self.module_id, "tensor": self.tensor,
            "chained": self.chained, "unchained_reason": self.unchained_reason,
            "error": self.error,
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
        """The earliest boundary outside tolerance — where to start looking."""
        return next((s for s in self.steps if not s.passed), None)

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
        values = [r.top1 for r in self.reports if r.tokens]
        return sum(values) / len(values) if values else 0.0

    def render(self) -> str:
        lines = [r.render() for r in self.reports]
        agreed = sum(1 for r in self.reports if r.tokens_agree)
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
    #: Graph tensor name -> whether the module that produced it held its reference.
    sound: dict[str, bool] = {}
    #: How many modules still ahead of us consume each tensor, so a feature map can
    #: be released the moment nothing wants it. One per module is gigabytes at long
    #: context, and holding them all to the end reserves memory for nothing.
    waiting = _consumer_counts(graph, order, by_id)
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
            if upstream is not None and args:
                if sound.get(upstream, True):
                    step.chained, step.unchained_reason = _edge_holds(
                        carried[upstream], args[0])
                else:
                    # The producer already drifted, so a mismatch here is that drift
                    # arriving rather than a plan edge that was never real.
                    step.chained = True
                if step.chained:
                    args = (carried[upstream],) + tuple(args[1:])
            impl = load_impl(impl_dir)
            weights = _weights_for(bundle, module_id, device)
            built = impl.build(bundle.config, weights, device)
            with torch.no_grad():
                output = built(*args, **kwargs)
            actual = first_tensor(output)
            reference = first_tensor(expected_output(records[-1], bundle.store, device=device))
            step.comparison = compare(actual, reference, module_id, tolerance)
            carried[produced] = actual
            sound[produced] = step.passed and sound.get(upstream, True)
            if upstream is not None:
                waiting[upstream] = waiting.get(upstream, 1) - 1
                if waiting[upstream] <= 0 and upstream != logits_tensor:
                    carried.pop(upstream, None)
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


def _consumer_counts(graph: PartitionGraph, order: list[str],
                     by_id: dict[str, Any]) -> dict[str, int]:
    """How many modules consume each tensor, so it can be freed once none remain."""
    counts: dict[str, int] = {}
    for module_id in order:
        module = by_id.get(module_id)
        if module is None:
            continue
        for tensor in module.inputs:
            counts[tensor] = counts.get(tensor, 0) + 1
    return counts


def _edge_holds(carried: Any, recorded: Any) -> tuple[bool, str]:
    """Whether the carried value is the tensor this module was actually given.

    A plan's edges are a claim about dataflow, and a split layer's residual add lives
    in the parent module rather than in either half — so the claim can be wrong in a
    way that has nothing to do with any implementation. Checking it here is what lets
    the drift measurement mean something.
    """
    if carried is None or recorded is None:
        return False, "nothing recorded to compare the edge against"
    if tuple(carried.shape) != tuple(recorded.shape):
        return False, (f"shape {tuple(carried.shape)} != the recorded input's "
                       f"{tuple(recorded.shape)}")
    match = compare(carried, recorded, "edge", Tolerance(rtol=1.0, atol=1.0,
                                                        min_pass_fraction=0.0,
                                                        min_cosine=EDGE_COSINE))
    if match.passed:
        return True, ""
    return False, (f"the module's recorded input is not this tensor "
                   f"(cosine {match.cosine:.4f}); something between them belongs to "
                   "no module — a residual add, most likely")


def _weights_for(bundle: TraceBundle, module_id: str, device: str) -> dict[str, Any]:
    from model_partition.runtime.module_runner import load_named_weights

    weights = load_named_weights(bundle, module_id, device=device)
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
