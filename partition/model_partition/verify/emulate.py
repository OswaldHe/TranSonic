# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Emulated end-to-end inference from partition artifacts.

Assembles the model from dumped per-module weights only, checks every module
boundary against the trace during the first forward, then generates tokens from
the final module's logits and has them judged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from model_partition.hardware import move_to_device
from model_partition.planner.graph import PartitionGraph
from model_partition.runtime.module_runner import (
    TraceBundle,
    expected_output,
    first_tensor,
)
from model_partition.runtime.streaming import (
    FillReport,
    capture_boundaries,
    fill_from_dumps,
    generate,
    place_across_devices,
)
from model_partition.verify.judge import Judge, StubJudge, Verdict
from model_partition.verify.modules import plan_owned_parameters, poison_parameters
from model_partition.verify.numerics import Comparison, Tolerance, compare


@dataclass
class EmulationOutcome:
    """One sample run end to end through the assembled model."""

    sample_id: str
    prompt: str = ""
    token_ids: list[int] = field(default_factory=list)
    text: str = ""
    boundary_checks: list[Comparison] = field(default_factory=list)
    verdict: Verdict | None = None
    error: str = ""

    @property
    def boundaries_passed(self) -> bool:
        return all(c.passed for c in self.boundary_checks)

    def passed(self, min_score: int = 4) -> bool:
        if self.error or not self.token_ids:
            return False
        if not self.boundaries_passed:
            return False
        return self.verdict is not None and self.verdict.passed(min_score)

    def failed_boundaries(self) -> list[Comparison]:
        return [c for c in self.boundary_checks if not c.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "prompt": self.prompt,
            "token_ids": self.token_ids,
            "text": self.text,
            "error": self.error,
            "boundaries_passed": self.boundaries_passed,
            "failed_boundaries": [c.to_dict() for c in self.failed_boundaries()],
            "verdict": self.verdict.to_dict() if self.verdict else None,
        }


@dataclass
class EmulationReport:
    """All emulated samples plus how the model was assembled."""

    outcomes: list[EmulationOutcome] = field(default_factory=list)
    fill: FillReport | None = None
    min_score: int = 4

    @property
    def passed(self) -> bool:
        return bool(self.outcomes) and all(o.passed(self.min_score) for o in self.outcomes)

    @property
    def failures(self) -> list[EmulationOutcome]:
        return [o for o in self.outcomes if not o.passed(self.min_score)]

    def mean_score(self) -> float:
        scores = [o.verdict.score for o in self.outcomes if o.verdict]
        return sum(scores) / len(scores) if scores else 0.0

    def render(self) -> str:
        lines: list[str] = []
        if self.fill:
            lines.append(self.fill.summary())
        for outcome in self.outcomes:
            status = "ok" if outcome.passed(self.min_score) else "FAIL"
            score = outcome.verdict.score if outcome.verdict else 0
            lines.append(f"[{outcome.sample_id}] {status} judge={score}/5 "
                         f"boundaries={'ok' if outcome.boundaries_passed else 'FAIL'}")
            if outcome.error:
                lines.append(f"    error: {outcome.error}")
            for comparison in outcome.failed_boundaries()[:5]:
                lines.append(f"    {comparison.summary()}")
            if outcome.verdict and outcome.verdict.reason:
                lines.append(f"    judge: {outcome.verdict.reason}")
        lines.append(f"{len(self.outcomes) - len(self.failures)}/{len(self.outcomes)} samples passed")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "mean_score": self.mean_score(),
            "n_samples": len(self.outcomes),
            "n_failed": len(self.failures),
            "fill": {
                "applied": self.fill.applied,
                "missing": self.fill.missing[:32],
                "complete": self.fill.complete,
            } if self.fill else None,
            "outcomes": [o.to_dict() for o in self.outcomes],
        }


@dataclass
class EmulationInput:
    """One prompt to emulate."""

    sample_id: str
    input_ids: Any
    prompt: str = ""


def emulate(
    build_model: Callable[[], Any],
    bundle: TraceBundle,
    graph: PartitionGraph,
    inputs: list[EmulationInput],
    decode: Callable[[list[int]], str] | None = None,
    judge: Judge | None = None,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    seed: int | None = 0,
    eos_token_id: int | None = None,
    device: str = "cpu",
    tolerance: Tolerance | None = None,
    min_score: int = 4,
    strict_fill: bool = True,
    check_boundaries: bool = True,
    place_max_memory: dict | None = None,
) -> EmulationReport:
    """Assemble from dumps, generate, and judge.

    ``place_max_memory`` spreads the assembled model across GPU and host for the
    generation pass, for a model too large to hold on the GPU whole.
    """
    judge = judge or StubJudge()
    report = EmulationReport(min_score=min_score)

    model, device = move_to_device(build_model(), device)
    poison_parameters(model, only=plan_owned_parameters(model, graph))
    report.fill = fill_from_dumps(model, bundle, graph, device=device, strict=strict_fill)
    if place_max_memory:
        model, device = place_across_devices(model, place_max_memory)

    for item in inputs:
        outcome = EmulationOutcome(sample_id=item.sample_id, prompt=item.prompt)
        handles, sink = capture_boundaries(model, graph) if check_boundaries else ([], {})
        try:
            token_ids, _ = generate(
                model, item.input_ids.to(device) if hasattr(item.input_ids, "to") else item.input_ids,
                max_new_tokens=max_new_tokens, temperature=temperature, seed=seed,
                eos_token_id=eos_token_id,
            )
            outcome.token_ids = token_ids
        except Exception as exc:
            outcome.error = f"generation failed: {exc}"
        finally:
            for handle in handles:
                handle.remove()

        if check_boundaries and not outcome.error:
            outcome.boundary_checks = _check_boundaries(
                sink, bundle, graph, item.sample_id, device, tolerance,
            )

        if outcome.token_ids:
            outcome.text = decode(outcome.token_ids) if decode else " ".join(map(str, outcome.token_ids))
            outcome.verdict = judge.judge(item.prompt or "(token ids)", outcome.text, item.sample_id)
        report.outcomes.append(outcome)
    return report


def _check_boundaries(
    sink: dict[str, Any],
    bundle: TraceBundle,
    graph: PartitionGraph,
    sample_id: str,
    device: str,
    tolerance: Tolerance | None,
) -> list[Comparison]:
    """Compare captured module outputs against the traced ones.

    Only the first forward is compared: later generation steps run on a longer
    sequence than the trace recorded. The hook fires on the module's *last*
    submodule, so the matching record is that submodule's, not the group's first.
    """
    comparisons: list[Comparison] = []
    for module_id, observed in sink.items():
        records = bundle.select(module_id=module_id, sample_id=sample_id, step=0)
        if not records:
            continue
        try:
            last_submodule = graph.by_id(module_id).submodules[-1]
        except (KeyError, IndexError):
            last_submodule = records[-1].submodule
        matching = [r for r in records if r.submodule == last_submodule] or records[-1:]
        record = matching[-1]
        reference = first_tensor(expected_output(record, bundle.store, device=device))
        actual = first_tensor(observed)
        if reference is None or actual is None:
            continue
        if tuple(actual.shape) != tuple(reference.shape):
            # A sliced long-context dump covers only head+tail positions.
            actual = _match_slice(actual, reference, record, bundle)
        comparisons.append(compare(actual, reference, f"boundary:{module_id}", tolerance))
    return comparisons


def _match_slice(actual: Any, reference: Any, record: Any, bundle: TraceBundle) -> Any:
    """Reduce a full tensor to the head/tail window a sliced dump recorded."""
    import torch

    names = record.tensor_names()
    entry = next((e for e in bundle.store.entries
                  if e.name in names and e.role == "output" and e.slice_info), None)
    if entry is None:
        return actual
    info = entry.slice_info
    axis, head, tail = info["axis"], info["head"], info["tail"]
    if axis >= actual.dim() or actual.shape[axis] < head + tail:
        return actual
    length = actual.shape[axis]
    index = torch.cat([
        torch.arange(head, device=actual.device),
        torch.arange(length - tail, length, device=actual.device),
    ])
    return actual.index_select(axis, index)
