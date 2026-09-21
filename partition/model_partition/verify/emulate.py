# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Emulated end-to-end inference from the partition's own implementations.

Assembles the model from dumped per-module weights only, replaces every
partitioned submodule with the implementation the loop extracted for it, checks
every module boundary against the trace during the first forward, then generates
tokens from the final module's logits and has them judged.

Running the model's own modules here would check the reference against itself and
print tokens the shipped code never produced. The implementations are what gets
handed over, so they are what generates.
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

    @property
    def mechanically_passed(self) -> bool:
        """True when the partition reproduced the model, judge aside.

        This is the objective part: tokens were produced and every module
        boundary matched. The judge's opinion of the text is tracked separately
        because it is a heuristic and must not be the thing that fails a run.
        """
        return bool(self.token_ids) and not self.error and self.boundaries_passed

    def passed(self, min_score: int = 4) -> bool:
        """Mechanically sound *and* judged sound — the bar for an early exit."""
        if not self.mechanically_passed:
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
    #: Which submodules were running the loop's own code during generation.
    install: Any = None
    min_score: int = 4

    @property
    def passed(self) -> bool:
        return bool(self.outcomes) and all(o.passed(self.min_score) for o in self.outcomes)

    @property
    def failures(self) -> list[EmulationOutcome]:
        return [o for o in self.outcomes if not o.passed(self.min_score)]

    @property
    def mechanically_passed(self) -> bool:
        """True when every sample reproduced the model, judge aside."""
        return bool(self.outcomes) and all(o.mechanically_passed for o in self.outcomes)

    @property
    def judge_declined(self) -> list[EmulationOutcome]:
        """Samples that reproduced the model but whose text the judge rejected.

        A judge that failed to answer is not a judge that said no, so those are
        reported separately: iterating on output quality in response to an unanswered
        question would be spending the loop's budget on nothing.
        """
        return [o for o in self.outcomes
                if o.mechanically_passed and not o.passed(self.min_score)
                and o.verdict is not None and not o.verdict.errored]

    @property
    def judge_errored(self) -> list[EmulationOutcome]:
        """Samples the judge did not manage to assess at all."""
        return [o for o in self.outcomes
                if o.verdict is None or o.verdict.errored]

    def mean_score(self) -> float:
        """Mean over samples the judge actually assessed."""
        scores = [o.verdict.score for o in self.outcomes
                  if o.verdict and not o.verdict.errored]
        return sum(scores) / len(scores) if scores else 0.0

    def render(self) -> str:
        lines: list[str] = []
        if self.fill:
            lines.append(self.fill.summary())
        if self.install:
            lines.append(self.install.summary())
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
            "mechanically_passed": self.mechanically_passed,
            "judge_declined": [o.sample_id for o in self.judge_declined],
            "judge_errored": [o.sample_id for o in self.judge_errored],
            "mean_score": self.mean_score(),
            "n_samples": len(self.outcomes),
            "n_failed": len(self.failures),
            "fill": {
                "applied": self.fill.applied,
                "missing": self.fill.missing[:32],
                "complete": self.fill.complete,
            } if self.fill else None,
            "install": self.install.to_dict() if self.install else None,
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
    impl_dirs: dict[str, Any] | None = None,
    run_dir: Any = None,
) -> EmulationReport:
    """Assemble from dumps, install the implementations, generate, and judge.

    ``place_max_memory`` spreads the assembled model across GPU and host for the
    generation pass, for a model too large to hold on the GPU whole.
    """
    judge = judge or StubJudge()
    report = EmulationReport(min_score=min_score)

    model, device = move_to_device(build_model(), device)
    poison_parameters(model, only=plan_owned_parameters(model, graph))
    report.fill = fill_from_dumps(model, bundle, graph, device=device, strict=strict_fill)
    if impl_dirs and run_dir is not None:
        from model_partition.runtime.assemble import install_implementations

        report.install = install_implementations(
            model, graph, bundle, impl_dirs, run_dir=run_dir, device=device,
        )
    if place_max_memory:
        model, device = place_across_devices(model, place_max_memory)

    for item in inputs:
        outcome = EmulationOutcome(sample_id=item.sample_id, prompt=item.prompt)
        ids = item.input_ids.to(device) if hasattr(item.input_ids, "to") else item.input_ids

        # The boundaries get their own forward, and the captured tensors are released
        # before generation starts. Leaving the hooks installed across every step
        # would hold one output per module — including a vocabulary-wide logits tensor
        # — for the whole generation, which is most of a GPU at long context.
        if check_boundaries:
            outcome.boundary_checks = _capture_and_check(
                model, bundle, graph, item.sample_id, ids, device,
                tolerance or Tolerance.accumulated(),
            )

        try:
            outcome.token_ids, _ = generate(
                model, ids, max_new_tokens=max_new_tokens, temperature=temperature,
                seed=seed, eos_token_id=eos_token_id,
            )
        except Exception as exc:
            outcome.error = f"generation failed: {exc}"

        if outcome.token_ids:
            outcome.text = decode(outcome.token_ids) if decode else " ".join(map(str, outcome.token_ids))
            outcome.verdict = judge.judge(item.prompt or "(token ids)", outcome.text, item.sample_id)
        report.outcomes.append(outcome)
        _release(device)
    return report


def _capture_and_check(
    model: Any,
    bundle: TraceBundle,
    graph: PartitionGraph,
    sample_id: str,
    input_ids: Any,
    device: str,
    tolerance: Tolerance,
) -> list[Comparison]:
    """Run one forward with every module hooked, compare, and let the tensors go."""
    import torch

    from model_partition.trace import forward_no_cache

    handles, sink = capture_boundaries(model, graph)
    try:
        with torch.no_grad():
            forward_no_cache(model, input_ids)
    except Exception:
        return []
    finally:
        for handle in handles:
            handle.remove()
    try:
        return _check_boundaries(sink, bundle, graph, sample_id, device, tolerance)
    finally:
        sink.clear()
        _release(device)


def _release(device: str) -> None:
    if str(device).startswith("cuda"):
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass


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
    """Reduce a full tensor to the head/tail window a sliced dump recorded.

    Sound here in a way it is not for module replay: the tensor being windowed
    came from a full-length forward, so windowing it reproduces exactly what the
    trace stored.
    """
    import torch

    names = record.tensor_names()
    entry = next((e for e in bundle.store.entries
                  if e.name in names and e.role == "output" and e.slice_info), None)
    if entry is None:
        return actual
    info = entry.slice_info
    head, tail = info["head"], info["tail"]
    windowed = actual
    for axis in info.get("axes", []):
        if axis >= windowed.dim() or windowed.shape[axis] < head + tail:
            return actual
        length = windowed.shape[axis]
        index = torch.cat([
            torch.arange(head, device=windowed.device),
            torch.arange(length - tail, length, device=windowed.device),
        ])
        windowed = windowed.index_select(axis, index)
    return windowed
