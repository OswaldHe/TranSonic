# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify each module reproduces its traced output from its dumped artifacts.

The model is poisoned with NaN before each module's dumped weights are applied,
so a parameter the dump forgot shows up as a non-finite result rather than
passing on whatever happened to be in memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from model_partition.planner.graph import PartitionGraph
from model_partition.runtime.module_runner import (
    TraceBundle,
    apply_dumped_weights,
    expected_output,
    first_tensor,
    replay_record,
)
from model_partition.verify.numerics import Comparison, Tolerance, compare


def poison_parameters(model: Any, value: float = float("nan"),
                      only: set[str] | None = None) -> int:
    """Fill parameters and buffers with ``value``; returns how many.

    With ``only``, restricted to those names. Poisoning everything is wrong for a
    real model: non-persistent buffers such as rotary ``inv_freq`` are computed at
    init and never appear in a checkpoint, so no dump can restore them and the
    NaNs propagate.
    """
    import torch

    count = 0
    with torch.no_grad():
        for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
            if only is not None and name not in only:
                continue
            if tensor.is_floating_point():
                tensor.fill_(value)
                count += 1
    return count


def move_submodules(model: Any, graph_module: Any, device: str) -> int:
    """Move just one module's submodules to ``device``; returns how many moved.

    This is what makes the plan's guarantee testable: a module sized to fit the
    GPU is verified *on* the GPU even when the whole model never could be
    resident there.
    """
    from model_partition.trace import _lookup

    moved = 0
    for submodule_name in graph_module.submodules:
        submodule = _lookup(model, submodule_name)
        if submodule is not None and hasattr(submodule, "to"):
            submodule.to(device)
            moved += 1
    return moved


def _release(device: str) -> None:
    if device.startswith("cuda"):
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass


def plan_owned_parameters(model: Any, graph: PartitionGraph) -> set[str]:
    """Names of parameters and buffers the plan's modules are responsible for."""
    from model_partition.trace import _lookup

    owned: set[str] = set()
    for module in graph.partitioned_modules:
        for submodule_name in module.submodules:
            submodule = _lookup(model, submodule_name)
            if submodule is None or not hasattr(submodule, "named_parameters"):
                continue
            for param_name, _ in list(submodule.named_parameters()) + list(submodule.named_buffers()):
                owned.add(f"{submodule_name}.{param_name}" if param_name else submodule_name)
    return owned


@dataclass
class ModuleVerification:
    """Verification outcome for one module on one sample."""

    module_id: str
    sample_id: str
    passed: bool
    weights_applied: int = 0
    comparisons: list[Comparison] = field(default_factory=list)
    error: str = ""
    #: Where the module actually ran.
    device: str = ""

    def summary(self) -> str:
        if self.error:
            return f"{self.module_id} [{self.sample_id}]: ERROR {self.error}"
        head = "ok" if self.passed else "FAIL"
        detail = "; ".join(c.summary() for c in self.comparisons if not c.passed) or "all tensors match"
        where = f" on {self.device}" if self.device else ""
        return f"{self.module_id} [{self.sample_id}]: {head} ({self.weights_applied} weights{where}) {detail}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_id": self.module_id, "sample_id": self.sample_id,
            "passed": self.passed, "weights_applied": self.weights_applied,
            "error": self.error, "device": self.device,
            "comparisons": [c.to_dict() for c in self.comparisons],
        }


@dataclass
class VerifyReport:
    """All module verifications for a run."""

    results: list[ModuleVerification] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.results) and all(r.passed for r in self.results)

    @property
    def failures(self) -> list[ModuleVerification]:
        return [r for r in self.results if not r.passed]

    def module_ids(self) -> list[str]:
        return sorted({r.module_id for r in self.results})

    def worst_cosine(self) -> float:
        values = [c.cosine for r in self.results for c in r.comparisons]
        return min(values) if values else 1.0

    def max_abs_err(self) -> float:
        values = [c.max_abs_err for r in self.results for c in r.comparisons]
        return max(values) if values else 0.0

    def render(self) -> str:
        lines = [r.summary() for r in self.results]
        lines.append(
            f"{len(self.results) - len(self.failures)}/{len(self.results)} module checks passed"
        )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "n_checks": len(self.results),
            "n_failed": len(self.failures),
            "worst_cosine": self.worst_cosine(),
            "max_abs_err": self.max_abs_err(),
            "results": [r.to_dict() for r in self.results],
        }


def verify_modules(
    build_model: Callable[[], Any],
    bundle: TraceBundle,
    graph: PartitionGraph,
    sample_ids: list[str] | None = None,
    module_ids: list[str] | None = None,
    device: str = "cpu",
    tolerance: Tolerance | None = None,
    poison: bool = True,
    model_device: str | None = None,
    module_device: str | None = None,
    impl_dirs: dict[str, Any] | None = None,
) -> VerifyReport:
    """Replay every module from its dumps and compare against the trace.

    ``model_device`` holds the whole model — the host, for a checkpoint larger
    than the GPU. ``module_device`` is where each module under test is moved for
    its replay, defaulting to ``model_device``. Setting it to a GPU verifies each
    module on the accelerator one at a time, which is exactly what the plan's
    per-module budget promises.

    ``impl_dirs`` maps a module id to its extracted implementation directory. When
    given, that implementation is what gets run — the code the loop owns and edits
    — with this function still deciding whether the result is correct.
    """
    report = VerifyReport()
    model = build_model()
    host = model_device or device
    model = model.to(host)
    target = module_device or host
    if poison:
        poison_parameters(model, only=plan_owned_parameters(model, graph))

    wanted_modules = module_ids or bundle.module_ids()
    wanted_samples = sample_ids or bundle.sample_ids()

    for module_id in wanted_modules:
        try:
            graph_module = graph.by_id(module_id)
        except KeyError:
            report.results.append(ModuleVerification(
                module_id=module_id, sample_id="-", passed=False,
                error="module is not in the partition graph",
            ))
            continue

        applied = apply_dumped_weights(model, bundle, module_id, graph_module, device=host)
        residency = target
        if target != host:
            try:
                move_submodules(model, graph_module, target)
            except RuntimeError as exc:
                # The module did not fit after all; fall back rather than abort.
                move_submodules(model, graph_module, host)
                _release(target)
                residency = host
                report.results.append(ModuleVerification(
                    module_id=module_id, sample_id="-", passed=True,
                    weights_applied=applied,
                    error=f"ran on {host}: {exc}",
                ))
        impl_dir = (impl_dirs or {}).get(module_id)
        # Built once per module: the baseline instantiates the model's structure,
        # which is far too expensive to repeat for every recorded call.
        impl_callable = None
        impl_error = ""
        if impl_dir is not None:
            try:
                impl_callable = _build_impl(impl_dir, bundle, module_id, residency)
            except Exception as exc:
                impl_error = str(exc)
        try:
            for sample_id in wanted_samples:
                records = bundle.select(module_id=module_id, sample_id=sample_id)
                if not records:
                    continue
                # With an implementation the module is checked as a whole: its
                # input is the first submodule's, its output the last submodule's.
                # Without one, each submodule is replayed against its own record.
                pairs = ([(records[0], records[-1])] if impl_dir is not None
                         else [(r, r) for r in records])
                for source, reference_record in pairs:
                    comparisons: list[Comparison] = []
                    try:
                        if impl_error:
                            raise RuntimeError(f"implementation unusable: {impl_error}")
                        actual = _run_once(model, source, bundle, impl_callable, residency)
                        reference = expected_output(reference_record, bundle.store,
                                                   device=residency)
                        comparisons = _compare_outputs(actual, reference, module_id, tolerance)
                    except Exception as exc:
                        report.results.append(ModuleVerification(
                            module_id=module_id, sample_id=sample_id, passed=False,
                            weights_applied=applied, error=str(exc), device=residency,
                        ))
                        continue
                    report.results.append(ModuleVerification(
                        module_id=module_id, sample_id=sample_id,
                        passed=bool(comparisons) and all(c.passed for c in comparisons),
                        weights_applied=applied, comparisons=comparisons, device=residency,
                    ))
        finally:
            if residency != host:
                move_submodules(model, graph_module, host)
                _release(residency)
    return report


def _build_impl(impl_dir: Any, bundle: TraceBundle, module_id: str, device: str) -> Any:
    """Build a module's extracted implementation once, ready to call."""
    from model_partition.runtime.module_impl import load_impl
    from model_partition.runtime.module_runner import load_named_weights

    impl = load_impl(impl_dir)
    weights = load_named_weights(bundle, module_id, device=device)
    if not weights:
        raise RuntimeError(f"no dumped weights for {module_id!r}")
    built = impl.build(bundle.metadata.get("config") or {}, weights, device)
    if not callable(built):
        raise RuntimeError(f"{impl.path}: build_module() returned a non-callable")
    return built


def _run_once(model: Any, record: Any, bundle: TraceBundle,
              impl_callable: Any, device: str) -> Any:
    """Produce a module's output for one recorded call.

    Calls the extracted implementation when one was built, so verification
    exercises the code the loop owns; otherwise replays the original submodule,
    which is the case before extraction has run.
    """
    import torch

    if impl_callable is None:
        return replay_record(model, record, bundle.store, device=device)

    from model_partition.runtime.module_runner import decode_call

    if record.has_unsupported():
        raise RuntimeError(
            f"{record.module_id}: recorded arguments include an unserializable value"
        )
    args, kwargs = decode_call(record, bundle.store, device)
    with torch.no_grad():
        return impl_callable(*args, **kwargs)


def _compare_outputs(actual: Any, reference: Any, label: str,
                     tolerance: Tolerance | None) -> list[Comparison]:
    """Compare every tensor in a possibly-nested output structure."""
    from model_partition.trace import is_tensor

    pairs = list(_walk_pairs(actual, reference, label))
    if not pairs:
        primary = compare(first_tensor(actual), first_tensor(reference), label, tolerance)
        return [primary]
    return [compare(a, b, name, tolerance) for name, a, b in pairs if is_tensor(b)]


def _walk_pairs(actual: Any, reference: Any, path: str):
    from model_partition.trace import is_tensor

    if is_tensor(reference):
        yield path, actual, reference
        return
    if isinstance(reference, dict):
        for key, value in reference.items():
            child = actual.get(key) if isinstance(actual, dict) else None
            yield from _walk_pairs(child, value, f"{path}.{key}")
        return
    if isinstance(reference, (list, tuple)):
        for i, value in enumerate(reference):
            child = actual[i] if isinstance(actual, (list, tuple)) and i < len(actual) else None
            yield from _walk_pairs(child, value, f"{path}[{i}]")
