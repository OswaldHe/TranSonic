# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify each module reproduces its traced output from its dumped artifacts.

A parameter the dump forgot must not pass on whatever happened to be in memory, so
the model's own weights are made unusable first: poisoned with NaN, or — for a model
too large to hold, whose weights are placeholders to begin with — left as
placeholders, which raise on any read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from model_partition.hardware import format_bytes
from model_partition.planner.graph import PartitionGraph
from model_partition.runtime.launcher import weights_bytes
from model_partition.runtime.module_runner import (
    TraceBundle,
    apply_named_weights,
    apply_state,
    expected_output,
    load_named_weights,
    replay_record,
)
from model_partition.verify.numerics import Comparison, Tolerance, compare_outputs


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


def move_submodules(model: Any, graph_module: Any, device: str,
                    partial: bool = False) -> int:
    """Move just one module's submodules to ``device``; returns how many moved.

    This is what makes the plan's guarantee testable: a module sized to fit the
    GPU is verified *on* the GPU even when the whole model never could be
    resident there.

    ``partial`` is for the module the measurement already says will not go whole: each
    submodule is placed with whatever fits and the rest left on the host, its boundaries
    wrapped. An indivisible 94.4 GiB table needs exactly that — moved wholesale it OOMs,
    and the resulting failure reads as "partition it further", which is the one thing
    that cannot help a single table.
    """
    from model_partition.runtime.launcher import place
    from model_partition.trace import _lookup

    moved = 0
    for submodule_name in graph_module.submodules:
        submodule = _lookup(model, submodule_name)
        if submodule is None:
            continue
        if partial:
            place(submodule, device)
            moved += 1
        elif hasattr(submodule, "to"):
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
    #: Set when no numeric check was possible. Neither a pass nor a failure: it is
    #: reported so missing coverage is visible rather than assumed away.
    skipped: str = ""
    #: The module would not fit the device it was sized for. A plan failure rather
    #: than a numeric one, so the loop repartitions instead of editing arithmetic.
    oversized: bool = False

    def summary(self) -> str:
        if self.skipped:
            return f"{self.module_id} [{self.sample_id}]: SKIP {self.skipped}"
        if self.error:
            return f"{self.module_id} [{self.sample_id}]: ERROR {self.error}"
        head = "ok" if self.passed else "FAIL"
        detail = "; ".join(c.summary() for c in self.comparisons if not c.passed) or "all tensors match"
        where = f" on {self.device}" if self.device else ""
        return (f"{self.module_id} [{self.sample_id}]: {head} "
                f"({self.weights_applied} weights{where}) {detail}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_id": self.module_id, "sample_id": self.sample_id,
            "passed": self.passed, "weights_applied": self.weights_applied,
            "error": self.error, "device": self.device, "skipped": self.skipped,
            "oversized": self.oversized,
            "comparisons": [c.to_dict() for c in self.comparisons],
        }


@dataclass
class VerifyReport:
    """All module verifications for a run."""

    results: list[ModuleVerification] = field(default_factory=list)
    #: What the run had to do differently, per module — where a check ran when it could
    #: not run where it was asked to.
    notes: list[str] = field(default_factory=list)

    @property
    def checked(self) -> list[ModuleVerification]:
        """Results that carry a verdict. Skips are reported, never counted."""
        return [r for r in self.results if not r.skipped]

    @property
    def skipped(self) -> list[ModuleVerification]:
        return [r for r in self.results if r.skipped]

    @property
    def passed(self) -> bool:
        return bool(self.checked) and all(r.passed for r in self.checked)

    @property
    def failures(self) -> list[ModuleVerification]:
        return [r for r in self.checked if not r.passed]

    @property
    def oversized(self) -> list[ModuleVerification]:
        """Failures that are about capacity, so the plan is what needs changing."""
        return [r for r in self.failures if r.oversized]

    def module_ids(self) -> list[str]:
        return sorted({r.module_id for r in self.results})

    def worst_cosine(self) -> float:
        values = [c.cosine for r in self.checked for c in r.comparisons]
        return min(values) if values else 1.0

    def max_abs_err(self) -> float:
        values = [c.max_abs_err for r in self.checked for c in r.comparisons]
        return max(values) if values else 0.0

    def render(self) -> str:
        lines = [r.summary() for r in self.results] + list(self.notes)
        checked = len(self.checked)
        lines.append(f"{checked - len(self.failures)}/{checked} module checks passed")
        if self.skipped:
            lines.append(f"{len(self.skipped)} check(s) skipped, unverified")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "n_checks": len(self.checked),
            "n_failed": len(self.failures),
            "n_skipped": len(self.skipped),
            "n_oversized": len(self.oversized),
            "worst_cosine": self.worst_cosine(),
            "max_abs_err": self.max_abs_err(),
            "notes": list(self.notes),
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
    target = module_device or host
    # A streamed model's weights are placeholders until each module reads its own, so
    # there is nothing to move and nothing to poison — and a placeholder is the stronger
    # guarantee anyway: reading one raises rather than returning a number. Each module
    # under test still gets its real weights from the recording below.
    placeholders = _has_placeholders(model)
    if not placeholders:
        model = model.to(host)
        if poison:
            poison_parameters(model, only=plan_owned_parameters(model, graph))
    else:
        report.notes.append(
            "the model's weights are placeholders, so each module was checked on the "
            "weights the trace recorded for it and nothing could fall back to the model"
        )

    # Every module of the plan, not only the ones the trace has records for. A module
    # with no records — a functional node, or one whose submodule was never called — is
    # a hole in the coverage, and reporting it is the difference between "verified" and
    # "verified the parts that happened to be recorded".
    planned = [m.id for m in graph.partitioned_modules]
    recorded = bundle.module_ids()
    wanted_modules = module_ids or planned + [m for m in recorded if m not in set(planned)]
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

        if graph_module.functional:
            report.results.append(ModuleVerification(
                module_id=module_id, sample_id="-", passed=False,
                skipped="functional module: no submodule, so the trace holds no "
                        "reference to compare against",
            ))
            continue

        # Once extraction has run, the implementation is the thing under test. A
        # module extraction missed has to be reported as unverified: replaying the
        # model's own submodule instead would compare the model against itself.
        impl_dir = (impl_dirs or {}).get(module_id)
        if impl_dirs is not None and impl_dir is None:
            report.results.append(ModuleVerification(
                module_id=module_id, sample_id="-", passed=False,
                error="extraction produced no implementation for this module, so "
                      "there is nothing of the loop's own to run",
            ))
            continue

        weights = load_named_weights(bundle, module_id, device=host)
        # How many recorded tensors this check ran on. Against a resident model they go
        # into it; against placeholders there is nothing to put them in — `copy_` onto a
        # placeholder succeeds and changes nothing — so they go to the implementation
        # alone, which is what is being verified either way.
        applied = (len(weights) if placeholders
                   else apply_named_weights(model, weights, graph_module))
        residency = target
        fits = _weights_fit(target, weights)
        if not fits:
            # Not a broken promise and not something partitioning fixes: one 94.4 GiB
            # embedding table is one piece of work whatever it is cut into. The check
            # still runs on the accelerator — the launcher leaves the table on the host
            # and moves what crosses it — because the fp8 GEMM beside it runs nowhere
            # else.
            report.notes.append(
                f"{module_id}: {format_bytes(weights_bytes(weights))} of weights, more "
                f"than {target} holds; the oversized ones stay on {host}"
            )
        if target != host and not placeholders:
            try:
                # Partially when the measurement above says it will not go whole.
                move_submodules(model, graph_module, target, partial=not fits)
            except RuntimeError as exc:
                # The plan said this module would fit and it did not. That is a
                # partition failure, not something to work around by finishing the
                # check on the host and calling the promise kept. An indivisible module
                # that will not go even partially is a different statement, and says so.
                move_submodules(model, graph_module, host)
                _release(target)
                advice = ("Partition it further." if fits else
                          "It cannot be split further, so this machine cannot check it.")
                report.results.append(ModuleVerification(
                    module_id=module_id, sample_id="-", passed=False,
                    weights_applied=applied, device=target, oversized=fits,
                    error=f"will not load on {target}: {exc}. {advice}",
                ))
                continue
        # Built once per (module, branch): constructing a module from its source and
        # loading its weights is too expensive to repeat for every recorded call.
        builder = (_impl_builder(impl_dir, weights, bundle, module_id, residency)
                   if impl_dir else None)
        # Only a parallel group has a branch to select; naming one for a sequential
        # group would ask for a prefix of the module.
        branch = graph_module.is_parallel
        try:
            if not bundle.select(module_id=module_id):
                # In the plan, absent from the trace: nothing was recorded for it, so
                # there is nothing to check it against. Reported rather than skipped
                # over, because a module nobody verified is not a module that passed.
                report.results.append(ModuleVerification(
                    module_id=module_id, sample_id="-", passed=False,
                    weights_applied=applied, device=residency,
                    skipped="the trace holds no record of this module, so it has no "
                            "reference to check against",
                ))
                continue
            for sample_id in wanted_samples:
                records = bundle.select(module_id=module_id, sample_id=sample_id)
                if not records:
                    continue
                runs = _record_runs(graph_module, records, with_impl=builder is not None)
                for run in runs:
                    source, reference_record = run[0], run[-1]
                    if source.sliced or reference_record.sliced:
                        report.results.append(ModuleVerification(
                            module_id=module_id, sample_id=sample_id, passed=False,
                            skipped="windowed long-context dump: the recorded output is "
                                    "not a function of the recorded input",
                        ))
                        continue
                    # A group run as one computation is handed every argument any of the
                    # submodules in *this* invocation received.
                    group = run
                    comparisons: list[Comparison] = []
                    try:
                        actual = _run_once(model, source, bundle, builder, residency,
                                           branch=branch, group=group)
                        reference = expected_output(reference_record, bundle.store,
                                                   device=residency)
                        comparisons = compare_outputs(actual, reference, module_id, tolerance)
                        del actual, reference
                    except Exception as exc:
                        # A module that will not run on the accelerator is a partition
                        # problem, not an arithmetic one: the plan promised it would
                        # fit. Report it as such — the loop's answer is to cut the
                        # module smaller, not to quietly finish the check somewhere
                        # slower and call the promise kept.
                        oversized = _is_out_of_memory(exc)
                        needed = _check_bytes(group + [reference_record], bundle)
                        detail = (f"{residency} ran out of memory; the module's tensors "
                                  f"are about {format_bytes(needed)} for this sample. "
                                  "Partition it further." if oversized else str(exc))
                        report.results.append(ModuleVerification(
                            module_id=module_id, sample_id=sample_id, passed=False,
                            weights_applied=applied, error=detail, device=residency,
                            oversized=oversized,
                        ))
                        _release(residency)
                        continue
                    report.results.append(ModuleVerification(
                        module_id=module_id, sample_id=sample_id,
                        passed=bool(comparisons) and all(c.passed for c in comparisons),
                        weights_applied=applied, comparisons=comparisons, device=residency,
                    ))
        finally:
            if residency != host and not placeholders:
                move_submodules(model, graph_module, host)
            del weights
            _release(target)
    return report


def _weights_fit(device: str, weights: dict[str, Any]) -> bool:
    """Whether one module's weights have room on ``device`` as things stand."""
    if not device.startswith("cuda"):
        return True
    try:
        import torch

        free, _total = torch.cuda.mem_get_info(torch.device(device))
    except Exception:  # pragma: no cover - no CUDA on the machine running the tests
        return True
    # The module also needs its recorded inputs, its output and whatever it allocates
    # between them, so the weights alone do not get the whole card.
    return weights_bytes(weights) < free * 0.8


def _has_placeholders(model: Any) -> bool:
    """True when the model's weights are placeholders rather than values.

    Which is how a model larger than this machine gets built at all: each module reads
    its own for its forward and releases them after.
    """
    for tensor in list(model.parameters()) + list(model.buffers()):
        if getattr(tensor, "is_meta", False):
            return True
    return False


def _check_bytes(records: list[Any], bundle: TraceBundle) -> int:
    """Bytes of recorded tensors one check has to hold.

    Read from the manifest rather than measured, because the point is to decide
    whether to allocate at all.
    """
    wanted: set[str] = set()
    for record in records:
        wanted.update(record.tensor_names())
    by_name = {entry.name: entry for entry in bundle.store.entries}
    return sum(by_name[name].nbytes for name in wanted if name in by_name)


def _is_out_of_memory(exc: Exception) -> bool:
    """True for an accelerator allocation failure, whatever torch version raised it."""
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except (ImportError, AttributeError):  # pragma: no cover
        pass
    return "out of memory" in str(exc).lower()


def _record_runs(graph_module: Any, records: list[Any], with_impl: bool) -> list[list[Any]]:
    """The records behind each independent check of one module on one sample.

    A sequential group run through its implementation is one computation: input from the
    first submodule's call, reference from the last submodule's output. That holds across
    *distinct* submodules. A submodule the pass called more than once is not a longer
    chain — it is the group run again, and DeepSeek's draft head does exactly that, calling
    `markov_head` once per drafted position on the token the previous call sampled. Taking
    the first call's input with the last call's output would check neither: the module is
    pure, so the mismatch is entirely the pairing's. Each invocation therefore gets its own
    run, which is also what makes the repeat-call dumps worth keeping apart.

    A parallel group is not a chain at all, since each expert sees its own routed tokens,
    and without an implementation each submodule is replayed against its own record either
    way — one record per run in both cases.
    """
    if not with_impl or graph_module.is_parallel:
        return [[record] for record in records]
    runs: list[list[Any]] = []
    current: list[Any] = []
    seen: set[Any] = set()
    for record in records:
        if record.submodule in seen:
            runs.append(current)
            current, seen = [], set()
        current.append(record)
        seen.add(record.submodule)
    if current:
        runs.append(current)
    return runs


def _impl_builder(impl_dir: Any, weights: dict[str, Any], bundle: TraceBundle,
                  module_id: str, device: str):
    """Return ``submodule -> callable``, memoized, or a raiser if unusable.

    Takes the weights already loaded for the live model rather than reading them
    again: a module's parameters are the same tensors either way, and ``copy_``
    moves them to wherever the implementation was built.
    """
    from model_partition.runtime.launcher import load_source
    from model_partition.runtime.module_impl import load_impl

    cache: dict[str | None, Any] = {}

    def build(submodule: str | None):
        if submodule in cache:
            built = cache[submodule]
            if isinstance(built, Exception):
                raise built
            return built
        try:
            impl = load_impl(impl_dir)
            if not weights:
                raise RuntimeError(f"no weights available for {module_id!r}")
            built = impl.build(bundle.config, weights, device, submodule=submodule)
            if not callable(built):
                raise RuntimeError(f"{impl.path}: build_module() returned a non-callable")
        except Exception as exc:
            wrapped = RuntimeError(f"implementation unusable: {exc}")
            cache[submodule] = wrapped
            raise wrapped from exc
        cache[submodule] = built
        return built

    # The module's own `source.py`, so a caller can put back the cross-module state a
    # recorded call ran on: that is where the module reads it from.
    build.source = lambda: load_source(impl_dir, device=device)
    return build


def _run_once(model: Any, record: Any, bundle: TraceBundle, builder: Any, device: str,
              branch: bool = False, group: list[Any] | None = None) -> Any:
    """Produce a module's output for one recorded call.

    Calls the extracted implementation when there is one, so verification
    exercises the code the loop owns; otherwise replays the original submodule,
    which is the case before extraction has run.
    """
    import torch

    if builder is None:
        return replay_record(model, record, bundle.store, device=device)

    from model_partition.runtime.module_runner import decode_group_call

    if record.has_unsupported():
        raise RuntimeError(
            f"{record.module_id}: recorded arguments include an unserializable value"
        )
    impl_callable = builder(record.submodule if branch else None)
    # The state this call read, put back where the module reads it. A consumer of
    # DeepSeek's shared compressed KV has no argument naming it.
    if getattr(record, "state", None):
        apply_state(builder.source(), record, bundle.store, device)
    args, kwargs = decode_group_call(group or [record], bundle.store, device)
    with torch.no_grad():
        return impl_callable(*args, **kwargs)
