# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a floorplan against a system and report a latency for each workload.

This is the executable the exploration loop measures. It loads the floorplan, the system
YAML, the partition graph and the agent-written cost models, walks the module DAG once per
workload, and prints the metrics the harness reads.

The workload grid is fixed here rather than in the floorplan, because it is what the run is
being judged on — a scheme that could choose its own benchmark would choose an easy one. It
spans three axes: phase (prefill/decode) x context length (128/8192) x batch size
(1/4/8/32), so sixteen points.

Batch is an axis rather than a constant because the two phases respond to it in opposite
directions, and the decisions that follow reverse with it. A 1-token decode step has nothing
in flight to fill a pipeline with, so every stage boundary is a bubble and depth is pure
cost; at batch 32 the same pipeline fills and the same depth is nearly free. Meanwhile the
KV cache grows linearly with batch, so a residency choice that fits at batch 1 and 8192
tokens can be infeasible at batch 32. Measuring only at batch 1 optimizes the worst case of
a tradeoff instead of the tradeoff.

Prefill is chunked and decode is not, which is where pipeline parallelism earns or loses its
keep. The simulator does not special-case any of this — it falls out of running the DAG once
per chunk and letting engine occupancy and stage ordering serialize what shares a unit.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from floorplan.parser import Hardware, ceil_div, load_system
from floorplan.schema import (
    BATCH_SIZES,
    CONTEXT_LENGTHS,
    WEIGHT_PARTITION_DIMS,
    Floorplan,
)
from floorplan.sim import api
from floorplan.sim.api import Context, CostModelError, Shard, Workload
from floorplan.sim.engine import Schedule, Trace
from floorplan.sim.memory import MemoryLedger, scope_key_for


def _workloads() -> tuple[Workload, ...]:
    """The grid every floorplan is measured on: phase x context length x batch size.

    Sixteen points, built rather than listed so the three axes stay in one place. Batch is an
    axis because the two phases respond to it in opposite directions, and a single-batch
    benchmark hides that: at batch 1 a decode step has one token in flight and every pipeline
    stage boundary is a bubble, while at batch 32 the same pipeline fills and the same plan
    looks entirely different. A scheme tuned only at batch 1 is tuned for the worst case of a
    decision that reverses.
    """
    points: list[Workload] = []
    for phase in ("prefill", "decode"):
        for context in CONTEXT_LENGTHS:
            for batch in BATCH_SIZES:
                points.append(Workload(
                    name=f"{phase}_{context}_b{batch}",
                    phase=phase,
                    context_tokens=context,
                    new_tokens=context if phase == "prefill" else 1,
                    batch=batch,
                ))
    return tuple(points)


#: The sixteen points the loop is measured at, and the metric each one publishes.
WORKLOADS: tuple[Workload, ...] = _workloads()

#: By name, because positional indexing into a grid breaks the moment an axis gains a value.
WORKLOADS_BY_NAME: dict[str, Workload] = {w.name: w for w in WORKLOADS}

#: The workload that exercises the most: longest context, largest batch, prefill. Used wherever
#: a single representative point is wanted (the memory report, the invariant suite's spot
#: checks) so those follow the grid instead of hardcoding a position in it.
HEAVIEST = f"prefill_{max(CONTEXT_LENGTHS)}_b{max(BATCH_SIZES)}"


def workload(name: str) -> Workload:
    """One workload by name, with the available names in the error."""
    try:
        return WORKLOADS_BY_NAME[name]
    except KeyError:
        raise SimulationError(
            f"no workload '{name}'. Known: {', '.join(sorted(WORKLOADS_BY_NAME))}"
        ) from None

#: Module kinds outside the text-only inference path. The vision tower is present in the
#: partition graph and is not part of `tokens -> logits`, so a floorplan neither places it
#: nor is faulted for omitting it.
EXCLUDED_KINDS = frozenset({"vision"})


class SimulationError(RuntimeError):
    """The simulation cannot run, or the plan it was given is not deployable."""


@dataclass
class WorkloadResult:
    """One workload's outcome."""

    workload: Workload
    seconds: float
    trace: Trace
    ledger: MemoryLedger
    chunks: int

    @property
    def milliseconds(self) -> float:
        return self.seconds * 1e3


# ---------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------
def load_graph(artifact: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """``(modules by id, the graph itself)`` from a partition artifact."""
    path = Path(artifact) / "plan" / "partition_graph.yaml"
    if not path.exists():
        raise SimulationError(f"no partition graph at {path}")
    graph = yaml.safe_load(path.read_text())
    modules = {str(m["id"]): m for m in graph.get("modules", [])}
    if not modules:
        raise SimulationError(f"{path} declares no modules")
    return modules, graph


def load_model_config(artifact: Path) -> dict[str, Any]:
    """The model's ``config.json``, from whichever module directory carries one.

    Every module directory holds the same config — it is the model's, not the module's —
    so the first one found is the right one, and looking rather than hardcoding a group
    name keeps this working when the artifact's groups change.
    """
    for candidate in sorted((Path(artifact) / "modules").glob("*/config.json")):
        return json.loads(candidate.read_text())
    raise SimulationError(f"no modules/*/config.json under {artifact}")


def load_cost_models(project: Path) -> list[str]:
    """Import every cost model under ``<project>/sim/modules/``. Returns their names.

    Imported by path rather than as a package so the project directory needs no
    ``__init__.py`` plumbing and the agent can add a file without touching anything else.
    """
    api.clear_registry()
    directory = Path(project) / "sim" / "modules"
    if not directory.is_dir():
        raise SimulationError(
            f"no cost models at {directory}. `floorplan build` writes them; without them "
            f"there is nothing to simulate"
        )
    paths = sorted(p for p in directory.glob("*.py") if not p.name.startswith("_"))
    if not paths:
        raise SimulationError(f"{directory} holds no cost models")
    for path in paths:
        spec = importlib.util.spec_from_file_location(f"_floorplan_cost_{path.stem}", path)
        if spec is None or spec.loader is None:
            raise SimulationError(f"cannot import {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            raise SimulationError(f"{path.name} failed to import: {exc}") from exc
    return [getattr(m, "name", type(m).__name__) for m in api.registry()]


def load_constraints(project: Path):
    """The agent's implementation of the platform's prose constraints, if present.

    Optional in mechanism, required in practice: the invariant suite reports every
    ``constraints_text`` item nothing implements. Returning None here rather than raising
    keeps a missing file from masquerading as a simulator bug.
    """
    path = Path(project) / "sim" / "constraints.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("_floorplan_constraints", path)
    if spec is None or spec.loader is None:
        raise SimulationError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise SimulationError(f"sim/constraints.py failed to import: {exc}") from exc
    return module


# ---------------------------------------------------------------------------------------
# The module DAG
# ---------------------------------------------------------------------------------------
def topological_order(modules: dict[str, dict[str, Any]]) -> list[str]:
    """Module ids in an order where every producer precedes its consumers.

    Producers are found through tensor names: a module consuming ``h.3`` waits for whoever
    produces ``h.3``. An input nothing produces is an entry tensor (``tokens``) and is not
    a dependency. Ties break on the id, so the order is stable across runs — which the
    determinism invariant depends on.
    """
    produced_by: dict[str, str] = {}
    for module_id, entry in modules.items():
        for tensor in entry.get("outputs") or []:
            produced_by[str(tensor)] = module_id

    dependencies: dict[str, set[str]] = {}
    for module_id, entry in modules.items():
        deps = {
            produced_by[str(tensor)]
            for tensor in (entry.get("inputs") or [])
            if str(tensor) in produced_by and produced_by[str(tensor)] != module_id
        }
        dependencies[module_id] = deps

    ready = sorted(m for m, deps in dependencies.items() if not deps)
    order: list[str] = []
    remaining = {m: set(deps) for m, deps in dependencies.items()}
    while ready:
        current = ready.pop(0)
        order.append(current)
        del remaining[current]
        newly: list[str] = []
        for module_id, deps in remaining.items():
            if current in deps:
                deps.discard(current)
                if not deps:
                    newly.append(module_id)
        ready = sorted(set(ready) | set(newly))

    if remaining:
        stuck = sorted(remaining)[:6]
        raise SimulationError(
            f"the module graph has a cycle involving {', '.join(stuck)}"
            f"{' ...' if len(remaining) > 6 else ''}. "
            f"{len(remaining)} module(s) could not be ordered"
        )
    return order


def deployable_modules(modules: dict[str, dict[str, Any]]) -> set[str]:
    """Every module a floorplan is required to place."""
    return {
        module_id for module_id, entry in modules.items()
        if str(entry.get("kind")) not in EXCLUDED_KINDS
    }


# ---------------------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------------------
def weight_divisor(placement: Any) -> int:
    """How many ways this placement's *parameters* are actually divided.

    Only the weight-partitioning dimensions count. A batch or sequence split partitions the
    activations and leaves every participant holding the module's full weights, so dividing by
    its factor understates residency: a batch-64 placement was charged 1/64 of its real
    weights, which let a plan overflow a bank while the capacity gate reported that it fit, and
    made data parallelism look almost free.
    """
    divisor = 1
    for split in placement.splits:
        if split.dim in WEIGHT_PARTITION_DIMS:
            divisor *= split.factor
    return divisor


def charge_weights(plan: Floorplan, modules: dict[str, dict[str, Any]], ledger: MemoryLedger) -> None:
    """Account every placement's parameter bytes. Deterministic — no cost model involved.

    A tiered placement charges the backing tier for all of its bytes and the cache tier for
    the resident fraction, because a cache is a copy: putting 10% of Engram in HBM does not
    remove 10% of it from host DRAM.
    """
    for placement in plan.placements:
        entry = modules.get(placement.module)
        if entry is None:
            raise SimulationError(
                f"placement names '{placement.module}', which is not in the partition graph"
            )
        total = int(entry.get("param_bytes") or 0) * placement.fraction
        per_shard = ceil_div(int(total), weight_divisor(placement))
        for unit in placement.units:
            residency = placement.residency
            ledger.add_weights(
                residency.tier,
                scope_key_for(residency.tier, unit, residency.backing_device),
                per_shard,
                placement.module,
            )
            if residency.resident_fraction < 1.0 and residency.cache_tier:
                ledger.add_weights(
                    residency.cache_tier,
                    scope_key_for(residency.cache_tier, unit, residency.backing_device),
                    int(per_shard * residency.resident_fraction),
                    f"{placement.module} (cache)",
                )


def simulate_workload(
    plan: Floorplan,
    hardware: Hardware,
    modules: dict[str, dict[str, Any]],
    config: dict[str, Any],
    workload: Workload,
    order: list[str] | None = None,
) -> WorkloadResult:
    """Run one workload and return its timeline and its memory ledger."""
    order = order or topological_order(modules)
    placed = defaultdict(list)
    for placement in plan.placements:
        placed[placement.module].append(placement)

    schedule = Schedule()
    ledger = MemoryLedger()
    charge_weights(plan, modules, ledger)

    chunk_tokens = (
        min(plan.runtime.prefill_chunk_tokens, workload.new_tokens)
        if workload.is_prefill() and plan.runtime.pipeline_chunks
        else workload.new_tokens
    )
    if workload.is_prefill():
        chunks = ceil_div(workload.new_tokens, chunk_tokens)
        micro_batch = workload.batch
    else:
        # A decode step has one token per sample, so the only way to put more than one item in
        # a deep pipeline is to split the *batch* into micro-batches — which is what
        # `runtime.decode_micro_batch` is for. It was never read, so every decode ran as a
        # single item walking all 43 layers, the stages stayed serialized at batch 32 exactly
        # as at batch 1, and the batch-dependent pipeline tradeoff the metric grid was added to
        # expose could not appear in any measurement.
        micro_batch = min(plan.runtime.decode_micro_batch, workload.batch)
        chunks = ceil_div(workload.batch, micro_batch)

    scratch: dict[str, Any] = {}
    # Completion indices per module, carried across the DAG so a consumer waits on its
    # producers' last ops. Reset per chunk: chunk 1's attention waits on chunk 1's norm,
    # and the serialization against chunk 0 comes from engine occupancy, not from here.
    for chunk_index in range(chunks):
        consumed_before = chunk_index * chunk_tokens
        this_chunk = min(chunk_tokens, workload.new_tokens - consumed_before)
        # In decode the chunk index walks the batch, not the prompt: each pass carries one
        # micro-batch of samples through every stage.
        this_batch = (
            workload.batch if workload.is_prefill()
            else min(micro_batch, workload.batch - chunk_index * micro_batch)
        )
        chunk_workload = Workload(
            name=workload.name,
            phase=workload.phase,
            context_tokens=(
                consumed_before + this_chunk if workload.is_prefill()
                else workload.context_tokens
            ),
            new_tokens=this_chunk if workload.is_prefill() else 1,
            batch=this_batch,
            resident_batch=workload.batch,
        )
        completions: dict[str, list[int]] = {}
        produced_by = {
            str(tensor): module_id
            for module_id, entry in modules.items()
            for tensor in (entry.get("outputs") or [])
        }
        # Op indices that closed each pipeline stage, so a later stage can wait on an earlier
        # one even where no tensor connects them.
        stage_completions: dict[int, list[int]] = {}

        for module_id in order:
            placements = placed.get(module_id)
            if not placements:
                continue
            entry = modules[module_id]
            data_deps: list[int] = []
            for tensor in entry.get("inputs") or []:
                producer = produced_by.get(str(tensor))
                if producer and producer != module_id:
                    data_deps.extend(completions.get(producer, []))

            model = api.resolve(module_id, entry)
            emitted: list[int] = []
            for placement in placements:
                # `stage` is a real ordering constraint, not a label. Without this it was
                # inert: ordering came only from data dependencies and unit contention, so
                # changing the documented control for pipeline depth could not move any
                # metric, and two plans with 4 and 16 stages on the same units were scheduled
                # identically.
                #
                # A placement in stage N waits for every stage below N to have closed. That is
                # what makes a deep pipeline cost something at decode, where one item in
                # flight leaves every stage boundary a bubble with nothing to fill it.
                #
                # Per placement, not per module: taking the minimum stage across a module's
                # placements let the fractional-placement form — parts of one module on
                # different stages — hand a later part the earliest part's dependencies, so it
                # started too soon and paid for none of the depth it declared.
                deps = list(data_deps)
                for stage in sorted(s for s in stage_completions if s < placement.stage):
                    deps.extend(stage_completions[stage])
                group = tuple(placement.units)
                placement_ops: list[int] = []
                for shard_index, unit in enumerate(placement.units):
                    total = int(entry.get("param_bytes") or 0) * placement.fraction
                    shard = Shard(
                        module=module_id,
                        kind=str(entry.get("kind", "other")),
                        unit=unit,
                        shard_index=shard_index,
                        shard_count=placement.shard_count(),
                        splits=tuple(placement.splits),
                        fraction=placement.fraction,
                        param_bytes=ceil_div(int(total), weight_divisor(placement)),
                        module_activation_bytes=int(entry.get("activation_bytes") or 0),
                        residency=placement.residency,
                        stage=placement.stage,
                        overlap_collectives=placement.overlap_collectives,
                        graph_entry=entry,
                        group=group,
                    )
                    context = Context(
                        hardware=hardware,
                        schedule=schedule,
                        ledger=ledger,
                        workload=chunk_workload,
                        shard=shard,
                        deps=tuple(deps),
                        config=config,
                        scratch=scratch,
                    )
                    try:
                        produced = model.emit(context)
                    except CostModelError:
                        raise
                    except Exception as exc:
                        raise SimulationError(
                            f"cost model '{getattr(model, 'name', model)}' failed on "
                            f"{module_id} shard {shard_index} at {unit} "
                            f"({workload.name}): {type(exc).__name__}: {exc}"
                        ) from exc
                    if produced is None:
                        raise SimulationError(
                            f"cost model '{getattr(model, 'name', model)}' returned None "
                            f"for {module_id}; emit() must return the op indices "
                            f"downstream modules wait on"
                        )
                    placement_ops.extend(int(i) for i in produced)
                emitted.extend(placement_ops)
                # Each placement's ops close *its* stage. Recording every op under every stage
                # a module touched made a module spanning stages 3 and 7 appear to have
                # completed stage 3 only once its stage-7 work was done, so the two stages
                # could never overlap.
                stage_completions.setdefault(placement.stage, []).extend(placement_ops)
            completions[module_id] = emitted

    return WorkloadResult(
        workload=workload,
        seconds=schedule.trace.makespan(),
        trace=schedule.trace,
        ledger=ledger,
        chunks=chunks,
    )


def simulate(
    plan_path: Path,
    artifact: Path,
    project: Path,
    systems_dir: Path | None = None,
) -> tuple[dict[str, WorkloadResult], Hardware]:
    """Run every workload. Raises if the plan is not deployable."""
    plan = Floorplan.load(plan_path)
    system = load_system(plan.target, systems_dir)
    hardware = Hardware.from_system(system)

    unresolved = hardware.unresolved()
    if unresolved:
        raise SimulationError(
            f"target '{hardware.name}' has {len(unresolved)} unmeasured value(s): "
            f"{', '.join(unresolved[:6])}{' ...' if len(unresolved) > 6 else ''}. "
            f"Run `autohelix floorplan probe` first. The simulator will not substitute "
            f"peak for a missing efficiency, because that would make compute free"
        )

    modules, _graph = load_graph(artifact)
    config = load_model_config(artifact)
    plan.validate_against(system, modules, config)

    required = deployable_modules(modules)
    missing = sorted(required - plan.modules())
    if missing:
        raise SimulationError(
            f"{len(missing)} module(s) are not deployed: {', '.join(missing[:8])}"
            f"{' ...' if len(missing) > 8 else ''}. Every module on the tokens -> logits "
            f"path must be placed for the run to reach the end"
        )
    extra = sorted(plan.modules() - set(modules))
    if extra:
        raise SimulationError(
            f"floorplan places module(s) that do not exist: {', '.join(extra[:8])}"
        )

    load_cost_models(project)
    constraints = load_constraints(project)
    if constraints is not None and hasattr(constraints, "check"):
        problems = constraints.check(hardware, plan) or []
        if problems:
            raise SimulationError(
                "the plan violates a platform constraint:\n  "
                + "\n  ".join(str(p) for p in problems)
            )

    order = topological_order(modules)
    results: dict[str, WorkloadResult] = {}
    for workload in WORKLOADS:
        result = simulate_workload(plan, hardware, modules, config, workload, order)
        violations = result.ledger.violations(hardware)
        if violations:
            raise SimulationError(
                f"{workload.name}: the plan does not fit:\n  " + "\n  ".join(violations)
            )
        results[workload.name] = result
    return results, hardware


# ---------------------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------------------
def write_trace(results: dict[str, WorkloadResult], hardware: Hardware, out: Path) -> None:
    """Dump the evidence the ranking step and the reviewer read.

    Deliberately verbose: per-engine utilization, per-link volume, the critical path and
    the memory high-water mark. A report that says one scheme beats another has to be able
    to point at which of these differs.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "target": hardware.name,
        "hardware": hardware.describe(),
        "workloads": {},
    }
    for name, result in results.items():
        trace = result.trace
        unit, busy = trace.busiest_unit()
        by_module = trace.by_module()
        payload["workloads"][name] = {
            "milliseconds": result.milliseconds,
            "chunks": result.chunks,
            "ops": len(trace.scheduled),
            "utilization": {k: round(v, 4) for k, v in sorted(trace.utilization().items())},
            "link_bytes": dict(sorted(trace.link_bytes.items())),
            "collective_ms": trace.collective_seconds * 1e3,
            "overlapped_ms": trace.overlapped_seconds * 1e3,
            "busiest_unit": {"unit": unit, "busy_ms": busy * 1e3},
            "top_modules_ms": {
                module: round(seconds * 1e3, 4)
                for module, seconds in sorted(
                    by_module.items(), key=lambda kv: -kv[1]
                )[:20]
            },
            "critical_path": [
                {
                    "op": entry.op.name,
                    "module": entry.op.module,
                    "unit": entry.op.unit,
                    "engine": entry.op.engine,
                    "start_ms": round(entry.start * 1e3, 6),
                    "finish_ms": round(entry.finish * 1e3, 6),
                }
                for entry in trace.critical_path()[:40]
            ],
            "memory": {
                tier: {
                    "worst_scope": scope,
                    "used_bytes": used,
                    "capacity_bytes": capacity,
                    "fill": round(used / capacity, 4) if capacity else None,
                }
                for tier, (scope, used, capacity)
                in sorted(result.ledger.high_water(hardware).items())
            },
        }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m floorplan.sim.runner",
        description="Simulate a floorplan and report a latency per workload.",
    )
    parser.add_argument("--plan", type=Path, default=Path("floorplan.yaml"),
                        help="the floorplan to simulate (default: floorplan.yaml)")
    parser.add_argument("--artifact", type=Path, required=True,
                        help="the partition artifact holding plan/partition_graph.yaml")
    parser.add_argument("--project", type=Path, default=Path("."),
                        help="the project holding sim/ (default: the working directory)")
    parser.add_argument("--trace", type=Path, default=None,
                        help="write the full trace here as JSON")
    parser.add_argument("--systems", type=Path, default=None,
                        help="override the systems/ directory")
    parser.add_argument("--quiet", action="store_true", help="metrics only")
    args = parser.parse_args(argv)

    try:
        results, hardware = simulate(args.plan, args.artifact, args.project, args.systems)
    except (SimulationError, CostModelError) as exc:
        print(f"simulation failed: {exc}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(hardware.describe())
        print()
        for name in (w.name for w in WORKLOADS):
            result = results[name]
            util = result.trace.utilization()
            busiest = result.trace.busiest_unit()
            print(
                f"{name:<14} {result.milliseconds:9.3f} ms  "
                f"{result.chunks:>2} chunk(s)  {len(result.trace.scheduled):>6} ops  "
                f"busiest {busiest[0] or '-'} {busiest[1] * 1e3:.2f} ms"
            )
            if util:
                print(
                    "               "
                    + "  ".join(f"{k} {v * 100:.0f}%" for k, v in sorted(util.items()))
                )
        print()
        print(results[HEAVIEST].ledger.report(hardware))
        print()

    # The harness reads these. One per workload, and the names match preset.yaml's metrics.
    for name in (w.name for w in WORKLOADS):
        print(f"##autohelix[{name}_ms={results[name].milliseconds:.6f}]")

    if args.trace:
        write_trace(results, hardware, args.trace)
        if not args.quiet:
            print(f"\ntrace written to {args.trace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
