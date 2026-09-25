# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The gate for step 2: is this simulator self-consistent enough to explore with?

The exploration loop's metrics are only as meaningful as the cost models underneath them, and
nothing here can check whether those models are *right* — there is no ground truth to compare
against, by design, since the only measured module latencies available describe unoptimized
kernels and would bias the whole run. So this suite checks the next best thing: that the
simulator behaves the way a cost model of a real machine has to behave, whatever the
constants are.

That distinction matters for reading a failure. A red invariant here does not mean a number
is off by 20%; it means the simulator would mislead the loop about the *direction* of a
change, which is the only thing the loop actually uses it for. A simulator where doubling
tensor parallelism does not reduce per-shard compute cannot rank tensor-parallel schemes at
all, however well calibrated its matmul rate is.

  a  models present       every module on the inference path has a cost model
  b  baseline simulates    the generated baseline runs and publishes four positive metrics
  c  deterministic         two identical runs produce identical metrics
  d  no free modules       every placed module contributes measurable work
  e  splits scale          doubling a split halves per-shard compute and adds communication
  f  context costs         longer contexts cost more, and superlinearly in prefill
  g  constraints covered   every numbered item in constraints_text is implemented
  h  costs are derived     no cost model fabricates a duration or reads the clock
  i  hardware read-only    simulating does not mutate the platform model
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from floorplan.checker import Check, Report
from floorplan.parser import Hardware, load_system
from floorplan.schema import Floorplan, Split
from floorplan.sim import api
from floorplan.sim.runner import (
    WORKLOADS,
    SimulationError,
    deployable_modules,
    load_cost_models,
    load_graph,
    load_model_config,
    simulate_workload,
    topological_order,
)

#: Constructs a cost model may not contain. The interface hands out seconds through
#: `matmul_seconds`, `elementwise_seconds`, `dma_seconds`, `gather_seconds` and
#: `collective_cost`; a model that reaches past them is either fabricating a duration or
#: making itself non-deterministic, and either one quietly invalidates the run.
BANNED_PATTERNS: tuple[tuple[str, str], str] | tuple = (
    (r"\bimport\s+time\b|\btime\s*\.\s*(time|perf_counter|monotonic)\b",
     "reads the clock — a cost model must be a pure function of shapes"),
    (r"\bimport\s+random\b|\brandom\s*\.\s*\w+", "uses randomness — runs must be reproducible"),
    (r"\bnumpy\s*\.\s*random\b", "uses randomness — runs must be reproducible"),
    (r"\.autohelix/memory|loop-\d+-|latency_ms", "reads recorded kernel latencies, which are "
                                                 "measurements of unoptimized kernels"),
    (r"\bDeepSeek-V4\.1-Flash-Trainium\b", "reads the bootstrapped kernels, which are "
                                           "out of scope for cost modeling"),
)

#: How far from ideal a scaling invariant may land before it counts as broken. Generous:
#: the point is direction, not accuracy. A split that halves work should land between 1.5x
#: and 2.5x, not at 1.0x or 4x.
SCALING_LOW = 1.5
SCALING_HIGH = 2.5


def _simulate(
    plan: Floorplan,
    hardware: Hardware,
    modules: dict[str, dict[str, Any]],
    config: dict[str, Any],
    workload_name: str,
    order: list[str],
):
    workload = next(w for w in WORKLOADS if w.name == workload_name)
    return simulate_workload(plan, hardware, modules, config, workload, order)


def check_models(modules: dict[str, dict[str, Any]]) -> Check:
    """(a) Every module on the inference path is claimed by some cost model."""
    check = Check("a", "models present", True)
    registered = api.registry()
    if not registered:
        return check.fail("no cost models are registered; sim/modules/ is empty or silent")
    unclaimed: list[str] = []
    for module_id in sorted(deployable_modules(modules)):
        try:
            api.resolve(module_id, modules[module_id])
        except api.CostModelError:
            unclaimed.append(module_id)
    if unclaimed:
        check.fail(
            f"{len(unclaimed)} module(s) have no cost model: "
            f"{', '.join(unclaimed[:10])}{' ...' if len(unclaimed) > 10 else ''}"
        )
    check.findings.append(
        f"{len(registered)} model(s): "
        f"{', '.join(getattr(m, 'name', type(m).__name__) for m in registered)}"
    )
    return check


def check_baseline(results: dict[str, Any] | None, error: str) -> Check:
    """(b) The generated baseline runs and publishes four positive metrics."""
    check = Check("b", "baseline simulates", True)
    if results is None:
        return check.fail(f"the baseline did not simulate: {error}")
    for workload in WORKLOADS:
        result = results.get(workload.name)
        if result is None:
            check.fail(f"{workload.name} produced no result")
        elif result.seconds <= 0:
            check.fail(f"{workload.name} took {result.seconds}s; a latency must be positive")
    if check.passed:
        check.findings.append(", ".join(
            f"{w.name} {results[w.name].milliseconds:.2f}ms" for w in WORKLOADS
        ))
    return check


def check_determinism(
    plan: Floorplan, hardware: Hardware, modules: Any, config: Any, order: list[str],
    results: dict[str, Any] | None,
) -> Check:
    """(c) A second run of every workload reproduces the first exactly."""
    check = Check("c", "deterministic", True)
    if results is None:
        return check.fail("not checked: the baseline did not simulate")
    for workload in WORKLOADS:
        again = _simulate(plan, hardware, modules, config, workload.name, order)
        before, after = results[workload.name].seconds, again.seconds
        if before != after:
            check.fail(
                f"{workload.name} changed between identical runs: "
                f"{before * 1e3:.6f}ms -> {after * 1e3:.6f}ms"
            )
    return check


def check_no_free_modules(plan: Floorplan, results: dict[str, Any] | None) -> Check:
    """(d) Every placed module contributes measurable work.

    Two failures, and the second is the likelier one. A module costing exactly zero is
    almost never a fast module; and a module that emitted no ops at all never appears in the
    trace, so it has to be found by its absence — an ``emit`` that returns early looks like
    success to everything else in the system. Either way the loop would then place that
    module anywhere for free, which is a cost model bug wearing the costume of a discovery.
    """
    check = Check("d", "no free modules", True)
    if results is None:
        return check.fail("not checked: the baseline did not simulate")
    worst = results[WORKLOADS[1].name]           # prefill_8192, where everything runs
    by_module = worst.trace.by_module()

    free = sorted(module for module, seconds in by_module.items() if seconds <= 0)
    if free:
        check.fail(
            f"{len(free)} module(s) cost nothing: {', '.join(free[:10])}"
            f"{' ...' if len(free) > 10 else ''}"
        )
    silent = sorted(plan.modules() - set(by_module))
    if silent:
        check.fail(
            f"{len(silent)} placed module(s) emitted no ops at all: "
            f"{', '.join(silent[:10])}{' ...' if len(silent) > 10 else ''}"
        )
    if check.passed:
        check.findings.append(f"{len(by_module)} module(s) contributed work")
    return check


def check_split_scaling(
    plan: Floorplan, hardware: Hardware, modules: Any, config: Any, order: list[str],
) -> Check:
    """(e) Doubling a split halves per-shard compute and puts something on the wire.

    Built by rewriting one attention module's head split from 4 to 2 and comparing that
    module's busiest shard. Per-shard, not total: the total is roughly conserved, which is
    the point — parallelism moves work, it does not delete it.
    """
    check = Check("e", "splits scale", True)
    target = next(
        (p for p in plan.placements
         if p.module.endswith(".attention")
         and any(s.dim == "head" and s.factor >= 4 for s in p.splits)),
        None,
    )
    if target is None:
        return check.fail(
            "no attention placement split at least 4 ways along 'head' in the baseline; "
            "cannot test split scaling"
        )

    def per_shard_seconds(factor: int) -> tuple[float, int]:
        variant = copy.deepcopy(plan)
        for placement in variant.placements:
            if placement.module != target.module:
                continue
            placement.splits = [
                Split(dim="head", factor=factor, collective="allreduce")
            ]
            placement.units = list(target.units[:factor])
        variant.validate()
        result = _simulate(variant, hardware, modules, config, "prefill_8192", order)
        compute = sum(
            entry.op.seconds
            for entry in result.trace.scheduled
            if entry.op.module == target.module and entry.op.engine == "tensor"
        )
        wire = sum(
            entry.op.bytes_moved
            for entry in result.trace.scheduled
            if entry.op.module == target.module and entry.op.engine == "cc"
        )
        return (compute / max(factor, 1), wire)

    try:
        narrow, narrow_wire = per_shard_seconds(2)
        wide, wide_wire = per_shard_seconds(4)
    except (SimulationError, api.CostModelError) as exc:
        return check.fail(f"a variant plan failed to simulate: {exc}")

    if wide <= 0 or narrow <= 0:
        return check.fail(
            f"{target.module} used no tensor-engine time at one of the split factors "
            f"(2-way {narrow * 1e3:.4f}ms, 4-way {wide * 1e3:.4f}ms)"
        )
    ratio = narrow / wide
    if not SCALING_LOW <= ratio <= SCALING_HIGH:
        check.fail(
            f"{target.module}: going from a 2-way to a 4-way head split changed per-shard "
            f"tensor time by {ratio:.2f}x, expected about 2x "
            f"(between {SCALING_LOW} and {SCALING_HIGH}). The cost model is not dividing "
            f"the work by the split factor"
        )
    if wide_wire <= narrow_wire:
        check.fail(
            f"{target.module}: a 4-way split moved {wide_wire} bytes and a 2-way split "
            f"{narrow_wire} — a wider split must communicate more, not less"
        )
    check.findings.append(
        f"{target.module}: per-shard tensor time 2-way -> 4-way is {ratio:.2f}x, "
        f"wire bytes {narrow_wire} -> {wide_wire}"
    )
    return check


def check_context_scaling(results: dict[str, Any] | None) -> Check:
    """(f) A longer context costs more, and superlinearly in prefill.

    64x the tokens through an attention stack is at least 64x the work and, with any
    quadratic term at all, appreciably more. Requiring only 8x leaves room for a chunked
    prefill — which flattens the quadratic within a chunk — while still failing a cost model
    that ignores sequence length, which is the mistake that would make every
    context-parallel scheme look pointless.
    """
    check = Check("f", "context costs", True)
    if results is None:
        return check.fail("not checked: the baseline did not simulate")
    short_prefill = results["prefill_128"].seconds
    long_prefill = results["prefill_8192"].seconds
    short_decode = results["decode_128"].seconds
    long_decode = results["decode_8192"].seconds

    if long_prefill <= short_prefill:
        check.fail(
            f"prefill_8192 ({long_prefill * 1e3:.3f}ms) is not slower than prefill_128 "
            f"({short_prefill * 1e3:.3f}ms)"
        )
    elif long_prefill < short_prefill * 8:
        check.fail(
            f"prefill_8192 is only {long_prefill / short_prefill:.1f}x prefill_128 for 64x "
            f"the tokens; the cost models are nearly insensitive to sequence length"
        )
    if long_decode <= short_decode:
        check.fail(
            f"decode_8192 ({long_decode * 1e3:.3f}ms) is not slower than decode_128 "
            f"({short_decode * 1e3:.3f}ms) — a longer KV cache costs more to attend over"
        )
    if check.passed:
        check.findings.append(
            f"prefill {long_prefill / short_prefill:.1f}x, "
            f"decode {long_decode / short_decode:.2f}x"
        )
    return check


def check_constraints_covered(project: Path, hardware: Hardware) -> Check:
    """(g) Every numbered item in ``constraints_text`` is implemented and cites its number.

    The prose half of the platform description is the half a parser cannot enforce, so the
    only available check is bookkeeping: each item has to be referenced from the code that
    implements it. An item nothing cites is not necessarily unimplemented — but it is
    unreviewable, which for a hidden-gate loop is the same problem.
    """
    check = Check("g", "constraints covered", True)
    items = re.findall(r"^\s{0,3}(\d+)\.\s", hardware.constraints_text, re.MULTILINE)
    if not items:
        return check.fail("the system YAML's constraints_text has no numbered items")

    sources: list[str] = []
    for relative in ("sim/constraints.py",):
        path = project / relative
        if path.exists():
            sources.append(path.read_text())
    for path in sorted((project / "sim" / "modules").glob("*.py")):
        sources.append(path.read_text())
    if not sources:
        return check.fail("no sim/ sources to check for constraint citations")
    blob = "\n".join(sources)

    uncited = [
        number for number in items
        if not re.search(rf"constraint\s*{number}\b|item\s*{number}\b|#\s*{number}\b", blob,
                         re.IGNORECASE)
    ]
    if uncited:
        check.fail(
            f"constraints_text item(s) {', '.join(uncited)} are not cited anywhere under "
            f"sim/. Cite the item number in the code that implements it, or say in "
            f"sim/constraints.py why it needs no implementation"
        )
    check.findings.append(f"{len(items)} item(s), {len(items) - len(uncited)} cited")
    return check


def check_costs_derived(project: Path) -> Check:
    """(h) No cost model fabricates a duration or reads the clock."""
    check = Check("h", "costs are derived", True)
    paths = sorted((project / "sim").rglob("*.py"))
    if not paths:
        return check.fail("no cost models to scan")
    for path in paths:
        text = path.read_text()
        relative = path.relative_to(project)
        for pattern, why in BANNED_PATTERNS:
            match = re.search(pattern, text)
            if match:
                line = text[:match.start()].count("\n") + 1
                check.fail(f"{relative}:{line} {why} (found '{match.group(0).strip()}')")
        # A raw `ctx.op(...)` with a literal duration bypasses the costing helpers. The
        # helpers exist so the rates come from the system YAML and not from a cost model's
        # imagination; a literal here is the one way to get around that.
        for match in re.finditer(r"\.op\(\s*[^)]*?,\s*[\"']\w+[\"']\s*,\s*([0-9.eE+-]+)\s*[,)]",
                                 text):
            literal = match.group(1)
            try:
                if float(literal) == 0.0:
                    continue          # a zero-duration marker op is harmless
            except ValueError:
                continue
            line = text[:match.start()].count("\n") + 1
            check.fail(
                f"{relative}:{line} passes the literal duration {literal} to op(). "
                f"Durations must come from matmul_seconds/elementwise_seconds/"
                f"dma_seconds/gather_seconds so the rates stay in the system YAML"
            )
    return check


def check_hardware_readonly(hardware: Hardware, before: str) -> Check:
    """(i) Simulating did not mutate the platform model."""
    check = Check("i", "hardware read-only", True)
    after = json.dumps(hardware.raw, sort_keys=True, default=str)
    if after != before:
        check.fail(
            "the platform model changed during simulation. A cost model has written to "
            "ctx.hardware; it is shared across every shard and workload, so a mutation "
            "makes results depend on evaluation order"
        )
    return check


# ---------------------------------------------------------------------------------------
def run(project: Path, artifact: Path, systems_dir: Path | None = None,
        target: str | None = None) -> Report:
    """Run the whole suite against a project's simulator."""
    started = time.monotonic()

    from floorplan import baseline as baseline_module

    modules, _graph = load_graph(artifact)
    config = load_model_config(artifact)

    plan_path = project / "floorplan.yaml"
    if plan_path.exists():
        plan = Floorplan.load(plan_path)
        chosen = plan.target
    else:
        chosen = target or "trn2-16device"
        plan = None
    system = load_system(chosen, systems_dir)
    hardware = Hardware.from_system(system)
    if plan is None:
        plan = baseline_module.build(modules, hardware)

    unresolved = hardware.unresolved()
    if unresolved:
        return Report(
            passed=False,
            checks=[Check("a", "models present", False, [
                f"target '{chosen}' has {len(unresolved)} unmeasured value(s) "
                f"({', '.join(unresolved[:4])}...). Run `autohelix floorplan probe` first"
            ])],
            seconds=time.monotonic() - started,
        )

    try:
        load_cost_models(project)
    except SimulationError as exc:
        # The state every build run starts in: `sim/modules/` is empty, so there is nothing to
        # import. That is the suite's *answer*, not a crash — the build loop's reviewer reads
        # this report to decide what iteration 2 should do, and a traceback tells it nothing.
        return Report(
            passed=False,
            checks=[Check("a", "models present", False, [
                str(exc),
                "Write one cost model per module archetype under sim/modules/, each calling "
                "floorplan.sim.api.register(). Until one exists, nothing else can be checked.",
            ])],
            seconds=time.monotonic() - started,
        )

    order = topological_order(modules)
    before = json.dumps(hardware.raw, sort_keys=True, default=str)

    results: dict[str, Any] | None = {}
    error = ""
    try:
        for workload in WORKLOADS:
            results[workload.name] = _simulate(
                plan, hardware, modules, config, workload.name, order,
            )
    except (SimulationError, api.CostModelError) as exc:
        results, error = None, f"{type(exc).__name__}: {exc}"

    checks = [
        check_models(modules),
        check_baseline(results, error),
        check_determinism(plan, hardware, modules, config, order, results),
        check_no_free_modules(plan, results),
        (check_split_scaling(plan, hardware, modules, config, order) if results
         else Check("e", "splits scale", False, ["not checked: the baseline did not simulate"])),
        check_context_scaling(results),
        check_constraints_covered(project, hardware),
        check_costs_derived(project),
        check_hardware_readonly(hardware, before),
    ]
    checks.sort(key=lambda c: c.check)
    metrics = (
        {f"{w.name}_ms": results[w.name].milliseconds for w in WORKLOADS} if results else {}
    )
    return Report(
        passed=all(c.passed for c in checks),
        checks=checks,
        metrics=metrics,
        seconds=time.monotonic() - started,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m floorplan.invariants",
        description="Check that a project's simulator is self-consistent.",
    )
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--systems", type=Path, default=None)
    parser.add_argument("--target", default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    report = run(args.project.resolve(), args.artifact.resolve(), args.systems, args.target)
    for check in report.checks:
        print(f"  [{check.check}] {'PASS' if check.passed else 'FAIL'}  {check.title}")
        for finding in check.findings:
            for index, line in enumerate(str(finding).splitlines()):
                print(f"          {'- ' if index == 0 else '  '}{line}")
    print()
    print(f"  {'PASS' if report.passed else 'FAIL'} in {report.seconds:.1f}s")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report.to_dict(), indent=2))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
