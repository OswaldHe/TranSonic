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
  b  baseline simulates    the baseline runs and publishes a positive metric per point
  c  deterministic         two identical runs produce identical metrics
  d  no free modules       every placed module contributes measurable work
  e  splits scale          doubling a split halves per-shard compute and adds communication
  f  context costs         longer contexts cost more, and superlinearly in prefill
  g  constraints covered   every numbered item in constraints_text is implemented
  h  costs are derived     no cost model fabricates a duration or reads the clock
  i  hardware read-only    simulating does not mutate the platform model
  j  batch costs more      a larger batch is slower, and prefill scales with it
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
from floorplan.schema import BATCH_SIZES, CONTEXT_LENGTHS, Floorplan, Split
from floorplan.sim import api
from floorplan.sim.runner import (
    HEAVIEST,
    WORKLOADS,
    SimulationError,
    deployable_modules,
    load_cost_models,
    load_graph,
    load_model_config,
    simulate_workload,
    topological_order,
    workload,
)

#: Constructs a cost model may not contain. The interface hands out seconds through
#: `matmul_seconds`, `elementwise_seconds`, `dma_seconds`, `gather_seconds` and
#: `collective_cost`; a model that reaches past them is either fabricating a duration or
#: making itself non-deterministic, and either one quietly invalidates the run.
#: Anchored at a statement position, because an unanchored `\bimport\s+time\b` matches the
#: English phrase "at import time" — which appears in a docstring and made this check fail on
#: prose. A banned-construct scanner that fires on comments is worse than none: it trains the
#: next iteration to work around a phantom.
BANNED_PATTERNS: tuple[tuple[str, str], str] | tuple = (
    (r"^\s*(?:import\s+time\b|from\s+time\s+import)|\btime\s*\.\s*(time|perf_counter|monotonic)\s*\(",
     "reads the clock — a cost model must be a pure function of shapes"),
    (r"^\s*(?:import\s+random\b|from\s+random\s+import)|\brandom\s*\.\s*\w+\s*\(",
     "uses randomness — runs must be reproducible"),
    (r"\bnumpy\s*\.\s*random\b|\bnp\s*\.\s*random\b",
     "uses randomness — runs must be reproducible"),
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
    """Run one named workload.

    Goes through ``runner.workload`` so a stale name fails with the name in the message. It
    used to be ``next(w for w in WORKLOADS if ...)``, which raises a bare ``StopIteration`` —
    and when the workloads gained the batch suffix, two checks here kept their old names and
    the whole suite crashed with no indication of which string was wrong. A build iteration was
    recorded as failing for that reason rather than for anything the agent had done.
    """
    return simulate_workload(
        plan, hardware, modules, config, workload(workload_name), order,
    )


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
    """(b) The generated baseline runs and publishes a positive metric at every point."""
    check = Check("b", "baseline simulates", True)
    if results is None:
        return check.fail(f"the baseline did not simulate: {error}")
    for point in WORKLOADS:
        result = results.get(point.name)
        if result is None:
            check.fail(f"{point.name} produced no result")
        elif result.seconds <= 0:
            check.fail(f"{point.name} took {result.seconds}s; a latency must be positive")
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
    for point in WORKLOADS:
        again = _simulate(plan, hardware, modules, config, point.name, order)
        before, after = results[point.name].seconds, again.seconds
        if before != after:
            check.fail(
                f"{point.name} changed between identical runs: "
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
    worst = results[HEAVIEST]      # the longest, largest point: everything runs there
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
        result = _simulate(variant, hardware, modules, config, HEAVIEST, order)
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
    """(f) A longer context costs the attention modules more, superlinearly in prefill.

    Measured on the attention modules' own busy time, not on the end-to-end latency. The
    end-to-end comparison was wrong and failed a correct simulator: on a 16-stage baseline a
    128-token prefill is almost entirely pipeline fill — 16 stage traversals for one chunk — so
    it is expensive for reasons that have nothing to do with context, and the 8192-token case
    came out only 3.9x more expensive for 64x the tokens. That measures the baseline's pipeline
    geometry, not whether the cost models read ``workload.context_tokens``.

    Attention busy time isolates the thing under test: it is where the quadratic term lives, it
    is summed across units so parallel execution does not hide it, and it is unaffected by how
    many stages the plan happens to use.
    """
    check = Check("f", "context costs", True)
    if results is None:
        return check.fail("not checked: the baseline did not simulate")
    short, long = min(CONTEXT_LENGTHS), max(CONTEXT_LENGTHS)
    batch = min(BATCH_SIZES)

    def attention_seconds(name: str) -> float:
        return sum(
            seconds for module, seconds in results[name].trace.by_module().items()
            if "attention" in module
        )

    short_prefill = attention_seconds(f"prefill_{short}_b{batch}")
    long_prefill = attention_seconds(f"prefill_{long}_b{batch}")
    short_decode = attention_seconds(f"decode_{short}_b{batch}")
    long_decode = attention_seconds(f"decode_{long}_b{batch}")
    tokens_ratio = long / short

    if not short_prefill or not short_decode:
        return check.fail(
            "the attention modules used no time at the short context, so there is nothing to "
            "compare — either no module id contains 'attention' or its cost model emits nothing"
        )

    if long_prefill <= short_prefill:
        check.fail(
            f"attention at {long} tokens ({long_prefill * 1e3:.3f}ms) is not slower than at "
            f"{short} ({short_prefill * 1e3:.3f}ms) in prefill"
        )
    elif long_prefill < short_prefill * tokens_ratio * 0.5:
        check.fail(
            f"attention in prefill is only {long_prefill / short_prefill:.1f}x from {short} to "
            f"{long} tokens, against {tokens_ratio:.0f}x the tokens. Prefill attention is at "
            f"least linear in tokens and quadratic in context, so the cost models are not "
            f"reading workload.context_tokens"
        )
    if long_decode <= short_decode:
        check.fail(
            f"attention at {long} tokens ({long_decode * 1e3:.3f}ms) is not slower than at "
            f"{short} ({short_decode * 1e3:.3f}ms) in decode — a longer KV cache costs more to "
            f"attend over"
        )
    if check.passed:
        check.findings.append(
            f"attention busy time {short}->{long} tokens: "
            f"prefill {long_prefill / short_prefill:.1f}x, decode {long_decode / short_decode:.2f}x"
        )
    return check


def check_batch_scaling(results: dict[str, Any] | None) -> Check:
    """(j) A larger batch costs more, and the two phases respond to it differently.

    Batch is an axis precisely because the answers reverse along it, so a cost model blind to
    ``ctx.workload.batch`` would make every point in a batch column identical and the four
    batch sizes would be four copies of one measurement. That is worse than not having the
    axis: it would look like evidence that batch does not matter.

    Prefill is required to scale roughly with batch — 32x the samples is 32x the tokens, so a
    sublinear result means the model is ignoring batch somewhere. Decode is only required to be
    monotone, because per-step decode work is dominated by weight movement that is amortized
    across the batch, so its growth is genuinely much slower than linear.
    """
    check = Check("j", "batch costs more", True)
    if results is None:
        return check.fail("not checked: the baseline did not simulate")

    for phase, context in (("prefill", max(CONTEXT_LENGTHS)), ("decode", max(CONTEXT_LENGTHS))):
        series = [
            (batch, results[f"{phase}_{context}_b{batch}"].seconds)
            for batch in BATCH_SIZES
            if f"{phase}_{context}_b{batch}" in results
        ]
        if len(series) < 2:
            check.fail(f"{phase}_{context}: fewer than two batch sizes produced a result")
            continue
        for (small, fast), (large, slow) in zip(series, series[1:]):
            if slow <= fast:
                check.fail(
                    f"{phase}_{context}: batch {large} ({slow * 1e3:.3f}ms) is not slower than "
                    f"batch {small} ({fast * 1e3:.3f}ms). The cost models are insensitive to "
                    f"batch, so the four batch columns are four copies of one measurement"
                )
                break
        else:
            smallest, largest = series[0], series[-1]
            ratio = largest[1] / smallest[1] if smallest[1] else 0.0
            if phase == "prefill":
                expected = largest[0] / smallest[0]
                if ratio < expected * 0.5:
                    check.fail(
                        f"prefill_{context}: batch {largest[0]} is only {ratio:.1f}x batch "
                        f"{smallest[0]} for {expected:.0f}x the samples. Prefill work is "
                        f"proportional to tokens, so this is sublinear by too much to be "
                        f"pipeline fill"
                    )
            check.findings.append(
                f"{phase}_{context}: batch {smallest[0]}->{largest[0]} costs {ratio:.2f}x"
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


def agent_written_sources(project: Path) -> list[Path]:
    """The files the agent writes, and only those.

    `sim/framework/` holds read-only copies of the framework, and scanning them polices the
    wrong code: the framework may legitimately do things a cost model may not, and it is frozen
    anyway. The first version globbed all of `sim/`, so the reference copy of `api.py` failed the
    banned-construct check on a docstring phrase.
    """
    sources = sorted((project / "sim" / "modules").glob("*.py"))
    constraints = project / "sim" / "constraints.py"
    if constraints.exists():
        sources.append(constraints)
    return [path for path in sources if path.name != "__init__.py"]


def check_costs_derived(project: Path) -> Check:
    """(h) No cost model fabricates a duration or reads the clock."""
    check = Check("h", "costs are derived", True)
    paths = agent_written_sources(project)
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


def hardware_fingerprint(hardware: Hardware) -> str:
    """Everything a cost model could mutate and thereby change a later cost.

    Not just ``raw``. ``Hardware.from_system`` builds fresh mutable mappings — ``efficiency``,
    ``compute``, ``tiers`` — and writing to one of those through ``ctx.hardware`` leaves the
    source YAML untouched. A model could raise ``matmul_bf16`` mid-run, speed up every shard
    evaluated afterwards, and pass the check meant to catch exactly that.
    """
    return json.dumps(
        {
            "raw": hardware.raw,
            "efficiency": hardware.efficiency,
            "compute": hardware.compute,
            "tiers": {name: vars(tier) for name, tier in hardware.tiers.items()},
            "links": [
                hardware.intra_device_bandwidth_bytes_per_s,
                hardware.intra_device_latency_us,
                hardware.inter_device_bandwidth_bytes_per_s,
                hardware.inter_device_hop_latency_us,
            ],
            "topology": [hardware.topology_kind, hardware.topology_shape, hardware.topology_wrap],
        },
        sort_keys=True, default=str,
    )


def check_hardware_readonly(hardware: Hardware, before: str) -> Check:
    """(i) Simulating did not mutate the platform model."""
    check = Check("i", "hardware read-only", True)
    if hardware_fingerprint(hardware) != before:
        check.fail(
            "the platform model changed during simulation. A cost model has written to "
            "ctx.hardware — its efficiency coefficients, compute rates, tiers or links — and "
            "that object is shared across every shard and workload, so a mutation makes "
            "results depend on evaluation order"
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
    before = hardware_fingerprint(hardware)

    results: dict[str, Any] | None = {}
    error = ""
    try:
        for point in WORKLOADS:
            results[point.name] = _simulate(
                plan, hardware, modules, config, point.name, order,
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
        check_batch_scaling(results),
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
