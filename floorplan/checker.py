# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The gate the exploration loop runs, and the agent never sees.

Seven checks, all of which must pass. Like `bootstrap/nki_checker.py`, this file stays in
the package and is named only by `preset.yaml`'s constraint command, so no copy of it
reaches the iteration worktree. That makes the preset's `goal` and this module a pair:
anything enforced here and unstated there is a trap rather than a requirement, and
`tests/test_floorplan_checker.py` fails if they drift.

The checks exist in a particular order because a failure early makes a later one
unknowable. A plan that does not parse cannot be checked for coverage; a plan whose
simulator has been edited cannot be trusted to report capacity honestly.

  a  plan well-formed     parses, and is legal for this hardware and this model
  b  coverage             every module on the tokens -> logits path is placed, fractions to 1
  c  dependencies         every placed module's producers are placed too
  d  frozen platform      sim/ and systems/ are byte-identical to the build manifest
  e  capacity             no memory tier over capacity, at any point of the workload grid
  f  simulates            every workload completes and publishes a metric
  g  deterministic        a second run produces identical metrics

Check (d) is the load-bearing one for honesty. The exploration loop's editable scope is
`floorplan.yaml` alone, and AutoHelix reverts out-of-scope edits before constraints run —
but reverting depends on git noticing, so (d) verifies the bytes directly against hashes
recorded when the simulator was frozen. A cost model quietly made cheaper is the one change
that would make every metric in the run meaningless.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

MANIFEST_RELATIVE = Path(".autohelix") / "floorplan" / "manifest.json"

#: Paths whose bytes are frozen once `floorplan build` finishes, relative to the project.
#:
#: ``systems`` is here as well as ``sim`` and it matters just as much. The cost models are the
#: obvious thing to make cheaper, but `systems/probed.yaml` holds the achieved-efficiency
#: coefficients every cost model divides by — raising `matmul_bf16` from 0.30 to 0.90 would
#: make every plan three times faster without touching a line of `sim/`. Both are the
#: simulator; only `floorplan.yaml` is the search.
FROZEN_TREES = ("sim", "systems")

#: The only file an iteration may change.
EDITABLE = ("floorplan.yaml",)


@dataclass
class Check:
    """One verdict."""

    check: str
    title: str
    passed: bool
    findings: list[str] = field(default_factory=list)

    def fail(self, message: str) -> "Check":
        self.passed = False
        self.findings.append(message)
        return self


@dataclass
class Report:
    passed: bool
    checks: list[Check]
    metrics: dict[str, float] = field(default_factory=dict)
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": [asdict(c) for c in self.checks],
            "metrics": self.metrics,
            "seconds": round(self.seconds, 2),
        }


def find_manifest(start: Path) -> Path:
    """Walk up from ``start`` to the manifest.

    An iteration worktree lives at ``<project>/.autohelix/worktrees/iter-N``, inside the
    project, so the manifest is always above it. That is what lets the constraint command in
    `preset.yaml` be a fixed string with no per-project path in it.
    """
    current = start.resolve()
    for candidate in [current, *current.parents]:
        manifest = candidate / MANIFEST_RELATIVE
        if manifest.exists():
            return manifest
    raise FileNotFoundError(
        f"no {MANIFEST_RELATIVE} at or above {start}. `floorplan init` writes it"
    )


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_tree(root: Path, trees: tuple[str, ...] = FROZEN_TREES) -> dict[str, str]:
    """``{relative path: sha256}`` for every ``.py`` and ``.yaml`` under ``trees``.

    Sorted, so the mapping is stable. ``__pycache__`` is skipped: it is generated, it
    differs between interpreters, and freezing it would fail the check for no reason.
    """
    out: dict[str, str] = {}
    for tree in trees:
        base = root / tree
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.suffix not in {".py", ".yaml", ".yml", ".json", ".md"}:
                continue
            out[str(path.relative_to(root))] = hash_file(path)
    return out


def framework_paths() -> list[Path]:
    """The installed simulator framework modules — the code that actually runs.

    This exists because freezing the project's files alone was not the promise it looked like.
    The project holds the agent-written cost models, but `run_simulation` invokes
    ``python -m floorplan.sim.runner``, which imports the framework from the *installed
    package*. Hashing only the project meant check (d) could pass while the code producing the
    metrics had changed underneath it — an upgrade, a local edit, a different venv — and the
    reproducibility and tamper-protection the gate advertises would not have held.
    """
    from floorplan import parser as parser_module
    from floorplan.sim import api, collectives, engine, memory, runner

    return sorted(
        Path(module.__file__).resolve()
        for module in (engine, collectives, memory, api, runner, parser_module)
        if getattr(module, "__file__", None)
    )


def hash_framework() -> dict[str, str]:
    """``{absolute path: sha256}`` for the installed framework."""
    return {str(path): hash_file(path) for path in framework_paths()}


# ---------------------------------------------------------------------------------------
# The checks
# ---------------------------------------------------------------------------------------
def check_plan(repo: Path, manifest: dict[str, Any]) -> tuple[Check, Any, Any, Any]:
    """(a) The floorplan parses and is legal for this hardware and this model."""
    from floorplan.parser import load_system
    from floorplan.schema import Floorplan, FloorplanError
    from floorplan.sim.runner import SimulationError, load_graph

    check = Check("a", "plan well-formed", True)
    plan = system = modules = None
    plan_path = repo / EDITABLE[0]
    if not plan_path.exists():
        return (check.fail(f"{EDITABLE[0]} is missing"), None, None, None)
    try:
        plan = Floorplan.load(plan_path)
    except FloorplanError as exc:
        return (check.fail(f"{EDITABLE[0]}: {exc}"), None, None, None)

    systems_dir = Path(manifest["systems_dir"]) if manifest.get("systems_dir") else None
    try:
        system = load_system(plan.target, systems_dir)
    except Exception as exc:
        return (check.fail(f"target '{plan.target}': {exc}"), plan, None, None)

    expected = manifest.get("target")
    if expected and plan.target != expected:
        check.fail(
            f"target is '{plan.target}' but this project was built for '{expected}'. "
            f"The cost models were calibrated against that platform"
        )

    try:
        modules, _ = load_graph(Path(manifest["artifact"]))
    except SimulationError as exc:
        return (check.fail(str(exc)), plan, system, None)

    try:
        plan.validate_against(system, modules)
    except FloorplanError as exc:
        check.fail(str(exc))
    return (check, plan, system, modules)


def check_coverage(plan: Any, modules: Any) -> Check:
    """(b) Every module on the inference path is deployed, and wholly."""
    from floorplan.sim.runner import deployable_modules

    check = Check("b", "coverage", True)
    if plan is None or modules is None:
        return check.fail("not checked: the plan or the graph did not load")

    required = deployable_modules(modules)
    placed = plan.modules()
    missing = sorted(required - placed)
    if missing:
        check.fail(
            f"{len(missing)} module(s) not deployed: {', '.join(missing[:10])}"
            f"{' ...' if len(missing) > 10 else ''}"
        )
    unknown = sorted(placed - set(modules))
    if unknown:
        check.fail(f"placed but not in the graph: {', '.join(unknown[:10])}")
    excluded = sorted(placed & (set(modules) - required))
    if excluded:
        check.fail(
            f"placed but off the tokens -> logits path: {', '.join(excluded[:10])}. "
            f"The vision tower is out of scope"
        )

    totals: dict[str, float] = {}
    for placement in plan.placements:
        totals[placement.module] = totals.get(placement.module, 0.0) + placement.fraction
    partial = sorted(m for m, total in totals.items() if abs(total - 1.0) > 1e-6)
    if partial:
        detail = ", ".join(f"{m} ({totals[m]:g})" for m in partial[:6])
        check.fail(f"fractions do not sum to 1 for: {detail}")
    return check


def check_dependencies(plan: Any, modules: Any) -> Check:
    """(c) Every placed module's producers are placed, and the graph still orders."""
    from floorplan.sim.runner import SimulationError, topological_order

    check = Check("c", "dependencies", True)
    if plan is None or modules is None:
        return check.fail("not checked: the plan or the graph did not load")

    produced_by = {
        str(tensor): module_id
        for module_id, entry in modules.items()
        for tensor in (entry.get("outputs") or [])
    }
    placed = plan.modules()
    for module_id in sorted(placed):
        entry = modules.get(module_id)
        if entry is None:
            continue
        for tensor in entry.get("inputs") or []:
            producer = produced_by.get(str(tensor))
            if producer and producer != module_id and producer not in placed:
                check.fail(
                    f"{module_id} consumes '{tensor}', produced by {producer}, "
                    f"which is not placed"
                )
    try:
        topological_order(modules)
    except SimulationError as exc:
        check.fail(str(exc))
    return check


def check_frozen(repo: Path, manifest: dict[str, Any]) -> Check:
    """(d) The simulator and the platform description are byte-identical to the build.

    Two sets, because the simulator lives in two places: the agent-written cost models and the
    platform YAML inside the project, and the framework inside the installed package. Verifying
    only the first would leave the code that computes every metric unchecked.
    """
    check = Check("d", "frozen platform", True)
    recorded = manifest.get("hashes") or {}
    if not recorded:
        return check.fail(
            "the manifest records no hashes; the platform was never frozen. "
            "`floorplan build` writes them when it finishes"
        )
    current = hash_tree(repo)
    for relative, digest in sorted(recorded.items()):
        actual = current.get(relative)
        if actual is None:
            check.fail(f"{relative} is missing — it was frozen at build time")
        elif actual != digest:
            check.fail(
                f"{relative} has been modified. The simulator is frozen during "
                f"exploration; only {EDITABLE[0]} is editable"
            )
    for relative in sorted(set(current) - set(recorded)):
        check.fail(f"{relative} was added after the platform was frozen")

    framework = manifest.get("framework_hashes") or {}
    if not framework:
        check.fail(
            "the manifest records no framework hashes, so the code that actually computes the "
            "metrics is unverified. Re-freeze with a current `floorplan build`"
        )
    else:
        live = hash_framework()
        for path, digest in sorted(framework.items()):
            actual = live.get(path)
            if actual is None:
                check.fail(
                    f"{path} is no longer part of the installed framework — the simulator "
                    f"that runs is not the one that was frozen"
                )
            elif actual != digest:
                check.fail(
                    f"{path} has changed since the freeze. This is the framework the metrics "
                    f"are computed by; results are not comparable to earlier iterations"
                )
        for path in sorted(set(live) - set(framework)):
            check.fail(f"{path} was added to the framework after the freeze")

        # And the project's reading copy must equal what executes. Without this the agent could
        # be reading one simulator while another computed every metric — the exact failure that
        # made copying the framework a bad idea in the first place.
        for path in framework_paths():
            mirror = repo / "sim" / "framework" / path.name
            if not mirror.exists():
                check.fail(
                    f"sim/framework/{path.name} is missing, so the framework you can read is "
                    f"not the framework that runs"
                )
            elif hash_file(mirror) != live.get(str(path)):
                check.fail(
                    f"sim/framework/{path.name} differs from the installed {path.name}. The "
                    f"reference copy and the executing code have diverged"
                )
    return check


def run_simulation(
    repo: Path, manifest: dict[str, Any], timeout: int, trace: Path | None,
) -> tuple[bool, dict[str, float], str]:
    """Run the simulator once as a subprocess. ``(ok, metrics, combined output)``.

    A subprocess rather than an in-process call, for two reasons: a cost model that loops
    forever is a timeout rather than a hung gate, and the exit status the loop's constraint
    reports is the simulator's own.
    """
    command = [
        sys.executable, "-m", "floorplan.sim.runner",
        "--plan", str(repo / EDITABLE[0]),
        "--artifact", str(manifest["artifact"]),
        "--project", str(repo),
    ]
    if manifest.get("systems_dir"):
        command += ["--systems", str(manifest["systems_dir"])]
    if trace is not None:
        command += ["--trace", str(trace)]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, cwd=repo,
        )
    except subprocess.TimeoutExpired:
        return (False, {}, f"the simulator did not finish within {timeout}s")
    output = (completed.stdout or "") + (completed.stderr or "")
    metrics: dict[str, float] = {}
    for line in (completed.stdout or "").splitlines():
        line = line.strip()
        if line.startswith("##autohelix[") and line.endswith("]"):
            body = line[len("##autohelix["):-1]
            name, _, value = body.partition("=")
            try:
                metrics[name.strip()] = float(value)
            except ValueError:
                continue
    return (completed.returncode == 0, metrics, output)


def check_simulates(
    repo: Path, manifest: dict[str, Any], timeout: int, trace: Path | None,
) -> tuple[Check, dict[str, float], str]:
    """(f) Every workload completes and publishes a metric."""
    from floorplan.sim.runner import WORKLOADS

    check = Check("f", "simulates", True)
    ok, metrics, output = run_simulation(repo, manifest, timeout, trace)
    if not ok:
        tail = "\n".join(output.strip().splitlines()[-12:])
        check.fail(f"the simulator exited non-zero:\n{tail}")
    expected = {f"{w.name}_ms" for w in WORKLOADS}
    missing = sorted(expected - set(metrics))
    if missing:
        check.fail(f"no metric published for: {', '.join(missing)}")
    for name, value in sorted(metrics.items()):
        if value <= 0:
            check.fail(f"{name} is {value}; a latency must be positive")
    return (check, metrics, output)


def check_capacity(output: str) -> Check:
    """(e) No tier over capacity, in any workload.

    Read from the simulator's own report rather than recomputed: the simulator refuses to
    produce a metric for a plan that does not fit, so a green (f) already implies this. It
    is a separate check because "does not fit" and "does not run" are different problems for
    whoever reads the verdict, and conflating them wastes an iteration.
    """
    check = Check("e", "capacity", True)
    if "does not fit" in output:
        for line in output.splitlines():
            if "exceeds" in line:
                check.fail(line.strip())
        if check.passed:
            check.fail("the simulator reported a capacity failure")
    return check


def check_determinism(
    repo: Path, manifest: dict[str, Any], timeout: int, first: dict[str, float],
) -> Check:
    """(g) A second run produces identical metrics.

    Cheap, and it catches the one class of bug that would invalidate a whole run silently:
    a cost model that reads the clock, iterates a set, or keeps state between workloads.
    Without this, a 3% "improvement" could be noise from dictionary ordering.
    """
    check = Check("g", "deterministic", True)
    if not first:
        return check.fail("not checked: the first run published no metrics")
    ok, second, _ = run_simulation(repo, manifest, timeout, None)
    if not ok:
        return check.fail("the second run failed while the first succeeded")
    for name in sorted(first):
        before, after = first[name], second.get(name)
        if after is None:
            check.fail(f"{name} was not published by the second run")
        elif before != after:
            check.fail(
                f"{name} changed between identical runs: {before:.6f} -> {after:.6f}. "
                f"A cost model is reading the clock, iterating an unordered collection, "
                f"or carrying state between workloads"
            )
    return check


# ---------------------------------------------------------------------------------------
def run(repo: Path, timeout: int = 1200, trace: Path | None = None) -> Report:
    started = time.monotonic()
    manifest_path = find_manifest(repo)
    manifest = json.loads(manifest_path.read_text())

    plan_check, plan, _system, modules = check_plan(repo, manifest)
    checks = [
        plan_check,
        check_coverage(plan, modules),
        check_dependencies(plan, modules),
        check_frozen(repo, manifest),
    ]
    metrics: dict[str, float] = {}
    if all(c.passed for c in checks):
        sim_check, metrics, output = check_simulates(repo, manifest, timeout, trace)
        checks.append(check_capacity(output))
        checks.append(sim_check)
        checks.append(check_determinism(repo, manifest, timeout, metrics))
    else:
        blocked = "not checked: an earlier check failed"
        checks.append(Check("e", "capacity", False, [blocked]))
        checks.append(Check("f", "simulates", False, [blocked]))
        checks.append(Check("g", "deterministic", False, [blocked]))

    checks.sort(key=lambda c: c.check)
    return Report(
        passed=all(c.passed for c in checks),
        checks=checks,
        metrics=metrics,
        seconds=time.monotonic() - started,
    )


def archive_candidate(repo: Path, report: Report) -> Path | None:
    """Copy the plan the gate just validated into the project's durable archive.

    Done here, from inside the iteration worktree, because this is the only moment the plan
    exists in a place that survives. AutoHelix deletes the worktree and its branch in a
    ``finally`` block, and a rejected iteration is never committed — so recovering it
    afterwards from git is impossible, and the first attempt to do so silently archived the
    final accepted plan for every iteration instead.

    Writes to the *project*, which the checker locates by walking up to the manifest, so a
    worktree's plan lands outside the worktree. Named by content hash: an iteration that
    changes nothing does not accumulate duplicates, and the same plan validated twice is one
    entry.
    """
    if not report.metrics:
        return None
    plan_path = repo / EDITABLE[0]
    if not plan_path.exists():
        return None
    try:
        project = find_manifest(repo).parent.parent.parent
    except FileNotFoundError:
        return None

    digest = hash_file(plan_path)[:12]
    destination = project / "schemes" / "candidates" / f"plan-{digest}.yaml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    header = ["# Captured by the gate after this plan passed every feasibility check.", "#"]
    for name, value in sorted(report.metrics.items()):
        header.append(f"#   {name}: {value:.4f}")
    header.append("#")
    header.append("# Whether the metric gate then accepted it is recorded in the sidecar JSON.")
    destination.write_text("\n".join(header) + "\n\n" + plan_path.read_text())
    destination.with_suffix(".json").write_text(json.dumps({
        "plan_sha256": hash_file(plan_path),
        "metrics": report.metrics,
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "passed_gate": report.passed,
    }, indent=2, sort_keys=True))
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m floorplan.checker",
        description="Gate one iteration of the floorplan exploration loop.",
    )
    parser.add_argument("--repo", type=Path, default=Path("."),
                        help="the candidate project (default: the working directory)")
    parser.add_argument("--json", type=Path, default=None, help="write the verdict here")
    parser.add_argument("--timeout", type=int, default=1200,
                        help="seconds one simulator run may take (default: 1200)")
    parser.add_argument("--trace", type=Path, default=None,
                        help="have the simulator write its trace here")
    parser.add_argument("--archive", action="store_true",
                        help="copy a plan that passes into the project's candidate archive, "
                             "before the iteration worktree is discarded")
    args = parser.parse_args(argv)

    try:
        report = run(args.repo.resolve(), args.timeout, args.trace)
    except FileNotFoundError as exc:
        print(f"floorplan checker: {exc}", file=sys.stderr)
        return 2

    if args.archive and report.passed:
        kept = archive_candidate(args.repo.resolve(), report)
        if kept is not None:
            print(f"  archived {kept}")

    for check in report.checks:
        mark = "PASS" if check.passed else "FAIL"
        print(f"  [{check.check}] {mark}  {check.title}")
        for finding in check.findings:
            for index, line in enumerate(str(finding).splitlines()):
                print(f"          {'- ' if index == 0 else '  '}{line}")
    if report.metrics:
        print()
        for name, value in sorted(report.metrics.items()):
            print(f"  {name:<18} {value:10.3f}")
    print()
    print(f"  {'PASS' if report.passed else 'FAIL'} in {report.seconds:.1f}s")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report.to_dict(), indent=2))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
