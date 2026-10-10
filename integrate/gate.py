# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Running every constraint once, and writing down which hold.

This is the one thing that decides whether an integration is done. It runs each declared
constraint — script and agent alike — in the order the config lists them, writes a verdict JSON,
and reports the count as the loop's metric.

Script constraints go through `autohelix.checks.run_constraint`, not a local subprocess call.
That is where the behaviour an operator actually needs already lives: it kills the whole process
group on timeout, so a timed-out pytest does not keep running detached, and it builds the same
worktree environment every other autohelix check runs under. Agent constraints go through
`integrate.agentcheck`.

**It exits 0 even when constraints fail**, under `--advisory`, and that is the central
difference from `optimization`'s gate. There, a failing gate means a broken candidate and
rejecting it is right. Here a failing constraint is the *normal* state for most of the run — it
is what the loop exists to fix — so a nonzero exit would reject every iteration and discard its
progress, and the loop could only succeed by accident on a single iteration. What prevents a
regression instead is the metric: `constraints_passing` can never go down, enforced by the
loop's acceptance rule.

Without `--advisory` it exits 1 when anything fails, which is what an operator wants by hand or
from CI.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from integrate.config import GATE_JSON, METRIC, Constraint, IntegrateConfig


@dataclass
class ConstraintResult:
    """What running one constraint produced."""

    name: str
    kind: str
    passed: bool
    detail: str = ""
    findings: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    returncode: int | None = None
    seconds: float = 0.0
    skipped: bool = False
    output: str = ""

    def line(self) -> str:
        if self.skipped:
            return f"  [skip] {self.name}"
        mark = "PASS" if self.passed else "FAIL"
        return f"  [{mark}] {self.name} ({self.kind}, {self.seconds:.1f}s) {self.detail}".rstrip()


def run_script_constraint(constraint: Constraint, repo: Path) -> ConstraintResult:
    """Run a script constraint through autohelix's own constraint runner."""
    from autohelix.checks import run_constraint as autohelix_run_constraint

    started = time.monotonic()
    result = autohelix_run_constraint(
        constraint.command,
        cwd=repo,
        timeout=constraint.timeout,
        # Gives the command the same worktree environment every other autohelix check gets.
        project_path=repo,
    )
    seconds = time.monotonic() - started
    output = (result.output or "").strip()
    # The tail, not the head: a failing command's reason is almost always at the end.
    detail = "" if result.passed else (output.splitlines()[-1][:200] if output else "no output")
    return ConstraintResult(
        name=constraint.name, kind="script", passed=result.passed, detail=detail,
        findings=[] if result.passed else [detail],
        returncode=result.return_code, seconds=seconds,
        # Bounded: this lands in a JSON the loop reads back every iteration.
        output=output[-4000:],
    )


def run_one(constraint: Constraint, repo: Path, agent: dict | None = None) -> ConstraintResult:
    """Run one constraint of either kind."""
    if constraint.kind == "agent":
        from integrate.agentcheck import run_agent_constraint

        verdict = run_agent_constraint(constraint, repo, agent=agent)
        return ConstraintResult(
            name=verdict.name, kind="agent", passed=verdict.passed,
            detail=(verdict.error or verdict.reason)[:200],
            findings=verdict.findings, evidence=verdict.evidence,
            seconds=verdict.seconds,
        )
    return run_script_constraint(constraint, repo)


def evaluate(
    config: IntegrateConfig, repo: Path, stop_early: bool = False
) -> tuple[list[ConstraintResult], dict]:
    """Run every constraint and return the results plus the verdict payload."""
    results: list[ConstraintResult] = []
    stopped = False
    for constraint in config.constraints:
        if stopped:
            results.append(
                ConstraintResult(
                    name=constraint.name, kind=constraint.kind, passed=False, skipped=True,
                    detail="not run: an earlier constraint failed under --stop-early",
                )
            )
            continue
        result = run_one(constraint, repo, agent=config.agent)
        results.append(result)
        if stop_early and not result.passed:
            stopped = True

    passing = sum(1 for r in results if r.passed)
    payload = {
        "target_id": config.target_id,
        "repo": str(repo),
        METRIC: passing,
        "constraints_total": len(results),
        "satisfied": passing == len(results) and len(results) > 0,
        "stopped_early": stopped,
        "constraints": [asdict(r) for r in results],
    }
    return results, payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="the integrate config")
    parser.add_argument("--json", required=True, help="where to write the verdict")
    parser.add_argument("--repo", default=None,
                        help="what to check (default: the working directory, which is the "
                             "iteration's worktree when the loop invokes this)")
    parser.add_argument("--advisory", action="store_true",
                        help="exit 0 even when constraints fail. How the loop runs it: a failing "
                             "constraint is the normal state, and rejecting on it would discard "
                             "the iteration's progress.")
    parser.add_argument("--stop-early", action="store_true",
                        help="stop at the first failure instead of running them all. Makes the "
                             "count a lower bound, so not for the loop.")
    args = parser.parse_args(argv)

    config = IntegrateConfig.load(Path(args.config).parent, args.config)
    # Default to cwd, not config.repo: the loop runs this inside a worktree, and checking the
    # operator's repo from there would judge the wrong tree entirely.
    repo = Path(args.repo).resolve() if args.repo else Path.cwd()

    results, payload = evaluate(config, repo, stop_early=args.stop_early)

    serialized = json.dumps(payload, indent=2) + "\n"
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(serialized)

    # Mirror it into the project's own state dir. The loop invokes this with the iteration's
    # worktree as the working directory, so `--json` lands inside a tree that is discarded when
    # the iteration ends — and the loop's termination check and the next iteration's prompt both
    # need the verdict that was just produced, not the one from whenever the repo was last
    # checked by hand. Safe to write: `.autohelix/` is gitignored, so this never dirties the
    # tree, and the gate is pipeline-owned rather than agent-owned.
    mirror = (config.project_path / GATE_JSON).resolve()
    if mirror != out.resolve():
        mirror.parent.mkdir(parents=True, exist_ok=True)
        mirror.write_text(serialized)

    passing, total = payload[METRIC], payload["constraints_total"]
    print(f"constraints: {passing}/{total} satisfied  (repo: {repo})")
    for result in results:
        print(result.line())
    for result in results:
        for item in result.evidence[:2]:
            print(f"    evidence [{result.name}]: {item}")
    print(f"\n##autohelix[{METRIC}={passing}]")
    print(f"##autohelix[constraints_total={total}]")
    if payload["satisfied"]:
        print("##autohelix[satisfied=1]")
        print("\nevery constraint is satisfied.")
    else:
        failing = [r.name for r in results if not r.passed]
        print("##autohelix[satisfied=0]")
        print(f"\nstill failing: {', '.join(failing)}")

    if args.advisory:
        return 0
    return 0 if payload["satisfied"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
