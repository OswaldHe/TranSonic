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
regression instead is the metric: `constraints_passing` — the *script* constraints that hold —
can never go down, enforced by the loop's acceptance rule. The verdict counts the agent
constraints separately, in `all_constraints_passing`, which is reported and ranked on but not
gated: a judge can reach a different verdict over an unchanged tree, and that is not something
to discard an iteration's work for. `satisfied` still means every constraint, both kinds.

Without `--advisory` it exits 1 when anything fails, which is what an operator wants by hand or
from CI.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from integrate.config import GATE_JSON, METRIC, METRIC_ALL, Constraint, IntegrateConfig


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
    #: Any `##autohelix[name=value]` this constraint printed, parsed from its *whole* output
    #: before `output` is truncated. What makes a measurement a constraint already produces
    #: usable as the loop's metric: `integrate run` reads these back rather than running a
    #: benchmark command of its own, so a verifier that takes an hour runs once an iteration
    #: instead of twice. See `integrate/readback.py`.
    metrics: dict[str, float] = field(default_factory=dict)

    def line(self) -> str:
        if self.skipped:
            return f"  [skip] {self.name}"
        mark = "PASS" if self.passed else "FAIL"
        return f"  [{mark}] {self.name} ({self.kind}, {self.seconds:.1f}s) {self.detail}".rstrip()


def parse_structured_metrics(output: str) -> dict[str, float]:
    """Any `##autohelix[name=value]` in a constraint's output.

    AutoHelix's own parser, so the protocol a constraint prints is the one every other part of
    the tool reads — a second implementation here would be a second set of accepted spellings.
    """
    from autohelix.checks import parse_structured_metrics as parse

    return parse(output or "")


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
        # Parsed from the *whole* output, above the truncation. A constraint that prints its
        # metrics early — anything more than 4000 characters before it exits — would otherwise
        # have them silently cut off, and a metric the loop gates on must not depend on how
        # chatty the command was afterwards.
        metrics=parse_structured_metrics(output),
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


def _collect_metrics(results: list[ConstraintResult]) -> dict[str, float]:
    """Every constraint's parsed metrics in one dict, in the order they were run.

    A later constraint printing a name an earlier one already printed wins, which is the only
    resolution that does not need a rule: the operator sees both in `metric_sources` and can
    rename one. Skipped constraints contribute nothing, so a `--stop-early` run reports fewer
    metrics rather than stale ones.
    """
    out: dict[str, float] = {}
    for result in results:
        if not result.skipped:
            out.update(result.metrics)
    return out


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
    script = [r for r in results if r.kind == "script"]
    payload = {
        "target_id": config.target_id,
        "repo": str(repo),
        # Every measurement the constraints printed, flattened, with the constraint that produced
        # each one recorded beside it. Flattened because a metric name is the loop's handle on a
        # number and two constraints printing the same name is a config mistake worth seeing, not
        # a namespace to resolve.
        "metrics": _collect_metrics(results),
        "metric_sources": {
            name: r.name for r in results for name in r.metrics
        },
        # The gated count and its denominator: script constraints only, deterministic, so a drop
        # is the iteration's doing. See `METRIC`.
        METRIC: sum(1 for r in script if r.passed),
        "script_constraints_total": len(script),
        # The whole bar, which is what `satisfied` and the loop's exit condition are about.
        METRIC_ALL: passing,
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

    passing, total = payload[METRIC_ALL], payload["constraints_total"]
    script, script_total = payload[METRIC], payload["script_constraints_total"]
    print(f"constraints: {passing}/{total} satisfied  (repo: {repo})")
    print(f"             {script}/{script_total} of them script, which is the gated count")
    for result in results:
        print(result.line())
    for result in results:
        for item in result.evidence[:2]:
            print(f"    evidence [{result.name}]: {item}")
    print(f"\n##autohelix[{METRIC}={script}]")
    print(f"##autohelix[{METRIC_ALL}={passing}]")
    print(f"##autohelix[script_constraints_total={script_total}]")
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
