# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The two stages of `autohelix integrate`'s first pass.

`init` makes a repo ready and refuses to pretend it is when it is not. `run_baseline` drives the
loop. Everything either reads is in the config; nothing here takes a flag the config cannot
express, so a run is reproducible from the file.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from integrate.config import GATE_JSON, IntegrateConfig
from integrate.loop import STATE_DIR


class StageError(RuntimeError):
    """A stage could not proceed, with the reason in the message."""


@dataclass
class StageOutcome:
    """Whether a stage succeeded, and what to say about it."""

    ok: bool
    detail: str = ""


class IntegratePass:
    """`init` and `run-baseline`, over one config."""

    def __init__(
        self,
        config: IntegrateConfig,
        console: Console | None = None,
        verbose: bool = False,
    ) -> None:
        self.config = config
        self.console = console or Console()
        self.verbose = verbose

    # ------------------------------------------------------------------ stage: init

    def init(self) -> StageOutcome:
        """Validate the config, check the repo, and make the workspace.

        Deliberately strict about the repo. The loop works in git worktrees cut from it, so a
        dirty tree or an uncommitted editable file is not a warning — it is a run that will
        either fail at iteration 1 or, worse, silently measure the operator's uncommitted work
        as if an iteration had produced it.
        """
        config = self.config
        problems = config.validate()
        if problems:
            for problem in problems:
                self.console.print(f"  [red]FAIL[/red] {problem}")
            raise StageError(f"the config has {len(problems)} problem(s); fix them and re-run")

        self.console.print("  [green]OK[/green] the config is complete")
        for warning in config.warnings():
            self.console.print(f"  [yellow]note[/yellow] {warning}")

        repo_problems = self._check_repo()
        if repo_problems:
            for problem in repo_problems:
                self.console.print(f"  [red]FAIL[/red] {problem}")
            raise StageError("the repo is not ready; see above")
        self.console.print(f"  [green]OK[/green] {config.project_path} is a clean git repo")

        state = config.project_path / STATE_DIR
        state.mkdir(parents=True, exist_ok=True)
        (state / "notes").mkdir(exist_ok=True)
        loop_config = config.write_loop_config()

        self.console.print(f"  [green]OK[/green] workspace at {state}")
        self.console.print(f"  [green]OK[/green] derived loop config at {loop_config}")
        self._describe_constraints()
        if config.stage == "run":
            self._describe_metrics()
        stage_command = "run-baseline" if config.stage == "baseline" else "run"
        self.console.print(
            f"\n  Next: [cyan]autohelix integrate {stage_command}[/cyan]"
            f"{'' if config.source is None else f' --config {config.source}'}"
        )
        return StageOutcome(True, f"initialized {config.target_id}")

    def _check_repo(self) -> list[str]:
        """Everything about the repo that would stop the loop.

        Uses `Sandbox`'s own checks rather than a second `git status` parse. An earlier cut
        hand-rolled one and was *stricter* than the thing it gates: it rejected untracked files,
        which `Sandbox.ensure_clean_working_tree` deliberately ignores — so `init` refused to
        start over the config file it had just written, and over a `__pycache__` left by a
        constraint command. Checking with the same code the Harness will means `init` passing is
        a real statement that the loop can start.
        """
        from autohelix.sandbox import Sandbox

        repo = self.config.project_path
        if not (repo / ".git").exists():
            return [f"{repo} is not a git repository"]

        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True,
        )
        if head.returncode != 0:
            return [
                f"{repo} has no commits yet. The loop needs a HEAD to branch from — commit the "
                "starting state first."
            ]

        out: list[str] = []
        sandbox = Sandbox(repo)
        try:
            sandbox.ensure_clean_working_tree()
        except RuntimeError as exc:
            out.append(str(exc))
        try:
            sandbox.ensure_editable_files_ready(self.config.editable)
        except RuntimeError as exc:
            out.append(
                f"{exc}\n        Every path in `scope.editable` has to be tracked and committed; "
                "an empty file is fine."
            )
        return out

    def _describe_constraints(self) -> None:
        self.console.print(f"\n  {len(self.config.constraints)} constraint(s), in order:")
        for i, constraint in enumerate(self.config.constraints, start=1):
            self.console.print(f"    {i}. [cyan]{constraint.name}[/cyan]  ({constraint.kind})")
            if constraint.kind == "script":
                self.console.print(f"       `{constraint.command}`")
            else:
                model = constraint.model or "claude-sonnet-5"
                effort = constraint.effort or "medium"
                first = constraint.criteria.strip().splitlines()[0] if constraint.criteria else ""
                self.console.print(f"       judged by {model} (effort {effort}): {first}")

    def _describe_metrics(self) -> None:
        """What the run stage will measure, and what rejects an iteration."""
        config = self.config
        self.console.print(f"\n  {len(config.metric_names)} metric(s):")
        for measurement in config.measurements:
            source = (
                "read back from the constraints' own output" if measurement.from_constraints
                else f"`{measurement.command}`"
            )
            for name, direction in measurement.values.items():
                gated = (
                    f"rejects over {config.max_regression_pct:g}% regression"
                    if measurement.gate else "recorded, not gated"
                )
                self.console.print(
                    f"    [cyan]{name}[/cyan]  ({direction} is better, {gated})"
                )
            self.console.print(f"       {source}")
        from integrate.schedule import Schedule

        schedule = Schedule.from_config(
            config.iteration_constraints, max_iterations=config.iterations
        )
        if schedule.slots:
            self.console.print(f"\n  {len(schedule.slots)} iteration slot(s):")
            for slot in schedule.slots:
                first = (slot.text.strip().splitlines() or [""])[0]
                self.console.print(f"    {slot.label}: {first[:72]}")

    # ------------------------------------------------------------------ stage: run-baseline

    def run_baseline(self, max_iterations: int | None = None) -> StageOutcome:
        """Run the loop until every constraint holds."""
        from integrate.loop import BaselineLoop

        config = self.config
        problems = config.validate()
        if problems:
            for problem in problems:
                self.console.print(f"  [red]FAIL[/red] {problem}")
            raise StageError("the config has problems; run `autohelix integrate check`")
        if not (config.project_path / STATE_DIR).is_dir():
            raise StageError(
                f"no workspace at {config.project_path / STATE_DIR}. Run "
                "`autohelix integrate init` first."
            )

        loop = BaselineLoop(config, console=self.console, verbose=self.verbose)
        loop.run(max_iterations=max_iterations)

        if loop.satisfied():
            return StageOutcome(
                True, f"all {len(config.constraints)} constraint(s) satisfied"
            )
        failing = loop.failing_names()
        return StageOutcome(
            False,
            f"{len(failing)} constraint(s) still failing: {', '.join(failing) or 'unknown'}",
        )

    # ------------------------------------------------------------------ stage: run

    def run(self, max_iterations: int | None = None) -> StageOutcome:
        """Run the loop for its budget, improving the metrics while the constraints hold."""
        from integrate.runloop import RunLoop

        config = self.config
        problems = config.validate()
        if problems:
            for problem in problems:
                self.console.print(f"  [red]FAIL[/red] {problem}")
            raise StageError(
                "the config has problems; run `autohelix integrate check --stage run`"
            )
        for warning in config.warnings():
            self.console.print(f"  [yellow]note[/yellow] {warning}")

        state = config.project_path / STATE_DIR
        state.mkdir(parents=True, exist_ok=True)
        (state / "notes").mkdir(exist_ok=True)

        foreign = self._foreign_history()
        if foreign:
            raise StageError(foreign)

        loop = RunLoop(config, console=self.console, verbose=self.verbose)
        loop.run(max_iterations=max_iterations)

        summary = json.loads((config.project_path / STATE_DIR / "run-summary.json").read_text())
        improved = [
            f"{name} {entry['improvement_pct']:+.1f}%"
            for name, entry in (summary.get("metrics") or {}).items()
            if entry.get("improvement_pct") is not None
        ]
        failing = loop.failing_names()
        if failing:
            return StageOutcome(
                False,
                f"{len(failing)} constraint(s) failing at the end: {', '.join(failing)}"
                + (f"; {', '.join(improved)}" if improved else ""),
            )
        return StageOutcome(
            True,
            ", ".join(improved) if improved else "no metric moved",
        )

    def _foreign_history(self) -> str:
        """Why this stage cannot start on the history already in the repo, or "".

        The Harness keeps one history per project, so a `run` started in a repo that still holds
        the baseline stage's iterations would continue its numbering — iteration 14 of a
        13-iteration budget is already spent — and compare its metrics against records that have
        none. Upstream notices the config changed and offers to archive interactively, which is
        no use to a run started from a terminal nobody is watching. So it is refused here, with
        the command that fixes it.
        """
        history = self.config.project_path / ".autohelix" / "history.jsonl"
        if not history.is_file():
            return ""
        from autohelix.history import History

        records = [r for r in History(self.config.project_path).load() if r.iteration > 0]
        if not records:
            return ""
        names = set(self.config.metric_names)
        if any(names & set(r.metrics or {}) for r in records):
            return ""  # this stage's own history: resume
        return (
            f"{len(records)} iteration(s) already recorded in {history}, none of them carrying "
            f"this stage's metrics ({', '.join(sorted(names)) or 'none declared'}). That is "
            "another run's history — most likely `run-baseline`'s — and continuing it would "
            "count its iterations against this budget and judge these metrics against records "
            "that have none.\n"
            f"        Archive it first: `autohelix clear -p {self.config.project_path}` "
            "(it moves the state aside rather than deleting it), then start this stage."
        )

    # ------------------------------------------------------------------ one-shot gate

    def gate(self, stop_early: bool = False) -> StageOutcome:
        """Run every constraint once against the repo as it stands, and report."""
        from integrate import gate as gate_mod

        results, payload = gate_mod.evaluate(
            self.config, self.config.project_path, stop_early=stop_early
        )
        out = self.config.project_path / GATE_JSON
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n")

        for result in results:
            colour = "green" if result.passed else ("dim" if result.skipped else "red")
            mark = "PASS" if result.passed else ("skip" if result.skipped else "FAIL")
            self.console.print(
                f"  [{colour}]{mark}[/{colour}] {result.name} "
                f"({result.kind}, {result.seconds:.1f}s) {result.detail}"
            )
            for finding in result.findings[:3]:
                self.console.print(f"        {finding}")

        passing, total = payload["all_constraints_passing"], payload["constraints_total"]
        if payload["satisfied"]:
            self.console.print(f"\n  [green]{passing}/{total} — every constraint holds[/green]")
            return StageOutcome(True, f"{passing}/{total}")
        failing = [r.name for r in results if not r.passed]
        self.console.print(
            f"\n  [yellow]{passing}/{total}[/yellow] — still failing: {', '.join(failing)}"
        )
        return StageOutcome(False, f"{passing}/{total}")

    # ------------------------------------------------------------------ reporting

    def summary_path(self) -> Path:
        name = (
            "baseline-summary.json" if self.config.stage == "baseline" else "run-summary.json"
        )
        return self.config.project_path / STATE_DIR / name
