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
        self.console.print(
            "\n  Next: [cyan]autohelix integrate run-baseline[/cyan]"
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

        passing, total = payload["constraints_passing"], payload["constraints_total"]
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
        return self.config.project_path / STATE_DIR / "baseline-summary.json"
