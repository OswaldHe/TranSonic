# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The bootstrap loop: AutoHelix's iteration cycle, re-aimed at a gate that starts red.

AutoHelix's loop assumes a working codebase being made better: constraints pass at the
baseline, a metric is measured, and an iteration that breaks a constraint is thrown away.
Bootstrapping a kernel inverts all three. There is no metric — there is nothing to measure
until a kernel exists. The gate fails at the baseline by construction. And discarding a
failing iteration would discard every iteration, so the loop could never accumulate the
thing it is trying to build.

So this subclasses :class:`~autohelix.harness.Harness` and changes exactly four things,
reusing its agent plumbing, worktrees, scope enforcement, display and logging:

* **every iteration is merged.** Work ratchets forward whether or not the gate passes;
  the verdict is recorded beside the commit instead of deciding its fate.
* **the reviewer runs every iteration.** Upstream runs it only after a constraint passes,
  which here would be never — and "what is left to satisfy the gate" is only a useful
  question while something is still missing.
* **a failing gate never stops the run.** Only the budgets do.
* **a passing gate does.** All six checks green means the module is bootstrapped; further
  iterations would be optimization, which is a different loop with a metric in it.

The baseline is gated too, so iteration 1 opens with a review of what the stub is missing
rather than with nothing.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text

from autohelix.checks import ConstraintResult, run_constraints
from autohelix.dashboard import generate_dashboard
from autohelix.harness import AutoHelixRunError, Harness, _IterationSpinner
from autohelix.history import IterationResult
from autohelix.sandbox import Worktree

from bootstrap.materialize import CHECKS_REL, MANIFEST_REL
from bootstrap.preset import PRESET_PATH, load_prompt_template

#: Where verdicts are kept for the operator, outside the agent's worktree.
REPORT_DIR = "bootstrap"


@dataclass
class Verdict:
    """What the gate said about one iteration."""

    passed: bool
    passing: list[str]
    failing: list[str]
    report: str
    payload: dict[str, Any]

    @property
    def total(self) -> int:
        return len(self.passing) + len(self.failing)

    def summary(self) -> str:
        if self.passed:
            return f"gate: all {self.total} checks pass"
        return (
            f"gate: {len(self.passing)}/{self.total} passing"
            + (f", failing {', '.join(self.failing)}" if self.failing else "")
        )


def read_verdict(results: list[ConstraintResult], checks_path: Path) -> Verdict:
    """Turn the gate's output into a verdict.

    The JSON the gate writes is preferred; the exit code is the fallback. A gate that
    timed out or crashed writes nothing, and then there is exactly one thing to say —
    everything is failing and here is the output — rather than a fabricated breakdown.
    """
    output = "\n".join(r.output for r in results).strip()
    passed_exit = bool(results) and all(r.passed for r in results)
    payload: dict[str, Any] = {}
    if checks_path.is_file():
        try:
            payload = json.loads(checks_path.read_text())
        except json.JSONDecodeError:
            payload = {}
    checks = payload.get("checks") or []
    if checks:
        return Verdict(
            passed=bool(payload.get("passed")) and passed_exit,
            passing=[c["check"] for c in checks if c.get("passed")],
            failing=[c["check"] for c in checks if not c.get("passed")],
            report=payload.get("report") or output,
            payload=payload,
        )
    return Verdict(
        passed=passed_exit, passing=[], failing=["?"] if not passed_exit else [],
        report=output or "the gate produced no output", payload=payload,
    )


class BootstrapLoop(Harness):
    """The loop `autohelix bootstrap run` drives."""

    def __init__(
        self, repo: Path, verbose: bool = False, heartbeat_seconds: int = 30,
        config_file: Path | None = None,
    ) -> None:
        repo = Path(repo).resolve()
        # The preset is read from the package, not from the repo. Every module repo runs
        # under the same fixed config, so there is nothing per-repo to install and nothing
        # that can drift; and because the file never lands in the repo, the gate's command
        # line is not in the worktree for the agent to read.
        super().__init__(
            repo,
            verbose=verbose,
            heartbeat_seconds=heartbeat_seconds,
            config_file=config_file or PRESET_PATH,
        )
        self.reports_dir = self.project_path / ".autohelix" / REPORT_DIR
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self._verdicts: dict[int, Verdict] = {}

    # -- reviewer -----------------------------------------------------------------

    def _build_reviewer_prompt(self, worktree_path: Path) -> str:
        """The stock reviewer prompt, plus where this iteration's verdict is.

        The reviewer runs in the same worktree the gate just ran in, so the verdict is
        beside it. Handing over the path rather than the contents keeps the prompt short
        and lets the reviewer read the findings it cares about.
        """
        if not self.config.reviewer:
            return ""
        review_path = worktree_path / ".autohelix" / "review.md"
        return (
            f"{self.config.reviewer.prompt.strip()}\n\n"
            f"This iteration's gate verdict: {worktree_path / CHECKS_REL}\n"
            f"Write your review to {review_path}"
        )

    # -- the gate -----------------------------------------------------------------

    def _run_gate(self, cwd: Path, iteration: int) -> Verdict:
        """Run the single constraint and record what it said."""
        self.log.info(f"iter {iteration} gate started")
        results = run_constraints(self.config, cwd, project_path=self.project_path)
        verdict = read_verdict(results, cwd / CHECKS_REL)
        self._verdicts[iteration] = verdict
        self.log.info(f"iter {iteration} {verdict.summary()}")

        (self.reports_dir / f"iter-{iteration}.txt").write_text(verdict.report)
        if verdict.payload:
            (self.reports_dir / f"iter-{iteration}.json").write_text(
                json.dumps(verdict.payload, indent=2)
            )
        return verdict

    def _print_header(self, max_iter: int, start_iter: int = 1) -> None:
        """The stock header, with the goal escaped before Rich renders it.

        The header prints the goal as markup, and this goal quotes the marker lines the
        agent must emit — `##autohelix[latency_ms=...]`. Rich reads those brackets as a
        style tag and drops them, so the operator's copy of the specification would be
        missing exactly the part that is easiest to get wrong.
        """
        original = self.config.goal
        self.config.goal = escape(original)
        try:
            super()._print_header(max_iter, start_iter)
        finally:
            self.config.goal = original

    def _show_verdict(self, verdict: Verdict) -> None:
        colour = "green" if verdict.passed else "yellow"
        self.console.print(f"  [{colour}]{verdict.summary()}[/{colour}]")
        if self.verbose or not verdict.passed:
            # Wrapped in Text, which is literal, because the report is full of square
            # brackets that are not Rich tags: as markup, `##autohelix[latency_ms=...]`
            # renders as `##autohelix`, turning the finding that names the marker the agent
            # must print into nonsense.
            self.console.print(Panel(
                Text(verdict.report.strip() or "(no output)"),
                title="gate", border_style=colour,
            ))

    # -- prompt -------------------------------------------------------------------

    def build_prompt(self, iteration: int, worktree_dir: Path) -> str:
        """The bootstrap prompt, with a history summary that says how the gate is doing.

        Two departures from upstream. The template comes from the package rather than from
        `.autohelix/prompt.md` in the repo — it is fixed, like the preset, and the stock
        template would not do because it renders the constraint commands and the constraint
        command names the checker the agent must not see.

        And the history summary is rebuilt: upstream reports metrics for an accepted
        iteration, and here every iteration is accepted and there are no metrics, so it
        would say nothing at all.
        """
        from autohelix.prompt_template import build_prompt_variables, render_template

        variables = build_prompt_variables(self.config, self.history, iteration, worktree_dir)
        prompt = render_template(load_prompt_template(), variables)
        lines = []
        for result in self.history.get_recent(5):
            verdict = self._verdicts.get(result.iteration)
            state = verdict.summary() if verdict else self._stored_summary(result)
            label = "baseline" if result.iteration == 0 else f"iter {result.iteration}"
            lines.append(f"- {label}: {state}")
        if lines:
            prompt = prompt.replace(
                "Recent history:", "Recent history:\n" + "\n".join(lines), 1,
            )
        return prompt

    @staticmethod
    def _stored_summary(result: IterationResult) -> str:
        """A past iteration's gate state, from history when it is not in memory."""
        if result.reason:
            return result.reason
        count = result.metrics.get("checks_passing")
        return f"gate: {count:g} check(s) passing" if count is not None else "no verdict recorded"

    # -- iteration ----------------------------------------------------------------

    def run_iteration(self, iteration: int) -> IterationResult:
        """One bootstrap iteration: agent, gate, review, merge — in that order, always.

        The merge is unconditional. That is the whole difference from upstream, and it is
        what makes a loop against a red gate possible: iteration N+1 starts from N's
        kernel instead of from the stub.
        """
        iter_start = time.monotonic()
        self.log.info(f"iter {iteration} started")

        worktree = self.sandbox.create_worktree(iteration)
        self.sandbox.prepare_worktree(worktree)
        (worktree.working_dir / ".autohelix" / "notes").mkdir(parents=True, exist_ok=True)

        agent_usage: dict = {}
        try:
            prompt = self.build_prompt(iteration, worktree.working_dir)
            if self.verbose:
                self.console.print(Panel(
                    prompt.strip(), title="[bold]Agent Prompt[/bold]", border_style="dim",
                ))
            agent_result = self.run_agent(worktree.working_dir, prompt, iteration)
            agent_usage = agent_result.usage or {}
            if not agent_result.success:
                detail = agent_result.error or f"agent exited with status {agent_result.exit_code}"
                raise AutoHelixRunError(f"Iteration {iteration} agent failed: {detail}")

            # Agents sometimes commit inside the worktree; reopen those so the gate and
            # the scope check see the whole candidate diff.
            reopened = self.sandbox.uncommit_agent_changes(worktree)
            if reopened:
                self.log.info(f"iter {iteration} normalized {reopened} agent-created commit(s)")

            effective_editable = self.sandbox.resolve_editable(
                self.config.editable, self.config.frozen, cwd=worktree.working_dir,
            )
            if effective_editable is not None:
                reverted = self.sandbox.revert_out_of_scope(worktree, effective_editable)
                if reverted:
                    self.log.info(f"iter {iteration} reverted {len(reverted)} out-of-scope file(s)")
                    self.console.print(
                        f"  [yellow]![/yellow] Reverted {len(reverted)} file(s) outside "
                        f"editable scope: {', '.join(reverted[:4])}"
                    )

            spinner = _IterationSpinner(iteration)
            spinner.set_phase("checks")
            with Live(spinner, console=self.console, refresh_per_second=10, transient=True):
                verdict = self._run_gate(worktree.working_dir, iteration)
            self._show_verdict(verdict)

            if self.config.reviewer and not self.run_reviewer(worktree, iteration):
                # Advisory, as upstream: a reviewer failure must not cost the iteration
                # its work. Recorded so a persistent failure is visible in the reports.
                self.console.print("  [yellow]![/yellow] Reviewer failed")
                (self.reports_dir / f"iter-{iteration}-review-failed").write_text("")

            commit = self.sandbox.merge_worktree(
                worktree,
                editable=effective_editable,
                message=self._commit_message(iteration, verdict, worktree.working_dir),
            )
            result = IterationResult(
                iteration=iteration,
                accepted=True,
                metrics={"checks_passing": float(len(verdict.passing))},
                commit=commit,
                reason=verdict.summary(),
                failure_output=None if verdict.passed else "\n".join(
                    verdict.report.strip().splitlines()[-20:]
                ),
                usage=agent_usage,
            )
        except AutoHelixRunError:
            raise
        except Exception as exc:
            self.log.info(f"iter {iteration} unexpected failure — {exc}")
            raise AutoHelixRunError(f"Iteration {iteration} failed unexpectedly: {exc}") from exc
        finally:
            self.sandbox.save_notes(worktree)
            if worktree.path.exists():
                self.sandbox.discard_worktree(worktree)

        result.usage["wall_clock_ms"] = int((time.monotonic() - iter_start) * 1000)
        self.log.info(
            f"iter {iteration} merged — commit {result.commit or 'no change'} "
            f"({result.usage['wall_clock_ms'] / 1000:.0f}s total)"
        )
        self.history.append(result)
        try:
            generate_dashboard(self.project_path, self.config.metric_directions())
        except Exception:
            pass
        return result

    def _commit_message(self, iteration: int, verdict: Verdict, worktree_path: Path) -> str:
        """A commit message that records the gate state, since the commit is not a pass.

        Every iteration lands, so `git log` is the record of how the kernel got here — and
        a line per commit saying which checks were green at the time is what makes that log
        readable later.
        """
        summary_file = worktree_path / ".autohelix" / "commit_summary.txt"
        headline = ""
        if summary_file.is_file():
            headline = summary_file.read_text().strip().splitlines()[0][:72] if summary_file.read_text().strip() else ""
        subject = headline or f"Bootstrap iteration {iteration}"
        body = [
            "",
            verdict.summary() + ".",
        ]
        if verdict.passing:
            body.append(f"Passing: {', '.join(verdict.passing)}")
        if verdict.failing:
            body.append(f"Failing: {', '.join(verdict.failing)}")
        return "\n".join([f"iter {iteration}: {subject}", *body])

    # -- baseline -----------------------------------------------------------------

    def _capture_baseline(self) -> None:
        """Gate the stub, then review it.

        Upstream captures metrics here and raises if one is missing. There are no metrics,
        and the useful baseline is the verdict: iteration 1 should open knowing which of
        the six checks the stub already satisfies and which it does not.
        """
        spinner = _IterationSpinner(0)
        spinner.set_phase("checks")
        with Live(spinner, console=self.console, refresh_per_second=10, transient=True):
            verdict = self._run_gate(self.project_path, 0)
        self._show_verdict(verdict)

        self.history.append(IterationResult(
            iteration=0, accepted=True,
            metrics={"checks_passing": float(len(verdict.passing))},
            reason=verdict.summary(),
        ))
        if self.config.reviewer:
            self._run_baseline_review()
        try:
            generate_dashboard(self.project_path, self.config.metric_directions())
        except Exception:
            pass

    def _run_baseline_review(self) -> None:
        """Review the stub, with the baseline verdict placed where the reviewer looks.

        The gate ran in the project root, not in a worktree, so its verdict is not where
        the reviewer expects. Copying it in is what lets one reviewer prompt serve the
        baseline and every iteration alike.
        """
        if not self.config.reviewer:
            return
        worktree = self.sandbox.create_worktree(0)
        self.sandbox.prepare_worktree(worktree)
        (worktree.working_dir / ".autohelix" / "notes").mkdir(parents=True, exist_ok=True)
        source = self.reports_dir / "iter-0.json"
        if source.is_file():
            target = worktree.working_dir / CHECKS_REL
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        try:
            if not self.run_reviewer(worktree, 0):
                self.console.print("  [yellow]![/yellow] Baseline review failed, continuing")
        finally:
            self.sandbox.discard_worktree(worktree)

    # -- preflight ----------------------------------------------------------------

    def _validate_environment(self) -> bool:
        """Check the fixed constraint command can actually run, before spending an agent on it.

        The command in `preset.yaml` says `python`, because a fixed file cannot name this
        machine's interpreter. That is fine and it is also the one thing about the preset that
        can be wrong at runtime: an unactivated venv gives a `python` that either cannot
        import `bootstrap` or — worse, because it fails later and less legibly — cannot import
        torch, which the gate needs to run `inference.py`. Both are cheap to detect here and
        expensive to discover after an iteration's work.
        """
        if not (self.project_path / MANIFEST_REL).is_file():
            self.console.print(
                f"[red]Error:[/red] {self.project_path} has no {MANIFEST_REL}. "
                f"Run `autohelix bootstrap init` first."
            )
            return False

        probe = subprocess.run(
            ["python", "-c",
             "import bootstrap.nki_checker, torch, torch_neuronx; print('ok')"],
            capture_output=True, text=True,
        )
        if probe.returncode != 0:
            missing = probe.stderr.strip().splitlines()[-1:] or ["unknown import error"]
            self.console.print(
                f"[red]Error:[/red] the gate runs `python -m bootstrap.nki_checker`, but "
                f"`python` on this PATH cannot import what it needs:\n  {missing[0]}\n"
                f"Activate the environment autohelix is installed in and try again."
            )
            return False
        return True

    # -- the run ------------------------------------------------------------------

    def run(self, max_iterations: int | None = None) -> bool:
        """Iterate until the gate passes or the budget runs out. True if it passed."""
        from autohelix.checks import preflight_check
        from autohelix.state import ensure_gitignore_entry

        max_iter = max_iterations or self.config.max_iterations
        start_iter = self.history.get_last_iteration() + 1

        if not self._check_active_config():
            return False
        ensure_gitignore_entry(self.project_path, ".autohelix/")
        try:
            self.sandbox.ensure_clean_working_tree()
            self.sandbox.ensure_editable_files_ready(self.config.editable)
        except RuntimeError as exc:
            self.console.print(f"[red]Error:[/red] {exc}")
            return False

        if not self._validate_environment():
            return False

        self._print_header(max_iter, start_iter)

        if start_iter == 1:
            issues = preflight_check(self.config, self._raw_config, self.project_path)
            for issue in issues:
                if issue.level == "error":
                    self.console.print(f"  [red]FAIL[/red] {issue.message}")
            if any(i.level == "error" for i in issues):
                self.console.print("\n[red]Preflight checks failed.[/red]")
                return False

        if start_iter == 1 and not any(r.iteration == 0 for r in self.history.load()):
            self._capture_baseline()

        cumulative_cost = 0.0
        loop_start = time.monotonic()
        passed = False
        iteration = start_iter - 1
        try:
            for iteration in range(start_iter, max_iter + 1):
                result = self.run_iteration(iteration)
                self._print_iteration_summary(result)

                verdict = self._verdicts.get(iteration)
                if verdict and verdict.passed:
                    passed = True
                    self.console.print(
                        f"\n[green]Bootstrapped.[/green] All checks pass at iteration "
                        f"{iteration}; the module has a kernel and a validator."
                    )
                    break

                cumulative_cost += (result.usage or {}).get("cost_usd", 0)
                if self.config.max_cost_usd is not None and cumulative_cost >= self.config.max_cost_usd:
                    self.console.print("\n[yellow]Cost budget reached.[/yellow]")
                    break
                if (self.config.max_time_seconds is not None
                        and time.monotonic() - loop_start >= self.config.max_time_seconds):
                    self.console.print("\n[yellow]Time budget reached.[/yellow]")
                    break
        except KeyboardInterrupt:
            self.console.print(f"\n\n[yellow]Interrupted at iteration {iteration}. Progress saved.[/yellow]")

        if not passed:
            self.console.print(
                "\n[yellow]Not bootstrapped yet.[/yellow] The gate is still failing; "
                "run again to continue from where this left off."
            )
        self._print_summary()
        return passed
