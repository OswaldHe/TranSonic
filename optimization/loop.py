# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The optimization loop: AutoHelix's iteration cycle with a per-iteration constraint.

Unlike `bootstrap`, this is close to upstream. The baseline is green, there is a metric, and a
rejected iteration is discarded — all three as `autohelix run` has them. Three things change:

1. **A per-iteration constraint runs before anything else.** The slot governing this iteration is
   checked as a static read of `source.py`, in milliseconds, before the ~4 minute device run. A
   violating iteration is rejected without spending the hardware on it.
2. **The prompt carries the slot's prose.** The agent is told what it may use this iteration and
   that a script it cannot see will check.
3. **Every constraint-passing candidate is archived.** The 5% metric gate governs what the next
   iteration builds on; it does not govern what is kept. Iterations 7-8 are set aside for
   aggressive exploration, and an experiment that lands 6% slower is rejected — but its code is
   worth reading afterwards, so it is recovered from its branch and stored with its metrics.

The deliverable is the *best* accepted commit, not `HEAD`. With 5% of slack an accepted iteration
can be slower than the best one, so after ten iterations `HEAD` is not necessarily the fastest
thing the run produced. `best_commit()` is what the next stage consumes.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from rich.markup import escape
from rich.panel import Panel
from rich.text import Text

from autohelix.dashboard import generate_dashboard
from autohelix.harness import Harness
from autohelix.history import IterationResult
from optimization import constraints as cons

#: The metric every stage of this pipeline optimizes. Fixed rather than configurable: the
#: constraint checkers, the latency bounds and the report all name it, and a configurable metric
#: name would make each of those a place it could be spelled differently.
METRIC = "latency_ms"

#: Where per-iteration records are kept, outside the agent's worktree.
REPORT_DIR = Path(".autohelix") / "optimization"

#: Where rejected-but-interesting candidates are recovered to.
CANDIDATES_DIR = REPORT_DIR / "candidates"


class OptimizationLoop(Harness):
    """The loop `autohelix optimize run` and `optimize run-full` both drive."""

    def __init__(
        self,
        repo: Path,
        config_file: Path,
        stage: str = "submodule",
        verbose: bool = False,
        heartbeat_seconds: int = 30,
    ) -> None:
        repo = Path(repo).resolve()
        super().__init__(
            repo, verbose=verbose, heartbeat_seconds=heartbeat_seconds, config_file=config_file,
        )
        self.stage = stage
        self.reports_dir = self.project_path / REPORT_DIR
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.schedule = self._load_schedule()
        self._slot_verdicts: dict[int, cons.SlotVerdict] = {}

    # -- configuration -------------------------------------------------------------

    def _load_schedule(self) -> cons.Schedule:
        """Parse `iteration_constraints:` out of the raw config.

        Read from `_raw_config` rather than `Config`: upstream's dataclass has no field for it, and
        adding one would put a stage-specific concept into the shared config for every AutoHelix
        user. The key is declared in `KNOWN_TOP_LEVEL_KEYS` so it does not warn as a typo.
        """
        schedule = cons.Schedule.from_config(
            (self._raw_config or {}).get("iteration_constraints"),
            max_iterations=self.config.max_iterations,
        )
        for warning in schedule.validate(self.config.max_iterations):
            self.console.print(f"  [yellow]![/yellow] {warning}")
        return schedule

    # -- prompt --------------------------------------------------------------------

    def build_prompt(self, iteration: int, worktree_dir: Path) -> str:
        """Upstream's prompt plus this iteration's constraint and the standing bounds."""
        from autohelix.prompt_template import build_prompt_variables, render_template

        from optimization import presets

        variables = build_prompt_variables(self.config, self.history, iteration, worktree_dir)
        variables["iteration_constraint"] = self.schedule.describe_for_prompt(iteration)
        variables["constraint_schedule"] = self.schedule.summary_table()
        variables["stage"] = self.stage
        variables["metric"] = METRIC
        best = self.history.get_best_metrics(self.config.metric_directions()).get(METRIC)
        variables["best_so_far"] = f"{best[0]:g}" if best else ""
        # The packaged template, not `.autohelix/prompt.md`'s default: the stock one renders the
        # constraint commands, and here constraint zero is the hidden slot checker.
        return render_template(presets.project_prompt(self.project_path), variables)

    def _print_header(self, max_iter: int, start_iter: int = 1) -> None:
        """The stock header with the goal escaped, since the goal quotes `##autohelix[...]`.

        Rich reads those brackets as a style tag and drops them, which would remove exactly the
        part of the specification that is easiest to get wrong.
        """
        original = self.config.goal
        self.config.goal = escape(original)
        try:
            super()._print_header(max_iter, start_iter)
        finally:
            self.config.goal = original
        if self.schedule.slots:
            self.console.print(Panel(
                Text(self.schedule.summary_table()),
                title="[bold]Constraint schedule[/bold]", border_style="dim",
            ))

    # -- iteration -----------------------------------------------------------------

    def run_iteration(self, iteration: int) -> IterationResult:
        """One iteration: agent, slot check, then upstream's constraints/metric/review path.

        The slot check is spliced in between scope enforcement and the constraints. Everything
        after it is `Harness.run_iteration`, reached by letting the base class run when the slot
        passes — rather than copied, so the acceptance logic, the metric gates and the reviewer
        stay in one place.
        """
        slot = self.schedule.slot_for(iteration)
        if slot is None or not (slot.has_text and slot.enforce):
            return super().run_iteration(iteration)

        # A governed iteration needs the agent's work in hand before the slot can be judged, so the
        # cheapest correct structure is to let the base class do everything and check the slot
        # inside the constraint phase. `run_constraints` is where that hook goes.
        return self._run_governed_iteration(iteration)

    def _run_governed_iteration(self, iteration: int) -> IterationResult:
        """`Harness.run_iteration` with the slot checker prepended to the constraint list.

        Implemented by temporarily prefixing `config.constraints` with the checker command. That
        keeps one copy of the iteration body: the slot becomes constraint zero, so a violation is
        rejected by exactly the path a failing constraint already takes, with the same reporting,
        the same discard and the same failure output carried into the next prompt.
        """
        from autohelix.config import ConstraintCommand

        slot = self.schedule.slot_for(iteration)
        assert slot is not None  # guarded by the caller
        checker = cons.checker_path(self.project_path, slot)
        if not checker.is_file():
            self.console.print(
                f"  [yellow]![/yellow] slot {slot.label} has no compiled checker at "
                f"{checker.name}; stated in the prompt but not enforced this iteration"
            )
            return super().run_iteration(iteration)

        report = cons.report_path(self.project_path, iteration)
        report.parent.mkdir(parents=True, exist_ok=True)
        if report.exists():
            report.unlink()  # a stale verdict from a re-run would be read as this iteration's
        command = cons.CHECKER_COMMAND.format(checker=checker, report=report)

        original = list(self.config.constraints)
        self.config.constraints = [ConstraintCommand(command=command, timeout=120), *original]
        try:
            result = super().run_iteration(iteration)
        finally:
            self.config.constraints = original

        # The checker's own JSON report is the authority on whether the slot was followed, not the
        # iteration's outcome: a rejection from the real constraints or the metric gate is an
        # ordinary failure that happened to run after a passing slot check, and recording it as a
        # violation would blame the schedule for it.
        verdict = cons.read_slot_verdict(
            iteration, slot.label, report, result.failure_output or "",
            return_code=0 if report.is_file() else 1,
        )
        self._slot_verdicts[iteration] = verdict
        if not verdict.passed:
            result.reason = f"constraint slot {slot.label} violated"
            self.console.print(f"  [red]{verdict.summary()}[/red]")
            if verdict.findings:
                self.console.print(Panel(
                    Text("\n".join(verdict.findings)),
                    title=f"constraint slot {slot.label}", border_style="red",
                ))
        elif verdict.checked:
            self.console.print(f"  [green]{verdict.summary()}[/green]")
        return result

    # -- candidate archive ---------------------------------------------------------

    def archive_candidate(self, iteration: int, result: IterationResult) -> Path | None:
        """Keep a constraint-passing candidate's `source.py` and metrics, accepted or not.

        The metric gate decides what the next iteration builds on. It should not decide what the
        run *retains*: iterations 7-8 exist to try something the disciplined iterations cannot, and
        an experiment 6% slower is rejected while still being the most informative thing in the run.
        Recovered from the iteration's branch, since the worktree is gone by now.
        """
        branch = f"{self.sandbox.branch_prefix}-iter-{iteration}"
        target = self.project_path / CANDIDATES_DIR / f"iter-{iteration}"
        try:
            show = subprocess.run(
                ["git", "show", f"{branch}:source.py"],
                cwd=self.project_path, capture_output=True, text=True, check=False,
            )
            if show.returncode != 0:
                return None
            target.mkdir(parents=True, exist_ok=True)
            (target / "source.py").write_text(show.stdout)
            (target / "metrics.json").write_text(json.dumps({
                "iteration": iteration,
                "accepted": result.accepted,
                "metrics": result.metrics,
                "reason": result.reason,
                "slot": (self.schedule.slot_for(iteration).label
                         if self.schedule.slot_for(iteration) else None),
            }, indent=2))
            return target
        except OSError:
            return None

    # -- the deliverable -----------------------------------------------------------

    def best_commit(self) -> tuple[str | None, float | None]:
        """The commit with the lowest latency among accepted iterations, and that latency.

        What the next stage consumes. With 5% of slack `HEAD` can be slower than the best accepted
        iteration, so handing `HEAD` on would quietly give away part of what the loop achieved.
        """
        best_commit: str | None = None
        best_value: float | None = None
        for record in self.history.load():
            if not record.accepted or not record.commit:
                continue
            value = record.metrics.get(METRIC)
            if value is None:
                continue
            if best_value is None or value < best_value:
                best_commit, best_value = record.commit, value
        return best_commit, best_value

    def write_stage_summary(self) -> Path:
        """What the stage achieved, for the report and for the next stage to read."""
        best_commit, best_value = self.best_commit()
        baseline = next(
            (r.metrics.get(METRIC) for r in self.history.load() if r.iteration == 0), None,
        )
        payload: dict[str, Any] = {
            "stage": self.stage,
            "metric": METRIC,
            "baseline_ms": baseline,
            "best_ms": best_value,
            "best_commit": best_commit,
            "speedup": (baseline / best_value) if (baseline and best_value) else None,
            "iterations": [
                {
                    "iteration": r.iteration,
                    "accepted": r.accepted,
                    METRIC: r.metrics.get(METRIC),
                    "reason": r.reason,
                    "slot": (self.schedule.slot_for(r.iteration).label
                             if self.schedule.slot_for(r.iteration) else None),
                    "slot_followed": (self._slot_verdicts[r.iteration].passed
                                      if r.iteration in self._slot_verdicts else None),
                }
                for r in self.history.load()
            ],
            "schedule": [s.to_dict() for s in self.schedule.slots],
        }
        path = self.reports_dir / f"{self.stage}-summary.json"
        path.write_text(json.dumps(payload, indent=2))
        return path

    # -- the run -------------------------------------------------------------------

    def run(self, max_iterations: int | None = None) -> None:
        """Upstream's run, with candidates archived and a stage summary written at the end."""
        super().run(max_iterations)
        for record in self.history.load():
            if record.iteration > 0:
                self.archive_candidate(record.iteration, record)
        path = self.write_stage_summary()
        best_commit, best_value = self.best_commit()
        if best_value is not None:
            self.console.print(
                f"\n[green]Best accepted:[/green] {best_value:g} ms at "
                f"{(best_commit or '')[:12]} — recorded in {path.name}"
            )
        try:
            generate_dashboard(self.project_path, self.config.metric_directions())
        except Exception:
            pass
