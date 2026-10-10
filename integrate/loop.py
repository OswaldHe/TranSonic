# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The baseline loop: iterate until every constraint is satisfied.

`OptimizationLoop` runs a fixed budget of iterations against a kernel that already works and
keeps whichever is fastest. This runs an *unbounded* number against a repo that does not work
yet and stops the moment it does.

Everything an iteration consists of is `Harness`'s and is not re-implemented here: the preflight,
the gitignore and clean-tree checks, the worktree, the agent run, the constraint and metric
commands, the reviewer, the acceptance decision, the commit, the history and the dashboard.
This subclass contributes exactly three things.

**A termination condition instead of a count.** `Harness.run` ends at `max_iterations` or on a
budget valve. Here the normal ending is the gate reporting every constraint satisfied, so `run`
drives `Harness.run` one iteration at a time and checks after each. Driving it rather than
copying it means the valves, the resume logic and the baseline capture keep working as they do
everywhere else.

**A failing baseline.** `_capture_baseline` records iteration 0's metrics without checking
constraints, so a baseline with nothing passing records as `constraints_passing: 0` and the run
proceeds. That is the starting condition, not an error — which is why this is a separate pass
rather than a mode of `optimize`.

**A prompt that says what is still failing.** The agent is told which constraints hold, and for
the ones that do not, the findings and evidence the gate or the judge already wrote down.
"""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

from autohelix.harness import Harness
from integrate.config import METRIC, IntegrateConfig

#: Where this pass keeps what it owns, inside the repo's gitignored `.autohelix/`.
STATE_DIR = ".autohelix/integrate"


class BaselineLoop(Harness):
    """An agent loop that runs until the declared constraints hold."""

    def __init__(
        self,
        config: IntegrateConfig,
        console: Console | None = None,
        verbose: bool = False,
    ) -> None:
        self.integrate_config = config
        super().__init__(
            project_path=config.project_path,
            verbose=verbose,
            config_file=config.write_loop_config(),
        )
        if console is not None:
            self.console = console
        self._header_shown = False

    # ------------------------------------------------------------------ state

    @property
    def gate_json(self) -> Path:
        return self.project_path / STATE_DIR / "gate.json"

    def read_gate(self) -> dict:
        """The last verdict the gate wrote, or an empty dict if it has not run."""
        try:
            return json.loads(self.gate_json.read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    def best_passing(self) -> int:
        """The most constraints any recorded iteration satisfied."""
        return max(
            (
                int(r.metrics.get(METRIC, 0))
                for r in self.history.load()
                if METRIC in r.metrics
            ),
            default=0,
        )

    def satisfied(self) -> bool:
        """True when some iteration satisfied every declared constraint.

        Decided on the *recorded metric*, not on the verdict file. The metric is captured by the
        Harness from the worktree the iteration actually ran in; the verdict file is a mirror
        written by the gate for its detail. An earlier cut of this read the file and compared it
        to the config, which looked equivalent and was not: with the gate running inside a
        worktree, the file in the project could be from any earlier check, so a finished run
        read as unfinished and the loop spent its whole budget after already succeeding.

        Compared against the config's own count, so a config that gained a constraint mid-run
        cannot look finished on a verdict that predates it.
        """
        total = len(self.integrate_config.constraints)
        return total > 0 and self.best_passing() >= total

    def failing_names(self) -> list[str]:
        payload = self.read_gate()
        return [
            str(c.get("name")) for c in (payload.get("constraints") or [])
            if not c.get("passed")
        ]

    # ------------------------------------------------------------------ prompt

    def build_prompt(self, iteration: int, worktree_dir: Path) -> str:
        """The per-iteration prompt: the goal, what is still failing, and what to read.

        The failing set comes from the previous iteration's verdict, so the agent is told what
        is actually blocking rather than left to rediscover it. Agent-constraint findings travel
        with it — a judge's findings are the most actionable thing in the run, and dropping them
        would make the next iteration guess at a verdict already written down.
        """
        from integrate import presets

        config = self.integrate_config
        payload = self.read_gate()
        results = payload.get("constraints") or []

        status: list[str] = []
        if results:
            status.append(
                f"As of the last check, {payload.get(METRIC, 0)} of "
                f"{payload.get('constraints_total', len(results))} constraint(s) hold.\n"
            )
            for entry in results:
                mark = "HOLDS" if entry.get("passed") else "FAILS"
                status.append(f"- [{mark}] {entry.get('name')} ({entry.get('kind')})")
                if entry.get("passed"):
                    continue
                detail = str(entry.get("detail") or "").strip()
                if detail:
                    status.append(f"    {detail}")
                for finding in (entry.get("findings") or [])[:4]:
                    status.append(f"    finding: {finding}")
                for item in (entry.get("evidence") or [])[:2]:
                    status.append(f"    evidence: {item}")
        else:
            status.append(
                "The gate has not produced a verdict yet, so this is the first look at the repo."
            )

        declared: list[str] = []
        for constraint in config.constraints:
            declared.append(f"- **{constraint.name}** ({constraint.kind})")
            if constraint.rationale:
                declared.append(f"    why: {constraint.rationale.strip()}")
            if constraint.kind == "script":
                declared.append(f"    checked by: `{constraint.command}`")
            else:
                declared.append(f"    an agent is asked: {constraint.prompt.strip()}")
                declared.append(f"    and passes it when: {constraint.criteria.strip()}")

        return presets.project_prompt(self.project_path).format(
            iteration=iteration,
            goal=config.goal.strip(),
            status="\n".join(status),
            constraints="\n".join(declared),
            editable="\n".join(f"- `{e}`" for e in config.editable),
        )

    # ------------------------------------------------------------------ display

    def _print_header(self, max_iter: int, start_iter: int = 1) -> None:
        """Replace the Harness header with one that says what this pass is doing.

        Printed once. `run` drives `Harness.run` per iteration, so the inherited header would
        otherwise reprint on every one.
        """
        if self._header_shown:
            return
        self._header_shown = True
        config = self.integrate_config
        kinds: dict[str, int] = {}
        for constraint in config.constraints:
            kinds[constraint.kind] = kinds.get(constraint.kind, 0) + 1
        shape = ", ".join(f"{n} {k}" for k, n in sorted(kinds.items()))
        bound = (
            f"{config.max_iterations} iteration(s)" if config.max_iterations
            else "unbounded — runs until the constraints hold"
        )
        self.console.print(
            f"\n[bold]integrate: {config.target_id}[/bold]\n"
            f"  repo        {config.project_path}\n"
            f"  constraints {len(config.constraints)} ({shape})\n"
            f"  budget      {bound}"
        )
        # From the native Config, which is where `budget.max_cost_usd` and `budget.time` are
        # parsed — this pass does not re-model them.
        if self.config.max_cost_usd is not None:
            self.console.print(f"  cost valve  ${self.config.max_cost_usd:g}")
        if self.config.max_time_seconds is not None:
            self.console.print(f"  time valve  {self.config.max_time_seconds}s")
        self.console.print()

    def _print_summary(self) -> None:
        """Suppressed per-iteration; `run` prints it once at the end via `super()`."""
        if getattr(self, "_summarizing", False):
            super()._print_summary()

    # ------------------------------------------------------------------ run

    def run(self, max_iterations: int | None = None) -> None:
        """Drive `Harness.run` one iteration at a time until the constraints hold.

        One iteration per call, because `Harness.run`'s only exits are its budget and its
        valves, and the condition this pass cares about has to be checked after each iteration
        before the next is spent. Everything inside an iteration stays `Harness`'s.

        A call that does not advance the history means `Harness.run` declined to iterate — a
        tripped valve, a failed preflight, a dirty tree, an interrupt — and it has already said
        why, so this stops rather than calling it again forever.
        """
        config = self.integrate_config
        cap = max_iterations or config.max_iterations or self.config.max_iterations
        self._ensure_baseline_verdict()

        while True:
            done = self.history.get_last_iteration()
            if done >= cap:
                self.console.print(
                    f"\n[yellow]Stopping: reached the {cap}-iteration backstop without "
                    f"satisfying every constraint.[/yellow] "
                    f"Still failing: {', '.join(self.failing_names()) or 'unknown'}"
                )
                break

            super().run(max_iterations=done + 1)

            if self.history.get_last_iteration() == done:
                # Harness.run stopped without iterating and has already reported the reason.
                break
            if self.satisfied():
                self._announce_success(self.history.get_last_iteration())
                break
            # The baseline alone can satisfy everything, in which case the first call captured
            # iteration 0 and nothing else; `satisfied()` above has already caught that.

        self._summarizing = True
        self._print_summary()
        self.write_summary()

    def _ensure_baseline_verdict(self) -> None:
        """Evaluate the constraints against the repo once, before the loop starts.

        `Harness._capture_baseline` runs the metric commands but not the constraints, and this
        pass's metric command only *reads back* a verdict the gate wrote. So with no verdict on
        disk the baseline produces no metric and the Harness aborts the run — which is how this
        first failed: the starting condition this pass is built around looked like a broken
        config.

        So the starting count is measured here, against the repo itself rather than a worktree.
        It is also worth having for its own sake: "you start at 1 of 3" is the first thing an
        operator wants to know, and it turns the failing baseline into a recorded fact.
        """
        if self.gate_json.is_file() or self.history.get_last_iteration() > 0:
            return
        from integrate import gate as gate_mod

        config = self.integrate_config
        self.console.print("  measuring the baseline against the repo...")
        results, payload = gate_mod.evaluate(config, config.project_path)
        self.gate_json.parent.mkdir(parents=True, exist_ok=True)
        self.gate_json.write_text(json.dumps(payload, indent=2) + "\n")
        passing, total = payload[METRIC], payload["constraints_total"]
        colour = "green" if payload["satisfied"] else "yellow"
        self.console.print(
            f"  baseline    [{colour}]{passing}/{total}[/{colour}] constraint(s) hold"
            + (
                ""
                if payload["satisfied"]
                else f"; failing: {', '.join(r.name for r in results if not r.passed)}"
            )
        )

    def _announce_success(self, iteration: int) -> None:
        total = len(self.integrate_config.constraints)
        self.console.print(
            f"\n[green]All {total} constraint(s) satisfied at iteration {iteration}.[/green]"
        )
        for entry in self.read_gate().get("constraints") or []:
            if entry.get("kind") == "agent":
                self.console.print(
                    f"  [dim]{entry.get('name')}: "
                    f"{str(entry.get('detail') or '').strip()}[/dim]"
                )

    # ------------------------------------------------------------------ summary

    def write_summary(self) -> Path:
        """What this run reached, for the next stage and for a human."""
        history = self.history.load()
        best = max((r.metrics.get(METRIC, 0) for r in history if r.iteration > 0), default=0)
        summary = {
            "pass": "integrate",
            "stage": "baseline",
            "target_id": self.integrate_config.target_id,
            "metric": METRIC,
            "satisfied": self.satisfied(),
            "constraints_total": len(self.integrate_config.constraints),
            "best_passing": best,
            "still_failing": self.failing_names(),
            "iterations": [
                {
                    "iteration": r.iteration,
                    "accepted": r.accepted,
                    METRIC: r.metrics.get(METRIC),
                    "reason": r.reason,
                }
                for r in history
            ],
            "final_verdict": self.read_gate(),
        }
        path = self.project_path / STATE_DIR / "baseline-summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2) + "\n")
        return path
