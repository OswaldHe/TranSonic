# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The run loop: hold the constraints, improve the measurements.

`BaselineLoop` drives a repo that does not satisfy its constraints to one that does, and stops
the moment it holds. This is what comes after: the constraints already hold, and the loop's job
is to make something measurable better without losing them.

It is the standard AutoHelix loop, which is the point — a fixed iteration budget, metrics from
the operator's config, acceptance that rejects an iteration whose metric regressed. Three things
are this pass's:

**The constraints are the integrate pass's, both kinds.** A script constraint is a command and an
exit code; an agent constraint is a prompt, a criteria and a judge. One gate evaluates all of
them per iteration and writes a verdict, exactly as in the baseline stage, so the thing the
repo was driven to satisfy keeps being checked while it is being optimized.

**The measurements come out of that same gate.** The thing that decides whether the port still
works is usually also the thing that reports how fast it is — here a verifier that takes an hour
on a Trainium device. Running it once as a constraint and again as a benchmark would double the
most expensive part of an iteration and let the two runs disagree about which number to keep. So
a constraint's `##autohelix[...]` output is the metric, read back from the verdict.

**Iterations can be given different guidance.** `iteration_constraints:` attaches prose to
iteration ranges — `optimization/`'s mechanism, as guidance only here. See `integrate/schedule.py`.

What rejects an iteration, in one place:

| | |
|---|---|
| a script constraint stopped holding | the `constraints_passing` gate, zero tolerance |
| a gated metric regressed more than the allowance | its metric gate, `acceptance.max_regression_pct` |
| the reviewer said so | the reviewer, as everywhere in AutoHelix |

An agent constraint flipping to FAIL does *not* reject, for the reason `integrate.config.METRIC`
records: a judge can reach a different verdict on an unchanged tree. It is reported, it reaches
the next iteration's prompt, and the stage's summary says the run ended with it failing.
"""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

from autohelix.harness import AutoHelixRunError, Harness
from integrate import schedule as sched
from integrate.config import METRIC, METRIC_ALL, IntegrateConfig
from integrate.loop import MAX_AGENT_FAILURES, STATE_DIR


class RunLoop(Harness):
    """An agent loop that improves the operator's metrics while the constraints keep holding."""

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
        self.schedule = self._load_schedule()
        self._header_shown = False

    # ------------------------------------------------------------------ state

    def _load_schedule(self) -> sched.Schedule:
        try:
            out = sched.Schedule.from_config(
                self.integrate_config.iteration_constraints,
                max_iterations=self.integrate_config.iterations,
            )
        except sched.ScheduleError as exc:
            # Reported rather than raised: `validate` already refuses to start a run on a broken
            # schedule, so reaching here means the config changed under a resume.
            self.console.print(f"  [yellow]![/yellow] iteration_constraints: {exc}")
            return sched.Schedule()
        for warning in out.validate(self.integrate_config.iterations):
            self.console.print(f"  [yellow]![/yellow] {warning}")
        return out

    @property
    def gate_json(self) -> Path:
        return self.project_path / STATE_DIR / "gate.json"

    def read_gate(self) -> dict:
        try:
            return json.loads(self.gate_json.read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    def failing_names(self) -> list[str]:
        return [
            str(c.get("name")) for c in (self.read_gate().get("constraints") or [])
            if not c.get("passed")
        ]

    def best_metrics(self) -> dict[str, tuple[float, int]]:
        """The best value of each metric so far, and which iteration produced it."""
        return self.history.get_best_metrics(self.config.metric_directions())

    # ------------------------------------------------------------------ prompt

    def build_prompt(self, iteration: int, worktree_dir: Path) -> str:
        """The per-iteration prompt: the goal, the numbers, the constraints, this slot's text."""
        from integrate import presets

        config = self.integrate_config
        best = self.best_metrics()
        directions = self.config.metric_directions()

        rows: list[str] = []
        for name in config.metric_names:
            entry = best.get(name)
            arrow = "lower is better" if directions.get(name) == "lower" else "higher is better"
            gated = name in config.gated_metric_names
            where = (
                f"best so far {entry[0]:.6g} (iteration {entry[1]})" if entry
                else "not measured yet"
            )
            rows.append(
                f"- **{name}** — {arrow}, {where}"
                + (
                    f"; a regression over {config.max_regression_pct:g}% rejects the iteration"
                    if gated else "; recorded, never gated"
                )
            )

        status: list[str] = []
        payload = self.read_gate()
        results = payload.get("constraints") or []
        if results:
            held = sum(1 for e in results if e.get("passed"))
            status.append(
                f"At the last check {held} of "
                f"{payload.get('constraints_total', len(results))} constraint(s) held."
            )
            for entry in results:
                if entry.get("passed"):
                    continue
                status.append(
                    f"- FAILING: {entry.get('name')} ({entry.get('kind')}) — "
                    f"{str(entry.get('detail') or '').strip()}"
                )
                for finding in (entry.get("findings") or [])[:3]:
                    status.append(f"    finding: {finding}")
        else:
            status.append("The gate has not produced a verdict yet.")

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

        return presets.project_run_prompt(self.project_path).format(
            iteration=iteration,
            iterations=config.iterations,
            goal=config.goal.strip(),
            metrics="\n".join(rows) or "(none declared)",
            status="\n".join(status),
            constraints="\n".join(declared),
            iteration_constraint=self.schedule.describe_for_prompt(iteration),
            constraint_schedule=self.schedule.summary_table(),
            regression_allowance=f"{config.max_regression_pct:g}",
            editable="\n".join(f"- `{e}`" for e in config.editable),
        )

    # ------------------------------------------------------------------ display

    def _print_header(self, max_iter: int, start_iter: int = 1) -> None:
        if self._header_shown:
            return
        self._header_shown = True
        config = self.integrate_config
        self.console.print(
            f"\n[bold]integrate run: {config.target_id}[/bold]\n"
            f"  repo        {config.project_path}\n"
            f"  constraints {len(config.constraints)} (held while optimizing)\n"
            f"  metrics     {', '.join(config.metric_names) or 'none'}\n"
            f"  allowance   {config.max_regression_pct:g}% regression rejects an iteration\n"
            f"  budget      {config.iterations} iteration(s), "
            f"starting at {start_iter}"
        )
        if self.config.iteration_time_seconds is not None:
            self.console.print(f"  per-iter    {self.config.iteration_time_seconds}s")
        if self.config.max_cost_usd is not None:
            self.console.print(f"  cost valve  ${self.config.max_cost_usd:g}")
        if self.schedule.slots:
            self.console.print(f"  schedule    {len(self.schedule.slots)} slot(s)")
        self.console.print()

    # ------------------------------------------------------------------ run

    def run(self, max_iterations: int | None = None) -> None:
        """Drive `Harness.run` one iteration at a time, absorbing a dead agent.

        The only reason this is not a plain `Harness.run` call: a transient agent death ends the
        whole run upstream, which is right for a short budget and wrong for iterations that take
        hours. Same policy as `BaselineLoop`, and the same bound on it.
        """
        config = self.integrate_config
        cap = max_iterations or config.iterations
        self._ensure_baseline_verdict()

        agent_failures = 0
        while True:
            done = self.history.get_last_iteration()
            if done >= cap:
                break
            try:
                super().run(max_iterations=done + 1)
            except AutoHelixRunError as exc:
                if "agent failed" not in str(exc):
                    raise
                agent_failures += 1
                self.log.warn(f"iter {done + 1} agent died ({exc}); attempt {agent_failures}")
                self.console.print(
                    f"\n[yellow]Iteration {done + 1}'s agent died[/yellow] — {exc}\n"
                    f"  Retrying ({agent_failures} of {MAX_AGENT_FAILURES})."
                )
                if agent_failures >= MAX_AGENT_FAILURES:
                    self.console.print(
                        f"\n[red]Stopping: {agent_failures} agent deaths in a row.[/red]"
                    )
                    break
                continue
            agent_failures = 0
            if self.history.get_last_iteration() == done:
                # Harness.run stopped without iterating and has already said why.
                break

        self.write_summary()

    def _ensure_baseline_verdict(self) -> None:
        """Evaluate the constraints against the repo once, before the loop starts.

        The same reason as `BaselineLoop`'s: this pass's metric command reads a verdict back
        rather than producing one, so with no verdict on disk the baseline capture has nothing to
        read and the Harness aborts. Here it matters twice over, because the baseline's *metrics*
        — the numbers every later iteration is judged against — come out of this evaluation.
        """
        if self.gate_json.is_file() or self.history.get_last_iteration() > 0:
            return
        from integrate import gate as gate_mod

        config = self.integrate_config
        self.console.print(
            "  measuring the baseline against the repo (this runs every constraint)..."
        )
        results, payload = gate_mod.evaluate(config, config.project_path)
        self.gate_json.parent.mkdir(parents=True, exist_ok=True)
        self.gate_json.write_text(json.dumps(payload, indent=2) + "\n")

        passing, total = payload[METRIC_ALL], payload["constraints_total"]
        colour = "green" if payload["satisfied"] else "yellow"
        self.console.print(
            f"  baseline    [{colour}]{passing}/{total}[/{colour}] constraint(s) hold"
            + (
                ""
                if payload["satisfied"]
                else f"; failing: {', '.join(r.name for r in results if not r.passed)}"
            )
        )
        measured = payload.get("metrics") or {}
        missing = [n for n in config.metric_names if n not in measured]
        for name in config.metric_names:
            if name in measured:
                self.console.print(f"              {name} = {measured[name]:.6g}")
        if missing:
            # Said here rather than discovered at the baseline capture, which reports it as a
            # missing metric without saying that a constraint was supposed to print it.
            self.console.print(
                f"  [yellow]![/yellow] no constraint printed ##autohelix[...] for: "
                f"{', '.join(missing)}. The run will stop at the baseline capture — a metric "
                f"this loop gates on has to come out of a constraint's own output."
            )

    # ------------------------------------------------------------------ summary

    def write_summary(self) -> Path:
        """What this run reached, for a human and for whatever comes after."""
        history = self.history.load()
        directions = self.config.metric_directions()
        best = self.history.get_best_metrics(directions)
        baseline = next((r for r in history if r.iteration == 0), None)
        config = self.integrate_config

        improvements: dict[str, dict[str, float | int | None]] = {}
        for name in config.metric_names:
            start = (baseline.metrics or {}).get(name) if baseline else None
            entry = best.get(name)
            pct: float | None = None
            if start not in (None, 0) and entry is not None:
                delta = (start - entry[0]) / start * 100
                pct = delta if directions.get(name) == "lower" else -delta
            improvements[name] = {
                "baseline": start,
                "best": entry[0] if entry else None,
                "best_iteration": entry[1] if entry else None,
                "improvement_pct": pct,
            }

        summary = {
            "pass": "integrate",
            "stage": "run",
            "target_id": config.target_id,
            "iterations_run": len([r for r in history if r.iteration > 0]),
            "iterations_budget": config.iterations,
            "accepted": [r.iteration for r in history if r.iteration > 0 and r.accepted],
            "gated_metrics": config.gated_metric_names,
            "max_regression_pct": config.max_regression_pct,
            "metrics": improvements,
            "constraints_total": len(config.constraints),
            "still_failing": self.failing_names(),
            "iterations": [
                {
                    "iteration": r.iteration,
                    "accepted": r.accepted,
                    "reason": r.reason,
                    "metrics": r.metrics,
                    "slot": (
                        s.label if (s := self.schedule.slot_for(r.iteration)) else None
                    ),
                }
                for r in history
            ],
            "final_verdict": self.read_gate(),
        }
        path = self.project_path / STATE_DIR / "run-summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2) + "\n")
        return path
