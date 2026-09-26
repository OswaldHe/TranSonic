# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The five stages, in order, and the seams between them.

    init       read the config, project the floorplan placement onto one device, make the workspace
    submodule  a one-shot agent cuts the module down to one rank; the generic gate accepts the repo
    run        the optimization loop, N iterations under the per-iteration constraint schedule
    assemble   a one-shot agent rebuilds all four ranks with a collective; the semantic gate accepts
    run-full   the optimization loop again, on the whole module

Two things about the seams are load-bearing.

**The preparation stages are one-shot agents with retries, not loops.** An iteration loop makes sense
when there is a metric to improve and a working baseline to improve from. A half-materialized repo
has neither: it is not a worse starting point than the last attempt, it is not a starting point. So a
failed attempt is thrown away whole and retried from the scaffolding, and after `preparation.retries`
the stage stops and reports rather than handing the next loop a baseline that does not hold.

**The constraint compiler runs before the loop's first iteration, once.** A checker written at the top
of iteration 7 could be written around what iteration 6 already did.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from autohelix.agents import AgentConfig, AgentEvent, create_agent
from optimization import constraints as cons
from optimization import materialize, presets
from optimization.config import PipelineConfig
from optimization.loop import METRIC, OptimizationLoop
from optimization.projection import Projection, ProjectionError, project_module

#: Where the pipeline's own records go, under the workspace root: the projection, each preparation
#: attempt's log and verdict, and the final report.
STATE_DIR = ".optimization"


class StageError(RuntimeError):
    """A stage could not complete, and the next one must not start."""


@dataclass
class StageOutcome:
    """What one stage achieved."""

    stage: str
    ok: bool
    detail: str = ""
    payload: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.payload is None:
            self.payload = {}


class Pipeline:
    """The five stages, driven from one pipeline config."""

    def __init__(self, config: PipelineConfig, console: Console | None = None,
                 verbose: bool = False) -> None:
        self.config = config
        self.console = console or Console()
        self.verbose = verbose
        self.state_dir = config.workspace_root / STATE_DIR
        self._projection: Projection | None = None

    # -- shared --------------------------------------------------------------------

    def _record(self, name: str, payload: dict[str, Any]) -> Path:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        path = self.state_dir / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2))
        return path

    def _read_record(self, name: str) -> dict[str, Any]:
        path = self.state_dir / f"{name}.json"
        if not path.is_file():
            return {}
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return {}

    def projection(self) -> Projection:
        """The module's placement, projected onto one device.

        Computed once and recorded, so every later stage reads the same projection even if the
        scheme file changes underneath the run.
        """
        if self._projection is not None:
            return self._projection
        try:
            projection = project_module(
                self.config.scheme, self.config.module_id, self.config.target_units,
            )
        except ProjectionError as exc:
            raise StageError(str(exc)) from exc

        if projection.diverges and self.config.on_oversized == "error":
            raise StageError(
                f"{self.config.module_id} is placed on {projection.planned_units} unit(s) across "
                f"{projection.planned_devices} device(s) in {self.config.scheme.name}, which does "
                f"not fit one device, and floorplan.on_oversized is 'error'.\n"
                f"Set it to 'project' to narrow the split to "
                f"{' * '.join(f'{f.dim}x{f.factor}' for f in projection.projected)} and record the "
                f"divergence, or pick a module the scheme already places on one device."
            )
        self._projection = projection
        return projection

    def _preflight(self) -> list[str]:
        """What has to be true before any device work starts.

        Checked up front because each of these fails *late* otherwise — four minutes into an
        iteration, or in the middle of a four-rank run — and the failure looks like the agent's
        fault rather than the environment's.
        """
        problems = self.config.validate()
        probe = subprocess.run(
            [sys.executable, "-c",
             "import optimization.submodule_checker, optimization.module_checker, "
             "torch, torch_neuronx; print('ok')"],
            capture_output=True, text=True,
        )
        if probe.returncode != 0:
            tail = (probe.stderr.strip().splitlines() or ["unknown import error"])[-1]
            problems.append(
                f"the gate runs `python -m optimization.*_checker`, but this interpreter cannot "
                f"import what it needs: {tail}. Activate {self.config.venv or 'the venv'}"
            )
        for tool in ("torchrun", "neuron-explorer"):
            if shutil.which(tool) is None:
                problems.append(f"'{tool}' is not on PATH; stage 4 and 5 cannot run without it")
        return problems

    # -- the one-shot preparation agents -------------------------------------------

    def _run_one_shot(
        self, repo: Path, prompt: str, label: str, timeout: str, model: str | None = None,
    ) -> bool:
        """Run a preparation agent once in the repo itself, not in a worktree.

        No worktree, because there is nothing to isolate: the repo is being built, a failed attempt
        is discarded whole, and scope enforcement has nothing to enforce against yet.
        """
        from autohelix.iteration_time import parse_duration

        log_dir = self.state_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{label}.log"

        agent_config = AgentConfig(
            type=self.config.agent_type, model=model,
            timeout_seconds=parse_duration(timeout),
        )
        agent = create_agent(agent_config)

        last = {"text": ""}

        def on_event(event: AgentEvent) -> None:
            if event.text:
                last["text"] = event.text.strip().splitlines()[-1][:110] if event.text.strip() else ""
            if self.verbose and event.text:
                self.console.print(f"  [dim]{event.text.strip()[:160]}[/dim]")

        self.console.print(f"  running the {label} agent (log: {log_path})")
        started = time.monotonic()
        result = agent.run(
            worktree_path=repo, prompt=prompt, iteration=0,
            log_path=log_path, event_callback=on_event, project_path=repo,
        )
        elapsed = time.monotonic() - started
        if not result.success:
            self.console.print(
                f"  [red]the {label} agent failed[/red] after {elapsed:.0f}s: "
                f"{result.error or f'exit {result.exit_code}'}"
            )
            return False
        self.console.print(f"  the {label} agent finished in {elapsed / 60:.1f} min")
        return True

    def _run_gate(self, repo: Path, module: str, label: str) -> tuple[bool, str]:
        """Run one of the two gates by hand, outside a loop, and report what it said."""
        verdict_path = repo / ".autohelix" / "optimization" / f"{label}-gate.json"
        verdict_path.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [sys.executable, "-m", module, "--repo", ".", "--json", str(verdict_path)],
            cwd=repo, capture_output=True, text=True,
        )
        report = result.stdout + result.stderr
        colour = "green" if result.returncode == 0 else "yellow"
        self.console.print(Panel(Text(report.strip() or "(no output)"),
                                 title=f"{label} gate", border_style=colour))
        return result.returncode == 0, report

    # -- stage: init ---------------------------------------------------------------

    def init(self) -> StageOutcome:
        """Validate the config, project the placement, and make the workspace."""
        problems = self._preflight()
        if problems:
            for problem in problems:
                self.console.print(f"  [red]FAIL[/red] {problem}")
            raise StageError(f"{len(problems)} configuration problem(s); nothing was created")

        projection = self.projection()
        self.config.workspace_root.mkdir(parents=True, exist_ok=True)
        payload = {
            "module": self.config.module_id,
            "scheme": str(self.config.scheme),
            "bootstrap_repo": str(self.config.bootstrap_repo),
            "projection": projection.to_dict(),
            "submodule_repo": str(self.config.submodule_repo),
            "full_repo": str(self.config.full_repo),
        }
        self._record("projection", payload)

        self.console.print(Panel(Text(projection.describe().strip()),
                                 title="projection", border_style="cyan"))
        if projection.diverges:
            self.console.print(
                f"  [yellow]note:[/yellow] this projection diverges from the ranked plan — a rank "
                f"holds {projection.weight_residency_ratio():g}x the planned weight bytes"
            )
        return StageOutcome("init", True, f"projected to {projection.projected_units} rank(s)",
                            payload)

    # -- stage: submodule ----------------------------------------------------------

    def submodule(self) -> StageOutcome:
        """Cut the module down to one rank, and accept the repo only when the gate agrees."""
        projection = self.projection()
        repo = self.config.submodule_repo
        prompt_template = presets.submodule_prompt()

        for attempt in range(1, self.config.preparation_retries + 1):
            self.console.print(f"\n[bold]submodule, attempt {attempt}[/bold] → {repo}")
            if repo.exists():
                shutil.rmtree(repo)
            materialize.materialize_submodule(
                repo=repo, bootstrap_repo=self.config.bootstrap_repo,
                artifact=self.config.artifact, projection=projection,
                module_id=self.config.module_id,
            )
            prompt = _render(prompt_template, {
                "repo": str(repo), "module": self.config.module_id,
                "entry_point": "kernel",
                "dim": (projection.projected[0].dim if projection.projected else "none"),
                "factor": projection.projected_units,
            })
            if not self._run_one_shot(repo, prompt, "submodule",
                                      self.config.preparation_timeout):
                continue

            ok, report = self._run_gate(repo, "optimization.submodule_checker", "submodule")
            self._record(f"submodule-attempt-{attempt}", {"passed": ok, "report": report})
            if ok:
                materialize.git_init(repo, f"Submodule baseline: one rank of {self.config.module_id}")
                latency = _latency_from_gate(
                    repo / ".autohelix" / "optimization" / "submodule-gate.json"
                )
                self.console.print(
                    f"  [green]submodule accepted[/green] — baseline {latency or float('nan'):g} ms"
                )
                return StageOutcome("submodule", True, f"baseline {latency} ms",
                                    {"latency_ms": latency, "attempt": attempt})

        raise StageError(
            f"the submodule repo did not pass its gate in {self.config.preparation_retries} "
            f"attempt(s). The reports are in {self.state_dir}; read the last one before raising the "
            f"retry count, since a repeated failure is usually the prompt or the module, not luck."
        )

    # -- stage: the constraint compiler --------------------------------------------

    def compile_constraints(self, stage: str) -> StageOutcome:
        """Write one checker per enforceable slot, before the loop's first iteration."""
        spec = self.config.submodule if stage == "submodule" else self.config.full
        repo = self.config.submodule_repo if stage == "submodule" else self.config.full_repo
        schedule = spec.schedule
        enforceable = schedule.enforceable()
        if not enforceable:
            self.console.print("  no enforceable constraint slots; nothing to compile")
            return StageOutcome("compile-constraints", True, "no slots", {})

        target = repo / cons.CONSTRAINTS_REL
        target.mkdir(parents=True, exist_ok=True)
        lines = []
        for slot in schedule.slots:
            path = cons.checker_path(repo, slot)
            if slot.has_text and slot.enforce:
                lines.append(
                    f"### Slot `{slot.label}` — iterations {slot.iterations}\n"
                    f"Write its checker to `{path}`.\n\n"
                    f"Its constraint, verbatim:\n\n"
                    f"```\n{slot.text.strip()}\n```\n"
                )
            else:
                lines.append(
                    f"### Slot `{slot.label}` — iterations {slot.iterations}\n"
                    f"Unconstrained (no text, or `enforce: false`). No checker needed.\n"
                )
        prompt = _render(presets.compiler_prompt(), {
            "slots": "\n".join(lines),
            "contract": cons.CHECKER_CONTRACT,
            "max_iterations": spec.iterations,
        })

        if not self._run_one_shot(repo, prompt, f"compiler-{stage}",
                                  self.config.compiler_timeout, self.config.compiler_model):
            raise StageError("the constraint compiler failed; the schedule would go unenforced")

        compiled: list[cons.CompiledSlot] = []
        problems: list[str] = []
        for slot in enforceable:
            path = cons.checker_path(repo, slot)
            findings = cons.validate_checker_source(path)
            if findings:
                problems += [f"slot {slot.label}: {f}" for f in findings]
                continue
            compiled.append(cons.CompiledSlot(
                label=slot.label, iterations=slot.iterations, path=path,
                sha256=cons.sha256_file(path),
            ))
        if problems:
            for problem in problems:
                self.console.print(f"  [red]FAIL[/red] {problem}")
            raise StageError(
                f"{len(problems)} compiled checker(s) are unusable. A schedule that cannot be "
                f"enforced is worse than no schedule: the prompts would claim a constraint the "
                f"loop does not apply."
            )
        cons.write_manifest(repo, compiled, schedule)
        self.console.print(
            f"  [green]compiled {len(compiled)} checker(s)[/green]: "
            f"{', '.join(c.label for c in compiled)}"
        )
        return StageOutcome("compile-constraints", True, f"{len(compiled)} checker(s)",
                            {"slots": [c.to_dict() for c in compiled]})

    # -- stage: the loops ----------------------------------------------------------

    def run_loop(self, stage: str) -> StageOutcome:
        """Run the optimization loop for one stage."""
        repo = self.config.submodule_repo if stage == "submodule" else self.config.full_repo
        if not repo.is_dir():
            raise StageError(
                f"{repo} does not exist — run `autohelix optimize "
                f"{'submodule' if stage == 'submodule' else 'assemble'}` first"
            )
        manifest = cons.read_manifest(repo)
        spec = self.config.submodule if stage == "submodule" else self.config.full
        if spec.schedule.enforceable() and not manifest.get("slots"):
            self.console.print(
                "  [yellow]![/yellow] no compiled checkers found; compiling them now"
            )
            self.compile_constraints(stage)
        for finding in cons.verify_manifest(repo):
            raise StageError(f"a compiled checker has changed since it was written: {finding}")

        config_path = self.config.write_loop_config(stage)
        loop = OptimizationLoop(repo, config_file=config_path, stage=stage, verbose=self.verbose)
        loop.run()
        best_commit, best_value = loop.best_commit()
        return StageOutcome(
            stage=f"run-{stage}", ok=best_value is not None,
            detail=f"best {best_value} ms" if best_value else "no accepted iteration",
            payload={"best_commit": best_commit, "best_ms": best_value},
        )

    # -- stage: assemble -----------------------------------------------------------

    def measure_baselines(self) -> tuple[float, float]:
        """Re-measure the two numbers stage 5 is bounded by, on this host, now.

        Not copied from the bootstrap log or from stage 3's history. Both bounds are comparisons
        against measurements taken at other times, and a latency from a different day on a possibly
        different toolchain is the kind of number that makes a bound quietly meaningless.
        """
        self.console.print("\n[bold]re-measuring the two baselines on this host[/bold]")
        bootstrap_ms = self._measure(
            self.config.bootstrap_repo, "the bootstrapped module (1 core)", cores="1",
        )
        submodule_ms = self._measure(
            self.config.submodule_repo, "the optimized submodule (1 core)", cores="1",
        )
        self._record("baselines", {
            "bootstrap_latency_ms": bootstrap_ms,
            "submodule_latency_ms": submodule_ms,
            "overhead_ceiling_ms": round(submodule_ms * 1.10, 6),
            "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        return bootstrap_ms, submodule_ms

    def _measure(self, repo: Path, label: str, cores: str) -> float:
        from optimization import gate

        self.console.print(f"  measuring {label} in {repo}")
        outcome = gate.run_candidate(
            repo, [sys.executable, "inference.py"],
            env_overrides={"NEURON_RT_NUM_CORES": cores},
        )
        latency = gate.marker_value(outcome.output, METRIC)
        if outcome.return_code != 0 or latency is None:
            tail = "\n".join(outcome.output.strip().splitlines()[-15:])
            raise StageError(
                f"could not measure {label}: exit {outcome.return_code}, "
                f"latency {latency}\n{tail}"
            )
        self.console.print(f"    {latency:g} ms")
        return latency

    def assemble(self) -> StageOutcome:
        """Rebuild the whole module across all ranks, and accept it only when the gate agrees."""
        projection = self.projection()
        repo = self.config.full_repo
        submodule_repo = self.config.submodule_repo
        if not submodule_repo.is_dir():
            raise StageError(f"{submodule_repo} does not exist — run the submodule stages first")

        best_commit, best_ms = _best_from_summary(submodule_repo)
        bootstrap_ms, submodule_ms = self.measure_baselines()
        ranks = projection.projected_units
        prompt_template = presets.assemble_prompt()

        for attempt in range(1, self.config.preparation_retries + 1):
            self.console.print(f"\n[bold]assemble, attempt {attempt}[/bold] → {repo}")
            if repo.exists():
                shutil.rmtree(repo)
            materialize.materialize_full(
                repo=repo, bootstrap_repo=self.config.bootstrap_repo,
                artifact=self.config.artifact, submodule_repo=submodule_repo,
                projection=projection, module_id=self.config.module_id,
                bootstrap_latency_ms=bootstrap_ms, submodule_latency_ms=submodule_ms,
                best_commit=best_commit,
            )
            prompt = _render(prompt_template, {
                "repo": str(repo), "module": self.config.module_id, "entry_point": "kernel",
                "ranks": ranks, "last_rank": ranks - 1,
                "bootstrap_latency": f"{bootstrap_ms:g}",
                "submodule_latency": f"{submodule_ms:g}",
                "overhead_ceiling": f"{submodule_ms * 1.10:g}",
            })
            if not self._run_one_shot(repo, prompt, "assemble", self.config.preparation_timeout):
                continue

            ok, report = self._run_gate(repo, "optimization.module_checker", "module")
            self._record(f"assemble-attempt-{attempt}", {"passed": ok, "report": report})
            if ok:
                materialize.git_init(
                    repo, f"Assembly baseline: {self.config.module_id} on {ranks} ranks",
                )
                latency = _latency_from_gate(
                    repo / ".autohelix" / "optimization" / "module-gate.json"
                )
                self.console.print(
                    f"  [green]assembly accepted[/green] — baseline {latency or float('nan'):g} ms "
                    f"against {bootstrap_ms:g} ms bootstrapped"
                )
                return StageOutcome("assemble", True, f"baseline {latency} ms", {
                    "latency_ms": latency, "bootstrap_latency_ms": bootstrap_ms,
                    "submodule_latency_ms": submodule_ms, "submodule_best_ms": best_ms,
                    "attempt": attempt,
                })

        raise StageError(
            f"the assembly did not pass its gate in {self.config.preparation_retries} attempt(s).\n"
            f"The two bounds it has to clear are {bootstrap_ms:g} ms (faster than) and "
            f"{submodule_ms * 1.10:g} ms (no slower than). If the reports show it matching the "
            f"golden but missing a bound, the submodule's cut is the thing to revisit, not the "
            f"assembly."
        )

    # -- the whole thing -----------------------------------------------------------

    def all(self) -> list[StageOutcome]:
        outcomes = [self.init()]
        outcomes.append(self.submodule())
        outcomes.append(self.compile_constraints("submodule"))
        outcomes.append(self.run_loop("submodule"))
        outcomes.append(self.assemble())
        if self.config.full.schedule.enforceable():
            outcomes.append(self.compile_constraints("full"))
        outcomes.append(self.run_loop("full"))
        return outcomes


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _render(template: str, variables: dict[str, Any]) -> str:
    """Render a packaged prompt with Jinja, the same engine the loop's prompts use."""
    from autohelix.prompt_template import render_template

    return render_template(template, variables)


def _latency_from_gate(path: Path) -> float | None:
    from optimization.readback import read_latency

    latency, _ = read_latency(path)
    return latency


def _best_from_summary(repo: Path) -> tuple[str | None, float | None]:
    """The submodule stage's best accepted commit and latency, from the summary it wrote."""
    path = repo / ".autohelix" / "optimization" / "submodule-summary.json"
    if not path.is_file():
        return None, None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None, None
    return payload.get("best_commit"), payload.get("best_ms")
