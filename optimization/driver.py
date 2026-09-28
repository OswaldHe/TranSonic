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
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from autohelix.agents import AgentConfig, AgentEvent, create_agent
from optimization import constraints as cons
from optimization import custody, materialize, presets
from optimization import feedback as fb
from optimization.config import GATE_PYTHON, PipelineConfig
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

    def _set_aside(self, repo: Path, stage: str, attempt: int) -> None:
        """Move a failed attempt's repo aside rather than deleting it.

        The first real run reached 5 of 7 checks on attempt 1, failed on two gate bugs, and the work
        was gone before it could be read — attempt 2 had already overwritten it. A failed attempt is
        the most informative artifact a preparation stage produces.

        What it costs: the carried-in `module/tensors/` are hard links to the bootstrapped repo, so
        they add no disk, but the slices the agent cut for itself are its own bytes and those do —
        on the order of the submodule's share of the weights per attempt, 1.8 GB for a quarter of
        this MoE. `du` on the attic reports the hard links too, so it reads far larger than the space
        actually held. Only the most recent `preparation.retries` attempts are kept, so the ceiling
        is bounded rather than growing with the run.
        """
        self._prune_attempts(stage, keep=self.config.preparation_retries)
        if not repo.exists():
            return
        attic = self.state_dir / "attempts" / f"{stage}-{attempt - 1}"
        attic.parent.mkdir(parents=True, exist_ok=True)
        if attic.exists():
            shutil.rmtree(attic)
        repo.rename(attic)
        self.console.print(f"  [dim]previous attempt kept at {attic}[/dim]")

    def _prune_attempts(self, stage: str, keep: int) -> None:
        """Drop all but the newest ``keep`` set-aside attempts for one stage.

        Each one holds the tensor slices its agent cut, so without a ceiling a run that retries
        repeatedly across several invocations accumulates gigabytes of superseded work.
        """
        attic = self.state_dir / "attempts"
        if not attic.is_dir() or keep < 0:
            return
        existing = sorted(
            (p for p in attic.iterdir() if p.is_dir() and p.name.startswith(f"{stage}-")),
            key=lambda p: p.stat().st_mtime,
        )
        for path in existing[: max(len(existing) - keep, 0)]:
            shutil.rmtree(path, ignore_errors=True)
            self.console.print(f"  [dim]pruned superseded attempt {path.name}[/dim]")

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

        # `init` records the projection and the later stages are documented as using that frozen
        # result, so the recorded one is *the* projection whenever it exists. A separate
        # `optimize submodule` or `assemble` invocation builds a fresh Pipeline, and recomputing
        # from whatever the scheme says now would let a scheme edited mid-run re-plan the module:
        # the assembly would be built against a cut the submodule repo was never made for.
        recorded = (self._read_record("projection") or {}).get("projection") or {}
        try:
            projection = project_module(
                self.config.scheme, self.config.module_id, self.config.target_units,
            )
        except ProjectionError as exc:
            if not recorded:
                raise StageError(str(exc)) from exc
            # The scheme no longer projects at all. The workspace is still coherent, so the
            # recorded projection stands and the run continues from it.
            projection = Projection.from_dict(recorded)

        if recorded:
            frozen = Projection.from_dict(recorded)
            # Every field a later stage builds on, not just the width: `expert x4` and `head x4`
            # are the same number of ranks and entirely different kernels.
            drift = frozen.differences(projection)
            if drift:
                raise StageError(
                    f"{self.config.scheme.name} no longer projects {self.config.module_id} the way "
                    f"this workspace was initialized:\n  "
                    + "\n  ".join(drift)
                    + f"\nThe repos already built assume the recorded projection.\n"
                    f"Either restore the scheme, or start a fresh workspace.root for the new one."
                )
            projection = frozen

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
        for warning in self.config.warnings():
            self.console.print(f"  [yellow]note[/yellow] {warning}")
        # Probe the interpreter the *derived commands* invoke, which is bare `python` from PATH, not
        # `sys.executable`. Launched through an absolute venv console script, pipx, or any wrapper
        # that does not put its own bin first, the two are different interpreters — and this
        # preflight claimed to prove something about an interpreter no iteration ever runs.
        resolved = shutil.which(GATE_PYTHON)
        if resolved is None:
            problems.append(
                f"the gate and metric commands run `{GATE_PYTHON} -m optimization.*`, but "
                f"'{GATE_PYTHON}' is not on PATH. Activate {self.config.venv or 'the venv'}"
            )
        else:
            probe = subprocess.run(
                [resolved, "-c",
                 "import optimization.submodule_checker, optimization.module_checker, "
                 "torch, torch_neuronx; print('ok')"],
                capture_output=True, text=True,
            )
            if probe.returncode != 0:
                tail = (probe.stderr.strip().splitlines() or ["unknown import error"])[-1]
                note = ("" if Path(resolved).resolve() == Path(sys.executable).resolve() else
                        f" Note that this is not the interpreter running autohelix "
                        f"({sys.executable}); the commands use `{GATE_PYTHON}` from PATH.")
                problems.append(
                    f"the gate runs `{GATE_PYTHON} -m optimization.*_checker`, and {resolved} "
                    f"cannot import what it needs: {tail}. "
                    f"Activate {self.config.venv or 'the venv'}.{note}"
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
        self._require_expressible_cut(projection)  # before hours of work, not after
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

        source_tensors = materialize.fingerprint_tensors(self.config.bootstrap_repo / "tensors")
        self._record("source-tensors", {"tensors": source_tensors})
        self.console.print(f"  recorded {len(source_tensors)} source tensor hash(es)")

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

    def _verify_source_tensors(self, after: str) -> None:
        """Refuse to go on if the bootstrapped module's recorded bytes have changed.

        Run after each stage that gave an agent write access to a repo holding hard links to them.
        See `materialize.fingerprint_tensors` for why the record lives outside every repo.
        """
        record = (self._read_record("source-tensors") or {}).get("tensors") or {}
        if not record:
            return
        findings = materialize.verify_tensors(record, self.config.bootstrap_repo / "tensors")
        if not findings:
            return
        for finding in findings[:10]:
            self.console.print(f"  [red]FAIL[/red] {finding}")
        raise StageError(
            f"{len(findings)} of the bootstrapped module's recorded tensors changed during "
            f"{after}. They are hard-linked into the optimization repos, so a write through either "
            f"link reaches {self.config.bootstrap_repo}/tensors — and those bytes are the golden "
            f"every stage is judged against.\n"
            f"Restore them from the partition artifact before continuing; nothing measured after "
            f"this point means anything."
        )

    def _require_expressible_cut(self, projection: Projection) -> None:
        """Refuse a projection the declaration format cannot state, instead of mis-stating it.

        `submodule.json` carries one scalar `dim`, one `factor` and one reassembly operation, and the
        prompt used to be handed `projected[0].dim` with `projected_units` as its width. For a
        projection like `head x2 * hidden x2` that presents a four-way `head` cut when the head factor
        is two, and the two dimensions need different collectives to rejoin — so the agent would be
        asked for a cut that cannot be described, let alone verified.

        No module in the shipped DeepSeek schemes projects to more than one factor, so this refuses
        rather than guesses. Supporting mixed-dimensional cuts means giving the declaration per-rank
        coordinates and a reassembly per dimension, which is a real feature and not a default.
        """
        if len(projection.projected) <= 1:
            return
        shape = " * ".join(f.label() for f in projection.projected)
        raise StageError(
            f"{self.config.module_id} projects onto {shape}, which is more than one dimension.\n"
            f"The submodule declaration states a single `dim`, `factor` and reassembly operation, so "
            f"a mixed-dimensional cut cannot be described in it — and each dimension would need its "
            f"own collective to rejoin.\n"
            f"Pick a module whose projection is one dimension wide, or narrow this scheme's "
            f"placement for {self.config.module_id} to a single split."
        )

    def submodule(self) -> StageOutcome:
        """Cut the module down to one rank, and accept the repo only when the gate agrees."""
        projection = self.projection()
        self._require_expressible_cut(projection)
        repo = self.config.submodule_repo
        prompt_template = presets.submodule_prompt()

        for attempt in range(1, self.config.preparation_retries + 1):
            self.console.print(f"\n[bold]submodule, attempt {attempt}[/bold] → {repo}")
            self._set_aside(repo, "submodule", attempt)
            prepared = materialize.materialize_submodule(
                repo=repo, bootstrap_repo=self.config.bootstrap_repo,
                artifact=self.config.artifact, projection=projection,
                module_id=self.config.module_id,
            )
            for line in prepared.get("stripped") or []:
                self.console.print(f"  [dim]stripped {line}[/dim]")
            prompt = _render(prompt_template, {
                "repo": str(repo), "module": self.config.module_id,
                "entry_point": "kernel",
                "dim": (projection.projected[0].dim if projection.projected else "none"),
                "factor": projection.projected_units,
                "splits": " * ".join(f.label() for f in projection.projected) or "no split",
            })
            held = custody.take_custody(
                prepared, custody.SUBMODULE_OWNED,
                self.state_dir / "custody" / f"submodule-{attempt}.json",
            )
            if not self._run_one_shot(repo, prompt, f"submodule-{attempt}",
                                      self.config.preparation_timeout):
                continue

            for line in custody.restore(
                repo / ".autohelix" / "optimization" / "submodule.json", held,
            ):
                self.console.print(f"  [yellow]![/yellow] manifest: {line}")

            ok, report = self._run_gate(repo, "optimization.submodule_checker", "submodule")
            self._record(f"submodule-attempt-{attempt}", {"passed": ok, "report": report})
            if ok:
                materialize.git_init(repo, f"Submodule baseline: one rank of {self.config.module_id}")
                latency = _latency_from_gate(
                    repo / ".autohelix" / "optimization" / "submodule-gate.json"
                )
                self._verify_source_tensors("the submodule stage")
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
        spec = self.config.submodule if stage == "submodule" else self.config.full
        stale = cons.schedule_drift(repo, spec.schedule)
        if stale:
            for line in stale:
                self.console.print(f"  [yellow]![/yellow] {line}")
            # Recompiling rather than reusing. A checker compiled from different prose than the
            # prompt now carries is the worst of both: the agent is told one rule and judged by
            # another, and a slot whose checker is simply absent runs unenforced while the prompt
            # still claims it is checked.
            self.compile_constraints(stage)
        for finding in cons.verify_manifest(repo):
            raise StageError(f"a compiled checker has changed since it was written: {finding}")

        self._seed_baseline_verdict(repo, stage)
        config_path = self.config.write_loop_config(stage)
        loop = OptimizationLoop(repo, config_file=config_path, stage=stage, verbose=self.verbose)
        # `Harness.run()` reports a dirty repository, a missing editable file or rejected config
        # drift by printing and returning normally, so "it ran" cannot be read off the call. Count
        # the iterations it recorded instead: an aborted loop leaves the history where it was, and
        # `optimize all` would otherwise walk straight into assembling a submodule never optimized.
        before = len(loop.history.load())
        loop.run()
        after = len(loop.history.load())
        budgeted = (self.config.submodule if stage == "submodule" else self.config.full).iterations
        if after == before and budgeted > 0:
            raise StageError(
                f"the {stage} loop recorded no iteration, so it did not start. The usual causes "
                f"print above: the repository has uncommitted changes, a file named in "
                f"scope.editable is missing, or the derived config no longer matches the saved "
                f"run state."
            )

        best_commit, best_value = loop.best_commit()
        return StageOutcome(
            stage=f"run-{stage}", ok=best_value is not None,
            detail=f"best {best_value} ms" if best_value else "no accepted iteration",
            payload={"best_commit": best_commit, "best_ms": best_value},
        )

    def _seed_baseline_verdict(self, repo: Path, stage: str) -> None:
        """Put the acceptance gate's verdict where iteration 0's metric command will look.

        `Harness._capture_baseline` runs the *metric* commands and not the constraints, in the
        project root rather than a worktree. Here the metric is read back out of the gate's verdict
        — deliberately, so the gate is the only thing that ever runs the candidate — and at iteration
        0 no constraint has run, so there is no verdict and the loop aborts before its first
        iteration. That is what happened on the first attempt at stage 3.

        The verdict is not synthesized. The stage's acceptance gate ran the validator on exactly this
        code minutes earlier and wrote its measurement, including the freshness checks that prove the
        profile came from that run; this copies it to the filename the loop reads. Iteration 1
        overwrites it with its own.
        """
        name = "submodule-gate.json" if stage == "submodule" else "module-gate.json"
        source = repo / ".autohelix" / "optimization" / name
        target = repo / ".autohelix" / "optimization" / "gate.json"
        if not source.is_file():
            self.console.print(
                f"  [yellow]![/yellow] no {name} to seed the baseline from; iteration 0 will have "
                f"no measurement and the loop will refuse to start"
            )
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

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
        # The submodule is measured at the *commit the assembly is built from*, not at `HEAD`. With
        # 5% of regression slack those differ, and measuring one while assembling the other would
        # make the 1.1x ceiling describe code that is not in the assembly.
        best_commit, _ = _best_from_summary(self.config.submodule_repo)
        with self._at_commit(self.config.submodule_repo, best_commit) as repo:
            submodule_ms = self._measure(
                repo, f"the optimized submodule at {(best_commit or 'HEAD')[:12]} (1 core)",
                cores="1",
            )
        self._record("baselines", {
            "bootstrap_latency_ms": bootstrap_ms,
            "submodule_latency_ms": submodule_ms,
            "overhead_ceiling_ms": round(submodule_ms * 1.10, 6),
            "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        return bootstrap_ms, submodule_ms

    @contextmanager
    def _at_commit(self, repo: Path, commit: str | None):
        """Yield the repo with one file — `source.py` — at ``commit``, then put it back.

        Only `source.py`, because that is the only file an iteration may change; everything else at
        `HEAD` is identical to everything else at `commit` by construction. Swapping one file rather
        than checking out a commit keeps the profile artifacts, the tensors and `.autohelix/` exactly
        where the validator expects them, and leaves nothing to clean up if the measurement fails.
        """
        target = repo / "source.py"
        if not commit or not target.is_file():
            yield repo
            return
        shown = subprocess.run(
            ["git", "show", f"{commit}:source.py"],
            cwd=repo, capture_output=True, text=True, check=False,
        )
        if shown.returncode != 0:
            self.console.print(
                f"  [yellow]![/yellow] could not read source.py at {commit[:12]}; "
                f"measuring HEAD instead"
            )
            yield repo
            return
        original = target.read_text()
        if shown.stdout == original:
            yield repo
            return
        target.write_text(shown.stdout)
        try:
            yield repo
        finally:
            target.write_text(original)

    def _measure(self, repo: Path, label: str, cores: str) -> float:
        from optimization import candidate

        self.console.print(f"  measuring {label} in {repo}")
        outcome = candidate.run_candidate(
            repo, [sys.executable, "inference.py"],
            env_overrides={"NEURON_RT_NUM_CORES": cores},
        )
        latency = candidate.marker_value(outcome.output, METRIC)
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
            self._set_aside(repo, "assemble", attempt)
            prepared = materialize.materialize_full(
                repo=repo, bootstrap_repo=self.config.bootstrap_repo,
                artifact=self.config.artifact, submodule_repo=submodule_repo,
                projection=projection, module_id=self.config.module_id,
                bootstrap_latency_ms=bootstrap_ms, submodule_latency_ms=submodule_ms,
                best_commit=best_commit,
            )
            for line in prepared.get("stripped") or []:
                self.console.print(f"  [dim]stripped {line}[/dim]")
            prompt = _render(prompt_template, {
                "repo": str(repo), "module": self.config.module_id, "entry_point": "kernel",
                "ranks": ranks, "last_rank": ranks - 1,
                "bootstrap_latency": f"{bootstrap_ms:g}",
                "submodule_latency": f"{submodule_ms:g}",
                "overhead_ceiling": f"{submodule_ms * 1.10:g}",
            })
            held = custody.take_custody(
                prepared, custody.MODULE_OWNED,
                self.state_dir / "custody" / f"assemble-{attempt}.json",
            )
            if not self._run_one_shot(repo, prompt, f"assemble-{attempt}",
                                      self.config.preparation_timeout):
                continue

            for line in custody.restore(
                repo / ".autohelix" / "optimization" / "module.json", held,
            ):
                self.console.print(f"  [yellow]![/yellow] manifest: {line}")

            ok, report = self._run_gate(repo, "optimization.module_checker", "module")
            self._record(f"assemble-attempt-{attempt}", {"passed": ok, "report": report})
            if ok:
                materialize.git_init(
                    repo, f"Assembly baseline: {self.config.module_id} on {ranks} ranks",
                )
                latency = _latency_from_gate(
                    repo / ".autohelix" / "optimization" / "module-gate.json"
                )
                self._verify_source_tensors("the assembly stage")
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

    # -- stage: feedback -----------------------------------------------------------

    def feedback(self) -> StageOutcome:
        """One agent reads everything both loops wrote and reports what stopped them.

        Last, because it needs the whole run: a blocker is only a blocker once the iterations that
        might have worked around it have been spent. It reads rather than measures, so unlike every
        other stage it changes no repository and has no gate — `fb.validate_report` checks the
        deliverable's shape, and nothing checks whether a finding is true, because nothing could.
        """
        sub, full = self.config.submodule_repo, self.config.full_repo
        files = fb.corpus(sub, full)
        if not files:
            raise StageError(
                f"neither {sub.name} nor {full.name} has any notes or reviews to read, so there is "
                f"nothing to reconcile. Run the loops first."
            )

        listing = "\n".join(
            f"- `{repo}/.autohelix/` — {len(paths)} file(s): "
            + ", ".join(sorted({p.parent.name for p in paths}))
            for repo, paths in files.items()
        )
        words = fb.word_count(files)
        baselines = self._read_record("baselines")
        sub_summary = _read_summary(sub, "submodule")
        full_summary = _read_summary(full, "full")

        # The prompt promises the agent that REPORT.md is there to read, and in the `all()` flow
        # nothing had written it: `optimize report` is a separate command and not one of the stages.
        # Written here rather than in `all()` so the standalone command works the same way.
        try:
            from optimization.report import write_report

            write_report(self.config, self.console)
        except Exception as exc:  # the report is context, not a precondition
            self.console.print(f"  [yellow]![/yellow] could not write the run report: {exc}")

        deliverables = fb.snapshot_deliverables(sub, full)
        fb.repro_dir(self.config.workspace_root).mkdir(parents=True, exist_ok=True)
        prompt = _render(presets.feedback_prompt(), {
            "module": self.config.module_id,
            "corpus": listing,
            "words": f"{words:,}",
            "submodule_repo": str(sub),
            "full_repo": str(full),
            "workspace_root": str(self.config.workspace_root),
            "report_path": str(fb.report_path(self.config.workspace_root)),
            "repro_dir": fb.REPRO_DIR,
            "context_md": str(_context_md()),
            "levels": "\n".join(f"- **{key}** — {text}" for key, text in fb.LEVELS.items()),
            "ranks": self.projection().projected_units,
            "bootstrap_latency": _ms(baselines.get("bootstrap_latency_ms")),
            "submodule_latency": _ms(sub_summary.get("best_ms")),
            "full_latency": _ms(full_summary.get("best_ms")),
        })

        self.console.print(
            f"  reading {sum(len(p) for p in files.values())} note(s) and review(s), "
            f"~{words:,} words"
        )
        for attempt in range(1, self.config.preparation_retries + 1):
            ok = self._run_one_shot(
                self.config.workspace_root, prompt, f"feedback-{attempt}",
                self.config.feedback_timeout, model=self.config.feedback_model,
            )
            # The prompt asks this agent to test disputed claims on the device, and it runs at the
            # workspace root with both finished repositories writable under it — no worktree, because
            # it is not producing a candidate. So the deliverables are put back rather than trusted:
            # the first real run compiled inside the submodule repo and left artifacts in its
            # `build/`, which is harmless, but nothing was stopping it editing `source.py`.
            touched = fb.restore_deliverables(deliverables)
            problems = fb.validate_report(self.config.workspace_root)
            self._record(f"feedback-attempt-{attempt}",
                         {"agent_ok": ok, "problems": problems, "restored": touched})
            for name in touched:
                self.console.print(
                    f"  [yellow]![/yellow] the feedback agent changed {name}; restored it from the "
                    f"gated version"
                )
            if ok and not problems:
                counts = fb.summarize(self.config.workspace_root)
                levels = ", ".join(f"{k} {v}" for k, v in counts["by_level"].items())
                self.console.print(
                    f"  [green]feedback written[/green] — {counts['findings']} finding(s) "
                    f"({levels}) in {fb.REPORT_NAME}"
                )
                return StageOutcome("feedback", True,
                                    f"{counts['findings']} finding(s)", counts)
            for problem in problems:
                self.console.print(f"  [yellow]![/yellow] {problem}")

        raise StageError(
            f"the feedback report did not pass its structural check in "
            f"{self.config.preparation_retries} attempt(s). The last problems are listed above; "
            f"{fb.report_path(self.config.workspace_root)} is on disk to read either way."
        )

    # -- the whole thing -----------------------------------------------------------

    def all(self) -> list[StageOutcome]:
        """Every stage in order, stopping at the first that did not succeed.

        Each stage here is minutes to hours of device time, so a failure has to stop the sequence
        rather than be collected into the returned list: `run_loop` reports "no accepted iteration"
        as `ok=False` without raising, and the earlier version appended that and walked into
        assembly. The CLI cannot catch it either — what it receives from this method is a list, which
        has no `ok` to inspect.
        """
        outcomes: list[StageOutcome] = []

        def step(outcome: StageOutcome) -> None:
            outcomes.append(outcome)
            if not outcome.ok:
                raise StageError(
                    f"{outcome.stage} did not succeed ({outcome.detail}), so the remaining stages "
                    f"are not started. Fix that stage and re-run it on its own, then `optimize all` "
                    f"again — the finished stages are recorded and will not be redone."
                )

        step(self.init())
        step(self.submodule())
        step(self.compile_constraints("submodule"))
        step(self.run_loop("submodule"))
        step(self.assemble())
        if self.config.full.schedule.enforceable():
            step(self.compile_constraints("full"))
        step(self.run_loop("full"))
        step(self.feedback())
        return outcomes


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _render(template: str, variables: dict[str, Any]) -> str:
    """Render a packaged prompt with Jinja, the same engine the loop's prompts use."""
    from autohelix.prompt_template import render_template

    return render_template(template, variables)


def _read_summary(repo: Path, stage: str) -> dict[str, Any]:
    path = repo / ".autohelix" / "optimization" / f"{stage}-summary.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def _ms(value: Any) -> str:
    return f"{float(value):g} ms" if isinstance(value, (int, float)) else "not recorded"


def _context_md() -> Path:
    """This project's glossary, which the feedback agent is told to write against.

    Not `__file__/../../CONTEXT.md`: in an installed wheel that resolves to a `site-packages` path
    with nothing at it, so the prompt pointed its agent at a glossary that did not exist and the
    vocabulary step it is told to take had nothing to take. `pyproject.toml` force-includes the file
    into the package's templates, and the repository-root copy is the fallback for a source checkout
    where the package data is not present.
    """
    packaged = Path(__file__).resolve().parent / "templates" / "CONTEXT.md"
    if packaged.is_file():
        return packaged
    return Path(__file__).resolve().parent.parent / "CONTEXT.md"


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
