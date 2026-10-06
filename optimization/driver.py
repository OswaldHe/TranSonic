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

import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
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
from optimization import memory as mem
from optimization import feedback as fb
from optimization.config import GATE_PYTHON, PipelineConfig
from optimization.loop import METRIC, OptimizationLoop
from optimization.projection import Projection, ProjectionError, project_module

#: Where the pipeline's own records go, under the workspace root: the projection, each preparation
#: attempt's log and verdict, and the final report.
STATE_DIR = ".optimization"

#: How many times the constraint compiler may be asked for a slot's checker. Attempt 1 is the
#: plain request; the rest carry the validator's findings back. Three is enough for the failures
#: seen in practice, which are a single contract violation the findings name exactly.
COMPILER_ATTEMPTS = 3

#: Appended to the compiler's prompt on a retry. It names the files it has to fix and quotes the
#: contract again, because the violation is always a clause of the contract it did not apply.
COMPILER_RETRY = """

---

## Your previous attempt is on disk and it does not satisfy the contract

The checkers you wrote were validated and rejected for these reasons:

{{ problems }}

Rewrite the offending checker(s) in place, at the same paths, fixing exactly these findings and
changing nothing else about what they check. The constraint prose has not changed, and a checker
that now passes validation but stops judging the mechanism is worse than the one you wrote.

The most common cause is an import: the allowed list is short and deliberate, and anything outside
it has to be written inline instead. Here is the contract again, in full:

{{ contract }}
"""


#: The accuracy statistic both gates publish into their verdict, and the one the bars are pinned on.
ACCURACY_MARKER = "max_abs_err"


def _validator_sha256(repo: Path) -> str | None:
    """`inference.py`'s hash, or None if it is not there yet.

    The validator is frozen for the whole of the loop that follows a preparation stage — its hash
    is already a gate check — which makes it the one thing that identifies "the repo that passed"
    and stays true until the stage is rebuilt. `source.py` would not do: the loop rewrites it every
    iteration, so matching on it would refuse to resume exactly when resuming is wanted.
    """
    path = repo / "inference.py"
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
            # A set-aside repo holds an iteration worktree per iteration, each with its own
            # read-only memory snapshot, so a plain rmtree stops on the first one.
            mem.rmtree_unlocked(attic)
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
            # Unlocked, and not `ignore_errors` alone: a read-only snapshot made this give up
            # part-way and leave most of the attempt on disk, which is the opposite of pruning.
            mem.rmtree_unlocked(path, ignore_errors=True)
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
                    + "\nThe repos already built assume the recorded projection.\n"
                    "Either restore the scheme, or start a fresh workspace.root for the new one."
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

    def _seed_preparation_memory(self, repo: Path, spec: "mem.MemorySpec") -> str:
        """Seed the memory for a one-shot preparation agent, and return its prompt block.

        Seeded into the repo itself, because `_run_one_shot` runs the agent there rather than in a
        worktree. Safe to land under `.autohelix/`: it is gitignored, and `Sandbox.prepare_worktree`
        copies an explicit allowlist out of it (`notes`, `observations`, `logs`, `peer_notes`) that
        does not include `memory` — so a copy left here could not leak into a later iteration that
        opted out. `_drop_preparation_memory` removes it anyway once the agent is done, so the repo
        the next stage consumes does not carry a second copy or a read-only subtree.
        """
        if not spec.reads_at_preparation:
            return ""
        problem = mem.seed_problem(spec, repo)
        if problem:
            self.console.print(f"  [yellow]![/yellow] memory: {problem}")
            return ""
        seeded = mem.seed(spec, repo)
        if seeded:
            self.console.print(f"  memory: {seeded} file(s) from {spec.path} at {mem.SEEDED_REL}")
        else:
            self.console.print(f"  [yellow]![/yellow] memory: nothing seeded from {spec.path}")
        return mem.describe_for_preparation(spec, seeded)

    def _drop_preparation_memory(self, repo: Path) -> None:
        """Remove a seeded copy from a repo once its preparation agent has finished."""
        target = repo / mem.SEEDED_REL
        if not target.exists():
            return
        mem.unlock(repo)
        shutil.rmtree(target, ignore_errors=True)

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
        try:
            result = agent.run(
                worktree_path=repo, prompt=prompt, iteration=0,
                log_path=log_path, event_callback=on_event, project_path=repo,
            )
        finally:
            # Unconditional and a no-op when nothing was seeded, so every way out of a preparation
            # attempt — finished, failed, raised, retried — leaves the repo without a stale
            # read-only copy in it. A retry re-seeds from `_seed_preparation_memory`.
            self._drop_preparation_memory(repo)
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

    def submodule(self, rebuild: bool = False) -> StageOutcome:
        """Cut the module down to one rank, and accept the repo only when the gate agrees."""
        self._refuse_to_clobber("submodule", rebuild)
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
            memory_block = self._seed_preparation_memory(repo, self.config.submodule.memory)
            prompt = _render(prompt_template, {
                "repo": str(repo), "module": self.config.module_id,
                "entry_point": "kernel",
                "dim": (projection.projected[0].dim if projection.projected else "none"),
                "factor": projection.projected_units,
                "splits": " * ".join(f.label() for f in projection.projected) or "no split",
                "memory": memory_block,
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
            self._record(f"submodule-attempt-{attempt}", {
                "passed": ok, "report": report,
                "validator_sha256": _validator_sha256(repo),
            })
            if ok:
                for line in self._tighten_submodule_bar(repo):
                    self.console.print(f"  {line}")
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

        # The compiler is an agent, so a checker that breaks its contract is a normal outcome, not
        # an exceptional one — and the findings say exactly what to change. Hand them back and let
        # it fix its own work. This used to raise on the first attempt, which stopped a whole
        # pipeline over one `import operator` after the agent had already been paid for.
        compiled: list[cons.CompiledSlot] = []
        problems: list[str] = []
        for attempt in range(1, COMPILER_ATTEMPTS + 1):
            label = f"compiler-{stage}" if attempt == 1 else f"compiler-{stage}-retry-{attempt - 1}"
            this_prompt = prompt if not problems else prompt + _render(COMPILER_RETRY, {
                "problems": "\n".join(f"  - {p}" for p in problems),
                "contract": cons.CHECKER_CONTRACT,
            })
            if not self._run_one_shot(repo, this_prompt, label,
                                      self.config.compiler_timeout, self.config.compiler_model):
                raise StageError("the constraint compiler failed; the schedule would go unenforced")

            compiled, problems = [], []
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
            if not problems:
                break
            for problem in problems:
                self.console.print(f"  [red]FAIL[/red] {problem}")
            if attempt < COMPILER_ATTEMPTS:
                self.console.print(
                    f"  [yellow]retrying the constraint compiler with those findings "
                    f"({attempt}/{COMPILER_ATTEMPTS - 1})[/yellow]")

        if problems:
            # Out of attempts. An unvalidated checker is never enforced, so the unusable ones go --
            # `loop.run_iteration` then finds no checker, says so on every affected iteration, and
            # enforces nothing.
            unusable = [
                slot for slot in enforceable
                if not any(c.path == cons.checker_path(repo, slot) for c in compiled)
            ]
            for slot in unusable:
                path = cons.checker_path(repo, slot)
                if path.is_file():
                    path.unlink()
            # Which slots they were decides whether the stage may continue. Dropping a `soft`
            # slot's checker costs advice the loop was free to ignore. Dropping a `hard` one's
            # means the prompt still tells the agent the rule is mandatory while nothing checks it,
            # and candidates that violate it merge — the stage would report success having
            # delivered something weaker than the config asked for, which no later stage can
            # detect. So soft degrades and hard stops.
            hard = [slot for slot in unusable if slot.enforcement == "hard"]
            if hard:
                raise StageError(
                    f"{len(hard)} checker(s) for `enforcement: hard` slot(s) "
                    f"({', '.join(s.label for s in hard)}) are still unusable after "
                    f"{COMPILER_ATTEMPTS} attempt(s), and a hard slot may not run unchecked: the "
                    f"prompt would call the rule mandatory with nothing enforcing it.\n"
                    f"The compiler's findings are above. Either rewrite that slot's prose so a "
                    f"checker can be written against it, or set `enforcement: soft` to accept "
                    f"advice without a gate."
                )
            self.console.print(
                f"  [red]![/red] {len(problems)} checker(s) still unusable after "
                f"{COMPILER_ATTEMPTS} attempt(s); removed them. Those slots are soft, so they run "
                f"**unenforced** and every affected iteration will say so."
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

        # Before anything measures in the main repo, since a measurement rewrites these and the
        # loop will not start on a dirty tree.
        untracked = materialize.untrack_regenerated(repo)
        if untracked:
            self.console.print(
                f"  stopped tracking {len(untracked)} regenerated artifact(s) the toolchain "
                f"rewrites ({', '.join(untracked[:3])}{', …' if len(untracked) > 3 else ''})"
            )

        if stage == "submodule":
            lines = self._tighten_submodule_bar(repo)
            for line in lines:
                self.console.print(f"  {line}")
            # The validator only. The manifest beside it lives under `.autohelix/`, which the
            # repo's own `.gitignore` keeps out of the history.
            if lines and materialize.git_commit_paths(
                repo, "Re-pin the numerical bar to what this cut achieved", ["inference.py"],
            ):
                self.console.print("  committed the re-pinned bar, so the loop opens clean")
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
        # An exhausted budget is not an aborted start. `Harness.run()` walks an empty range when the
        # history already holds every budgeted iteration, so it returns with `after == before` —
        # exactly what an abort looks like from here. Re-running `optimize run` or `optimize all`
        # over a finished stage therefore failed instead of reusing the result it had already
        # produced. Told apart by whether the loop reached its last iteration, which an abort never
        # does.
        exhausted = self._last_iteration(loop) >= budgeted > 0
        if after == before and budgeted > 0 and not exhausted:
            raise StageError(
                f"the {stage} loop recorded no iteration, so it did not start. The usual causes "
                f"print above: the repository has uncommitted changes, a file named in "
                f"scope.editable is missing, or the derived config no longer matches the saved "
                f"run state."
            )
        if after == before and exhausted:
            self.console.print(
                f"  [dim]{stage}: all {budgeted} budgeted iteration(s) are already on record, so "
                f"the loop had nothing to run — reusing the recorded outcome[/dim]"
            )

        best_commit, best_value = loop.best_commit()
        # The best commit becomes the deliverable, not merely a metadata field. The metric gate
        # allows `acceptance.max_regression_pct`, so the last accepted iteration at HEAD can be
        # slower than an earlier accepted one — round 2 of the real stage 5 ended with HEAD at
        # 14.8703 ms while an earlier commit held 14.8456. Assembly reads the submodule's best commit
        # explicitly, but nothing downstream does that for the full repo, so the headline latency and
        # the `source.py` a reader opens described different code.
        if stage == "full" and best_commit:
            self._materialize_best(repo, best_commit, best_value)

        # **A timed-out loop is not a failed stage.** `step` in `all` turns `ok=False` into a
        # `StageError` and abandons every later stage, which is right when the loop produced nothing
        # usable — and wrong when the only thing that went wrong is that an agent ran over its
        # clock. The loop already behaves correctly in that case: a timeout is recorded as a
        # rejected iteration, nothing is merged, and the repo stays at the last accepted commit, so
        # there is always a kernel to hand on. With no accepted iteration at all that kernel is the
        # baseline cut, which `materialize_full` picks up anyway when `best_commit` is None.
        #
        # So an all-timeout loop reports ok and says so, rather than taking stages 4, 5, feedback
        # and report down with it. Any other reason for an empty loop still fails the stage.
        timed_out = sum(1 for r in loop.history.load()
                        if not r.accepted and "timed out" in (r.reason or ""))
        ok = best_value is not None or timed_out > 0
        if best_value is not None:
            detail = f"best {best_value} ms"
        elif timed_out:
            detail = (f"no accepted iteration; {timed_out} agent timeout(s) — carrying the "
                      f"baseline cut forward")
        else:
            detail = "no accepted iteration"
        return StageOutcome(
            stage=f"run-{stage}", ok=ok, detail=detail,
            payload={"best_commit": best_commit, "best_ms": best_value},
        )

    @staticmethod
    def _last_iteration(loop: OptimizationLoop) -> int:
        """The highest iteration number the history records, or 0 when it records none."""
        numbers = []
        for entry in loop.history.load():
            try:
                numbers.append(int(getattr(entry, "iteration", None) or entry["iteration"]))
            except (KeyError, TypeError, ValueError, IndexError):
                continue
        return max(numbers, default=0)

    def _materialize_best(self, repo: Path, commit: str, value: float | None) -> None:
        """Leave `source.py` at the stage's best accepted commit rather than at HEAD.

        A no-op when they are already the same code, which is the common case. Reported rather than
        silent: a reader who sees the headline number has to be able to tell that the file beside it
        was moved to match, and from where.
        """
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                              capture_output=True, text=True)
        if head.returncode == 0 and head.stdout.strip().startswith(commit[:12]):
            return
        current = (repo / "source.py").read_text() if (repo / "source.py").is_file() else None
        shown = subprocess.run(["git", "show", f"{commit}:source.py"], cwd=repo,
                               capture_output=True, text=True)
        if shown.returncode != 0 or shown.stdout == current:
            return
        (repo / "source.py").write_text(shown.stdout)
        shown_value = f"{value:g} ms" if value is not None else "the best accepted result"
        # Its own commit rather than `_restore_commit`, whose message says "start from" — this is the
        # stage finishing, not a round beginning. The identity flags match `_commit_tree`: a host with
        # no configured `user.email` would otherwise fail the commit and leave the tree dirty.
        self._commit_tree(
            repo, f"stage 5: deliver {commit[:12]}, the best accepted iteration ({shown_value})",
        )
        self.console.print(
            f"  [dim]source.py moved to {commit[:12]}, the best accepted iteration ({shown_value}); "
            f"HEAD held a slower accepted candidate within the regression allowance[/dim]"
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

    def measure_baselines(self) -> tuple[float, float, dict[str, float]]:
        """Re-measure the two numbers stage 5 is bounded by, on this host, now.

        Not copied from the bootstrap log or from stage 3's history. Both bounds are comparisons
        against measurements taken at other times, and a latency from a different day on a possibly
        different toolchain is the kind of number that makes a bound quietly meaningless.
        """
        self.console.print("\n[bold]re-measuring the two baselines on this host[/bold]")
        bootstrap_ms, achieved = self._measure(
            self.config.bootstrap_repo, "the bootstrapped module (1 core)", cores="1",
        )
        if achieved:
            self.console.print(
                "    achieved "
                + ", ".join(f"{k}={v:g}" for k, v in sorted(achieved.items()))
            )
        # The submodule is measured at the *commit the assembly is built from*, not at `HEAD`. With
        # 5% of regression slack those differ, and measuring one while assembling the other would
        # make the 1.1x ceiling describe code that is not in the assembly.
        best_commit, _ = _best_from_summary(self.config.submodule_repo)
        with self._at_commit(self.config.submodule_repo, best_commit) as repo:
            submodule_ms, _ = self._measure(
                repo, f"the optimized submodule at {(best_commit or 'HEAD')[:12]} (1 core)",
                cores="1",
            )
        self._record("baselines", {
            "bootstrap_latency_ms": bootstrap_ms,
            "submodule_latency_ms": submodule_ms,
            "overhead_ceiling_ms": round(submodule_ms * 1.10, 6),
            "bootstrap_achieved": achieved,
            "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        return bootstrap_ms, submodule_ms, achieved

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

    #: The numerics a validator prints beside its latency. Read from the same run, because they
    #: describe the same kernel on the same host — and because a bar derived from the recorded
    #: output alone is not a bar the pipeline can be held to. See `materialize.tighten_bar`.
    ACHIEVED_MARKERS = ("max_abs_err", "cosine", "pass_fraction")

    def _measure(self, repo: Path, label: str, cores: str) -> tuple[float, dict[str, float]]:
        """The validator's latency, and whatever numerics it printed alongside it."""
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
        achieved = {}
        for name in self.ACHIEVED_MARKERS:
            value = candidate.marker_value(outcome.output, name)
            if value is not None:
                achieved[name] = value
        self.console.print(f"    {latency:g} ms")
        return latency, achieved

    def assemble(self, rebuild: bool = False) -> StageOutcome:
        """Rebuild the whole module across all ranks, and accept it only when the gate agrees."""
        self._refuse_to_clobber("assemble", rebuild)
        projection = self.projection()
        repo = self.config.full_repo
        submodule_repo = self.config.submodule_repo
        if not submodule_repo.is_dir():
            raise StageError(f"{submodule_repo} does not exist — run the submodule stages first")

        best_commit, best_ms = _best_from_summary(submodule_repo)
        bootstrap_ms, submodule_ms, achieved = self.measure_baselines()
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
                best_commit=best_commit, achieved=achieved,
            )
            for line in prepared.get("stripped") or []:
                self.console.print(f"  [dim]stripped {line}[/dim]")
            memory_block = self._seed_preparation_memory(repo, self.config.full.memory)
            prompt = _render(prompt_template, {
                "repo": str(repo), "module": self.config.module_id, "entry_point": "kernel",
                "ranks": ranks, "last_rank": ranks - 1,
                "bootstrap_latency": f"{bootstrap_ms:g}",
                "submodule_latency": f"{submodule_ms:g}",
                "overhead_ceiling": f"{submodule_ms * 1.10:g}",
                "memory": memory_block,
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
            self._record(f"assemble-attempt-{attempt}", {
                "passed": ok, "report": report,
                "validator_sha256": _validator_sha256(repo),
            })
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

    # -- another round of stage 5 --------------------------------------------------

    def rerun_full(self, from_commit: str | None = None, note: str = "") -> StageOutcome:
        """Start a fresh round of the whole-module loop from the kernel the last round produced.

        The reason this exists: the first round's own notes are where the ideas for the second round
        come from. You read them, you learn that the routed-expert skip was blocked by a loop bound
        or that the scale multiply is the binding pass, you write that into a constraint slot — and
        then you need the loop to start again *from the kernel you already have*, not from the
        assembly baseline it started from the first time.

        So this archives the run state the way `autohelix clear` does and starts over, keeping three
        things deliberately:

        - **`notes/` and `reviews/` are carried forward.** Round 2's agent should read what round 1
          learned; that is the whole point. They are archived with everything else and then copied
          back, so the archive stays a complete record of the round.
        - **The kernel is the previous round's best accepted commit**, not `HEAD`. With regression
          slack those differ, and starting from a kernel slower than the one you have would spend the
          first iterations getting back to where you were.
        - **The baseline is re-measured by the gate** on that kernel. A recorded verdict from the old
          round describes different code, and every bound in the new round is stated against this
          number.
        """
        repo = self.config.full_repo
        if not (repo / "source.py").is_file():
            raise StageError(
                f"{repo} has no source.py, so there is no assembled module to re-run. "
                f"Run `optimize assemble` first."
            )

        previous = self._previous_round(repo)
        target = from_commit or previous.get("best_commit")
        previous_best = previous.get("best_ms")
        if not target:
            raise StageError(
                f"no previous round to continue from: {repo} has no recorded best commit.\n"
                f"Use `optimize run-full` for the first round, or pass --from-commit to name one."
            )

        # `_roll_round` returns the number it gave the round it just archived, so this one is the
        # next: printing the archive's number called the new round by the old round's name.
        round_number = 1 + self._roll_round(repo, note=note, previous=previous)
        self.console.print(
            f"  round {round_number}: starting from {str(target)[:12]}"
            + (f", the best of the last round at {previous_best:g} ms" if previous_best else "")
        )
        self._restore_commit(repo, target, round_number)

        # No explicit compile step: `run_loop` already recompiles when the prose differs from what
        # the manifest recorded, and the roll archived the old checkers, so a round whose prose
        # changed gets new ones and a re-run after a failed start does not pay for them twice.
        ok, report = self._run_gate(repo, "optimization.module_checker", "module")
        if not ok:
            raise StageError(
                f"the kernel this round starts from does not pass the module gate, so there is no "
                f"baseline to improve on. That kernel passed when it was accepted, so look for what "
                f"changed around it — the tensors, the manifest, the toolchain.\n{report[-800:]}"
            )
        latency = _latency_from_gate(repo / ".autohelix" / "optimization" / "module-gate.json")
        self.console.print(f"  round {round_number} baseline: {latency or float('nan'):g} ms")

        # The gate just ran the validator, which rewrites its outputs under `build/`, and those are
        # tracked — so the tree is dirty and `Harness.run` refuses to start. In the first-time flow
        # `assemble` happens to absorb this, because `git_init` on a fresh repo commits everything
        # the gate left; on an existing repo it returns early and nothing does.
        self._commit_tree(repo, f"round {round_number} baseline: {latency or float('nan'):g} ms")

        outcome = self.run_loop("full")
        self._record(f"round-{round_number}", {
            "stage": "full", "from_commit": target, "note": note,
            "baseline_ms": latency, "previous_best_ms": previous_best,
            "best_ms": outcome.payload.get("best_ms"),
            "best_commit": outcome.payload.get("best_commit"),
        })
        return outcome

    def _previous_round(self, repo: Path) -> dict[str, Any]:
        """The last round's summary, from the live file or from the newest archived round.

        The archive matters because this command can fail *after* rolling the state — a startup
        refusal from the loop, say — and re-running it then found no live summary and reported "no
        previous round to continue from" about a round it had itself just filed away.
        """
        live = _read_summary(repo, "full")
        if live.get("best_commit"):
            return live
        archive = repo / ".autohelix" / "archive"
        if not archive.is_dir():
            return live
        for directory in sorted(archive.iterdir(), reverse=True):
            summary = directory / "full-summary.json"
            if not summary.is_file():
                continue
            try:
                archived = json.loads(summary.read_text())
            except json.JSONDecodeError:
                continue
            if archived.get("best_commit"):
                self.console.print(f"  reading the last round's result from {directory.name}")
                return archived
        return live

    def _commit_tree(self, repo: Path, message: str) -> None:
        """Commit whatever is in the tree, so the loop starts on a clean one. A no-op when clean."""
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
        if not subprocess.run(["git", "status", "--porcelain"], cwd=repo,
                              capture_output=True, text=True).stdout.strip():
            return
        subprocess.run(
            ["git", "-c", "user.name=autohelix", "-c", "user.email=autohelix@localhost",
             "commit", "-q", "-m", message],
            cwd=repo, check=True,
        )

    #: Archived per round and then put back, because the next round's agent should start knowing what
    #: the last one learned — that is the whole reason to run a second round rather than a longer
    #: first one. Everything else `archive_state` moves (observations, logs, output, hints) belongs to
    #: the round that produced it.
    CARRIED_FORWARD = ("notes", "reviews")

    def _rounds_on_record(self, state: Path) -> int:
        """How many rounds have been archived, counted by the marker this writes.

        Counted by `round.json` rather than by archive directories, because `autohelix clear` writes
        archives here too and those are not rounds.
        """
        archive = state / "archive"
        if not archive.is_dir():
            return 0
        return sum(1 for d in archive.iterdir() if (d / "round.json").is_file())

    def _roll_round(self, repo: Path, note: str, previous: dict[str, Any]) -> int:
        """Archive the finished round, the way `autohelix clear` archives a run.

        Through `archive_state`, rather than moving files by hand: it is the convention an operator
        already knows, it writes to `.autohelix/archive/<timestamp>/`, and it covers the state a
        hand-rolled version forgets — `observations/`, `logs/`, `output/`, `hints.md`. Leaving those
        in place, which an earlier version of this did, carries one round's captured measurements and
        agent logs into the next round's directories.

        Two departures from `clear`. `notes/` and `reviews/` are copied back afterwards, so the
        archive is a complete record of the round *and* the next agent still reads what the last one
        learned. And the pipeline's own per-round state — the stage summary, the candidate archive,
        the compiled checkers — goes into the same directory, since `archive_state` does not know
        about it.

        Returns the number of the round it archived, not the next one. The kernel is untouched: the
        working tree keeps the previous round's best, which is what the new round starts from.
        """
        from autohelix.state import archive_state

        state = repo / ".autohelix"
        # Nothing to archive means this is a re-run after a start that did not reach iteration 1.
        # Rolling again would file an empty round and shift every later number by one.
        if not (state / "history.jsonl").exists():
            existing = self._rounds_on_record(state)
            self.console.print(f"  the previous round is already archived ({existing} on record)")
            return existing

        number = self._rounds_on_record(state) + 1
        before = {d.name for d in (state / "archive").iterdir()} if (state / "archive").is_dir() \
            else set()
        archive_state(repo, self.console)
        fresh = [d for d in (state / "archive").iterdir() if d.name not in before] \
            if (state / "archive").is_dir() else []
        if not fresh:
            # `archive_state` names a directory by the second, so this means two rounds were rolled
            # inside one second and the second would have merged into the first's archive. Refusing
            # beats silently mixing two rounds' records together.
            raise StageError(
                f"archiving round {number} produced no new directory under {state / 'archive'}. "
                f"A round was archived less than a second ago, so this one would have merged into "
                f"it. Wait a moment and re-run."
            )
        destination = max(fresh, key=lambda d: d.name)

        for relative in ("optimization/full-summary.json",
                         "optimization/candidates",
                         "optimization/constraints"):
            source = state / relative
            if not source.exists():
                continue
            moved = destination / Path(relative).name
            if moved.exists():
                shutil.rmtree(moved) if moved.is_dir() else moved.unlink()
            shutil.move(str(source), str(moved))

        (destination / "round.json").write_text(json.dumps({
            "round": number,
            "note": note,
            "closed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "best_ms": previous.get("best_ms"),
            "best_commit": previous.get("best_commit"),
        }, indent=2))

        for name in self.CARRIED_FORWARD:
            live = state / name
            archived = destination / name
            # Stamped with the round that wrote them, *before* the next round starts. Iteration
            # numbering restarts at 1 each round, so a carried-forward `iter-3.md` is the exact name
            # the new round's third iteration writes: round 2 overwrote round 1's notes for
            # iterations 1-4 and two of them are simply gone, which is evidence the feedback agent
            # was meant to reconcile. The suffix goes after the number so `_iteration_order` still
            # sorts a carried note next to the same iteration's new one.
            carried = self._stamp_round(live, number)
            if archived.is_dir():
                live.mkdir(parents=True, exist_ok=True)
                for item in archived.iterdir():
                    target = live / item.name
                    if item.is_dir():
                        shutil.copytree(item, target, dirs_exist_ok=True)
                    else:
                        shutil.copy2(item, target)
                carried += self._stamp_round(live, number)
            if carried:
                self.console.print(
                    f"  [dim]carried {carried} {name[:-1]}(s) forward into round {number + 1}, "
                    f"stamped round {number}[/dim]"
                )

        self.console.print(f"  round {number} archived at {destination}")
        return number

    @staticmethod
    def _stamp_round(directory: Path, number: int) -> int:
        """Rename a round's files so the next round's identical names cannot overwrite them.

        `iter-3.md` becomes `iter-3-round2.md`. Already-stamped files are left alone, so rolling a
        third round does not produce `iter-3-round2-round3.md`. Returns how many were renamed.
        """
        if not directory.is_dir():
            return 0
        stamped = 0
        for item in sorted(directory.iterdir()):
            if not item.is_file() or re.search(r"-round\d+(?=\.|$)", item.stem):
                continue
            target = item.with_name(f"{item.stem}-round{number}{item.suffix}")
            if target.exists():
                continue
            item.rename(target)
            stamped += 1
        return stamped

    def _restore_commit(self, repo: Path, commit: str, round_number: int) -> None:
        """Put `source.py` at `commit` and commit it, so the loop starts on a clean tree.

        Committing rather than leaving the working tree dirty because `Harness.run` refuses to start
        otherwise — and the commit is a real record of what this round began from.
        """
        shown = subprocess.run(["git", "show", f"{commit}:source.py"],
                               cwd=repo, capture_output=True, text=True)
        if shown.returncode != 0:
            raise StageError(
                f"cannot read source.py at {commit[:12]} in {repo}: {shown.stderr.strip()}"
            )
        target = repo / "source.py"
        if target.read_text() == shown.stdout:
            self.console.print("  the working tree is already that kernel")
        else:
            target.write_text(shown.stdout)
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
        status = subprocess.run(["git", "status", "--porcelain"],
                                cwd=repo, capture_output=True, text=True)
        if status.stdout.strip():
            subprocess.run(
                ["git", "commit", "-q", "-m",
                 f"round {round_number}: start from {commit[:12]}"],
                cwd=repo, check=True,
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

    def _tighten_submodule_bar(self, repo: Path) -> list[str]:
        """Re-pin the submodule's numerical bar to what its own cut achieved, before the loop runs.

        The bar the agent derived is a property of the rank's recorded output — `RTOL` times its
        largest element plus `ATOL` — so on a cut far better than that, the bar never binds and the
        loop can spend a large multiple of the achievable error with nothing objecting. That is what
        happened on `layers.2.attention`: the cut reached `max_abs_err = 0.0389` against a bar of
        0.7875, iteration 1 bought 4x of latency for 2.2x of error, and the assembly and stage 5
        then inherited it unchanged.

        Done here rather than by the agent because the cut can only be measured after the validator
        is written, and the validator may not move afterwards without the manifest and the recorded
        hash moving with it — which is why all three are rewritten together. The pipeline may do
        that; the agent may not, and `check_frozen_validator` is what keeps the distinction.

        10% of slack, calibrated on the runs that have finished: `00-Attention-B` ended +1.0% from
        its own baseline, `00-Attention-C` -3.5%, `24-Attention` -49.3%, and `02-Attention` +77.0%.

        Called from the submodule stage and again from the start of the loop, and idempotent across
        both: `tolerance_achieved` in the manifest is the record that it has already happened. Both
        call sites because reaching the re-pin used to mean running the submodule stage, and running
        the submodule stage means rebuilding the cut — so "restart the loop under a tighter bar"
        cost a tuned kernel and an hour of agent time to ask for.
        """
        manifest_path = repo / ".autohelix" / "optimization" / "submodule.json"
        validator = repo / "inference.py"
        if not (manifest_path.is_file() and validator.is_file()):
            return []
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            return []
        derived = manifest.get("tolerance") or {}
        if not derived:
            return []
        already = manifest.get("tolerance_achieved") or {}
        if already:
            return [
                f"bar already re-pinned: MAX_ABS_ERR {derived.get('MAX_ABS_ERR')} "
                f"(the cut measured {already.get(ACCURACY_MARKER)})"
            ]

        achieved = self._cut_accuracy(repo)
        if achieved is None:
            return [
                f"[yellow]![/yellow] no {ACCURACY_MARKER} for the cut, so the bar stays as the "
                f"agent derived it (MAX_ABS_ERR {derived.get('MAX_ABS_ERR')}) — the loop is free "
                f"to spend accuracy up to it"
            ]

        tightened = materialize.tighten_bar(derived, {ACCURACY_MARKER: achieved})
        if tightened == derived:
            return []

        text = validator.read_text()
        for name, value in tightened.items():
            text, n = re.subn(rf"^{name} = [0-9.eE+-]+$", f"{name} = {value}", text,
                              count=1, flags=re.M)
            if n != 1:
                return [f"[yellow]![/yellow] could not re-pin {name} in inference.py; "
                        f"leaving the bar as the agent derived it"]
        validator.write_text(text)
        manifest["tolerance"] = tightened
        manifest["tolerance_derived"] = derived
        manifest["tolerance_achieved"] = {ACCURACY_MARKER: achieved}
        frozen = manifest.get("frozen")
        if isinstance(frozen, dict) and "inference.py" in frozen:
            frozen["inference.py"] = _validator_sha256(repo)
        manifest_path.write_text(json.dumps(manifest, indent=2))
        self._refresh_recorded_validator("submodule", _validator_sha256(repo))
        # The two namespaces differ on purpose and are easy to confuse: the bar is keyed by the
        # constant names the validator declares (`MAX_ABS_ERR`), the verdict and the marker by the
        # statistic's own name (`max_abs_err`).
        return [
            f"[green]bar re-pinned[/green] to this cut's own accuracy: MAX_ABS_ERR "
            f"{derived.get('MAX_ABS_ERR')} -> {tightened.get('MAX_ABS_ERR'):g} "
            f"(the cut measured {achieved:g})"
        ]

    def _refresh_recorded_validator(self, stage: str, digest: str) -> None:
        """Re-record the frozen validator's hash in the stage's passing records.

        The pipeline is allowed to rewrite `inference.py` — re-pinning the bar is that — but
        `_prepared` tells a half-replaced stage from a finished one by comparing the repo's
        validator against the hash the passing record carries. Leave the record behind and the next
        `optimize all` reads its own edit as a replaced repo and rebuilds the stage, which is the
        destruction this was meant to prevent.
        """
        for record in sorted(self.state_dir.glob(f"{stage}-attempt-*.json")):
            try:
                payload = json.loads(record.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if not payload.get("passed") or payload.get("validator_sha256") == digest:
                continue
            payload["validator_sha256"] = digest
            record.write_text(json.dumps(payload, indent=2))

    def _cut_accuracy(self, repo: Path) -> float | None:
        """What the cut's own frozen validator measured for `ACCURACY_MARKER`, or None.

        `submodule-gate.json` and not `gate.json`. The loop points its per-iteration gate at
        `gate.json` (`config.write_loop_config`), so one iteration in, that file describes the
        newest candidate rather than the cut — and a bar re-pinned from it would be pinned to
        whatever the loop had already drifted to. `submodule-gate.json` is written once, by the
        submodule stage's own gate, and nothing afterwards touches it.

        A verdict written before that gate published this marker carries a latency and no accuracy.
        The validator is frozen and the cut is sitting in the repo, so measure it rather than let
        the loop run unbounded: one validator run against hours of iterations.
        """
        verdict = repo / ".autohelix" / "optimization" / "submodule-gate.json"
        try:
            recorded = json.loads(verdict.read_text()).get(ACCURACY_MARKER)
        except (OSError, json.JSONDecodeError):
            recorded = None
        if isinstance(recorded, (int, float)):
            return float(recorded)

        self.console.print(
            f"  {verdict.name} carries no {ACCURACY_MARKER}; running the frozen validator once to "
            f"find what the cut achieves"
        )
        try:
            _, measured = self._measure(repo, "the cut's accuracy", "1")
        except StageError as exc:
            self.console.print(f"  [yellow]![/yellow] {exc}")
            return None
        measured_value = measured.get(ACCURACY_MARKER)
        return float(measured_value) if isinstance(measured_value, (int, float)) else None

    def _refuse_to_clobber(self, stage: str, rebuild: bool) -> None:
        """Stop a preparation stage from archiving a repo whose own gate already passed.

        `submodule` and `assemble` both open with `_set_aside`, which moves the finished repo into
        `.optimization/attempts/` and materializes a fresh one for the agent to fill. On a stage
        that has not passed that is the whole point. On a stage that has, it costs the tuned kernel,
        its git history, its notes and reviews, and an hour of agent time to rebuild something
        nobody asked to change — recoverable only because `_set_aside` archives rather than deletes.

        So rebuilding is opt-in. Neither reason an operator reaches for `optimize submodule` on a
        passed stage needs it: re-pinning the numerical bar happens at the start of the loop
        (`_tighten_submodule_bar`), and re-running the loop is `optimize run`.
        """
        if rebuild or self._prepared(stage, quiet=True) is None:
            return
        repo = self.config.submodule_repo if stage == "submodule" else self.config.full_repo
        nxt = "run" if stage == "submodule" else "run-full"
        raise StageError(
            f"{stage} already passed its gate and {repo.name} is still there, so this would "
            f"archive it to {self.state_dir / 'attempts'} and have an agent build another one.\n"
            f"To re-run the loop over the repo you have: `optimize {nxt}`.\n"
            f"To change its numerical bar: that happens at the start of the loop now, so "
            f"`optimize {nxt}` is enough.\n"
            f"To replace the repo on purpose: `optimize {stage} --rebuild`."
        )

    def _prepared(self, stage: str, *, quiet: bool = False) -> StageOutcome | None:
        """The recorded outcome of a preparation stage that already passed, or None.

        `submodule` and `assemble` both open with `_set_aside`, so calling either one again moves
        the finished repo into `.optimization/attempts/` and builds a new one from the bootstrap.
        That is the right behaviour for `optimize submodule` run on purpose, and the wrong
        behaviour for `optimize all` run to pick a pipeline up where it stopped — which is what
        `all`'s own docstring promises. Resuming a run that died in stage 5 cost a tuned
        single-rank kernel and five iterations of history before this guard existed; they were
        recoverable only because `_set_aside` archives rather than deletes.

        Both halves have to hold. A record says the gate passed once; the repo being present says
        the thing it passed on is still there to run.
        """
        repo = self.config.submodule_repo if stage == "submodule" else self.config.full_repo
        if not (repo / "source.py").is_file():
            return None
        current = _validator_sha256(repo)
        for record in sorted(self.state_dir.glob(f"{stage}-attempt-*.json")):
            try:
                payload = json.loads(record.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if not payload.get("passed"):
                continue
            # A pass record alone is not enough. A rebuild interrupted after `_set_aside`
            # materialized a fresh repo but before its own record was written leaves an older
            # `passed: true` beside a repo of stubs, and skipping the gate for *that* is how a
            # half-replaced stage reaches the loop. The frozen validator's hash is what ties the
            # verdict to the repo it was a verdict about.
            recorded = payload.get("validator_sha256")
            if recorded is None and not quiet:
                self.console.print(
                    f"  [yellow]![/yellow] {record.name} records a pass but no validator hash "
                    f"(written before this was recorded); skipping {stage} on the weaker evidence "
                    f"that {repo.name} is present"
                )
            elif recorded is not None and recorded != current:
                if not quiet:
                    self.console.print(
                        f"  [yellow]![/yellow] {record.name} passed against a different "
                        f"inference.py than {repo.name} now has — rebuilding {stage} rather than "
                        f"trusting it"
                    )
                return None
            if not quiet:
                self.console.print(
                    f"  [green]{stage} already passed[/green] ({record.name}) and {repo.name} "
                    f"is present — skipping it. Run `optimize {stage} --rebuild` to replace it "
                    f"on purpose."
                )
            return StageOutcome(stage=stage, ok=True, detail="already recorded as passing")
        return None

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
        step(self._prepared("submodule") or self.submodule())
        step(self.compile_constraints("submodule"))
        step(self.run_loop("submodule"))
        step(self._prepared("assemble") or self.assemble())
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
