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
from autohelix.harness import AutoHelixRunError, Harness
from autohelix.history import IterationResult
from optimization import constraints as cons
from optimization import memory as mem
from optimization import feedback

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
        self.memory = self._load_memory()
        self._slot_verdicts: dict[int, cons.SlotVerdict] = {}
        #: Which iteration `_check_metric_gates` is judging. Upstream does not pass it, and the
        #: end-of-slot rule needs to know whose slot verdict to consult.
        self._current_iteration: int | None = None
        #: Each iteration's `source.py`, read in the last moment before its worktree is torn down.
        self._captured_source: dict[int, str] = {}
        self._install_candidate_capture()

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

    def _load_memory(self) -> mem.MemorySpec:
        """Parse `memory:` out of the raw config, for the same reason the schedule is read there.

        The path in the derived config is already absolute — `PipelineConfig` resolved it against
        the operator's config file — so no base directory is needed here.
        """
        try:
            spec = mem.MemorySpec.from_config(
                (self._raw_config or {}).get("memory"),
                max_iterations=self.config.max_iterations,
            )
        except mem.MemoryError_ as exc:
            self.console.print(f"  [yellow]![/yellow] memory: {exc}")
            return mem.MemorySpec()
        for warning in spec.validate():
            self.console.print(f"  [yellow]![/yellow] {warning}")
        return spec

    # -- prompt --------------------------------------------------------------------

    def build_prompt(self, iteration: int, worktree_dir: Path) -> str:
        """Upstream's prompt plus this iteration's constraint and the standing bounds."""
        from autohelix.prompt_template import build_prompt_variables, render_template

        from optimization import presets

        variables = build_prompt_variables(self.config, self.history, iteration, worktree_dir)
        variables["iteration_constraint"] = self.schedule.describe_for_prompt(iteration)
        # Seeded here rather than in `prepare_worktree`, because this is the one hook that knows
        # both the iteration number and the worktree — and the whole point is that some iterations
        # read the memory and others do not. An iteration that does not read it never sees the
        # directory at all, so it cannot be tempted into it by a path in its tree.
        variables["memory"] = self._seed_memory(iteration, worktree_dir)
        variables["constraint_schedule"] = self.schedule.summary_table()
        variables["stage"] = self.stage
        variables["metric"] = METRIC
        best = self.history.get_best_metrics(self.config.metric_directions()).get(METRIC)
        variables["best_so_far"] = f"{best[0]:g}" if best else ""
        # Taken from the config rather than written into the prompt as a number. The template said
        # "more than 5% above it is rejected", which is only true while the operator leaves
        # `acceptance.max_regression_pct` at its default — and an agent told the wrong allowance
        # either wastes an iteration it could have kept or throws away one it could not.
        allowance = next((g.max_regression_pct for g in self.config.acceptance.metric_gates
                          if g.metric == METRIC), None)
        variables["regression_allowance"] = f"{allowance:g}" if allowance is not None else ""
        # The packaged template, not `.autohelix/prompt.md`'s default: the stock one renders the
        # constraint commands, and here constraint zero is the hidden slot checker.
        return render_template(presets.project_prompt(self.project_path), variables)

    def _seed_memory(self, iteration: int, worktree_dir: Path) -> str:
        """Copy the memory in for an iteration that reads it, and return its prompt block."""
        if not self.memory.reads_at(iteration):
            return ""
        # Checked before copying, and reported as itself. "Nothing seeded" for a path that contains
        # the worktree would send the operator looking for an empty directory, when what actually
        # happened is that the copy was refused to stop it recursing into its own output.
        problem = mem.seed_problem(self.memory, worktree_dir)
        if problem:
            self.console.print(f"  [yellow]![/yellow] memory: {problem}")
            return ""
        seeded = mem.seed(self.memory, worktree_dir)
        if seeded:
            self.console.print(
                f"  memory: {seeded} file(s) from {self.memory.path} at {mem.SEEDED_REL}"
            )
        else:
            # Said out loud rather than passed over: the operator asked this iteration to start
            # from earlier work, and it is starting from nothing instead.
            self.console.print(
                f"  [yellow]![/yellow] memory: nothing seeded from {self.memory.path}"
            )
        return mem.describe_for_prompt(self.memory, iteration, seeded)

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

        The slot check is spliced in as constraint zero. Everything after it is
        `Harness.run_iteration` rather than a copy of it, so the acceptance logic, the metric gates
        and the reviewer stay in one place.
        """
        # `_check_metric_gates` needs to know which iteration it is judging, and upstream does not
        # pass it. Set before the branch so it is right for ungoverned iterations too.
        self._current_iteration = iteration

        slot = self.schedule.slot_for(iteration)
        try:
            if slot is None or not (slot.has_text and slot.enforce):
                result = super().run_iteration(iteration)
            else:
                result = self._run_governed_iteration(iteration)
        except AutoHelixRunError as exc:
            # A timed-out agent is a rejected iteration, not a failed run: its candidate was never
            # measured, its notes are already saved, and propagating only loses the later stages.
            if "timed out" not in str(exc):
                raise
            self.console.print(f"  [yellow]![/yellow] iteration {iteration} {exc}"
                               f" — recorded as rejected; the loop continues")
            self.log.warning(f"iter {iteration} agent timed out; recorded as rejected")
            result = IterationResult(
                iteration=iteration, accepted=False, metrics={},
                reason=f"agent timed out: {exc}",
            )
            self.history.append(result)
            return result
        archived = self.archive_candidate(iteration, result)
        if archived is not None and not result.accepted:
            self.console.print(f"  [dim]candidate kept at {archived}[/dim]")
        return result

    def _run_governed_iteration(self, iteration: int) -> IterationResult:
        """`Harness.run_iteration` with the slot checker prepended to the constraint list.

        Implemented by temporarily prefixing `config.constraints` with the checker command, so a
        violation is rejected by exactly the path a failing constraint already takes — same
        reporting, same discard, same failure output carried into the next prompt.

        On the slot's **last** iteration the checker runs `--advisory`: it writes the
        same verdict but exits 0, so the real constraints and the measurement still run, and the
        accept/reject decision moves to `_check_metric_gates`, which requires a strict improvement
        instead of allowing the usual 5%. By then the agent has had every iteration the slot allows,
        and discarding something correct and faster buys nothing.
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

        # `enforcement_for` is the single place the hard/soft decision is made, so what the prompt
        # told the agent and what acceptance applies cannot come apart.
        mode = slot.enforcement_for(iteration)
        advisory = mode == "soft"
        # Through the helper, which quotes both paths. Formatting `CHECKER_COMMAND` here instead —
        # which this did — bypasses the quoting on the one call site that actually runs, so a
        # workspace path with a space in it split the command and failed every governed iteration.
        command = cons.checker_command(checker, report, advisory=advisory)
        # Printed for every governed iteration, hard included. A round whose schedule said soft and
        # whose log said nothing is a round where the only record of which rule ran is the absence
        # of `--advisory` inside a wrapped command line, which is not somewhere an operator looks.
        if advisory:
            because = ("the last iteration of the slot" if slot.enforcement == "hard"
                       else "a soft slot")
            self.console.print(
                f"  [dim]slot {slot.label}: soft on iteration {iteration} ({because}), so the "
                f"constraint is checked but not fatal — acceptance needs a strict improvement "
                f"instead[/dim]"
            )
        else:
            self.console.print(
                f"  [dim]slot {slot.label}: hard on iteration {iteration}, so a violation rejects "
                f"the candidate before it is measured[/dim]"
            )

        original = list(self.config.constraints)
        self.config.constraints = [ConstraintCommand(command=command, timeout=180), *original]
        try:
            result = super().run_iteration(iteration)
        finally:
            self.config.constraints = original

        verdict = self._slot_verdict(iteration, slot, report, result.failure_output or "")
        if not verdict.passed:
            self.console.print(f"  [red]{verdict.summary()}[/red]")
            if verdict.findings:
                self.console.print(Panel(
                    Text("\n".join(verdict.findings)),
                    title=f"constraint slot {slot.label}", border_style="red",
                ))
            if not advisory:
                result.reason = f"constraint slot {slot.label} violated"
        elif verdict.checked:
            self.console.print(f"  [green]{verdict.summary()}[/green]")
        return result

    def _slot_verdict(
        self, iteration: int, slot: cons.Slot, report: Path, output: str,
    ) -> cons.SlotVerdict:
        """Read (and cache) whether one iteration followed its slot.

        The checker's own JSON report is the authority, not the iteration's outcome: a rejection
        from the real constraints or the metric gate is an ordinary failure that happened to run
        after a passing slot check, and recording it as a violation would blame the schedule.
        """
        cached = self._slot_verdicts.get(iteration)
        if cached is not None:
            return cached
        verdict = cons.read_slot_verdict(
            iteration, slot.label, report, output,
            return_code=0 if report.is_file() else 1,
        )
        self._slot_verdicts[iteration] = verdict
        return verdict

    def _slot_followed(self, iteration: int) -> bool | None:
        """Whether this iteration followed its slot: from memory, then from disk, then unknown.

        The disk fallback is what makes a resumed run's summary honest. `_slot_verdicts` only holds
        what *this* instance judged, so after a resume every earlier governed iteration reported as
        un-checked even though its verdict was still on disk.
        """
        cached = self._slot_verdicts.get(iteration)
        if cached is not None:
            return cached.passed
        slot = self.schedule.slot_for(iteration)
        stored = cons.persisted_verdict(self.project_path, iteration,
                                        slot.label if slot else None)
        return stored.passed if stored is not None else None

    # -- acceptance ----------------------------------------------------------------

    def _check_metric_gates(self, metrics: dict[str, float]) -> str | None:
        """Upstream's gate, plus the stricter rule for a violation the checker did not reject.

        A candidate reaches here having violated its constraint only when that iteration was soft —
        either a slot declared `enforcement: soft`, or the last iteration of a hard one. It does not
        get the regression slack a compliant iteration gets: it has to be **strictly faster** than
        the best accepted so far. Correct-and-faster is worth keeping; correct-and-merely-not-much-
        worse is not, when it also ignored the constraint. That is what keeps a soft constraint from
        being the same thing as no constraint.
        """
        iteration = getattr(self, "_current_iteration", None)
        verdict = self._slot_verdicts.get(iteration) if iteration is not None else None
        slot = self.schedule.slot_for(iteration) if iteration is not None else None

        if verdict is None and iteration is not None and slot is not None:
            # The gate runs inside `Harness.run_iteration`, before `_run_governed_iteration` gets
            # to read the report, so on the first pass the verdict has to be read here.
            report = cons.report_path(self.project_path, iteration)
            if report.is_file():
                verdict = self._slot_verdict(iteration, slot, report, "")

        if verdict is None or verdict.passed or not verdict.checked:
            return super()._check_metric_gates(metrics)

        value = metrics.get(METRIC)
        if value is None:
            return f"metric gate: {METRIC} not produced by benchmark"
        best = self.history.get_best_metrics(self.config.metric_directions()).get(METRIC)
        if best is None:
            return super()._check_metric_gates(metrics)
        best_value, best_iteration = best
        if value < best_value:
            self.console.print(
                f"  [yellow]kept anyway:[/yellow] slot {verdict.label} was not followed, but "
                f"{METRIC} improved {best_value:g} -> {value:g} ms and the correctness gate passed"
            )
            return None
        return (
            f"constraint slot {verdict.label} violated on its last iteration and {METRIC} did not "
            f"improve ({value:g} ms against the best {best_value:g} ms from iter {best_iteration}). "
            f"An iteration that ignores its constraint has to earn it with a strict improvement"
        )

    # -- review --------------------------------------------------------------------

    def run_reviewer(self, worktree, iteration: int) -> bool:
        """Upstream's reviewer, with the editable scope held to what the gate measured.

        The reviewer is a second write-capable agent in the same worktree, and it runs *after* scope
        reversion, the constraints and the metrics — then the iteration is staged and merged. So a
        reviewer that edits `source.py` while investigating it replaces the candidate that was gated
        with code nothing ever ran, and that is what gets merged and reported.

        Not solved by telling the reviewer not to: its whole job is adversarial reading, it has the
        tools to edit, and one stray `Edit` is indistinguishable in the result from a deliberate one.
        So the files are snapshotted and put back. Restoring rather than rejecting, because the
        review is not the candidate's fault and its verdict is still worth having.

        Overridden here rather than in `Harness`, because only this pipeline's contract says the
        measured bytes and the merged bytes must be identical.
        """
        # `Config.editable`, flat. The YAML nests it under `scope:`, but the dataclass does not, and
        # reading `config.scope.editable` raised `AttributeError` at the baseline review — before any
        # iteration, so the stage died on startup. The fallback covers a config that leaves `editable`
        # empty (whitelist unset means "everything but `frozen`"), where guarding nothing would leave
        # the reviewer able to rewrite the very files whose hashes are gate checks.
        names = list(self.config.editable) or list(feedback.DELIVERABLE_FILES)
        snapshot = {
            path: path.read_bytes()
            for path in (worktree.working_dir / name for name in names)
            if path.is_file()
        }
        try:
            return super().run_reviewer(worktree, iteration)
        finally:
            self._undo_reviewer_writes(worktree, snapshot)

    def _undo_reviewer_writes(self, worktree, snapshot: dict[Path, bytes]) -> None:
        """Put the worktree back to the bytes the gate measured, commits included.

        A byte snapshot of the deliverables is not enough on its own. `Harness.run_iteration` runs
        the reviewer *after* `uncommit_agent_changes` and `revert_out_of_scope` have already
        happened, and then `Sandbox.merge_worktree` does `git merge <worktree.branch>` — so while
        that merge only *stages* the editable files for its own commit, the merge itself carries
        every commit already on the branch. A reviewer that commits therefore reaches main with
        whatever it touched, in scope or out, and the working-tree restore below never sees it.

        So the reviewer gets the same three steps the agent gets, in the same order: reopen its
        commits, revert what is outside the scope, then restore the deliverables it was allowed to
        touch but must not have changed.
        """
        commits = 0
        try:
            commits = self.sandbox.uncommit_agent_changes(worktree)
        except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as exc:
            # This step is the only thing between a reviewer commit and the main branch, and when it
            # fails it fails *before* reporting what it found — so there is no way to tell "no
            # commits to reopen" from "commits I could not reopen". Carrying on would let
            # `merge_worktree` take the branch as it stands, reviewer commits and all, into the
            # deliverable that was just gated. Losing this iteration is the cheaper mistake.
            self.console.print(f"  [red]![/red] could not reopen reviewer commits: {exc}")
            raise RuntimeError(
                f"the reviewer's commits could not be reopened ({exc}), so this worktree cannot be "
                f"shown to be free of them and must not be merged. The candidate is still on its "
                f"branch; reopen or drop the reviewer's commits by hand and re-run the stage."
            ) from exc
        if commits:
            self.console.print(
                f"  [yellow]![/yellow] the reviewer created {commits} commit(s); reopened so scope "
                f"enforcement sees them instead of `git merge` carrying them into the deliverable"
            )
        try:
            effective = self.sandbox.resolve_editable(
                self.config.editable, self.config.frozen, cwd=worktree.working_dir,
            )
            if effective is not None:
                reverted = self.sandbox.revert_out_of_scope(worktree, effective)
                if reverted:
                    self.console.print(
                        f"  [yellow]![/yellow] the reviewer touched "
                        f"{', '.join(sorted(reverted)[:6])} outside the editable scope; reverted"
                    )
        except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as exc:
            self.console.print(f"  [red]![/red] could not revert the reviewer's out-of-scope work: {exc}")

        changed = [
            path for path, body in snapshot.items()
            if not path.is_file() or path.read_bytes() != body
        ]
        for path in changed:
            path.write_bytes(snapshot[path])
        if changed:
            self.console.print(
                f"  [yellow]![/yellow] the reviewer modified "
                f"{', '.join(p.name for p in changed)} and it was restored: the candidate that "
                f"is merged has to be the one the gate measured"
            )

    # -- candidate archive ---------------------------------------------------------

    def archive_candidate(self, iteration: int, result: IterationResult) -> Path | None:
        """Keep a candidate's `source.py` and metrics, accepted or not.

        The metric gate decides what the next iteration builds on. It should not decide what the
        run *retains*: iterations 7-8 exist to try something the disciplined iterations cannot, and
        an experiment 6% slower is rejected while still being the most informative thing in the run.

        Called from inside `run_iteration`, **before** the worktree is discarded. The obvious place
        is after the run, reading each candidate back out of its branch — and that silently archives
        nothing, because `Sandbox.remove_worktree` runs `git branch -D` in `Harness.run_iteration`'s
        `finally` block, so by the time the loop ends every iteration branch it would read is gone.
        """
        target = self.project_path / CANDIDATES_DIR / f"iter-{iteration}"
        source = self._candidate_source(iteration)
        if source is None:
            return None
        try:
            target.mkdir(parents=True, exist_ok=True)
            (target / "source.py").write_text(source)
            (target / "metrics.json").write_text(json.dumps({
                "iteration": iteration,
                "accepted": result.accepted,
                "metrics": result.metrics,
                "reason": result.reason,
                "slot": (self.schedule.slot_for(iteration).label
                         if self.schedule.slot_for(iteration) else None),
                "slot_followed": self._slot_followed(iteration),
            }, indent=2))
            return target
        except OSError:
            return None

    def _candidate_source(self, iteration: int) -> str | None:
        """This iteration's `source.py`, captured just before its worktree was torn down."""
        captured = self._captured_source.pop(iteration, None)
        if captured is not None:
            return captured
        # A merged iteration's code is also on the main branch, which outlives the worktree.
        shown = subprocess.run(
            ["git", "show", "HEAD:source.py"],
            cwd=self.project_path, capture_output=True, text=True, check=False,
        )
        return shown.stdout if shown.returncode == 0 else None

    def _install_candidate_capture(self) -> None:
        """Read each candidate's `source.py` in the last moment it exists.

        `Harness.run_iteration` tears the worktree down in a `finally` block, and
        `Sandbox.remove_worktree` deletes the iteration branch as well as the directory — so after
        the iteration returns there is nothing left to archive from, for a rejected iteration in
        particular. Wrapping the teardown is the one seam where the candidate is still on disk and
        the outcome is already decided.
        """
        original = self.sandbox.remove_worktree

        def remove(worktree_path: Path) -> None:
            # The project directory inside the worktree, which is the worktree root itself unless
            # AutoHelix runs from a subdirectory of a larger repository. Both the capture below and
            # the unlock after it are relative to it.
            path = Path(worktree_path)
            if self.sandbox.repo_prefix:
                path = path / self.sandbox.repo_prefix
            try:
                source = path / "source.py"
                iteration = self._current_iteration
                if iteration is not None and source.is_file():
                    self._captured_source[iteration] = source.read_text()
            except OSError:
                pass
            # The seeded memory is read-only down to its directory bits, and both
            # `git worktree remove --force` and the `shutil.rmtree` behind it need those bits back
            # to delete what is inside. Given back here, in the same seam, so a read-only snapshot
            # never turns into a teardown failure that strands a worktree — and given back at
            # `path`, where `seed` put it, since the worktree root is the wrong place to look under
            # a nested project and the directories would stay locked.
            mem.unlock(path)
            original(worktree_path)

        self.sandbox.remove_worktree = remove  # type: ignore[method-assign]

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
                    "slowest_rank_ms": r.metrics.get("slowest_rank_ms"),
                    "rank_spread_ms": r.metrics.get("rank_spread_ms"),
                    "reason": r.reason,
                    "slot": (self.schedule.slot_for(r.iteration).label
                             if self.schedule.slot_for(r.iteration) else None),
                    "slot_followed": self._slot_followed(r.iteration),
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
