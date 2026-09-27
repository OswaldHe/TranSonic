# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The per-iteration constraint schedule: different soft constraints for different iterations.

Upstream AutoHelix has one fixed set of constraints for a whole run. An optimization loop wants
something else: early iterations held to a narrow tool (write it in NKI, prove the device path),
middle ones allowed to reach for torch, a couple left completely free to try something wild, and
a return to the disciplined regime at the end to consolidate. Ten iterations of "do whatever you
like" converges on whatever the first iteration happened to try.

The schedule is prose, because the useful constraints are prose: "must use NKI, no torch" is a
sentence, not a predicate. Turning it into a predicate is a job, and it is given to an agent —
the *constraint compiler* — which runs once before iteration 1 and writes one checker script per
slot. The optimizing agent is given the prose in its prompt and never sees the script, exactly as
`bootstrap` and `floorplan` hide their gates: a constraint whose implementation is readable is a
constraint that gets read for loopholes rather than followed.

Three properties worth stating because each was a choice:

- **The compiler runs once, up front, not per iteration.** A checker written at the top of
  iteration 7 could be written to accommodate what iteration 6 already did. Writing all of them
  before any code exists makes the schedule a commitment rather than a running negotiation.
- **Slots are indexed by iteration number, not by accepted count.** If iteration 3 is rejected,
  iteration 4 still gets slot 4. The budget, the schedule and the transcript stay aligned, and a
  run is reproducible from its config.
- **A permissive slot still gets a checker.** "Allow both NKI and torch" has nothing to reject,
  so the compiler writes a checker that passes — but it writes one, and says in a comment why it
  passes everything. The alternative (no script for permissive slots) makes "the compiler decided
  there was nothing to check" indistinguishable from "the compiler failed to write a script".
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Where compiled checkers live, relative to the project. Inside `.autohelix/`, which is
#: gitignored and — crucially — *not* among the directories `Sandbox.prepare_worktree` seeds into
#: an iteration worktree, so a checker is never placed where the agent is working. As in
#: `floorplan`, treat that as a speed bump backed by the reviewer rather than a sandbox: a
#: determined agent can still walk up out of its worktree.
CONSTRAINTS_REL = Path(".autohelix") / "optimization" / "constraints"

#: The record of what the compiler produced, hashed so a later edit to a checker is detectable.
MANIFEST_REL = CONSTRAINTS_REL / "manifest.json"

#: What a checker is invoked as. Driven through `optimization.slotcheck` rather than directly, so
#: that the last iteration of an interval can run it `--advisory` — writing the same verdict but
#: exiting 0, which lets the measurement proceed and hands the accept/reject decision to the loop's
#: stricter end-of-interval rule. See `optimization/slotcheck.py`.
CHECKER_COMMAND = (
    "python -m optimization.slotcheck --checker {checker} --repo . --json {report}{advisory}"
)


def checker_command(checker: Path, report: Path, advisory: bool = False) -> str:
    """The command line for one slot's checker, with both paths shell-quoted.

    `autohelix.checks.run_constraint` executes with `shell=True`, and these are absolute paths under
    a workspace root the operator chose. One space in it and the command splits — every governed
    iteration would then fail before its work was even looked at.
    """
    return CHECKER_COMMAND.format(
        checker=shlex.quote(str(checker)),
        report=shlex.quote(str(report)),
        advisory=" --advisory" if advisory else "",
    )


class ScheduleError(ValueError):
    """The iteration-constraint schedule is not well formed."""


@dataclass
class Slot:
    """One range of iterations and the constraint text that governs them."""

    iterations: list[int]
    text: str = ""
    #: Set false to force a no-op checker regardless of what the text says. The escape hatch for
    #: text that is guidance for the agent rather than a rule to enforce.
    enforce: bool = True
    #: Whether `enforce` was written in the config. Tracked so "enforce: true with no text" can be
    #: warned about while a deliberately empty slot — the free-exploration ones — stays silent.
    enforce_explicit: bool = False
    #: Filled in by `compile_schedule`: the checker written for this slot.
    checker: str | None = None

    @property
    def last_iteration(self) -> int:
        """The final iteration this slot governs.

        The one where a violation stops being fatal: by then the agent has had every iteration the
        slot allows, so a correct-and-faster candidate that still misses the constraint is worth
        keeping. See `optimization/slotcheck.py` for the reasoning and the stricter rule that
        replaces the rejection.
        """
        return max(self.iterations)

    @property
    def label(self) -> str:
        """`1-3`, or `7` for a single iteration. Used in filenames and the prompt."""
        if len(self.iterations) == 1:
            return str(self.iterations[0])
        runs = _contiguous_runs(self.iterations)
        return ",".join(f"{a}-{b}" if a != b else str(a) for a, b in runs)

    @property
    def slug(self) -> str:
        return self.label.replace(",", "_")

    @property
    def has_text(self) -> bool:
        return bool(self.text.strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "iterations": list(self.iterations),
            "label": self.label,
            "text": self.text,
            "enforce": self.enforce,
            "checker": self.checker,
        }


@dataclass
class Schedule:
    """The whole schedule: slots covering some or all of a run's iterations."""

    slots: list[Slot] = field(default_factory=list)

    def slot_for(self, iteration: int) -> Slot | None:
        """The slot governing ``iteration``, or None when it is unconstrained."""
        for slot in self.slots:
            if iteration in slot.iterations:
                return slot
        return None

    def covered(self) -> set[int]:
        return {i for slot in self.slots for i in slot.iterations}

    def enforceable(self) -> list[Slot]:
        """Slots that need a checker written: the ones with text and enforcement on."""
        return [s for s in self.slots if s.has_text and s.enforce]

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, raw: Any, max_iterations: int | None = None) -> "Schedule":
        """Parse an ``iteration_constraints:`` block.

        Accepts either an explicit iteration list or a ``from``/``to`` range::

            iteration_constraints:
              - iterations: [1, 2, 3]
                text: |
                  Must use NKI...
              - from: 4
                to: 6
                text: |
                  Both NKI and torch-xla are allowed...

        An empty or absent block is a valid schedule with no slots — every iteration is then
        unconstrained, which is what stage 5 uses until its ranges are filled in.
        """
        if raw is None:
            return cls()
        if not isinstance(raw, list):
            raise ScheduleError(
                "'iteration_constraints' must be a list of ranges, each with 'iterations' "
                "(or 'from'/'to') and 'text'"
            )

        slots: list[Slot] = []
        seen: dict[int, str] = {}
        for index, entry in enumerate(raw):
            where = f"iteration_constraints[{index}]"
            if not isinstance(entry, dict):
                raise ScheduleError(f"{where}: must be a mapping")
            unknown = set(entry) - {"iterations", "from", "to", "text", "enforce"}
            if unknown:
                raise ScheduleError(
                    f"{where}: unknown key(s) {', '.join(sorted(unknown))}. "
                    f"Known: iterations, from, to, text, enforce"
                )

            iterations = _parse_iterations(entry, where)
            if max_iterations is not None:
                beyond = [i for i in iterations if i > max_iterations]
                if beyond:
                    raise ScheduleError(
                        f"{where}: names iteration(s) {beyond} but budget.iterations is "
                        f"{max_iterations}. Raise the budget or narrow the range — a slot that "
                        f"never runs is a constraint you will believe was applied"
                    )
            for i in iterations:
                if i in seen:
                    raise ScheduleError(
                        f"{where}: iteration {i} is already governed by slot '{seen[i]}'. "
                        f"Ranges may not overlap — one iteration, one constraint"
                    )

            slot = Slot(
                iterations=iterations,
                text=str(entry.get("text") or ""),
                enforce=bool(entry.get("enforce", True)),
                enforce_explicit="enforce" in entry,
            )
            for i in iterations:
                seen[i] = slot.label
            slots.append(slot)

        schedule = cls(slots=slots)
        return schedule

    def validate(self, max_iterations: int) -> list[str]:
        """Warnings, not errors: gaps in coverage are legal but usually a mistake."""
        warnings: list[str] = []
        missing = sorted(set(range(1, max_iterations + 1)) - self.covered())
        if missing and self.slots:
            warnings.append(
                f"iteration(s) {missing} have no constraint slot and will run unconstrained"
            )
        for slot in self.slots:
            # A textless slot is the intended way to write "explore freely here". Only an
            # *explicit* `enforce: true` with nothing to enforce is worth reporting.
            if slot.enforce_explicit and slot.enforce and not slot.has_text:
                warnings.append(
                    f"slot {slot.label} has enforce: true but no text — nothing to check"
                )
        return warnings

    # ------------------------------------------------------------------
    # The prompt's view
    # ------------------------------------------------------------------
    def describe_for_prompt(self, iteration: int) -> str:
        """What this iteration's agent is told about its constraint.

        The text verbatim, plus the fact that it is checked and that a violation costs the
        iteration. Stating the consequence matters: a constraint the agent reads as advice is one
        it will trade away for a faster kernel, which is exactly the trade the schedule exists to
        prevent.
        """
        slot = self.slot_for(iteration)
        if slot is None or not slot.has_text:
            return (
                "**No additional constraint on this iteration.** Explore freely: this is one of "
                "the iterations set aside for trying something the disciplined ones cannot."
            )
        lines = [
            f"**Constraint for iteration {iteration}** (slot {slot.label}):",
            "",
            slot.text.strip(),
        ]
        if slot.enforce and iteration < slot.last_iteration:
            lines += [
                "",
                "This is checked by a script you cannot see, before anything is measured. An "
                "iteration that does not follow it is rejected and its work discarded, however "
                "fast it is — so if you believe the constraint is wrong for this module, say so "
                "in your notes and follow it anyway.",
            ]
        elif slot.enforce:
            lines += [
                "",
                f"This is the last iteration this constraint governs, so it is checked but no "
                f"longer fatal. If you cannot satisfy it and you have something correct and "
                f"**strictly faster than the best so far**, it will still be kept — the usual 5% "
                f"of slack is what you give up, not the work. Follow the constraint if you can; "
                f"say in your notes why you could not if you did not.",
            ]
        return "\n".join(lines)

    def summary_table(self) -> str:
        """The whole schedule, for the operator's report and the project README."""
        if not self.slots:
            return "(no per-iteration constraints)"
        rows = ["| iterations | enforced | constraint |", "|---|---|---|"]
        for slot in self.slots:
            first = slot.text.strip().splitlines()[0] if slot.has_text else "—"
            enforced = "yes" if (slot.enforce and slot.has_text) else "no"
            rows.append(f"| {slot.label} | {enforced} | {first} |")
        return "\n".join(rows)


def _parse_iterations(entry: dict[str, Any], where: str) -> list[int]:
    raw = entry.get("iterations")
    if raw is not None:
        if isinstance(raw, int):
            raw = [raw]
        if not isinstance(raw, list) or not raw:
            raise ScheduleError(f"{where}: 'iterations' must be a non-empty list of integers")
        try:
            iterations = sorted({int(i) for i in raw})
        except (TypeError, ValueError) as exc:
            raise ScheduleError(f"{where}: 'iterations' must be integers ({exc})") from exc
    else:
        if "from" not in entry or "to" not in entry:
            raise ScheduleError(
                f"{where}: needs either 'iterations: [...]' or both 'from:' and 'to:'"
            )
        try:
            start, end = int(entry["from"]), int(entry["to"])
        except (TypeError, ValueError) as exc:
            raise ScheduleError(f"{where}: 'from'/'to' must be integers ({exc})") from exc
        if end < start:
            raise ScheduleError(f"{where}: 'to' ({end}) is before 'from' ({start})")
        iterations = list(range(start, end + 1))
    if any(i < 1 for i in iterations):
        raise ScheduleError(
            f"{where}: iterations are 1-based and iteration 0 is the baseline, which has no "
            f"agent and so cannot be constrained"
        )
    return iterations


def _contiguous_runs(values: list[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    for v in sorted(values):
        if runs and v == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], v)
        else:
            runs.append((v, v))
    return runs


# ----------------------------------------------------------------------
# Compiled checkers
# ----------------------------------------------------------------------
@dataclass
class CompiledSlot:
    """A slot whose checker has been written and hashed."""

    label: str
    iterations: list[int]
    path: Path
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "iterations": self.iterations,
            "checker": self.path.name,
            "sha256": self.sha256,
        }


def checker_path(project_path: Path, slot: Slot) -> Path:
    return project_path / CONSTRAINTS_REL / f"slot-{slot.slug}.py"


def report_path(project_path: Path, iteration: int) -> Path:
    return project_path / CONSTRAINTS_REL / f"iter-{iteration}.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(project_path: Path, compiled: list[CompiledSlot], schedule: Schedule) -> Path:
    """Record what was compiled, so a checker edited later is detectable.

    The schedule text is recorded alongside the hashes: the pair is the claim "these scripts
    implement this prose", and a report that shows both lets a reader judge it.
    """
    path = project_path / MANIFEST_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "slots": [c.to_dict() for c in compiled],
        "schedule": [s.to_dict() for s in schedule.slots],
    }, indent=2))
    return path


def read_manifest(project_path: Path) -> dict[str, Any]:
    path = project_path / MANIFEST_REL
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def schedule_drift(project_path: Path, schedule: Schedule) -> list[str]:
    """Why the compiled checkers no longer match the schedule in the config, if they do not.

    "Are there any compiled slots" is not the question. An operator who edits a slot's prose while
    keeping its iteration range leaves the manifest non-empty and every hash intact, and the old
    checker is then applied to a prompt that says something else — the agent told one rule and
    judged by another. Changing the *ranges* is worse: the manifest stays non-empty, the new slot's
    checker is simply absent, and the loop falls back to running that iteration unenforced while its
    prompt still claims a script is watching.
    """
    manifest = read_manifest(project_path)
    recorded = manifest.get("schedule")
    enforceable = schedule.enforceable()

    if not enforceable:
        return []
    if not manifest.get("slots"):
        return ["no compiled checkers found for the current schedule"]
    if recorded is None:
        return ["the checker manifest records no schedule, so it cannot be matched to this config"]

    by_label = {str(entry.get("label")): entry for entry in recorded if isinstance(entry, dict)}
    findings: list[str] = []
    for slot in enforceable:
        entry = by_label.get(slot.label)
        if entry is None:
            findings.append(f"slot {slot.label} has no compiled checker")
            continue
        if list(entry.get("iterations") or []) != list(slot.iterations):
            findings.append(f"slot {slot.label} now covers different iterations")
        if str(entry.get("text") or "").strip() != slot.text.strip():
            findings.append(f"slot {slot.label}'s constraint text has changed since it was compiled")
        if bool(entry.get("enforce", True)) != slot.enforce:
            findings.append(f"slot {slot.label}'s enforcement has been toggled")
    # The reverse direction compares against what was *compiled*, not against the recorded schedule.
    # The manifest records every slot, enforceable or not, so comparing the recorded schedule with
    # `enforceable()` reported each unconstrained slot as "compiled but no longer in the schedule" —
    # a finding that is never true of a slot that was never compiled, and which recompiled all the
    # checkers on every `run_loop` invocation. That cost two needless agent runs on the first real
    # pass before it was noticed.
    compiled = {
        str(entry.get("label")) for entry in (manifest.get("slots") or [])
        if isinstance(entry, dict)
    }
    for label in sorted(compiled - {s.label for s in enforceable}):
        findings.append(f"slot {label} was compiled but is no longer enforceable")
    return findings


def verify_manifest(project_path: Path) -> list[str]:
    """Findings if a compiled checker has been edited or removed since it was written."""
    manifest = read_manifest(project_path)
    findings: list[str] = []
    for entry in manifest.get("slots", []):
        path = project_path / CONSTRAINTS_REL / str(entry.get("checker", ""))
        if not path.is_file():
            findings.append(f"slot {entry.get('label')}: checker {path.name} is missing")
            continue
        actual = sha256_file(path)
        if actual != entry.get("sha256"):
            findings.append(
                f"slot {entry.get('label')}: checker {path.name} was edited after compilation "
                f"(recorded {str(entry.get('sha256'))[:12]}, found {actual[:12]})"
            )
    return findings


#: A checker's verdict, as the loop reads it back.
@dataclass
class SlotVerdict:
    """Whether one iteration followed its slot's constraint."""

    iteration: int
    label: str | None
    checked: bool
    passed: bool
    findings: list[str] = field(default_factory=list)
    output: str = ""

    def summary(self) -> str:
        if not self.checked:
            return "no constraint on this iteration"
        if self.passed:
            return f"slot {self.label}: followed"
        detail = "; ".join(self.findings[:3]) or "no finding reported"
        return f"slot {self.label}: VIOLATED — {detail}"


def read_slot_verdict(iteration: int, label: str | None, path: Path, output: str,
                      return_code: int) -> SlotVerdict:
    """Turn a checker's run into a verdict.

    The JSON it writes is preferred and the exit code is the fallback, so a checker that crashed
    before writing anything reports as a violation with its traceback rather than as a pass. A
    broken checker blocking the iteration is the safe direction: the alternative is a schedule
    that silently stops constraining anything.
    """
    payload: dict[str, Any] = {}
    if path.is_file():
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            payload = {}
    if payload:
        findings = [str(f) for f in (payload.get("findings") or [])]
        return SlotVerdict(
            iteration=iteration, label=label, checked=True,
            passed=bool(payload.get("passed")) and return_code == 0,
            findings=findings, output=output,
        )
    tail = "\n".join(output.strip().splitlines()[-10:])
    return SlotVerdict(
        iteration=iteration, label=label, checked=True, passed=return_code == 0,
        findings=[] if return_code == 0 else [
            f"the checker wrote no verdict and exited {return_code}",
            tail or "(no output)",
        ],
        output=output,
    )


#: The contract the compiler agent is held to. Extracted as a constant because it appears both in
#: the compiler's prompt and in the validation that its output is usable.
CHECKER_CONTRACT = """\
Each checker is a standalone Python script, run from the candidate repository's root:

    python <checker> --repo <dir> --json <report.json>

It must:
  - import nothing outside the standard library
  - read only files under `--repo` (the constraint is about the candidate, nothing else)
  - write `--json` as {"passed": true|false, "findings": ["...", ...]}
  - exit 0 when the constraint was followed and non-zero when it was not
  - finish in under 30 seconds: it is a static read of a source file, not a build or a run
  - never modify anything
"""

#: Either quote style: the checker is Python source, and a dict literal there is as likely to use
#: single quotes as the JSON it ends up writing is to use double.
_JSON_KEYS = re.compile(r"""['"](passed|findings)['"]""")


def validate_checker_source(path: Path) -> list[str]:
    """Cheap structural checks on a compiled checker, before the run depends on it.

    Not a proof that the checker implements its prose — nothing here can be. It catches the
    failures that would otherwise surface as a mid-run infrastructure error: a script that does
    not parse, does not take the arguments the loop passes, or never writes a verdict.
    """
    findings: list[str] = []
    if not path.is_file():
        return [f"{path.name} was not written"]
    source = path.read_text()
    try:
        compile(source, str(path), "exec")
    except SyntaxError as exc:
        return [f"{path.name} does not parse: {exc}"]
    for flag in ("--repo", "--json"):
        if flag not in source:
            findings.append(f"{path.name} never mentions {flag}, so the loop cannot drive it")
    if len(set(_JSON_KEYS.findall(source))) < 2:
        findings.append(
            f"{path.name} does not write both 'passed' and 'findings' into its JSON report"
        )
    return findings
