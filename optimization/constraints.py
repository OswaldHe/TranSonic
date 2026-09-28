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

import ast
import hashlib
import json
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from optimization.slotcheck import CHECKER_TIMEOUT_SECONDS

#: How an unfilled hint reads if one survives into a slot's prose. Matched as a prefix rather than
#: as the whole `<FILL IN>` of `config.PLACEHOLDER`, because a slot's hints carry a description
#: after the colon: `<FILL IN: module-specific guidance for iterations 9-10, if any.>`.
PLACEHOLDER_MARKER = "<FILL IN"

#: Where compiled checkers live, relative to the project. Inside `.autohelix/`, which is
#: gitignored and — crucially — *not* among the directories `Sandbox.prepare_worktree` seeds into
#: an iteration worktree, so a checker is never placed where the agent is working. As in
#: `floorplan`, treat that as a speed bump backed by the reviewer rather than a sandbox: a
#: determined agent can still walk up out of its worktree.
CONSTRAINTS_REL = Path(".autohelix") / "optimization" / "constraints"

#: The record of what the compiler produced, hashed so a later edit to a checker is detectable.
MANIFEST_REL = CONSTRAINTS_REL / "manifest.json"

#: What a checker is invoked as. Driven through `optimization.slotcheck` rather than directly, so
#: that a slot's last iteration can run it `--advisory` — writing the same verdict but
#: exiting 0, which lets the measurement proceed and hands the accept/reject decision to the loop's
#: stricter end-of-slot rule. See `optimization/slotcheck.py`.
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


#: How a slot's constraint is enforced, per iteration.
#:
#: - ``hard``  a violation rejects the iteration and its work is discarded.
#: - ``soft``  a violation is checked and recorded but never rejects on its own. Acceptance falls to
#:             the stricter metric rule: correct, and *strictly* faster than the best so far, or the
#:             iteration is rejected anyway. So a soft constraint still costs something to miss.
#: - ``off``   the text goes into the prompt as guidance and no checker is written.
ENFORCEMENT = ("hard", "soft", "off")

#: What the old boolean `enforce:` meant, kept working because configs in the wild use it.
_ENFORCE_ALIAS = {True: "hard", False: "off"}


@dataclass
class Slot:
    """The iterations one constraint governs, and how strictly it is enforced on each of them.

    A slot may be a single iteration, which is how per-iteration control is written: one slot per
    iteration, each with its own text and its own `enforcement`. The range form is shorthand for
    several iterations that happen to share a constraint.
    """

    iterations: list[int]
    text: str = ""
    #: One of `ENFORCEMENT`. See that constant for what each does.
    enforcement: str = "hard"
    #: Whether the last iteration of a `hard` slot spanning **more than one** iteration drops to
    #: `soft`. On by default: by then the agent has had every other iteration the slot allows, so
    #: discarding something correct and faster buys nothing. Set false for a constraint that must
    #: hold on every iteration of the slot without exception.
    #:
    #: It does not apply to a one-iteration slot, where the only iteration is also the last one:
    #: softening it would make `enforcement: hard` mean nothing, which is never what `at: 3` with
    #: `enforcement: hard` was asking for. The escape exists because earlier iterations under the
    #: same constraint came first; with no earlier iterations there is nothing it can be an escape
    #: from.
    soften_last: bool = True
    #: Whether enforcement was written in the config. Tracked so "enforced with no text" can be
    #: warned about while a deliberately empty slot — the free-exploration ones — stays silent.
    enforce_explicit: bool = False
    #: Filled in by `compile_schedule`: the checker written for this slot.
    checker: str | None = None

    @property
    def enforce(self) -> bool:
        """Whether this slot needs a checker at all. Retained for the old boolean's readers."""
        return self.enforcement != "off"

    def enforcement_for(self, iteration: int) -> str:
        """How this iteration is judged: one of `ENFORCEMENT`.

        The only place the three-way decision is made, so the prompt, the checker invocation and the
        acceptance rule cannot disagree about which iteration was hard and which was soft.
        """
        if not self.has_text or self.enforcement == "off":
            return "off"
        if self.enforcement == "soft":
            return "soft"
        if self.soften_last and len(self.iterations) > 1 and iteration >= self.last_iteration:
            return "soft"
        return "hard"

    @property
    def last_iteration(self) -> int:
        """The final iteration this slot governs."""
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
            "enforcement": self.enforcement,
            "soften_last": self.soften_last,
            "per_iteration": {str(i): self.enforcement_for(i) for i in sorted(self.iterations)},
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
            known = {"iterations", "at", "from", "to", "text", "enforce", "enforcement",
                     "soften_last"}
            unknown = set(entry) - known
            if unknown:
                raise ScheduleError(
                    f"{where}: unknown key(s) {', '.join(sorted(unknown))}. "
                    f"Known: {', '.join(sorted(known))}"
                )
            if "enforce" in entry and "enforcement" in entry:
                raise ScheduleError(
                    f"{where}: has both 'enforce' and 'enforcement'. Use 'enforcement' "
                    f"({'/'.join(ENFORCEMENT)}); 'enforce' is the older boolean form of it"
                )
            enforcement = entry.get("enforcement")
            if enforcement is None:
                enforcement = _ENFORCE_ALIAS.get(bool(entry.get("enforce", True)), "hard")
            elif isinstance(enforcement, bool):
                # YAML 1.1 reads bare `off` as False and `on` as True, so `enforcement: off` — the
                # spelling this file documents — arrives here as a boolean. Requiring quotes around
                # one of three documented values would be a trap, so both spellings are accepted.
                enforcement = _ENFORCE_ALIAS[enforcement]
            enforcement = str(enforcement).strip().lower()
            if enforcement not in ENFORCEMENT:
                raise ScheduleError(
                    f"{where}: enforcement must be one of {', '.join(ENFORCEMENT)}, "
                    f"not '{enforcement}'"
                )

            iterations = _parse_iterations(entry, where)
            if max_iterations is not None:
                beyond = [i for i in iterations if i > max_iterations]
                if beyond:
                    raise ScheduleError(
                        f"{where}: names iteration(s) {beyond} but budget.iterations is "
                        f"{max_iterations}. Raise the budget or narrow the slot — a slot that "
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
                enforcement=enforcement,
                soften_last=bool(entry.get("soften_last", True)),
                enforce_explicit=("enforce" in entry or "enforcement" in entry),
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
            # A `#` inside a `text: |` block is prompt content, not a YAML comment. The first
            # shipped template put its fill-in hints there, so a config used as delivered sent
            # "<FILL IN: module-specific guidance ...>" to the agent as part of its constraint.
            if PLACEHOLDER_MARKER in slot.text:
                warnings.append(
                    f"slot {slot.label} still contains a {PLACEHOLDER_MARKER} placeholder, and "
                    f"the whole of text: reaches the agent verbatim — fill it in or delete the line"
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
        mode = slot.enforcement_for(iteration)
        if mode == "hard":
            lines += [
                "",
                "**This is a hard constraint on this iteration.** It is checked by a script you "
                "cannot see, before anything is measured. An iteration that does not follow it is "
                "rejected and its work discarded, however fast it is — so if you believe the "
                "constraint is wrong for this module, say so in your notes and follow it anyway.",
            ]
        elif mode == "soft":
            last = iteration >= slot.last_iteration and slot.enforcement == "hard"
            lines += [
                "",
                "**This is a soft constraint on this iteration.** It is still checked by a script "
                "you cannot see, and the verdict is recorded and reported"
                + (", but this is the last iteration it governs, so a violation is no longer fatal"
                   if last else ", but a violation does not reject the iteration on its own")
                + ". What a violation costs is the regression slack: a candidate that misses the "
                  "constraint has to be correct and **strictly faster than the best so far** to be "
                  "kept, where a compliant one only has to stay within the allowance. Follow the "
                  "constraint if you can, and say in your notes why you could not if you did not.",
            ]
        else:
            lines += [
                "",
                "This is guidance rather than a rule: nothing checks it, and nothing rejects an "
                "iteration for missing it.",
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
    """Which iterations a slot governs, from `at:`, `iterations:` or `from:`/`to:`.

    `at: 7` is the single-iteration form, and it is the one to reach for when the point is to give
    one iteration its own constraint and its own enforcement rather than to describe a phase.
    """
    if "at" in entry:
        for other in ("iterations", "from", "to"):
            if other in entry:
                raise ScheduleError(f"{where}: 'at' names one iteration, so drop '{other}'")
        try:
            return [int(entry["at"])] if int(entry["at"]) >= 1 else _refuse_zero(where)
        except (TypeError, ValueError) as exc:
            raise ScheduleError(f"{where}: 'at' must be an integer ({exc})") from exc
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
        _refuse_zero(where)
    return iterations


def _refuse_zero(where: str) -> list[int]:
    raise ScheduleError(
        f"{where}: iterations are 1-based and iteration 0 is the baseline, which has no agent and "
        f"so cannot be constrained"
    )


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


def persisted_verdict(project_path: Path, iteration: int, label: str | None) -> SlotVerdict | None:
    """The verdict an earlier run of this iteration left on disk, or None if there is none.

    A resumed loop has an empty in-memory cache, so a summary written after a resume reported every
    iteration from before it as un-checked — the report showed `—` for iterations whose `iter-N.json`
    was sitting right there. Read as pass/fail only: the exit code that produced it is long gone, so
    a verdict is trusted here exactly as the checker wrote it.
    """
    path = report_path(project_path, iteration)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    if "passed" not in payload:
        return None
    return SlotVerdict(
        iteration=iteration, label=label, checked=True, passed=bool(payload.get("passed")),
        findings=[str(f) for f in (payload.get("findings") or [])],
    )


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
#: the compiler's prompt and in the validation that its output is usable. The timeout is
#: interpolated from where it is enforced: the prompt used to promise 30 seconds while
#: `run_checker` killed at 120, so a compiler that budgeted honestly budgeted for the wrong number.
CHECKER_CONTRACT = f"""\
Each checker is a standalone Python script, run from the candidate repository's root:

    python <checker> --repo <dir> --json <report.json>

It must:
  - import nothing outside the standard library
  - read only files under `--repo` (the constraint is about the candidate, nothing else)
  - write `--json` as {{"passed": true|false, "findings": ["...", ...]}}
  - exit 0 when the constraint was followed and non-zero when it was not
  - finish well inside {CHECKER_TIMEOUT_SECONDS}s, where it is killed: it reads one file statically
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

    It also holds the checker to the read-only half of `CHECKER_CONTRACT`, which used to be stated
    and unchecked. A checker runs *after* `revert_out_of_scope`, so a script that writes
    `source.py` writes the code that is then measured and merged as the candidate — the agent's
    constraint would be enforced by a program able to edit the thing it is judging.
    """
    findings: list[str] = []
    if not path.is_file():
        return [f"{path.name} was not written"]
    source = path.read_text()
    try:
        tree = ast.parse(source, str(path))
    except SyntaxError as exc:
        return [f"{path.name} does not parse: {exc}"]
    for flag in ("--repo", "--json"):
        if flag not in source:
            findings.append(f"{path.name} never mentions {flag}, so the loop cannot drive it")
    if len(set(_JSON_KEYS.findall(source))) < 2:
        findings.append(
            f"{path.name} does not write both 'passed' and 'findings' into its JSON report"
        )
    findings += _side_effect_findings(tree, path.name)
    return findings


#: Imports a checker may have. It reads one file and writes one JSON report, so the standard
#: library's text, parsing and path tools are the whole job. `subprocess` is absent deliberately:
#: a checker that can spawn a process can do anything this list is trying to prevent.
CHECKER_ALLOWED_IMPORTS = frozenset({
    "argparse", "ast", "json", "os", "pathlib", "re", "sys", "collections", "dataclasses",
    "itertools", "functools", "typing", "textwrap", "difflib", "tokenize", "io", "math",
    "string", "enum", "keyword", "symtable", "hashlib", "warnings",
})

#: Calls that delete, replace or execute. Writing is *not* here: a checker's whole output is the
#: JSON report, so it legitimately writes one file, and banning writes would reject every checker
#: the compiler can produce. What the candidate needs protecting from is a checker that removes or
#: overwrites existing files or runs something else — and the runtime guard in `slotcheck` catches
#: the mutation itself, whatever shape the code takes. This list is the cheap early warning.
CHECKER_BANNED_CALLS = frozenset({
    "remove", "unlink", "rmdir", "rmtree", "rename", "copy", "copy2", "copyfile", "copytree",
    "move", "chmod", "system", "popen", "spawnl", "spawnv", "execv", "execve",
    "check_call", "check_output", "Popen", "eval", "exec", "__import__", "import_module",
    "truncate", "symlink", "link",
})


def _side_effect_findings(tree: ast.Module, name: str) -> list[str]:
    """Static reasons this checker is not the read-only reader its contract promises."""
    findings: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            roots = ([a.name.split(".")[0] for a in node.names]
                     if isinstance(node, ast.Import)
                     else [(node.module or "").split(".")[0]])
            for root in roots:
                if root and root not in CHECKER_ALLOWED_IMPORTS:
                    findings.append(
                        f"{name}:{node.lineno} imports '{root}', which is outside the checker's "
                        f"allowed imports — it reads one file and writes one JSON report"
                    )
        elif isinstance(node, ast.Call):
            tail = _call_tail(node.func)
            if tail in CHECKER_BANNED_CALLS:
                findings.append(
                    f"{name}:{node.lineno} calls '{tail}', which can delete, overwrite or execute. "
                    f"A checker runs after out-of-scope changes are reverted, so anything it "
                    f"changes in the repository is measured as the candidate"
                )
    return findings


def _call_tail(func: ast.expr) -> str:
    """The last name in a call's callee: `a.b.rmtree` -> `rmtree`."""
    if isinstance(func, ast.Attribute):
        return func.attr
    return func.id if isinstance(func, ast.Name) else ""
