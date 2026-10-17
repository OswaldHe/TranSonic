# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The per-iteration constraint schedule: different guidance for different iterations.

A run with one fixed set of instructions converges on whatever the first iteration happened to
try. `optimization/` solved that with `iteration_constraints:` — prose attached to iteration
ranges, so early iterations can be held to one approach, middle ones opened up, and the last
ones spent consolidating. This is the same key, with the same spellings, for `integrate run`.

**Here they are guidance only.** `optimization/` compiles each slot into a checker script and can
reject an iteration that violates one. This pass does not: it already has a constraint list whose
verdicts gate acceptance, and a second enforcement path judged by a different mechanism would
make "why was this rejected" a two-document question. So a slot's text goes into the prompt for
the iterations it governs and nothing checks it. `enforcement:` is accepted — configs get copied
between passes — and `hard` warns rather than silently meaning something weaker.

Slots are indexed by iteration number, not by accepted count: if iteration 3 is rejected,
iteration 4 still gets slot 4. The schedule, the budget and the transcript stay aligned, and a
run is reproducible from its config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: How an unfilled slot reads, matched as a prefix because a slot's text carries a description
#: after the marker: `<FILL IN: what iterations 9-13 are for>`.
PLACEHOLDER_MARKER = "<FILL IN"

#: Accepted on a slot for compatibility with `optimization/`'s schedule, which this mirrors.
#: `hard` is the one that warns: nothing here enforces.
ENFORCEMENT = ("hard", "soft", "off")


class ScheduleError(ValueError):
    """The schedule is not well formed."""


@dataclass
class Slot:
    """The iterations one piece of guidance governs."""

    iterations: list[int]
    text: str = ""
    #: Carried for compatibility and reported; see the module docstring.
    enforcement: str = "soft"

    @property
    def label(self) -> str:
        """`1-3`, or `7` for a single iteration."""
        if len(self.iterations) == 1:
            return str(self.iterations[0])
        runs: list[tuple[int, int]] = []
        for i in sorted(self.iterations):
            if runs and i == runs[-1][1] + 1:
                runs[-1] = (runs[-1][0], i)
            else:
                runs.append((i, i))
        return ",".join(f"{a}-{b}" if a != b else str(a) for a, b in runs)

    @property
    def has_text(self) -> bool:
        return bool(self.text.strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "iterations": list(self.iterations),
            "label": self.label,
            "text": self.text,
            "enforcement": self.enforcement,
        }


@dataclass
class Schedule:
    """Slots covering some or all of a run's iterations."""

    slots: list[Slot] = field(default_factory=list)

    def slot_for(self, iteration: int) -> Slot | None:
        for slot in self.slots:
            if iteration in slot.iterations:
                return slot
        return None

    def covered(self) -> set[int]:
        return {i for slot in self.slots for i in slot.iterations}

    # ------------------------------------------------------------------ parsing

    @classmethod
    def from_config(cls, raw: Any, max_iterations: int | None = None) -> "Schedule":
        """Parse an `iteration_constraints:` block.

            iteration_constraints:
              - from: 1
                to: 4
                text: |
                  Measure before you change anything...
              - at: 5
                text: |
                  Spend this one on the kernel that the profile says dominates.

        An absent or empty block is a valid schedule with no slots: every iteration is then
        governed by the goal alone, which is what an unfilled config does.
        """
        if raw is None:
            return cls()
        if not isinstance(raw, list):
            raise ScheduleError(
                "`iteration_constraints:` must be a list of ranges, each with `iterations:` "
                "(or `at:`, or `from:`/`to:`) and `text:`"
            )

        slots: list[Slot] = []
        seen: dict[int, str] = {}
        for index, entry in enumerate(raw):
            where = f"iteration_constraints[{index}]"
            if not isinstance(entry, dict):
                raise ScheduleError(f"{where}: must be a mapping")
            known = {"iterations", "at", "from", "to", "text", "enforcement", "enforce"}
            unknown = set(entry) - known
            if unknown:
                raise ScheduleError(
                    f"{where}: unknown key(s) {', '.join(sorted(unknown))}. "
                    f"Known: {', '.join(sorted(known))}"
                )
            iterations = _parse_iterations(entry, where)
            if max_iterations is not None:
                beyond = [i for i in iterations if i > max_iterations]
                if beyond:
                    raise ScheduleError(
                        f"{where}: names iteration(s) {beyond} but `budget.iterations` is "
                        f"{max_iterations}. Raise the budget or narrow the slot — a slot that "
                        "never runs is guidance you will believe was given"
                    )
            for i in iterations:
                if i in seen:
                    raise ScheduleError(
                        f"{where}: iteration {i} is already governed by slot '{seen[i]}'. "
                        "Ranges may not overlap — one iteration, one slot"
                    )

            slot = Slot(
                iterations=iterations,
                text=str(entry.get("text") or ""),
                enforcement=_parse_enforcement(entry, where),
            )
            for i in iterations:
                seen[i] = slot.label
            slots.append(slot)
        return cls(slots=slots)

    def validate(self, max_iterations: int | None = None) -> list[str]:
        """Things worth saying about the schedule. None of them stop a run."""
        out: list[str] = []
        for slot in self.slots:
            if not slot.has_text:
                out.append(
                    f"iteration(s) {slot.label}: the slot has no `text:`, so those iterations "
                    "are governed by the goal alone"
                )
            elif PLACEHOLDER_MARKER in slot.text:
                out.append(
                    f"iteration(s) {slot.label}: the text still contains '{PLACEHOLDER_MARKER}' "
                    "— it goes into the prompt as written"
                )
            if slot.enforcement == "hard":
                out.append(
                    f"iteration(s) {slot.label}: `enforcement: hard` has no effect in this pass "
                    "— the schedule is guidance and only the `constraints:` list gates "
                    "acceptance. Express what must hold as a constraint."
                )
        if max_iterations:
            uncovered = sorted(set(range(1, max_iterations + 1)) - self.covered())
            if self.slots and uncovered:
                out.append(
                    f"iteration(s) {', '.join(str(i) for i in uncovered)} have no slot, so they "
                    "run on the goal alone"
                )
        return out

    # ------------------------------------------------------------------ rendering

    def describe_for_prompt(self, iteration: int) -> str:
        """This iteration's guidance, as the prompt's `{iteration_constraint}` block."""
        slot = self.slot_for(iteration)
        if slot is None or not slot.has_text:
            return ""
        scope = (
            f"iteration {iteration}" if len(slot.iterations) == 1
            else f"iterations {slot.label}, of which this is iteration {iteration}"
        )
        return (
            f"The operator set this for {scope}:\n\n{slot.text.strip()}\n\n"
            "It is direction for this iteration rather than a check: nothing measures it, and "
            "the constraints below are what decide whether your work is kept. Follow it anyway — "
            "it is there because a run that does whatever looks best each time repeats itself."
        )

    def summary_table(self) -> str:
        """The whole schedule, so an iteration can see where it sits in the run."""
        if not self.slots:
            return ""
        lines = ["| iterations | guidance |", "| --- | --- |"]
        for slot in self.slots:
            first = (slot.text.strip().splitlines() or [""])[0]
            lines.append(f"| {slot.label} | {first[:90]} |")
        return "\n".join(lines)


def _parse_iterations(entry: dict[str, Any], where: str) -> list[int]:
    """`iterations: [1,2]`, `at: 3`, or `from: 4` / `to: 6` — one of the three."""
    forms = [k for k in ("iterations", "at", "from", "to") if k in entry]
    if not forms:
        raise ScheduleError(f"{where}: needs `iterations:`, `at:`, or `from:`/`to:`")
    if "iterations" in entry and len(forms) > 1:
        raise ScheduleError(
            f"{where}: has both `iterations:` and {', '.join(k for k in forms if k != 'iterations')}"
            " — use one form"
        )
    if "at" in entry and ("from" in entry or "to" in entry):
        raise ScheduleError(f"{where}: has both `at:` and `from:`/`to:` — use one form")

    if "iterations" in entry:
        raw = entry["iterations"]
        if not isinstance(raw, list) or not raw:
            raise ScheduleError(f"{where}: `iterations:` must be a non-empty list")
        values = [_positive_int(v, f"{where}.iterations") for v in raw]
    elif "at" in entry:
        values = [_positive_int(entry["at"], f"{where}.at")]
    else:
        if "from" not in entry or "to" not in entry:
            raise ScheduleError(f"{where}: a range needs both `from:` and `to:`")
        start = _positive_int(entry["from"], f"{where}.from")
        end = _positive_int(entry["to"], f"{where}.to")
        if end < start:
            raise ScheduleError(f"{where}: `to: {end}` is before `from: {start}`")
        values = list(range(start, end + 1))

    out = sorted(set(values))
    if len(out) != len(values):
        raise ScheduleError(f"{where}: the same iteration is listed twice")
    return out


def _parse_enforcement(entry: dict[str, Any], where: str) -> str:
    """`enforcement:`, accepted for compatibility. See the module docstring."""
    raw = entry.get("enforcement")
    if raw is None:
        # YAML 1.1 reads bare `off`/`on` as booleans, so the older `enforce:` form arrives as one.
        if "enforce" in entry:
            return "hard" if bool(entry["enforce"]) else "off"
        return "soft"
    if isinstance(raw, bool):
        return "hard" if raw else "off"
    value = str(raw).strip().lower()
    if value not in ENFORCEMENT:
        raise ScheduleError(
            f"{where}: enforcement must be one of {', '.join(ENFORCEMENT)}, not '{value}'"
        )
    return value


def _positive_int(value: Any, where: str) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ScheduleError(f"{where}: '{value}' is not an integer") from exc
    if out < 1:
        raise ScheduleError(f"{where}: iterations are numbered from 1, got {out}")
    return out
