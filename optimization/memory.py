# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A directory of prior work an iteration may start from.

`bootstrap/memory.py` carries one bootstrapped module's kernel and notes to the next one, and it
does so automatically: every entry, every iteration, recorded by the loop after a run ends. That
shape is right there, because bootstrap's modules are interchangeable archetypes and the loop is
the only thing that knows when one finished.

Optimization is the other case. What is worth carrying is not another module — it is *this*
module's own earlier run: the submodule loop's notes when the assembly stalls, both loops' reviews
when a second round starts, the `source.py` a previous round ended on. That material is assembled
by the operator, who knows which run is worth reading, and it is wanted on *some* iterations and
not others: iteration 1 of a second round should start from what the first round learned, while a
free-exploration iteration deliberately should not, because the point of that slot is to reach
something the earlier run did not.

So this is the operator-driven counterpart:

- **The operator names the directory.** `memory.path` in the pipeline config. Nothing is recorded
  automatically and nothing is indexed by shape — the directory's own layout and its README are
  the index, because the operator built it for this module.
- **The schedule says which iterations read it.** Same selector grammar as
  `iteration_constraints` (`iterations:`, `from:`/`to:`, `at:`), plus `all`. An iteration that does
  not read it is not told it exists.
- **The operator writes the navigation prose.** `memory.prompt` goes into those iterations'
  prompts directly beneath the pointer to the seeded copy. A directory of twelve files with no
  word about which to open first is a directory the agent skims and abandons.

Seeded the same way bootstrap's is, and for the same two reasons: copied rather than pointed at,
so an iteration reads a snapshot that cannot change under it, and landed under `.autohelix/`,
which is gitignored and outside the editable scope, so it cannot be committed into the candidate
or edited on the way through.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from optimization.constraints import ScheduleError, _parse_iterations

#: Where the seeded copy appears inside an iteration worktree. Named literally in the prompt, so
#: it must not change casually. Under `.autohelix/` for the same reason bootstrap's is.
SEEDED_REL = Path(".autohelix") / "memory"

#: Read `iterations: all` as "every iteration of this stage", which is also what an omitted
#: selector means: pointing a stage at a memory directory and then having no iteration read it is
#: not a configuration anybody wants by accident.
EVERY = "all"

#: Names an operator's index conventionally takes. Listed first in the prompt's top-level
#: listing, because the whole point of an index is to be the thing opened first.
_INDEX_NAMES = ("README.md", "INDEX.md")

#: The keys that together name *which* iterations read the memory. Treated as one unit when a
#: stage block is merged over the top-level one: a stage that narrows the selection has to replace
#: the whole selector, not add a second one beside it, or `iterations: all` at the top level would
#: silently outrank `at: 3` in the stage.
_SELECTOR_KEYS = ("iterations", "at", "from", "to")


class MemoryError_(ValueError):
    """A memory block that cannot be honoured as written."""


@dataclass
class MemorySpec:
    """One stage's memory: where it is, who reads it, and how to navigate it."""

    path: Path | None = None
    prompt: str = ""
    #: Which iterations read it. Empty tuple with `every=False` means none.
    iterations: tuple[int, ...] = ()
    every: bool = False
    #: Kept so warnings can name the file the operator wrote, not the derived config.
    source: Path | None = field(default=None, compare=False)

    @property
    def enabled(self) -> bool:
        """Whether any iteration reads this. A path with no reader is disabled, not an error."""
        return self.path is not None and (self.every or bool(self.iterations))

    def reads_at(self, iteration: int) -> bool:
        if not self.enabled:
            return False
        return self.every or iteration in self.iterations

    # -- parsing -------------------------------------------------------------------

    @staticmethod
    def merge(shared: Any, stage: Any, where: str = "memory") -> dict[str, Any] | None:
        """One `memory:` block from a shared one and a stage's override of it.

        The point is that `path:` is written once. Both stages of a run almost always read the same
        directory — it is the same module's earlier work — so requiring the path twice is the
        redundancy this removes, while `iterations:` and `prompt:` stay per-stage because those are
        exactly what differs between optimizing one rank and optimizing four.

        The selector is replaced wholesale rather than key-by-key: see `_SELECTOR_KEYS`.
        """
        for name, block in (("shared", shared), ("stage", stage)):
            if block is not None and not isinstance(block, dict):
                raise MemoryError_(f"{where}: the {name} block must be a mapping")
        if not shared and not stage:
            return None
        merged = dict(shared or {})
        stage = dict(stage or {})
        if any(k in stage for k in _SELECTOR_KEYS):
            for key in _SELECTOR_KEYS:
                merged.pop(key, None)
        merged.update(stage)
        return merged

    @classmethod
    def from_config(
        cls,
        section: Any,
        *,
        max_iterations: int,
        base_dir: Path | None = None,
        where: str = "memory",
    ) -> MemorySpec:
        """Parse a `memory:` block. An absent or empty block is a disabled spec, not an error."""
        if section is None:
            return cls()
        if not isinstance(section, dict):
            raise MemoryError_(f"{where}: must be a mapping")

        known = {"path", "prompt", "iterations", "at", "from", "to"}
        unknown = set(section) - known
        if unknown:
            raise MemoryError_(
                f"{where}: unknown key(s) {', '.join(sorted(unknown))}. "
                f"Expected {', '.join(sorted(known))}"
            )

        raw_path = section.get("path")
        if raw_path is None or not str(raw_path).strip():
            # A block the operator started and did not finish. Refused rather than quietly
            # disabled, for either half: a `prompt:` with no `path:` puts navigation prose in no
            # prompt at all, and a selector with no `path:` names iterations that then run with no
            # memory and no warning — which is the case an operator hits by writing
            # `iterations:` in a stage and forgetting `path:` in the shared block.
            started = [
                key for key in ("prompt", *_SELECTOR_KEYS)
                if key in section and str(section.get(key) or "").strip() != ""
            ]
            if started:
                raise MemoryError_(
                    f"{where}: has {', '.join(repr(k) for k in started)} but no 'path:'. "
                    f"Set `path:` here or in the shared top-level `memory:` block, or remove "
                    f"these keys — as written, those iterations would run with no memory"
                )
            return cls()
        path = Path(str(raw_path)).expanduser()
        if not path.is_absolute() and base_dir is not None:
            path = (base_dir / path).resolve()

        every, iterations = cls._parse_selector(section, max_iterations, where)
        return cls(
            path=path,
            prompt=str(section.get("prompt") or ""),
            iterations=iterations,
            every=every,
            source=base_dir,
        )

    @staticmethod
    def _parse_selector(
        section: dict[str, Any], max_iterations: int, where: str,
    ) -> tuple[bool, tuple[int, ...]]:
        """`all`, or the same `at`/`iterations`/`from`/`to` grammar the schedule uses."""
        raw = section.get("iterations")
        if isinstance(raw, str):
            if raw.strip().lower() != EVERY:
                raise MemoryError_(
                    f"{where}: 'iterations' as a string must be '{EVERY}' (got {raw!r}); "
                    f"otherwise use a list, or 'from'/'to', or 'at'"
                )
            return True, ()
        if raw is None and not any(k in section for k in ("at", "from", "to")):
            return True, ()
        try:
            chosen = _parse_iterations(section, where)
        except ScheduleError as exc:
            raise MemoryError_(str(exc)) from exc
        beyond = [i for i in chosen if i > max_iterations]
        if beyond:
            raise MemoryError_(
                f"{where}: names iteration(s) {beyond} but the stage budget is {max_iterations}"
            )
        return False, tuple(chosen)

    # -- the derived config round trip ----------------------------------------------

    def to_payload(self) -> dict[str, Any]:
        """YAML-safe, because `derive_loop_config` writes it and the loop reads it back."""
        payload: dict[str, Any] = {"path": str(self.path)}
        payload["iterations"] = EVERY if self.every else list(self.iterations)
        if self.prompt:
            payload["prompt"] = self.prompt
        return payload

    # -- warnings ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """Problems worth telling the operator about before a stage starts, not errors.

        A missing directory is a warning rather than a refusal on purpose: the pipeline config is
        written before the memory is assembled, and a stage that refuses to start because a
        directory it will not read until iteration 4 does not exist yet is a worse failure than
        one that says so and carries on.
        """
        if not self.enabled:
            if self.path is not None:
                return [
                    f"memory.path is set to {self.path} but no iteration reads it — "
                    f"add `iterations: {EVERY}` or a selector"
                ]
            return []
        found: list[str] = []
        if not self.path.exists():
            found.append(f"memory.path {self.path} does not exist; those iterations get nothing")
        elif not self.path.is_dir():
            found.append(f"memory.path {self.path} is not a directory")
        elif not entry_names(self.path):
            found.append(f"memory.path {self.path} holds nothing to read")
        if not self.prompt.strip():
            found.append(
                "memory.prompt is empty — the agent is handed a directory with no word on how to "
                "navigate it, which is how memory gets skimmed and abandoned"
            )
        return found


# --------------------------------------------------------------------------------------
# reading the directory
# --------------------------------------------------------------------------------------


def entry_names(memory: Path) -> list[str]:
    """Top-level entries, index first then sorted — so the listing is stable and starts usefully.

    The index leads because the prompt tells the agent to open it first, and a listing that
    omitted it would contradict that. Dotfiles are dropped: they are the operator's, not material.
    """
    if not memory.is_dir():
        return []
    names = sorted(p.name for p in memory.iterdir() if not p.name.startswith("."))
    index = [n for n in _INDEX_NAMES if n in names]
    return index + [n for n in names if n not in index]


def seed_problem(spec: MemorySpec, worktree_dir: Path) -> str | None:
    """Why this memory cannot be seeded into this worktree, or None when it can.

    The one that matters is containment. `copytree` from a directory that *contains* its own
    destination copies its own output as it writes it: the recursion only stops at the path-length
    limit, and what it leaves behind first is a deeply nested partial tree and a lot of disk. An
    operator reaches this by pointing `path:` at the workspace root, or at the module repo, rather
    than at a directory beside them — which is a plausible mistake, not a contrived one.
    """
    if not spec.enabled:
        return None
    try:
        source = spec.path.resolve()
        target = (Path(worktree_dir) / SEEDED_REL).resolve()
    except OSError as exc:  # pragma: no cover - a path that cannot be resolved at all
        return f"memory.path {spec.path} could not be resolved: {exc}"
    if source == target:
        return f"memory.path {spec.path} *is* the seed destination"
    if target.is_relative_to(source):
        return (
            f"memory.path {spec.path} contains the worktree, so copying it would copy its own "
            f"output into itself. Point it at a directory beside the workspace root, not at one "
            f"above it"
        )
    if source.is_relative_to(target):
        return (
            f"memory.path {spec.path} is inside the seed destination {SEEDED_REL}, which is "
            f"replaced on every seed"
        )
    if not source.is_dir():
        return f"memory.path {spec.path} is not a readable directory"
    return None


def unlock(worktree_dir: Path) -> None:
    """Give the write bits back to a seeded tree, so it can be removed.

    `seed` takes the directory bits away as well as the files' — on Unix, write permission on the
    containing directory is enough to replace a `0444` file, so files alone do not make a snapshot
    read-only. The cost is that anything which later *deletes* the tree needs them back:
    `git worktree remove --force`, `shutil.rmtree`, and `seed`'s own replacement of a previous
    copy. Called from all three, so read-only never turns into a teardown failure.
    """
    root = Path(worktree_dir) / SEEDED_REL
    if not root.exists():
        return
    try:
        for path in [root, *root.rglob("*")]:
            if path.is_dir():
                path.chmod(0o755)
    except OSError:
        pass


def seed(spec: MemorySpec, worktree_dir: Path) -> int:
    """Copy the memory into a worktree, read-only. Returns files seeded, 0 when there is nothing.

    Read-only means the *directories* too, not only the files. Best-effort otherwise, like
    bootstrap's: a memory directory that cannot be read is a degraded iteration, not a failed one,
    and an iteration that lost its notes is still worth running.
    """
    if seed_problem(spec, worktree_dir) is not None:
        return 0
    try:
        target = Path(worktree_dir) / SEEDED_REL
        if target.exists():
            unlock(worktree_dir)
            shutil.rmtree(target)
        shutil.copytree(spec.path, target)
        count = 0
        # Depth-first, so a directory's contents are chmodded before the directory itself. chmod
        # on an existing entry needs only ownership, but doing it in this order keeps the tree
        # traversable at every step and makes the loop easy to reason about.
        for path in sorted(target.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if path.is_file():
                path.chmod(0o444)
                count += 1
            elif path.is_dir():
                path.chmod(0o555)
        target.chmod(0o555)
        return count
    except OSError:
        return 0


def describe_for_prompt(spec: MemorySpec, iteration: int, seeded: int) -> str:
    """The prompt block for an iteration that reads the memory. Empty string when it does not."""
    if not spec.reads_at(iteration) or seeded <= 0:
        return ""
    names = entry_names(spec.path)
    lines = [
        f"**Start from the memory in `{SEEDED_REL}/`.** {seeded} file(s) from earlier work on this "
        f"module, copied in read-only before you started. It is outside your editable scope, so "
        f"you cannot commit it and nothing you do to it reaches the candidate.",
        "",
    ]
    if names:
        shown = ", ".join(f"`{n}`" for n in names[:12])
        more = f", and {len(names) - 12} more" if len(names) > 12 else ""
        lines += [f"Top level: {shown}{more}.", ""]
    if spec.prompt.strip():
        lines += [spec.prompt.strip(), ""]
    lines += [
        "Read it as evidence, not as an answer. A measurement in there was taken on a different "
        "kernel, and a number that was true then can be false now — your own gate and your own "
        "profile are the only things that decide. Where it records something that *did not* work, "
        "that is the most valuable part: do not spend an iteration rediscovering it.",
    ]
    return "\n".join(lines)
