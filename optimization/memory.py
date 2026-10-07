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
import sys
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

#: `iterations: none` — no loop iteration reads it. Only useful beside `preparation: true`, which
#: is the "let the one-shot agent see it, leave the loop alone" case.
NONE = "none"

#: Names an operator's index conventionally takes. Listed first in the prompt's top-level
#: listing, because the whole point of an index is to be the thing opened first.
_INDEX_NAMES = ("README.md", "INDEX.md")

#: The keys that together name *which* iterations read the memory. Treated as one unit when a
#: stage block is merged over the top-level one: a stage that narrows the selection has to replace
#: the whole selector, not add a second one beside it, or `iterations: all` at the top level would
#: silently outrank `at: 3` in the stage.
_SELECTOR_KEYS = ("iterations", "at", "from", "to")

#: Said to everything that reads the memory, because the seed is per-run and what an agent leaves
#: behind is not. A stage-2 agent wrote ``.autohelix/memory/FEEDBACK.md`` rows 2 and 10 into
#: `source.py` and `SUBMODULE.md` — true while it ran, a dangling path for every later iteration the
#: operator left out of the selector and for anyone reading the repo afterwards. Nothing had told it
#: the directory was temporary, so the citation looked like an ordinary cross-reference.
TRANSIENCE = (
    "The pipeline copies `{rel}/` in for this run only and removes it afterwards, and iterations "
    "outside the schedule never receive it at all. Do not cite these paths in anything you leave "
    "behind, in a comment or in your notes. Restate what the file said in enough detail to act "
    "on, so that someone who never had this copy can follow you."
)


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
    #: Whether this stage's one-shot *preparation* agent reads it too — stage 2 for `submodule`,
    #: stage 4 for `full`. Those agents build the repo the matching loop then optimizes, and they
    #: are the runs that would otherwise re-derive the cut or the collective's placement from
    #: nothing, so they default to on: having configured memory for a stage, wanting its
    #: preparation agent blind to it is the unusual ask, not the usual one.
    preparation: bool = True
    #: Kept so warnings can name the file the operator wrote, not the derived config.
    source: Path | None = field(default=None, compare=False)
    #: Where the seeded copy lands in a worktree. Not operator-configurable — the prompts name it
    #: literally — but carried as a field so the pipeline's own stores, the feedback archive in
    #: particular, are seeded by this same read-only machinery into a destination of their own.
    dest: Path = SEEDED_REL

    @property
    def enabled(self) -> bool:
        """Whether anything reads this. A path with no reader at all is disabled, not an error."""
        return self.path is not None and (self.every or bool(self.iterations) or self.preparation)

    @property
    def reads_in_loop(self) -> bool:
        """Whether any loop iteration reads it, as opposed to only the preparation agent."""
        return self.path is not None and (self.every or bool(self.iterations))

    def reads_at(self, iteration: int) -> bool:
        # Iterations are 1-based and iteration 0 is the baseline, which has no agent to read
        # anything — the same rule `_parse_iterations` enforces when an operator writes `at: 0`.
        # Without this, `iterations: all` claimed to cover iteration 0 as well.
        if iteration < 1 or not self.reads_in_loop:
            return False
        return self.every or iteration in self.iterations

    @property
    def reads_at_preparation(self) -> bool:
        return self.path is not None and self.preparation

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

        known = {"path", "prompt", "iterations", "at", "from", "to", "preparation"}
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
                key for key in ("prompt", "preparation", *_SELECTOR_KEYS)
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
        raw_prep = section.get("preparation", True)
        if not isinstance(raw_prep, bool):
            raise MemoryError_(
                f"{where}: 'preparation' must be true or false (got {raw_prep!r}). It says whether "
                f"this stage's one-shot preparation agent reads the memory"
            )
        return cls(
            path=path,
            prompt=str(section.get("prompt") or ""),
            iterations=iterations,
            every=every,
            preparation=raw_prep,
            source=base_dir,
        )

    @staticmethod
    def _parse_selector(
        section: dict[str, Any], max_iterations: int, where: str,
    ) -> tuple[bool, tuple[int, ...]]:
        """`all`, or the same `at`/`iterations`/`from`/`to` grammar the schedule uses."""
        raw = section.get("iterations")
        if isinstance(raw, str):
            word = raw.strip().lower()
            # One selector form, never two. `iterations: all` with `at: 3` beside it used to take
            # the word and drop the range silently, so a config asking for memory at one iteration
            # got it at every one — the opposite of what it said, and 5 of 5 iterations paying for
            # a seed the operator wanted once. Ordinary selectors already reject conflicts.
            clash = sorted(k for k in ("at", "from", "to") if k in section)
            if clash:
                raise MemoryError_(
                    f"{where}: 'iterations: {word}' cannot be combined with "
                    f"{', '.join(repr(k) for k in clash)}. Use one form: the word "
                    f"'{EVERY}'/'{NONE}', or a list, or 'at', or 'from'/'to'"
                )
            if word == NONE:
                # Only useful beside `preparation: true`: let the one-shot agent read the memory
                # and leave the loop's iterations alone.
                return False, ()
            if word != EVERY:
                raise MemoryError_(
                    f"{where}: 'iterations' as a string must be '{EVERY}' or '{NONE}' "
                    f"(got {raw!r}); otherwise use a list, or 'from'/'to', or 'at'"
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
        if self.every:
            payload["iterations"] = EVERY
        elif self.iterations:
            payload["iterations"] = list(self.iterations)
        else:
            # `[]` would fail the "non-empty list" check on the way back in, and this spec is
            # reachable: `iterations: none` with `preparation: true`.
            payload["iterations"] = NONE
        payload["preparation"] = self.preparation
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
                    f"memory.path is set to {self.path} but nothing reads it — add "
                    f"`iterations: {EVERY}` or a selector, or `preparation: true`"
                ]
            return []
        found: list[str] = []
        if not self.path.exists():
            found.append(f"memory.path {self.path} does not exist; those iterations get nothing")
        elif not self.path.is_dir():
            found.append(f"memory.path {self.path} is not a directory")
        else:
            # `warnings()` runs in `_preflight`, before a stage starts, so an unreadable directory
            # has to come back as a warning. Raising here turns a permissions problem on one path
            # into a crash of the whole command.
            try:
                empty = not entry_names(self.path)
            except OSError as exc:
                found.append(f"memory.path {self.path} cannot be listed ({exc.strerror or exc})")
            else:
                if empty:
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
        target = (Path(worktree_dir) / spec.dest).resolve()
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
            f"memory.path {spec.path} is inside the seed destination {spec.dest}, which is "
            f"replaced on every seed"
        )
    if not source.is_dir():
        return f"memory.path {spec.path} is not a readable directory"
    return _symlink_problem(source, spec.path)


def _symlink_problem(source: Path, declared: Path) -> str | None:
    """Why a symlink in the memory tree makes it unseedable, or None when there is none.

    A snapshot has to be a snapshot. Following links copies whatever they point at — a link to the
    parent recurses to the path-length limit, a link to a large external tree copies all of it —
    and *preserving* them is worse than it looks: the agent reaches the operator's original through
    `.autohelix/memory/<link>` and can write to it, since the read-only chmod cannot apply to a
    link without changing the target's own mode. A tree of only links also reports zero files
    seeded while every target stays reachable.

    So neither copy policy is safe and the link itself is the problem. Refused with the path, since
    replacing it with a copy is a one-line fix and no behaviour here could guess which the operator
    meant.
    """
    try:
        found = [p for p in source.rglob("*") if p.is_symlink() and not _hidden(p, source)]
    except OSError as exc:
        return f"memory.path {declared} could not be walked: {exc}"
    if not found:
        return None
    shown = ", ".join(str(p.relative_to(source)) for p in found[:3])
    return (
        f"memory.path {declared} contains {len(found)} symlink(s) ({shown}"
        f"{', …' if len(found) > 3 else ''}). A seeded memory is a read-only snapshot, and a link "
        f"cannot be one: following it copies the target, and keeping it lets the iteration write "
        f"through to the original. Replace each link with a copy of what it points at"
    )


def _hidden(path: Path, root: Path) -> bool:
    """Whether `path` sits under a dot-prefixed name, and so is not copied into a snapshot."""
    return any(part.startswith(".") for part in path.relative_to(root).parts)


def skip_hidden(directory: str, names: list[str]) -> set[str]:
    """`copytree`'s ignore callback: leave the operator's dotfiles out of the snapshot.

    `entry_names` already treats a dot-prefixed entry as the operator's rather than as memory
    material and keeps it out of the prompt's listing. Copying it anyway is worse than
    inconsistent: a `memory.path` pointed at an earlier repo or a working directory carries `.git`,
    `.env` and whatever else lives there into a tree the agent can read, so a credential the
    operator never meant to share is seeded beside the notes and never mentioned.
    """
    del directory  # the signature `shutil.copytree` calls this with
    return {name for name in names if name.startswith(".")}


def unlock(worktree_dir: Path, *rels: Path) -> None:
    """Give the write bits back to one or more seeded trees, so they can be removed.

    `seed` takes the directory bits away as well as the files' — on Unix, write permission on the
    containing directory is enough to replace a `0444` file, so files alone do not make a snapshot
    read-only. The cost is that anything which later *deletes* the tree needs them back:
    `git worktree remove --force`, `shutil.rmtree`, and `seed`'s own replacement of a previous
    copy. Called from all three, so read-only never turns into a teardown failure.
    """
    for rel in (rels or (SEEDED_REL,)):
        _unlock_one(Path(worktree_dir) / rel)


def _unlock_one(root: Path) -> None:
    if not root.exists():
        return
    try:
        for path in [root, *root.rglob("*")]:
            # `is_dir` and `chmod` both follow symlinks, and `seed` now copies directory links as
            # links rather than following them — so without this an external target's mode is
            # changed during teardown, and a private 0700 directory becomes world-traversable.
            if path.is_symlink():
                continue
            if path.is_dir():
                path.chmod(0o755)
    except OSError:
        pass


def rmtree_unlocked(root: Path, ignore_errors: bool = False) -> None:
    """`shutil.rmtree`, giving back the write bits a read-only snapshot took away.

    `unlock` restores them for *one* worktree's snapshot, which is enough to tear that worktree
    down. A set-aside attempt is a whole repo, and every iteration worktree inside it carries its
    own snapshot — so removing one fails on the first read-only directory with a `PermissionError`
    naming a file nobody would connect to a memory snapshot. `02-Attention`'s stage 4 died on
    `REPORT.md`, eight hours into the run and after its loop had already succeeded.

    Driven by the error rather than by a search for snapshots, so it clears any read-only directory
    in the way and does not have to know where `seed` has been.
    """
    def grant(func, path, _exc):  # type: ignore[no-untyped-def]
        # It is the *containing* directory's write bit that permits unlinking, and a directory's
        # own bits that permit removing it once empty, so both are candidates. Symlinks are left
        # alone, since chmod follows them and the target is not ours.
        for candidate in (Path(path), Path(path).parent):
            try:
                if candidate.is_dir() and not candidate.is_symlink():
                    candidate.chmod(candidate.stat().st_mode | 0o700)
            except OSError:
                pass
        func(path)

    # `onexc` is 3.12; `onerror` takes the same three arguments with `sys.exc_info()` in the third
    # and is the spelling 3.11 has. The package supports both, so pick by what `rmtree` accepts
    # rather than by a version comparison.
    keyword = "onexc" if sys.version_info >= (3, 12) else "onerror"
    try:
        shutil.rmtree(root, **{keyword: grant})
    except OSError:
        if not ignore_errors:
            raise


def seed(spec: MemorySpec, worktree_dir: Path) -> int:
    """Copy the memory into a worktree, read-only. Returns files seeded, 0 when there is nothing.

    Read-only means the *directories* too, not only the files. Best-effort otherwise, like
    bootstrap's: a memory directory that cannot be read is a degraded iteration, not a failed one,
    and an iteration that lost its notes is still worth running.
    """
    if seed_problem(spec, worktree_dir) is not None:
        return 0
    try:
        target = Path(worktree_dir) / spec.dest
        if target.exists():
            unlock(worktree_dir, spec.dest)
            shutil.rmtree(target)
        # `symlinks=True` copies a link as a link instead of following it. Following is how a
        # memory holding a link to its own parent, or to a large external tree, both escapes
        # `seed_problem` — which compares only the two root paths — and copies far more than the
        # operator pointed at.
        shutil.copytree(spec.path, target, symlinks=True, ignore=skip_hidden)
        count = 0
        # Depth-first, so a directory's contents are chmodded before the directory itself. chmod
        # on an existing entry needs only ownership, but doing it in this order keeps the tree
        # traversable at every step and makes the loop easy to reason about.
        for path in sorted(target.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            # `Path.chmod` follows symlinks, so chmodding a copied link would change the mode of
            # whatever it points at — a file outside the snapshot, possibly outside the workspace.
            if path.is_symlink():
                continue
            if path.is_file():
                path.chmod(_locked(path, 0o444))
                count += 1
            elif path.is_dir():
                path.chmod(_locked(path, 0o555))
        target.chmod(_locked(target, 0o555))
        return count
    except OSError:
        # Returning 0 makes the caller omit the memory prompt, so the agent is told nothing about
        # a tree it can nonetheless read — and a snapshot abandoned mid-chmod is still writable.
        # Whatever landed has to go with the report of failure.
        try:
            if target.exists():
                unlock(worktree_dir, spec.dest)
                shutil.rmtree(target, ignore_errors=True)
        except OSError:
            pass
        return 0


def _locked(path: Path, fallback: int) -> int:
    """`path`'s own mode with every write bit cleared, or `fallback` if it cannot be read.

    Forcing 0444 and 0555 is the obvious way to lock a snapshot and it *widens* access: a memory
    file carried in at 0600 becomes readable by every local user on the host, and a 0700 directory
    becomes traversable by them. Read-only is the requirement; who may read was already decided by
    whoever owns the source.
    """
    try:
        return path.stat().st_mode & 0o7777 & ~0o222
    except OSError:
        return fallback


def describe_for_preparation(spec: MemorySpec, seeded: int) -> str:
    """The prompt block for a one-shot preparation agent — stage 2's cut, stage 4's assembly.

    Separate from `describe_for_prompt` because the framing is genuinely different. A loop
    iteration is told "start from this rather than from nothing"; a preparation agent is building
    the repo the loop will then optimize, so what it wants from the memory is the *shape* the
    earlier run settled on — which dimension the cut ran along, where the collective ended up —
    and it has no gate of its own to catch a number it inherited on faith.
    """
    if not spec.reads_at_preparation or seeded <= 0:
        return ""
    lines = [
        f"**Earlier work on this module is in `{spec.dest}/`.** It holds {seeded} read-only "
        f"file(s). That run already settled questions you are about to answer, and it also says "
        f"which ones it could not settle.",
        "",
        _listing(spec),
    ]
    if spec.prompt.strip():
        lines += [spec.prompt.strip(), ""]
    lines += [
        "What you are building has no gate behind it yet, so nothing downstream will catch a "
        "number you took on faith. Separate two kinds of thing. That run fitted its tile widths, "
        "buffer depths and loop bounds to its own shapes, so re-derive those from the repository "
        "in front of you. Technique — how it decodes a value, which engine runs which pass — "
        "usually survives a change of shape, so port it rather than leaving the loop to re-earn "
        "it. Say in your write-up which you took, which you re-derived and which you left, and "
        "why.",
        "",
        TRANSIENCE.format(rel=spec.dest),
    ]
    return "\n".join(lines)


def _listing(spec: MemorySpec) -> str:
    """The top-level entries as one prompt line, or empty when the directory is unreadable."""
    names = entry_names(spec.path) if spec.path else []
    if not names:
        return ""
    shown = ", ".join(f"`{n}`" for n in names[:12])
    more = f", and {len(names) - 12} more" if len(names) > 12 else ""
    return f"Top level: {shown}{more}.\n"


def describe_for_prompt(spec: MemorySpec, iteration: int, seeded: int) -> str:
    """The prompt block for an iteration that reads the memory. Empty string when it does not."""
    if not spec.reads_at(iteration) or seeded <= 0:
        return ""
    names = entry_names(spec.path)
    lines = [
        f"**Earlier work on this module is in `{spec.dest}/`.** The pipeline copied {seeded} "
        f"read-only file(s) there before you started, outside your editable scope, so nothing you "
        f"do to them reaches the candidate.",
        "",
    ]
    if names:
        shown = ", ".join(f"`{n}`" for n in names[:12])
        more = f", and {len(names) - 12} more" if len(names) > 12 else ""
        lines += [f"Top level: {shown}{more}.", ""]
    if spec.prompt.strip():
        lines += [spec.prompt.strip(), ""]
    lines += [
        "Read it as evidence rather than as an answer. An earlier kernel produced every "
        "measurement in there, so a number that held then can be false now, and only your own "
        "gate and your own profile decide. Where it records something that did not work, that is "
        "the most valuable part of it: rediscovering a dead end costs you an iteration.",
        "",
        TRANSIENCE.format(rel=spec.dest),
    ]
    return "\n".join(lines)
