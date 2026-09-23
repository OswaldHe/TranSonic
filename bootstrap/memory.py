# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""What one bootstrapped module leaves behind for the next one to read.

Modules repeat. A partition of a decoder splits into a handful of archetypes and then
instantiates each of them many times, so the fourth attention kernel is largely the third
one at a different shape, and every validator is the same twenty lines of load-trace-profile
-compare around a different tensor table. But each module is its own git repo with its own
history, so nothing crossed between them: iteration 1 re-derived from the frozen references
every time. Measured over four bootstraps, that first iteration cost $48 on average against
$10.50 for the ones after it, because the ones after it start from notes.

This makes that carry across repos. After a run ends, the module's kernel, its validator and
its notes are copied into a shared directory beside the repos, indexed by group and sample.
Before each iteration the directory is seeded into the worktree read-only, and the preset's
goal tells the agent to read the index first and work from the closest match.

Two things it is deliberately not:

- It is not a cache. Nothing is reused automatically; the agent reads it and decides. A
  kernel written for another module's shapes is a starting point, not an answer, and the
  frozen references remain the only specification.
- It is not the agent's to write. The loop records entries after a run, outside any
  iteration, so one iteration cannot plant something for the next to find — and the seeded
  copy is what the agent sees, not the directory itself.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: The shared directory, kept beside the module repos rather than inside one: it outlives any
#: single repo and is read by all of them. Dot-prefixed so it does not look like a module.
MEMORY_DIRNAME = ".bootstrap-memory"

#: Where the seeded copy appears inside an iteration worktree. Under `.autohelix/`, which is
#: gitignored and outside the editable scope, so it cannot be committed or edited into the
#: candidate. The preset's goal names this path literally, so it must not change casually.
SEEDED_REL = ".autohelix/memory"

INDEX_NAME = "INDEX.md"
ENTRY_NAME = "ENTRY.md"
META_NAME = "entry.json"


def location(repo: Path) -> Path:
    """The memory directory serving a module repo: its sibling, not its child."""
    return repo.resolve().parent / MEMORY_DIRNAME


@dataclass
class Entry:
    """One recorded module, as the index lists it."""

    name: str
    module_id: str
    group: str
    sample_id: str
    passed: bool
    iterations: int
    summary: str
    input_shape: str = ""
    output_shape: str = ""

    @classmethod
    def from_meta(cls, payload: dict[str, Any]) -> Entry:
        return cls(
            name=payload.get("name", ""), module_id=payload.get("module_id", ""),
            group=payload.get("group", ""), sample_id=payload.get("sample_id", ""),
            passed=bool(payload.get("passed")), iterations=int(payload.get("iterations", 0)),
            summary=payload.get("summary", ""), input_shape=payload.get("input_shape", ""),
            output_shape=payload.get("output_shape", ""),
        )


def entry_name(
    module_id: str, sample_id: str, step: int = 0, call_index: int = 0,
) -> str:
    """`layers.2.attention` + `long-needle-8192-0` -> `layers_2_attention__long-needle-8192-0`.

    Keyed by the *module*, not its group. A group is a deduplicated implementation shared by
    many modules — `00-Attention` serves thirty-one layers — so keying on it would make the
    second layer bootstrapped overwrite the first, which is exactly the accumulation this
    exists for. Step and call index join the key when they are not the default, because the
    same module at a different decode step is a different recorded invocation.
    """
    key = f"{module_id}__{sample_id}"
    if step:
        key += f"__s{step}"
    if call_index:
        key += f"__c{call_index}"
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
    return safe.strip("_")


def entries(memory: Path) -> list[Entry]:
    """Every recorded entry, newest-looking first is not meaningful so sort by name."""
    found: list[Entry] = []
    if not memory.is_dir():
        return found
    for meta in sorted(memory.glob(f"*/{META_NAME}")):
        try:
            found.append(Entry.from_meta(json.loads(meta.read_text())))
        except (json.JSONDecodeError, OSError, ValueError):
            continue
    return found


def _shape_of(manifest: dict[str, Any], role: str) -> str:
    for tensor in manifest.get("tensors") or []:
        if tensor.get("role") == role:
            return f"{tensor.get('dtype')}{tuple(tensor.get('shape') or ())}"
    return ""


def record(
    repo: Path, *, passed: bool, iterations: int, summary: str, memory: Path | None = None,
) -> Path | None:
    """Copy this repo's kernel, validator and notes into the shared memory.

    Returns the entry directory, or None when there is nothing worth recording. Never raises:
    a run that produced a kernel must not be reported as failed because the memory directory
    was read-only.
    """
    try:
        from bootstrap.materialize import load_manifest

        repo = repo.resolve()
        manifest = load_manifest(repo)
        group = manifest.get("artifact_group") or "unknown"
        module_id = manifest.get("module_id") or "unknown"
        sample_id = manifest.get("sample_id") or "unknown"

        target = (memory or location(repo)) / entry_name(
            module_id, sample_id, int(manifest.get("step") or 0),
            int(manifest.get("call_index") or 0),
        )
        # A re-run of the same module replaces its entry rather than accumulating duplicates;
        # accumulation is across *modules*, which is what makes the index worth reading.
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)

        for name in ("source.py", "inference.py"):
            source = repo / name
            if source.is_file():
                shutil.copy2(source, target / name)

        notes = repo / ".autohelix" / "notes"
        if notes.is_dir():
            shutil.copytree(notes, target / "notes", dirs_exist_ok=True)

        meta = {
            "name": target.name, "group": group, "module_id": module_id,
            "sample_id": sample_id, "passed": passed, "iterations": iterations,
            "summary": summary,
            "input_shape": _shape_of(manifest, "input"),
            "output_shape": _shape_of(manifest, "golden"),
            "submodules": manifest.get("submodules") or [],
            "composition": manifest.get("composition"),
            "state_included": manifest.get("state_included"),
            "tolerance": manifest.get("tolerance") or {},
        }
        (target / META_NAME).write_text(json.dumps(meta, indent=2) + "\n")
        (target / ENTRY_NAME).write_text(_render_entry(meta))
        write_index(target.parent)
        return target
    except Exception:  # noqa: BLE001 - recording is best-effort by design
        return None


def _render_entry(meta: dict[str, Any]) -> str:
    bar = meta.get("tolerance") or {}
    lines = [
        f"# {meta['module_id']} ({meta['group']}, sample `{meta['sample_id']}`)",
        "",
        f"- **{'bootstrapped' if meta['passed'] else 'did not pass'}** "
        f"in {meta['iterations']} iteration(s): {meta['summary']}",
        f"- chain: {' -> '.join(s for s in meta.get('submodules') or [] if s) or 'unknown'}",
        f"- input `{meta.get('input_shape')}` -> output `{meta.get('output_shape')}`",
        f"- composition: {meta.get('composition')}; "
        f"cross-module state included: {meta.get('state_included')}",
    ]
    if bar:
        lines.append("- bar: " + ", ".join(f"{k}={v:g}" for k, v in sorted(bar.items())))
    lines += [
        "",
        "`source.py` is the kernel this module ended on and `inference.py` its validator.",
        "`notes/` is what each iteration recorded about the device and the compiler.",
        "",
        "Read them as evidence from a *different* module. Where its shapes, dtypes or bar",
        "differ from yours, it is wrong for you — the frozen references in your own repo are",
        "the specification, and this is only a head start on reaching them.",
    ]
    return "\n".join(lines) + "\n"


def write_index(memory: Path) -> Path | None:
    """Rewrite `INDEX.md` so one read tells the agent what is worth opening."""
    if not memory.is_dir():
        return None
    found = entries(memory)
    lines = [
        "# Bootstrap memory",
        "",
        f"{len(found)} module(s) bootstrapped before this one. Each directory holds that",
        "module's kernel (`source.py`), its validator (`inference.py`) and its iteration",
        "notes. Skim this table, open the closest match, and adapt — do not paste: a kernel",
        "written for another shape or another numeric format is wrong here even when it looks",
        "right, and your own `reference_torch.py` is the only specification.",
        "",
        "| entry | module | sample | input -> output | result |",
        "|---|---|---|---|---|",
    ]
    for e in found:
        result = f"passed in {e.iterations} iter" if e.passed else f"failed after {e.iterations} iter"
        lines.append(
            f"| `{e.name}` | `{e.module_id}` | {e.sample_id} | "
            f"`{e.input_shape}` -> `{e.output_shape}` | {result} |"
        )
    if not found:
        lines.append("| _(empty)_ | | | | |")
    path = memory / INDEX_NAME
    path.write_text("\n".join(lines) + "\n")
    return path


def seed(repo: Path, worktree_dir: Path, memory: Path | None = None) -> int:
    """Copy the memory into a worktree, read-only to the agent. Returns entries seeded.

    Copied rather than pointed at, so an iteration reads a snapshot that cannot change under
    it and cannot be written back through. Best-effort: no memory is a normal state, and the
    first module ever bootstrapped has none.
    """
    try:
        source = memory or location(repo)
        if not source.is_dir():
            return 0
        found = entries(source)
        if not found:
            return 0
        target = worktree_dir / SEEDED_REL
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
        for path in target.rglob("*"):
            if path.is_file():
                path.chmod(0o444)
        return len(found)
    except Exception:  # noqa: BLE001 - seeding is best-effort by design
        return 0


def clear(memory: Path) -> int:
    """Delete every entry. Returns how many were removed."""
    count = len(entries(memory))
    if memory.is_dir():
        shutil.rmtree(memory)
    return count
