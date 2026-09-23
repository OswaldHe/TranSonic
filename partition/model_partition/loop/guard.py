# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep an agent inside its editable surface.

The agent runs with the whole run directory writable, and the prompt is not a
security boundary. Two things in that directory decide whether a module is
correct — the generated verifiers and the dumped reference tensors — so an agent
that edited them could make a broken partition pass. This snapshots them around
every agent call and puts back anything that moved.

Small text files are restored from their bytes. Tensor blobs are far too large to
copy, so they are fingerprinted by size and mtime and a change is reported as a
failure rather than repaired: an iteration that rewrote the references cannot be
trusted whatever else it did.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

#: Harness-owned text, restored byte for byte. Everything the agent may edit —
#: ``plan/partition_graph.yaml``, ``modules/*/inference.py`` and, when the surface is
#: ``compat``, ``compat/*.py`` — is absent here by design.
#:
#: ``runtime/**`` and ``calls.json`` are here because the *standalone* verifier runs on
#: them: the loop's own checks import the harness's copy of the comparator, but the
#: isolated check that gates publication imports the artifact's. An agent that weakened
#: the copied comparator, or pointed ``calls.json`` at another tensor, would leave the
#: in-process checks honest and the shipped artifact passing on nothing.
PROTECTED_FILES = (
    "run.yaml",
    "state.json",
    "modules/index.yaml",
    "modules/*/verify.py",
    "modules/*/meta.yaml",
    "modules/*/README.md",
    "modules/*/calls.json",
    "modules/*/config.json",
    "runtime/**/*.py",
    "trace/records.yaml",
    "trace/manifest.yaml",
    "reports/verify.json",
    "reports/chain.json",
    "reports/emulate.json",
)

#: Directories whose contents are checked for tampering but never copied.
FINGERPRINTED_DIRS = ("trace/activations", "trace/weights")


@dataclass
class GuardReport:
    """What the agent touched that it should not have."""

    restored: list[str] = field(default_factory=list)
    tampered: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.restored and not self.tampered

    def summary(self) -> str:
        if self.clean:
            return "agent stayed inside its editable surface"
        parts = []
        if self.restored:
            parts.append(f"restored {len(self.restored)} harness file(s): "
                         f"{', '.join(self.restored[:4])}")
        if self.tampered:
            parts.append(f"{len(self.tampered)} reference tensor(s) changed: "
                         f"{', '.join(self.tampered[:4])}")
        return "; ".join(parts)


@dataclass
class HarnessGuard:
    """A snapshot of the harness-owned files under one run directory."""

    root: Path
    contents: dict[str, bytes] = field(default_factory=dict)
    stamps: dict[str, tuple[int, int, int, int]] = field(default_factory=dict)

    @classmethod
    def capture(cls, root: str | Path) -> HarnessGuard:
        base = Path(root)
        guard = cls(root=base)
        for pattern in PROTECTED_FILES:
            for path in sorted(base.glob(pattern)):
                if path.is_file():
                    guard.contents[str(path.relative_to(base))] = path.read_bytes()
        for name in FINGERPRINTED_DIRS:
            directory = base / name
            if not directory.is_dir():
                continue
            for path in directory.rglob("*"):
                if path.is_file():
                    guard.stamps[str(path.relative_to(base))] = _stamp(path)
        return guard

    def restore(self) -> GuardReport:
        """Put harness files back and report anything unrepairable."""
        report = GuardReport()
        for name, data in self.contents.items():
            path = self.root / name
            if path.is_file() and path.read_bytes() == data:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            report.restored.append(name)
        for name, stamp in self.stamps.items():
            path = self.root / name
            if not path.is_file():
                report.tampered.append(name)
                continue
            if _stamp(path) != tuple(stamp):
                report.tampered.append(name)
        return report


def _stamp(path: Path) -> tuple[int, int, int, int]:
    """A fingerprint of a reference tensor that an agent cannot reproduce at will.

    Size and mtime alone are not enough: `utime()` sets mtime to anything, so a file
    could be rewritten and its stamp restored — and these files are the reference every
    check is measured against. Inode and *change* time close that: ctime moves on any
    write and no unprivileged process can set it back, and a rewrite-and-replace shows
    up as a new inode. Digesting tens of gigabytes of feature maps around every agent
    call would cost more than the call.
    """
    stat = path.stat()
    return (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)
