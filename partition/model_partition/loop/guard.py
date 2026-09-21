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
#: ``plan/partition_graph.yaml`` and ``modules/*/inference.py`` — is absent here
#: by design.
PROTECTED_FILES = (
    "run.yaml",
    "state.json",
    "modules/index.yaml",
    "modules/*/verify.py",
    "modules/*/meta.yaml",
    "modules/*/README.md",
    "trace/records.yaml",
    "trace/manifest.yaml",
    "reports/verify.json",
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
    stamps: dict[str, tuple[int, int]] = field(default_factory=dict)

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
                    stat = path.stat()
                    guard.stamps[str(path.relative_to(base))] = (stat.st_size, stat.st_mtime_ns)
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
            stat = path.stat()
            if (stat.st_size, stat.st_mtime_ns) != stamp:
                report.tampered.append(name)
        return report
