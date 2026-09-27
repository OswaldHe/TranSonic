# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keeping the fields the gate trusts out of the agent's reach.

A preparation agent runs *in* the repository it is building, and it is asked to finish the manifest
— the tensor record and the declaration are genuinely its work. But the same file also carries the
numerical bar, the recorded golden, the rank count and the two latency bounds, and the gate then
reads all of those back as ground truth. So the agent is writing its own examination paper.

Nothing suggests an agent would do this on purpose. It does not need to: a manifest rewritten from
scratch rather than edited loses whatever it did not think to copy, and a bar that quietly widened
by a factor of ten is indistinguishable in the gate's output from a bar that was always that wide.

The fix is custody, not trust. Materialization writes the pipeline-owned fields to a copy **outside**
the repo, and after the agent finishes they are restored over whatever is in the repo now. The agent
keeps every field that is its to decide; the fields that decide whether its work passes are put back
the way the pipeline recorded them. A field that changed is reported, because an agent that tried to
move the bar is worth knowing about even when the move has been undone.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Fields of the submodule manifest the pipeline owns. `tolerance`, `tensors` and the declaration
#: are the agent's — everything here is derived from the bootstrapped module or from the floorplan,
#: and is what the gate measures the agent's work against.
SUBMODULE_OWNED = (
    "module", "entry_point", "projection", "module_output", "module_tolerance", "module_tensors",
)

#: Fields of the assembly manifest the pipeline owns. `tensors`, `frozen` and the agent's own notes
#: are its to write; the bar, the golden, the rank count and both bounds are not.
MODULE_OWNED = (
    "module", "entry_point", "ranks", "projection", "tolerance", "module_output", "baselines",
    "submodule_commit",
)


@dataclass
class Custody:
    """The pipeline-owned half of a manifest, held outside the repo the agent writes in."""

    path: Path
    fields: dict[str, Any] = field(default_factory=dict)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.fields, indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: Path) -> "Custody":
        if not path.is_file():
            return cls(path=path, fields={})
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            return cls(path=path, fields={})
        return cls(path=path, fields=data if isinstance(data, dict) else {})


def take_custody(manifest: dict[str, Any], owned: tuple[str, ...], path: Path) -> Custody:
    """Record the pipeline-owned fields of a freshly materialized manifest, outside the repo."""
    custody = Custody(path=path, fields={k: manifest[k] for k in owned if k in manifest})
    custody.save()
    return custody


def restore(manifest_path: Path, custody: Custody) -> list[str]:
    """Put the pipeline-owned fields back, and report any the agent had changed.

    Returns one line per field that differed. Restoring rather than rejecting is deliberate: the
    common cause is a manifest rewritten from scratch, which is careless rather than dishonest, and
    failing the attempt over it would cost an hour of work for a field the pipeline can simply put
    back. The report is what makes a *deliberate* move visible.
    """
    if not custody.fields:
        return []
    try:
        current = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    except json.JSONDecodeError:
        current = {}
    if not isinstance(current, dict):
        current = {}

    changed: list[str] = []
    for key, value in custody.fields.items():
        if key not in current:
            changed.append(f"{key} was dropped from the manifest; restored")
        elif current[key] != value:
            changed.append(
                f"{key} was changed in the manifest; restored the value materialization recorded"
            )
        current[key] = value

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(current, indent=2))
    return changed
