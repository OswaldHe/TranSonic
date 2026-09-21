# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read a module's weights straight out of the checkpoint shards.

The alternative to dumping them. ``cache_weights: false`` keeps only the index of
which parameters each module owns and resolves their values here, so verification
still runs against real weights without writing a second copy of the checkpoint —
which is the difference between a 510 GB model fitting a 1 TB volume and not.

The dumps remain the default, and the only option for replaying a module on a
machine the checkpoint never reaches.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class WeightSourceError(RuntimeError):
    """Raised when a requested tensor is not in the checkpoint."""


@dataclass
class CheckpointWeights:
    """Resolves original parameter names to tensors in local safetensors shards."""

    root: Path
    shard_of: dict[str, str] = field(default_factory=dict)
    #: Names that needed a suffix match, for the run's notes.
    remapped: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_ingest(cls, result: Any) -> CheckpointWeights:
        return cls(
            root=Path(result.root),
            shard_of={entry.name: entry.shard for entry in result.index.entries},
        )

    def resolve(self, name: str) -> str | None:
        """Checkpoint tensor name for a model parameter name.

        Vendor code often names modules differently from the checkpoint, so an
        exact miss falls back to the longest suffix that identifies exactly one
        tensor — the same rule plan reconciliation uses.
        """
        if name in self.shard_of:
            return name
        parts = name.split(".")
        for start in range(1, len(parts)):
            suffix = "." + ".".join(parts[start:])
            matches = [n for n in self.shard_of if n.endswith(suffix)]
            if len(matches) == 1:
                self.remapped[name] = matches[0]
                return matches[0]
            if not matches:
                break
        return None

    def load(self, names: list[str], device: str = "cpu") -> dict[str, Any]:
        """Load the named parameters, keyed by the name that was asked for.

        Shards are opened once each and read lazily, so a module's weights cost one
        pass over the shards holding them rather than over the whole checkpoint.
        """
        from safetensors import safe_open

        wanted: dict[str, list[tuple[str, str]]] = defaultdict(list)
        missing: list[str] = []
        for name in names:
            resolved = self.resolve(name)
            if resolved is None:
                missing.append(name)
                continue
            wanted[self.shard_of[resolved]].append((name, resolved))
        if missing:
            raise WeightSourceError(
                f"{len(missing)} parameter(s) are not in the checkpoint: "
                f"{', '.join(missing[:8])}"
            )

        loaded: dict[str, Any] = {}
        for shard, pairs in wanted.items():
            path = self.root / shard
            if not path.is_file():
                raise WeightSourceError(
                    f"Shard {shard} is not present at {path}; fetch the weights first"
                )
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                for name, resolved in pairs:
                    loaded[name] = handle.get_tensor(resolved).to(device)
        return loaded
