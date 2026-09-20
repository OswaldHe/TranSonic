# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post-loop pruning: keep deduplicated code, keep tensors for sample layers only.

When layers share an implementation there is no reason to keep 64 copies of the
same weights and feature maps. Retention keeps a representative set of layers
plus every structurally unique module.

Selection is signature-aware, not purely index-based: keeping only layers 1, 5,
mid and last would discard the sole copy of a kernel variant in a hybrid stack
(Qwen3.5 alternates linear and full attention), so each distinct signature is
guaranteed a representative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from model_partition.hardware import format_bytes
from model_partition.planner.graph import PartitionGraph
from model_partition.runtime.module_runner import RECORDS_NAME, TraceBundle
from model_partition.sizing import ModelInventory


@dataclass
class RetentionPolicy:
    """Which layers survive pruning."""

    preferred_layers: tuple[int, ...] = (1, 5)
    keep_first: bool = True
    keep_mid: bool = True
    keep_last: bool = True
    #: Guarantee a representative for every distinct layer signature.
    keep_all_signatures: bool = True
    #: Modules with no layer index (embed, final norm, lm_head) always survive.
    keep_globals: bool = True

    def layers_to_keep(self, inventory: ModelInventory) -> list[int]:
        if not inventory.layers:
            return []
        indices = sorted(layer.index for layer in inventory.layers)
        keep = {i for i in self.preferred_layers if i in set(indices)}
        if self.keep_first:
            keep.add(indices[0])
        if self.keep_mid:
            keep.add(indices[len(indices) // 2])
        if self.keep_last:
            keep.add(indices[-1])
        if self.keep_all_signatures:
            for layer_indices in inventory.signature_groups().values():
                if not keep & set(layer_indices):
                    keep.add(min(layer_indices))
        return sorted(keep)


@dataclass
class RetentionPlan:
    """What pruning would keep and drop."""

    kept_layers: list[int] = field(default_factory=list)
    kept_modules: list[str] = field(default_factory=list)
    dropped_modules: list[str] = field(default_factory=list)
    bytes_kept: int = 0
    bytes_dropped: int = 0

    @property
    def total_bytes(self) -> int:
        return self.bytes_kept + self.bytes_dropped

    def render(self) -> str:
        lines = [
            f"kept layers      : {self.kept_layers}",
            f"kept modules     : {len(self.kept_modules)}",
            f"dropped modules  : {len(self.dropped_modules)}",
            f"bytes kept       : {format_bytes(self.bytes_kept)}",
            f"bytes reclaimed  : {format_bytes(self.bytes_dropped)}",
        ]
        if self.total_bytes:
            share = 100 * self.bytes_dropped / self.total_bytes
            lines.append(f"reduction        : {share:.1f}%")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kept_layers": list(self.kept_layers),
            "kept_modules": list(self.kept_modules),
            "dropped_modules": list(self.dropped_modules),
            "bytes_kept": self.bytes_kept,
            "bytes_dropped": self.bytes_dropped,
        }


def plan_retention(
    graph: PartitionGraph,
    inventory: ModelInventory,
    bundle: TraceBundle,
    policy: RetentionPolicy | None = None,
) -> RetentionPlan:
    """Decide which modules' tensor artifacts to keep."""
    policy = policy or RetentionPolicy()
    kept_layers = policy.layers_to_keep(inventory)
    kept_layer_set = set(kept_layers)

    keep: set[str] = set()
    for module in graph.partitioned_modules:
        if not module.layer_indices:
            if policy.keep_globals:
                keep.add(module.id)
            continue
        if kept_layer_set & set(module.layer_indices):
            keep.add(module.id)

    # No signature group may be emptied: a variant with no kept module keeps its first.
    for module_ids in graph.signature_groups().values():
        if not keep & set(module_ids):
            keep.add(sorted(module_ids)[0])

    plan = RetentionPlan(kept_layers=kept_layers)
    all_modules = [m.id for m in graph.partitioned_modules]
    plan.kept_modules = sorted(m for m in all_modules if m in keep)
    plan.dropped_modules = sorted(m for m in all_modules if m not in keep)

    for entry in bundle.store.entries:
        if entry.module_id is None or entry.module_id in keep:
            plan.bytes_kept += entry.nbytes
        else:
            plan.bytes_dropped += entry.nbytes
    return plan


@dataclass
class RetentionResult:
    """Outcome of applying a retention plan."""

    plan: RetentionPlan
    removed_files: int = 0
    bytes_freed: int = 0
    dry_run: bool = False

    def summary(self) -> str:
        verb = "would remove" if self.dry_run else "removed"
        return (f"{verb} {self.removed_files} file(s), "
                f"{format_bytes(self.bytes_freed)} reclaimed; "
                f"kept layers {self.plan.kept_layers}")


def apply_retention(
    bundle: TraceBundle,
    plan: RetentionPlan,
    dry_run: bool = False,
) -> RetentionResult:
    """Delete artifacts for dropped modules and rewrite the manifest.

    Hardlinked blobs are only freed when the last reference goes, so the reported
    figure counts bytes actually released.
    """
    result = RetentionResult(plan=plan, dry_run=dry_run)
    dropped = set(plan.dropped_modules)

    keep_entries = [e for e in bundle.store.entries if e.module_id is None or e.module_id not in dropped]
    drop_entries = [e for e in bundle.store.entries if e.module_id is not None and e.module_id in dropped]
    surviving_hashes = {e.sha256 for e in keep_entries}

    for entry in drop_entries:
        blob = bundle.store.blob_path(entry)
        sidecar = blob.with_suffix(".json")
        if not blob.is_file():
            continue
        # Only count bytes freed when no surviving entry shares the blob.
        if entry.sha256 not in surviving_hashes:
            result.bytes_freed += entry.nbytes
        result.removed_files += 1
        if not dry_run:
            blob.unlink(missing_ok=True)
            sidecar.unlink(missing_ok=True)

    if not dry_run:
        bundle.store.entries = keep_entries
        bundle.records = [r for r in bundle.records if r.module_id not in dropped]
        bundle.weights = {k: v for k, v in bundle.weights.items() if k not in dropped}
        bundle.metadata = {
            **bundle.metadata,
            "retention": plan.to_dict(),
        }
        bundle.save()
        _prune_empty_dirs(bundle.root)
    return result


def _prune_empty_dirs(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()


def write_retention_report(path: str | Path, result: RetentionResult) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump({
        "dry_run": result.dry_run,
        "removed_files": result.removed_files,
        "bytes_freed": result.bytes_freed,
        **result.plan.to_dict(),
    }, sort_keys=False))
    return target


__all__ = [
    "RECORDS_NAME", "RetentionPlan", "RetentionPolicy", "RetentionResult",
    "apply_retention", "plan_retention", "write_retention_report",
]
