# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reconcile plan submodule names against the instantiated model.

The plan is derived from checkpoint tensor names, which need not match the
module paths of the loaded model — a checkpoint may store
``model.language_model.layers.0`` while the instantiated model exposes
``model.layers.0``. Hooks attach to modules, so the plan is rewritten to the
paths that actually exist, resolved by longest unique suffix.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from model_partition.planner.graph import PartitionGraph


@dataclass
class ReconcileReport:
    """What reconciliation changed."""

    resolved: dict[str, str] = field(default_factory=dict)
    unresolved: list[str] = field(default_factory=list)
    ambiguous: dict[str, list[str]] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return any(old != new for old, new in self.resolved.items())

    def summary(self) -> str:
        renamed = sum(1 for old, new in self.resolved.items() if old != new)
        parts = [f"{renamed} submodule name(s) remapped"]
        if self.unresolved:
            parts.append(f"{len(self.unresolved)} unresolved: {', '.join(self.unresolved[:5])}")
        if self.ambiguous:
            parts.append(f"{len(self.ambiguous)} ambiguous")
        return "; ".join(parts)


def build_suffix_index(names: list[str]) -> dict[str, list[str]]:
    """Map every dotted suffix of every name to the names having it."""
    index: dict[str, list[str]] = defaultdict(list)
    for name in names:
        parts = name.split(".")
        for start in range(len(parts)):
            index[".".join(parts[start:])].append(name)
    return index


def resolve_name(wanted: str, known: set[str], suffix_index: dict[str, list[str]]) -> tuple[str | None, list[str]]:
    """Resolve one name. Returns ``(resolved_or_None, ambiguous_candidates)``."""
    if wanted in known:
        return wanted, []
    parts = wanted.split(".")
    ambiguous: list[str] = []
    for length in range(len(parts), 0, -1):
        candidates = suffix_index.get(".".join(parts[-length:]))
        if not candidates:
            continue
        if len(candidates) == 1:
            return candidates[0], []
        ambiguous = candidates
        break
    if not ambiguous:
        return None, []
    # Break a tie by the longest shared prefix, then the shortest name — which
    # maps a checkpoint's "model.language_model.norm" onto the model's "model.norm"
    # rather than onto some "...linear_attn.norm" deeper in the tree.
    best = min(ambiguous, key=lambda name: (-_shared_prefix(parts, name.split(".")),
                                            len(name.split(".")), name))
    return best, ambiguous


def _shared_prefix(left: list[str], right: list[str]) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def reconcile_submodules(graph: PartitionGraph, model: Any) -> ReconcileReport:
    """Rewrite the graph's submodule names in place to match ``model``."""
    names = [name for name, _ in model.named_modules() if name]
    known = set(names)
    suffix_index = build_suffix_index(names)
    report = ReconcileReport()

    for module in graph.partitioned_modules:
        rewritten: list[str] = []
        for wanted in module.submodules:
            resolved, ambiguous = resolve_name(wanted, known, suffix_index)
            if ambiguous:
                report.ambiguous[wanted] = ambiguous[:8]
            if resolved is None:
                report.unresolved.append(wanted)
                rewritten.append(wanted)
                continue
            report.resolved[wanted] = resolved
            rewritten.append(resolved)
        # Preserve order while dropping duplicates introduced by remapping.
        seen: set[str] = set()
        module.submodules = [n for n in rewritten if not (n in seen or seen.add(n))]
    return report
