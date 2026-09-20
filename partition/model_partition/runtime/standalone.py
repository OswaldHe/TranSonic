# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay a module from artifacts alone — no checkpoint required.

A module's dumped weights and inputs are sufficient to reproduce it; the only
other thing needed is the model's *structure*, which comes from the config and
code in the snapshot. So a 765 GB model's individual modules stay replayable on a
machine that could never hold the checkpoint, which is also what makes the dumps
useful to hand to kernel development.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from model_partition.layout import RunLayout
from model_partition.planner.graph import PartitionGraph
from model_partition.runtime.module_runner import (
    TraceBundle,
    apply_dumped_weights,
    expected_output,
    first_tensor,
    replay_record,
)
from model_partition.spec import parse_spec
from model_partition.verify.numerics import Comparison, compare


class StandaloneError(RuntimeError):
    """Raised when a run's artifacts are insufficient to replay a module."""


@dataclass
class LoadedRun:
    """A run's manifest, plan, and trace, opened from disk."""

    layout: RunLayout
    manifest: dict[str, Any]
    graph: PartitionGraph
    bundle: TraceBundle

    @property
    def spec(self):
        payload = self.manifest.get("spec")
        if not payload:
            raise StandaloneError(f"{self.layout.run_file} has no 'spec' section")
        return parse_spec(payload)


def load_run(run_dir: str | Path) -> LoadedRun:
    layout = RunLayout.at(run_dir)
    manifest = layout.read_run()
    if not layout.graph_path.is_file():
        raise StandaloneError(f"No partition graph at {layout.graph_path}")
    return LoadedRun(
        layout=layout,
        manifest=manifest,
        graph=PartitionGraph.load(layout.graph_path),
        bundle=TraceBundle.load(layout.trace_dir),
    )


def build_structure_only(spec, device: str = "cpu") -> Any:
    """Instantiate the model's structure with uninitialized storage.

    Meta-device construction reads only config and code, then ``to_empty`` gives
    real (garbage) storage that dumped weights overwrite.
    """
    from model_partition.ingest import ingest
    from model_partition.loaders import build_loader

    result = ingest(spec)
    loaded = build_loader(result).build_meta()
    model = loaded.model
    model.to_empty(device=device)
    model.eval()
    return model


def replay_module(
    run_dir: str | Path,
    module_id: str,
    sample_id: str | None = None,
    device: str = "cpu",
    model: Any = None,
) -> list[Comparison]:
    """Replay one module from artifacts and compare against its trace."""
    run = load_run(run_dir)
    graph_module = _module_or_raise(run.graph, module_id)
    sample = sample_id or (run.bundle.sample_ids() or [None])[0]
    records = run.bundle.select(module_id=module_id, sample_id=sample)
    if not records:
        raise StandaloneError(f"No trace records for module {module_id!r} sample {sample!r}")

    if model is None:
        from model_partition.verify.modules import poison_parameters

        model = build_structure_only(run.spec, device=device)
        poison_parameters(model)
    apply_dumped_weights(model, run.bundle, module_id, graph_module, device=device)

    results: list[Comparison] = []
    for record in records:
        actual = first_tensor(replay_record(model, record, run.bundle.store, device=device))
        reference = first_tensor(expected_output(record, run.bundle.store, device=device))
        results.append(compare(actual, reference, name=f"{module_id}@{sample}"))
    return results


def _module_or_raise(graph: PartitionGraph, module_id: str):
    try:
        return graph.by_id(module_id)
    except KeyError as exc:
        available = ", ".join(m.id for m in graph.partitioned_modules[:10])
        raise StandaloneError(f"Module {module_id!r} is not in the plan (have: {available}...)") from exc
