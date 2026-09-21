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
    replay_record,
)
from model_partition.spec import parse_spec
from model_partition.verify.numerics import Comparison, compare_outputs


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
    """Instantiate the model's structure without reading the checkpoint.

    Prefers normal from-config initialization so non-persistent buffers (rotary
    ``inv_freq``, masks) are correct; falls back to meta + ``to_empty`` when the
    model is too large to initialize, in which case such buffers hold garbage and
    only modules that receive everything through recorded arguments are reliable.
    """
    from model_partition.ingest import ingest
    from model_partition.loaders import build_loader

    # No tensor inventory: structure comes from config and code only.
    result = ingest(spec, with_index=False)
    loader = build_loader(result)
    try:
        return loader.build_config_only(device=device).model
    except Exception:
        model = loader.build_meta().model
        model.to_empty(device=device)
        model.eval()
        return model


def load_module(
    run_dir: str | Path,
    module_id: str,
    device: str = "cpu",
) -> tuple[LoadedRun, Any, Any]:
    """Build a module ready to run: structure from code, weights from the dumps.

    Returns ``(run, model, graph_module)``. The checkpoint is never read.
    """
    from model_partition.verify.modules import plan_owned_parameters, poison_parameters

    run = load_run(run_dir)
    graph_module = _module_or_raise(run.graph, module_id)
    model = build_structure_only(run.spec, device=device)
    poison_parameters(model, only=plan_owned_parameters(model, run.graph))
    applied = apply_dumped_weights(model, run.bundle, module_id, graph_module, device=device)
    if not applied:
        raise StandaloneError(
            f"No dumped weights were applied for {module_id!r}; the artifacts may "
            "have been pruned by retention"
        )
    return run, model, graph_module


def replay_module(
    run_dir: str | Path,
    module_id: str,
    sample_id: str | None = None,
    device: str = "cpu",
    tolerance: Any = None,
) -> list[Comparison]:
    """Run a module's inference and compare it against the dumped output."""
    run, model, _ = load_module(run_dir, module_id, device=device)
    sample = sample_id or (run.bundle.sample_ids() or [None])[0]
    records = _records_or_raise(run, module_id, sample)

    results: list[Comparison] = []
    for record in records:
        actual = replay_record(model, record, run.bundle.store, device=device)
        reference = expected_output(record, run.bundle.store, device=device)
        label = f"{module_id}@{sample}" if len(records) == 1 else f"{record.submodule}@{sample}"
        results.extend(compare_outputs(actual, reference, label, tolerance))
    return results


def verify_impl(
    run_dir: str | Path,
    impl_dir: str | Path,
    module_id: str,
    sample_id: str | None = None,
    device: str = "cpu",
    tolerance: Any = None,
) -> list[Comparison]:
    """Run a module's *extracted implementation* and compare against the dump.

    This is the check that matters: it exercises the code the loop owns, not the
    original model. The comparison itself stays here, out of the agent's reach —
    and covers every tensor the module returns, not only the first.
    """
    from model_partition.runtime.module_impl import load_impl, run_impl
    from model_partition.runtime.module_runner import decode_group_call, load_named_weights

    run = load_run(run_dir)
    graph_module = _module_or_raise(run.graph, module_id)
    sample = sample_id or (run.bundle.sample_ids() or [None])[0]
    records = _records_or_raise(run, module_id, sample)

    impl = load_impl(impl_dir)
    weights = load_named_weights(run.bundle, module_id, device=device)
    config = run.bundle.config or run.manifest.get("config") or (run.spec.overrides or {})

    # A parallel group's recorded calls are independent — one expert each — so
    # every one is checked. A sequential group is one computation from the first
    # call's input to the last call's output.
    pairs = ([(r, r) for r in records] if graph_module.is_parallel
             else [(records[0], records[-1])])
    results: list[Comparison] = []
    for source, reference_record in pairs:
        group = records if len(pairs) == 1 else [source]
        args, kwargs = decode_group_call(group, run.bundle.store, device)
        actual = run_impl(impl, config, weights, args, kwargs, device=device,
                          submodule=source.submodule if graph_module.is_parallel else None)
        reference = expected_output(reference_record, run.bundle.store, device=device)
        results.extend(compare_outputs(actual, reference, f"impl:{module_id}@{sample}",
                                       tolerance))
    return results


def _records_or_raise(run: LoadedRun, module_id: str, sample: str | None) -> list[Any]:
    records = run.bundle.select(module_id=module_id, sample_id=sample)
    if not records:
        raise StandaloneError(f"No trace records for module {module_id!r} sample {sample!r}")
    return records


def _module_or_raise(graph: PartitionGraph, module_id: str):
    try:
        return graph.by_id(module_id)
    except KeyError as exc:
        available = ", ".join(m.id for m in graph.partitioned_modules[:10])
        raise StandaloneError(f"Module {module_id!r} is not in the plan (have: {available}...)") from exc
