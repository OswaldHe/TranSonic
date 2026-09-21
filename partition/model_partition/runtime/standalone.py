# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay a module from artifacts alone — no checkpoint, no model.

A module directory holds everything needed to run one module: ``source.py`` is the
implementation, ``inference.py`` launches it, and the trace holds the weights and the
input it was recorded with. So a 765 GB model's individual modules stay replayable on
a machine that could never hold the checkpoint, which is what makes the dumps useful
to hand to kernel development.

What runs here is the extracted implementation, the same code the loop verifies and
the same code someone optimizing a kernel would edit — not the original model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from model_partition.layout import RunLayout
from model_partition.planner.graph import PartitionGraph
from model_partition.runtime.module_runner import TraceBundle, expected_output
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


def find_impl(run: LoadedRun, module_id: str) -> Path:
    """The module directory implementing ``module_id``, or raise saying why not."""
    from model_partition.runtime.module_impl import find_impl_dirs

    directories = find_impl_dirs(run.layout.modules_dir)
    directory = directories.get(module_id)
    if directory is None:
        raise StandaloneError(
            f"No extracted implementation for {module_id!r} under "
            f"{run.layout.modules_dir}; run extraction first"
        )
    return directory


def replay_module(
    run_dir: str | Path,
    module_id: str,
    sample_id: str | None = None,
    device: str = "cpu",
    tolerance: Any = None,
    impl_dir: str | Path | None = None,
) -> list[Comparison]:
    """Run a module's extracted implementation and compare against the dump.

    This exercises the code the loop owns and hands over — ``inference.py``
    launching ``source.py`` — not the original model, so a passing replay says the
    artifact reproduces the module on its own. The comparison stays here, out of the
    agent's reach, and covers every tensor the module returns rather than the first.
    """
    from model_partition.runtime.module_impl import load_impl, run_impl
    from model_partition.runtime.module_runner import decode_group_call, load_named_weights

    run = load_run(run_dir)
    graph_module = _module_or_raise(run.graph, module_id)
    sample = sample_id or (run.bundle.sample_ids() or [None])[0]
    records = _records_or_raise(run, module_id, sample)

    impl = load_impl(impl_dir if impl_dir is not None else find_impl(run, module_id))
    weights = load_named_weights(run.bundle, module_id, device=device)
    if not weights:
        raise StandaloneError(
            f"No dumped weights for {module_id!r}; the artifacts may have been "
            "pruned by retention"
        )
    config = run.bundle.config or run.manifest.get("config") or {}

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
        results.extend(compare_outputs(actual, reference, f"{module_id}@{sample}",
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
