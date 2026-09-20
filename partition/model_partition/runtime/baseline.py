# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The baseline an extracted implementation starts from.

A generated ``inference.py`` delegates here so it is correct the moment it is
written: the module's structure comes from the model's own config and code, and
its weights from the dumps. The agent replaces that delegation with an explicit
implementation when a module needs one — which is the point at which the loop's
numeric repair has something to act on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class BaselineError(RuntimeError):
    """Raised when the baseline cannot assemble a module."""


def build_from_dumps(
    config: dict[str, Any],
    weights: dict[str, Any],
    device: str = "cpu",
    run_dir: str | Path | None = None,
    module_ids: list[str] | None = None,
) -> Any:
    """Build a module's callable from its config and dumped weights.

    Returns a callable with the same signature the traced submodule had, so the
    recorded arguments apply unchanged.
    """
    from model_partition.runtime.standalone import build_structure_only, load_run
    from model_partition.trace import _lookup

    if run_dir is None or not module_ids:
        raise BaselineError("build_from_dumps needs a run directory and module ids")

    run = load_run(run_dir)
    module_id = next((m for m in module_ids if _in_graph(run.graph, m)), None)
    if module_id is None:
        raise BaselineError(f"None of {module_ids} are in the plan at {run_dir}")
    graph_module = run.graph.by_id(module_id)

    model = build_structure_only(run.spec, device=device)
    applied = _apply_named(model, graph_module, weights)
    if not applied:
        raise BaselineError(
            f"No weights matched {module_id!r}; expected keys like "
            f"{next(iter(weights), '(none supplied)')}"
        )

    targets = [_lookup(model, name) for name in graph_module.submodules]
    targets = [t for t in targets if t is not None and callable(t)]
    if not targets:
        raise BaselineError(f"{module_id!r} names no callable submodule")

    if len(targets) == 1:
        return targets[0]

    def run_sequence(*args: Any, **kwargs: Any) -> Any:
        """Run a multi-layer group as its sequence of submodule calls.

        Later layers get the first layer's kwargs, which is what a stack of like
        layers receives in a real forward: position embeddings and masks are
        shared, only the hidden state advances.
        """
        output = targets[0](*args, **kwargs)
        for target in targets[1:]:
            hidden = output[0] if isinstance(output, tuple) else output
            output = target(hidden, **kwargs)
        return output

    return run_sequence


def _in_graph(graph: Any, module_id: str) -> bool:
    try:
        graph.by_id(module_id)
    except KeyError:
        return False
    return True


def _apply_named(model: Any, graph_module: Any, weights: dict[str, Any]) -> int:
    """Copy weights keyed by original parameter name into the model."""
    import torch

    from model_partition.trace import _lookup

    applied = 0
    for submodule_name in graph_module.submodules:
        submodule = _lookup(model, submodule_name)
        if submodule is None or not hasattr(submodule, "named_parameters"):
            continue
        items = list(submodule.named_parameters()) + list(submodule.named_buffers())
        for param_name, tensor in items:
            full = f"{submodule_name}.{param_name}" if param_name else submodule_name
            candidate = weights.get(full)
            if candidate is None:
                continue
            with torch.no_grad():
                tensor.copy_(candidate.reshape(tensor.shape).to(tensor.dtype))
            applied += 1
    return applied
