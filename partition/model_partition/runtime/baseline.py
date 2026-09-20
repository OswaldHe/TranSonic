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


#: Instantiating a model's structure is the expensive part and is identical for
#: every module of a run, so it is built once per (run, device). Each module's
#: parameters are poisoned before its own weights are applied, so reuse cannot let
#: one module's weights stand in for another's missing dump.
_STRUCTURE_CACHE: dict[tuple[str, str], Any] = {}


def clear_structure_cache() -> None:
    """Drop cached structures. Call when a run's code or config changed."""
    _STRUCTURE_CACHE.clear()


def _structure(run: Any, device: str) -> Any:
    from model_partition.runtime.standalone import build_structure_only

    key = (str(run.layout.root), device)
    model = _STRUCTURE_CACHE.get(key)
    if model is None:
        model = build_structure_only(run.spec, device=device)
        _STRUCTURE_CACHE[key] = model
    return model


def _poison(model: Any, graph_module: Any) -> None:
    """NaN a module's parameters so a weight the dump omits cannot pass."""
    import torch

    from model_partition.trace import _lookup

    with torch.no_grad():
        for submodule_name in graph_module.submodules:
            submodule = _lookup(model, submodule_name)
            if submodule is None or not hasattr(submodule, "parameters"):
                continue
            for tensor in list(submodule.parameters()) + list(submodule.buffers()):
                if tensor.is_floating_point():
                    tensor.fill_(float("nan"))


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
    from model_partition.runtime.standalone import load_run
    from model_partition.trace import _lookup

    if run_dir is None or not module_ids:
        raise BaselineError("build_from_dumps needs a run directory and module ids")

    run = load_run(run_dir)
    module_id = _module_for_weights(run.graph, module_ids, weights)
    if module_id is None:
        raise BaselineError(
            f"Could not tell which of {module_ids} these weights belong to; "
            f"keys look like {next(iter(weights), '(none supplied)')}"
        )
    graph_module = run.graph.by_id(module_id)

    model = _structure(run, device)
    _poison(model, graph_module)
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


def _module_for_weights(graph: Any, module_ids: list[str],
                        weights: dict[str, Any]) -> str | None:
    """Which module of a group these weights belong to.

    One implementation serves every module sharing a signature, so the group alone
    does not say *which* instance is being run — layers 4-6 and layers 0-2 use the
    same code. The weight names do say: they are original parameter names, so the
    module whose submodules prefix the most of them is the one.
    """
    best: tuple[int, str] | None = None
    for module_id in module_ids:
        try:
            module = graph.by_id(module_id)
        except KeyError:
            continue
        matches = sum(1 for key in weights
                      if any(key.startswith(f"{s}.") or key == s for s in module.submodules))
        if matches and (best is None or matches > best[0]):
            best = (matches, module_id)
    return best[1] if best else None


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
