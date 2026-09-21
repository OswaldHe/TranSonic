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
    submodule: str | None = None,
) -> Any:
    """Build a module's callable from its config and dumped weights.

    Returns a callable with the same signature the traced submodule had, so the
    recorded arguments apply unchanged. ``submodule`` selects one branch of a
    parallel group — which expert of an expert group is being run.
    """
    from model_partition.runtime.module_runner import apply_named_weights
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
    applied = apply_named_weights(model, weights, graph_module)
    if not applied:
        raise BaselineError(
            f"No weights matched {module_id!r}; expected keys like "
            f"{next(iter(weights), '(none supplied)')}"
        )

    # Only a parallel group has branches to choose between. A sequential group runs
    # all of its submodules however the caller names them, or it would silently
    # compute a prefix of the module and compare it against the whole.
    names = ([submodule] if submodule and graph_module.is_parallel
             and submodule in graph_module.submodules
             else _execution_order(run, module_id, graph_module.submodules))
    targets = [_lookup(model, name) for name in names]
    targets = [t for t in targets if t is not None and callable(t)]
    if not targets:
        raise BaselineError(f"{module_id!r} names no callable submodule")

    if len(targets) == 1:
        return targets[0]

    if graph_module.is_parallel:
        raise BaselineError(
            f"{module_id!r} is a parallel group of {len(targets)} submodules "
            f"({graph_module.submodules[0]}, ...); one call runs one branch, so the "
            "caller must name which — pass submodule=<name> to build_module()"
        )

    def run_sequence(*args: Any, **kwargs: Any) -> Any:
        """Run a group as its sequence of submodule calls.

        Each submodule is given the flowing tensor plus whichever of the group's
        keywords it actually accepts. Filtering matters for a heterogeneous group: a
        normalization takes only the hidden state, while the attention after it needs
        the rotary embeddings and mask, and passing either one the other's arguments
        fails.
        """
        output = None
        flowing = args[0] if args else None
        rest = args[1:]
        for index, target in enumerate(targets):
            if index:
                flowing = output[0] if isinstance(output, tuple) else output
                rest = ()
            accepted = _accepted_kwargs(target, kwargs)
            output = target(flowing, *rest, **accepted)
        return output

    return run_sequence


def _accepted_kwargs(target: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """The subset of ``kwargs`` this submodule's forward declares.

    Everything when it takes ``**kwargs``, which is the common case for a decoder
    layer and means the filter costs nothing there. Either way the first parameter
    is dropped: the flowing tensor is passed positionally, and a group's merged
    keywords may well name it too — the trace records however the model called it,
    and most call it ``hidden_states=``.
    """
    import inspect

    forward = getattr(target, "forward", target)
    try:
        parameters = inspect.signature(forward).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins only
        return dict(kwargs)
    names = list(parameters)
    flowing = names[0] if names else None
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return {key: value for key, value in kwargs.items() if key != flowing}
    accepted = set(names[1:])
    return {key: value for key, value in kwargs.items() if key in accepted}


def _execution_order(run: Any, module_id: str, submodules: list[str]) -> list[str]:
    """Order a sequential group's submodules the way the model ran them.

    The plan lists submodules; the trace *observed* them. A norm and the projection
    it feeds must run in the model's order, and no ordering of names can be relied
    on to say which that is — so take it from the recording, falling back to the
    plan's order when a module has no records yet.
    """
    first_seen: dict[str, int] = {}
    for record in run.bundle.records:
        if record.module_id == module_id and record.submodule not in first_seen:
            first_seen[record.submodule] = record.order
    if not all(name in first_seen for name in submodules):
        return list(submodules)
    return sorted(submodules, key=lambda name: first_seen[name])


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
