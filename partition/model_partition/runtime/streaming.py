# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a whole model assembled exclusively from dumped per-module weights.

The emulation contract: build the model structure, fill every parameter from the
partition artifacts (never from the original checkpoint), then generate. If the
dumps are incomplete the fill reports exactly which parameters were left
unsatisfied instead of quietly falling back.

Module boundaries are captured on their own forward, then released before
generation: leaving the hooks installed would hold one output per module for every
step, and at long context the logits alone are gigabytes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from model_partition.planner.graph import PartitionGraph
from model_partition.runtime.module_runner import TraceBundle, load_named_weights
from model_partition.trace import _lookup


class StreamingError(RuntimeError):
    """Raised when a model cannot be assembled from its dumps."""


@dataclass
class FillReport:
    """Outcome of filling a model's parameters from dumps."""

    applied: int = 0
    missing: list[str] = field(default_factory=list)
    unclaimed: list[str] = field(default_factory=list)
    by_module: dict[str, int] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return not self.missing

    def summary(self) -> str:
        text = f"{self.applied} parameter(s) filled from dumps"
        if self.missing:
            preview = ", ".join(self.missing[:5])
            text += f"; {len(self.missing)} missing ({preview}{'...' if len(self.missing) > 5 else ''})"
        return text


def fill_from_dumps(
    model: Any,
    bundle: TraceBundle,
    graph: PartitionGraph,
    device: str = "cpu",
    strict: bool = True,
) -> FillReport:
    """Overwrite every partitioned module's parameters with its recorded values."""
    import torch

    report = FillReport()
    claimed: set[str] = set()

    for module in graph.partitioned_modules:
        # Do not skip a module with no dumped weights: if it owns parameters,
        # every one of them is missing, and that must be reported rather than
        # leaving them silently unclaimed.
        dumped = load_named_weights(bundle, module.id, device=device)
        count = 0
        for submodule_name in module.submodules:
            submodule = _lookup(model, submodule_name)
            if submodule is None or not hasattr(submodule, "named_parameters"):
                continue
            items = list(submodule.named_parameters()) + list(submodule.named_buffers())
            for param_name, tensor in items:
                full = f"{submodule_name}.{param_name}" if param_name else submodule_name
                claimed.add(full)
                candidate = dumped.get(full)
                if candidate is None:
                    report.missing.append(full)
                    continue
                with torch.no_grad():
                    tensor.copy_(candidate.reshape(tensor.shape).to(tensor.dtype))
                count += 1
        report.by_module[module.id] = count
        report.applied += count

    for name, _ in list(model.named_parameters()) + list(model.named_buffers()):
        if name not in claimed:
            report.unclaimed.append(name)

    if strict and not report.complete:
        raise StreamingError(
            f"Cannot assemble the model from dumps: {len(report.missing)} parameter(s) "
            f"have no dumped value ({', '.join(report.missing[:8])})"
        )
    return report


def place_across_devices(model: Any, max_memory: dict) -> tuple[Any, str]:
    """Spread an already-materialized model across GPU and host for the forward.

    Assembling from dumps needs writable parameters, which rules out loading with
    a device map. Dispatching afterwards gets the GPU back: the weights already
    exist, so placement only decides where each layer runs. Returns
    ``(model, input_device)``; on any failure the model is left untouched on its
    current device.
    """
    try:
        from accelerate import dispatch_model, infer_auto_device_map
    except ImportError:
        return model, _first_param_device(model)
    try:
        device_map = infer_auto_device_map(model, max_memory=max_memory)
        placed = dispatch_model(model, device_map=device_map)
    except Exception:
        return model, _first_param_device(model)
    return placed, _first_param_device(placed)


def _first_param_device(model: Any) -> str:
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return "cpu"


def capture_boundaries(model: Any, graph: PartitionGraph) -> tuple[list, dict[str, Any]]:
    """Hook every partitioned module to record its output on the *first* forward.

    Generation re-runs the forward on a growing sequence, so only the first pass
    is comparable with a trace taken at the prompt's length.
    """
    sink: dict[str, Any] = {}
    handles = []
    owners: dict[str, list[str]] = {}
    for module in graph.partitioned_modules:
        if module.submodules:
            owners.setdefault(module.submodules[-1], []).append(module.id)

    def make_hook(module_ids: list[str]):
        def hook(_module, _args, output):
            for module_id in module_ids:
                sink.setdefault(module_id, output)
        return hook

    for submodule_name, module_ids in owners.items():
        submodule = _lookup(model, submodule_name)
        if submodule is not None and hasattr(submodule, "register_forward_hook"):
            handles.append(submodule.register_forward_hook(make_hook(module_ids)))
    return handles, sink


def greedy_step(logits: Any, temperature: float = 0.0, seed: int | None = None) -> int:
    """Pick the next token id: argmax, or seeded sampling when temperature > 0."""
    import torch

    last = logits[0, -1] if logits.dim() == 3 else logits[-1]
    if temperature <= 0:
        return int(last.argmax())
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)
    probs = torch.softmax(last.float().cpu() / temperature, dim=-1)
    return int(torch.multinomial(probs, num_samples=1, generator=generator))


#: Names a causal LM uses for "apply the head to only the last N positions".
#: ``logits_to_keep`` in transformers 5, ``num_logits_to_keep`` before it.
LAST_LOGITS_ARGUMENTS = ("logits_to_keep", "num_logits_to_keep")


def _last_logits_argument(model: Any) -> str | None:
    """The keyword this model takes to skip computing logits it will not be asked for."""
    import inspect

    try:
        parameters = inspect.signature(model.forward).parameters
    except (AttributeError, TypeError, ValueError):
        return None
    return next((name for name in LAST_LOGITS_ARGUMENTS if name in parameters), None)


def generate(
    model: Any,
    input_ids: Any,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    seed: int | None = None,
    eos_token_id: int | None = None,
    logits_of: Any = None,
) -> tuple[list[int], Any]:
    """Greedy/sampled generation by re-running the forward each step.

    No KV cache: every step re-runs the full prefill on the extended sequence.
    Quadratic, but architecture-agnostic — it needs nothing from the model beyond
    ``model(input_ids) -> logits``, which is what keeps emulation independent of
    each architecture's cache implementation.

    Sampling reads only the final row, so the model is asked for only that row. A
    full ``[1, seq, vocab]`` tensor is 8 GB at 16k tokens with a large vocabulary and
    grows by a row every step, which the allocator cannot serve from the previous
    step's cached block — measured on a 1.6 GiB model, reserved memory climbed 10 →
    25 → 44 GiB and then started failing allocations. Keeping one row holds it flat
    at 3 GiB.
    """
    import torch

    from model_partition.trace import forward_no_cache

    extract = logits_of or (lambda out: out.logits if hasattr(out, "logits") else out)
    keep_last = _last_logits_argument(model)
    sequence = input_ids
    produced: list[int] = []
    last = None
    with torch.no_grad():
        for _ in range(max_new_tokens):
            output = (model(sequence, use_cache=False, **{keep_last: 1}) if keep_last
                      else forward_no_cache(model, sequence))
            logits = extract(output)
            last = logits[:, -1:].clone() if logits.dim() == 3 else logits[-1:].clone()
            del output, logits
            token = greedy_step(last, temperature=temperature, seed=seed)
            produced.append(token)
            if eos_token_id is not None and token == eos_token_id:
                break
            sequence = torch.cat(
                [sequence, torch.tensor([[token]], device=sequence.device, dtype=sequence.dtype)],
                dim=-1,
            )
    return produced, last
