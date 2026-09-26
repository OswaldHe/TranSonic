# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The iteration-0 floorplan: the straightforward scheme, generated rather than authored.

The 10% regression rule is measured against this, so it has to be reproducible. A baseline
an agent wrote on its first attempt would make every later comparison a comparison against
one agent's first guess, and re-running the loop would move the reference. This module
computes it from the partition graph and the system YAML by fixed rules, so the same inputs
give the same baseline and a reviewer can check it by reading forty lines.

Straightforward, not deliberately weak. It is what an engineer would do first:

- **Tensor parallel within a device, pipeline across devices.** Each layer's modules go on
  one device and are split four ways over its four logical NeuronCores. Collectives for
  those splits stay on the intra-device link, which is the fast one, and the only traffic
  crossing the torus is the activation handoff between consecutive layers — one hop, since
  consecutive stages are neighbours by construction.
- **Each module split along its natural dimension.** Attention over heads, FFN over
  experts, the vocabulary modules over vocab. That is what the dimension is for.
- **Engram in host DRAM.** Its 94.56 GiB does not fit a 24 GiB bank, and a quarter of it
  (23.64 GiB) fits only by leaving nothing for anything else. Host DRAM is where it goes if
  you are not being clever, and whether being clever pays is precisely what the loop is for.

What it does *not* do, so the loop has somewhere to go: nothing overlaps, no module is
tiered, no expert is replicated, and the pipeline is as deep as the device count.
"""

from __future__ import annotations

from typing import Any

from floorplan.parser import Hardware
from floorplan.schema import Address, Floorplan, Placement, Residency, Runtime, Split

#: The natural partition dimension and rejoining collective for each module kind.
#:
#: ``all_to_all`` for the MoE FFN rather than ``allreduce``: splitting 384 experts across
#: units means a token's six chosen experts are on units that do not hold that token, so the
#: traffic is an exchange, not a reduction.
BY_KIND: dict[str, tuple[str, str]] = {
    "attention": ("head", "allreduce"),
    "mlp":       ("expert", "all_to_all"),
    "embed":     ("vocab", "allreduce"),
    "lm_head":   ("vocab", "allgather"),
}

#: Same, for the modules the graph files under ``other``, keyed by an id fragment.
BY_ID: dict[str, tuple[str, str]] = {
    "engram": ("ngram", "allgather"),
    "hc_":    ("hidden", "allreduce"),
}

#: Modules small enough that splitting them costs more in collectives than it saves.
#: ``final_norm`` is 0 GiB of parameters; a four-way split of it buys nothing and adds an
#: allreduce to the critical path.
UNSPLIT_IDS = frozenset({"final_norm", "other_globals"})

#: Where Engram's weights go. See the module docstring.
ENGRAM_TIER = "host_dram"


def _natural_split(module_id: str, entry: dict[str, Any]) -> tuple[str, str] | None:
    """``(dim, collective)`` for this module, or None to leave it unsplit."""
    if module_id in UNSPLIT_IDS:
        return None
    for fragment in sorted(BY_ID, key=len, reverse=True):
        if fragment in module_id:
            return BY_ID[fragment]
    return BY_KIND.get(str(entry.get("kind")))


def build(
    modules: dict[str, dict[str, Any]],
    hardware: Hardware,
    excluded_kinds: frozenset[str] = frozenset({"vision"}),
) -> Floorplan:
    """Generate the baseline floorplan for this graph on this hardware."""
    device_count = len(hardware.devices)
    per_device = len(hardware.devices[0].logical_ncs)
    if device_count < 1 or per_device < 1:
        raise ValueError(f"'{hardware.name}' has no units to place onto")

    deployable = {
        module_id: entry for module_id, entry in modules.items()
        if str(entry.get("kind")) not in excluded_kinds
    }

    # Layers are pipelined in index order, so consecutive stages land on consecutive
    # devices and the handoff between them is one torus hop.
    layers = sorted({
        index
        for entry in deployable.values()
        for index in (entry.get("layer_indices") or [])
    })
    stage_of_layer = {
        layer: min(position * device_count // max(len(layers), 1), device_count - 1)
        for position, layer in enumerate(layers)
    }

    placements: list[Placement] = []
    for module_id in sorted(deployable):
        entry = deployable[module_id]
        layer_indices = entry.get("layer_indices") or []
        if layer_indices:
            device = stage_of_layer[min(layer_indices)]
        elif str(entry.get("kind")) in {"norm", "lm_head"} or module_id == "hc_collapse":
            device = device_count - 1      # the tail of the model
        else:
            device = 0                     # embed, hc_expand, globals — the head

        chosen = _natural_split(module_id, entry)
        if chosen is None:
            units = [Address(device, 0)]
            splits: list[Split] = []
        else:
            dim, collective = chosen
            units = [Address(device, index) for index in range(per_device)]
            splits = [Split(dim=dim, factor=per_device, collective=collective)]

        residency = (
            Residency(tier=ENGRAM_TIER) if "engram" in module_id else Residency(tier="hbm_bank")
        )
        placements.append(Placement(
            module=module_id,
            units=units,
            splits=splits,
            residency=residency,
            stage=device,
        ))

    plan = Floorplan(
        target=hardware.name,
        placements=placements,
        runtime=Runtime(prefill_chunk_tokens=2048, decode_micro_batch=1, pipeline_chunks=True),
        notes=(
            "Generated baseline: tensor parallel 4-way within each device, layers pipelined "
            "across devices in index order, Engram in host DRAM. Reproducible from the "
            "partition graph — see floorplan/baseline.py. Not hand-tuned."
        ),
    )
    plan.validate()
    return plan


HEADER = """\
# The iteration-0 floorplan, generated by floorplan/baseline.py.
#
# This is the only file you may edit, and it is the whole design space: placements, splits,
# residency, schedule order and the runtime knobs. Everything else — the simulator, the cost
# models, the system description — is frozen.
#
# The metrics measured against this file are the sixteen points of phase x context length x
# batch size, named {phase}_{context}_b{batch}_ms. An iteration is rejected if any of them is
# more than 10% worse than the best value that metric has reached, so an improvement in one
# that wrecks another does not count.
#
# Addresses are d<device>.l<logical_nc>: 16 devices, 4 logical NeuronCores each, 64 in all.
# A logical core has a 24 GiB HBM bank, not 96 — the 96 GiB is the device's four banks.
"""
