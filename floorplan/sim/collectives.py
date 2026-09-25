# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""What it costs to rejoin a split.

Every partition dimension except ``batch`` buys its parallelism with communication, so this
module is where a scheme's parallelism strategy is actually priced. It is deliberately a
small amount of well-known arithmetic — ring/tree cost models — rather than anything
clever, because the interesting variable is not the algorithm constant but *which units are
in the group*: the same allreduce over four logical cores inside one device and over four
devices at opposite corners of the torus differ by the link bandwidth and by the hop count,
and that difference is most of what a floorplan is choosing between.

Two properties the rest of the simulator relies on:

- A group entirely inside one device never pays an inter-device cost, and a group spanning
  devices always pays the worst hop distance in the group, not the average. A TP group laid
  out as a torus row costs less than the same group scattered, which is text constraint 5
  made arithmetic.
- A group inside a single *logical* core is free. The two physical NeuronCore-v3 at LNC=2
  share one address space (text constraint 2), so there is no transfer to charge.
"""

from __future__ import annotations

from dataclasses import dataclass

from floorplan.parser import Hardware
from floorplan.schema import Address

#: Bytes actually crossing the wire per participant, as a multiple of the tensor size, and
#: how many dependent steps the algorithm takes. Ring algorithms: N-1 steps of 1/N each.
#:
#: The step count matters as much as the volume at decode, where tensors are small and a
#: collective is latency-bound — a ring allreduce over 16 participants is 30 serialized
#: hops, and at a microsecond each that is 30 us whatever the bandwidth.
_ALGORITHMS: dict[str, tuple[float, int]] = {
    # kind:            (byte factor as multiple of (N-1)/N,  steps as multiple of (N-1))
    "allreduce":       (2.0, 2),   # reduce-scatter then allgather
    "allgather":       (1.0, 1),
    "reduce_scatter":  (1.0, 1),
    "all_to_all":      (1.0, 1),
    "p2p":             (1.0, 1),
}


@dataclass(frozen=True)
class CollectiveCost:
    """The priced collective."""

    seconds: float
    bytes_on_wire: int
    link_class: str
    hops: int
    participants: int

    def describe(self) -> str:
        return (
            f"{self.participants}-way over {self.link_class}"
            f"{f' ({self.hops} hops)' if self.hops else ''}: "
            f"{self.bytes_on_wire / 2 ** 20:.1f} MiB, {self.seconds * 1e3:.3f} ms"
        )


def classify(hardware: Hardware, units: list[str] | tuple[str, ...]) -> tuple[str, int]:
    """``(link_class, hops)`` for a group of unit addresses.

    ``link_class`` is ``none`` for a group that shares a logical core, ``intra_device``
    for one confined to a device, ``inter_device`` otherwise. ``hops`` is the largest
    torus distance in the group, which is the one that sets the pace.
    """
    parsed = [Address.parse(u) if isinstance(u, str) else u for u in units]
    logical = {(a.device, a.logical_nc) for a in parsed}
    if len(logical) <= 1:
        return ("none", 0)
    devices = sorted({a.device for a in parsed})
    if len(devices) == 1:
        return ("intra_device", 0)
    hops = max(
        hardware.hops(a, b)
        for index, a in enumerate(devices)
        for b in devices[index + 1:]
    )
    return ("inter_device", hops)


def cost(
    hardware: Hardware,
    kind: str,
    bytes_per_participant: int,
    units: list[str] | tuple[str, ...],
) -> CollectiveCost:
    """Price one collective over one group.

    ``bytes_per_participant`` is the size of the tensor each participant contributes — for
    an allreduce of a 10 MiB activation over 8 units, it is 10 MiB, not 80.
    """
    if kind == "none":
        return CollectiveCost(0.0, 0, "none", 0, len(set(units)))
    if kind not in _ALGORITHMS:
        raise ValueError(
            f"unknown collective '{kind}'. Known: none, {', '.join(sorted(_ALGORITHMS))}"
        )
    if bytes_per_participant < 0:
        raise ValueError(f"bytes_per_participant must be >= 0, got {bytes_per_participant}")

    link_class, hops = classify(hardware, units)
    participants = len({
        (a.device, a.logical_nc)
        for a in (Address.parse(u) if isinstance(u, str) else u for u in units)
    })
    if link_class == "none" or participants <= 1:
        # Inside one logical core: shared address space, nothing on the wire.
        return CollectiveCost(0.0, 0, "none", 0, participants)

    byte_factor, step_factor = _ALGORITHMS[kind]
    if kind == "p2p":
        # A point-to-point handoff is not a ring. The whole activation travels from sender to
        # receiver, so the `(N-1)/N` share does not apply — for the usual two-participant stage
        # boundary it charged half the payload and underpriced every cross-device pipeline
        # handoff by about 2x.
        bytes_on_wire = int(bytes_per_participant)
        steps = 1
    else:
        share = (participants - 1) / participants
        bytes_on_wire = int(bytes_per_participant * byte_factor * share)
        steps = step_factor * (participants - 1)

    bandwidth, latency_us = hardware.require_link(link_class)
    efficiency = hardware.require_efficiency(
        "collective_all_to_all" if kind == "all_to_all" else "collective_allreduce"
    )

    # Hops multiply latency, not bandwidth: an extra hop is another store-and-forward
    # delay, while the per-chip link bandwidth is what it is. On the intra-device link
    # hops is 0 and the term vanishes.
    by_bandwidth = bytes_on_wire / (bandwidth * efficiency)
    by_latency = steps * max(hops, 1) * latency_us * 1e-6
    return CollectiveCost(
        seconds=by_bandwidth + by_latency,
        bytes_on_wire=bytes_on_wire,
        link_class=link_class,
        hops=hops,
        participants=participants,
    )
