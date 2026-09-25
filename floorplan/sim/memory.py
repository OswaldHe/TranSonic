# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where the bytes are, and whether they fit.

Capacity is the constraint that decides most of this floorplan before latency gets a vote.
475 GiB of parameters fit comfortably in 1,536 GiB of device memory and do not fit at all
in the 24 GiB bank of one logical NeuronCore, so the question is never "does the model fit"
but "does every shard fit where it was put" — and the answer has to be checked per bank,
per device and per tier, not in aggregate. A scheme that balances latency beautifully while
putting 30 GiB on one bank is not a slow scheme, it is an impossible one.

Weight accounting is deterministic and lives here: ``param_bytes`` comes from the partition
graph and the split factors come from the floorplan, so the framework can compute it
without asking anybody's opinion. KV cache and activation peaks are model-specific and are
contributed by the agent-written cost models through ``add_kv`` and ``add_activation``.
The split follows the same rule as the rest of the platform: what the artifacts state, the
framework computes; what the model's semantics imply, the cost model supplies.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from floorplan.parser import Hardware, format_bytes
from floorplan.schema import Address

#: Which address a tier's capacity is accounted against. ``hbm_bank`` is per logical core,
#: ``device_hbm`` per device, the rest pooled instance-wide.
TIER_SCOPE = {
    "hbm_bank": "logical_nc",
    "device_hbm": "device",
    "peer_hbm": "instance",
    "host_dram": "instance",
    "nvme": "instance",
}


@dataclass
class Entry:
    """One contribution to one tier at one scope."""

    tier: str
    scope_key: str
    nbytes: int
    kind: str  # weights | kv | activation
    module: str
    #: Which shard of ``module`` this belongs to. Part of the peak-coalescing key, so two
    #: shards sharing a bank are charged separately while repeated chunk updates for the *same*
    #: shard coalesce. Without it, a module split across ``d0.l0.p0`` and ``d0.l0.p1`` — both in
    #: the same HBM-bank scope, each holding a different piece of the KV cache — had only the
    #: larger piece charged, so a plan could overflow the shared bank while the gate said it fit.
    shard: int = 0


@dataclass
class MemoryLedger:
    """Accumulates residency, then reports whether it fits.

    A ledger is per workload, because KV and activation peaks are workload-dependent while
    weights are not — an 8192-token context is where a plan that fits at 128 tokens stops
    fitting, and that is exactly the failure worth catching.
    """

    entries: list[Entry] = field(default_factory=list)

    def add_weights(self, tier: str, scope_key: str, nbytes: int, module: str) -> None:
        self._add(tier, scope_key, nbytes, "weights", module)

    def add_kv(self, tier: str, scope_key: str, nbytes: int, module: str,
               shard: int = 0) -> None:
        self._add(tier, scope_key, nbytes, "kv", module, shard)

    def add_activation(self, tier: str, scope_key: str, nbytes: int, module: str,
                       shard: int = 0) -> None:
        self._add(tier, scope_key, nbytes, "activation", module, shard)

    def _add(self, tier: str, scope_key: str, nbytes: int, kind: str, module: str,
             shard: int = 0) -> None:
        if nbytes < 0:
            raise ValueError(f"{module}: negative {kind} bytes ({nbytes})")
        if not nbytes:
            return
        if kind == "weights":
            self.entries.append(Entry(tier, scope_key, int(nbytes), kind, module, shard))
            return
        # KV and activations are a *peak*, not an accumulation. A chunked prefill calls a cost
        # model once per chunk with a growing context, so summing would report four KV caches
        # for a four-chunk prompt. Keeping the largest per
        # (tier, scope, kind, module, shard) lets a cost model charge naively every chunk and
        # still have the ledger mean what it says — while two distinct shards on one bank stay
        # separate, which they must, since they hold different pieces.
        for entry in self.entries:
            if (entry.tier, entry.scope_key, entry.kind, entry.module, entry.shard) == (
                tier, scope_key, kind, module, shard
            ):
                entry.nbytes = max(entry.nbytes, int(nbytes))
                return
        self.entries.append(Entry(tier, scope_key, int(nbytes), kind, module, shard))

    # ------------------------------------------------------------------
    def totals(self) -> dict[tuple[str, str], int]:
        """Bytes per (tier, scope_key), with activations counted as a concurrent peak.

        Weights and KV are persistent: every module's share occupies the scope for the whole
        run, so they sum. Activations are a *working set* — modules on a pipeline stage execute
        one after another and reuse the same scratch — so summing them across the dozens of
        layers sharing a bank reported the total of every workspace ever used rather than the
        largest live at once. That rejected feasible floorplans and pushed the search toward
        needless distribution.

        The rule here is the conservative approximation available without lifetime analysis:
        activations contribute the single largest per-scope entry, weights and KV contribute
        their sum. A schedule-aware peak would be better and needs the timeline, which this
        ledger does not see.
        """
        persistent: dict[tuple[str, str], int] = defaultdict(int)
        activation_peak: dict[tuple[str, str], int] = defaultdict(int)
        for entry in self.entries:
            key = (entry.tier, entry.scope_key)
            if entry.kind == "activation":
                activation_peak[key] = max(activation_peak[key], entry.nbytes)
            else:
                persistent[key] += entry.nbytes
        out: dict[tuple[str, str], int] = defaultdict(int)
        for key, nbytes in persistent.items():
            out[key] += nbytes
        for key, nbytes in activation_peak.items():
            out[key] += nbytes
        return dict(out)

    def by_kind(self) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for entry in self.entries:
            out[entry.kind] += entry.nbytes
        return dict(out)

    def high_water(self, hardware: Hardware) -> dict[str, tuple[str, int, int]]:
        """Per tier: ``(worst scope_key, bytes there, capacity)``.

        A tier's pressure is its worst scope, not its sum. ``device_hbm`` additionally
        inherits everything its banks hold, since a bank *is* device memory — accounting
        them separately would let a plan put 24 GiB on each of four banks and report the
        device as empty.
        """
        totals = self.totals()
        rolled = dict(totals)
        for (tier, scope_key), nbytes in totals.items():
            if tier != "hbm_bank":
                continue
            device = f"d{Address.parse(scope_key).device}"
            rolled[("device_hbm", device)] = rolled.get(("device_hbm", device), 0) + nbytes

        out: dict[str, tuple[str, int, int]] = {}
        for tier_name, tier in hardware.tiers.items():
            if tier.capacity_bytes is None:
                continue
            scoped = {k: v for (t, k), v in rolled.items() if t == tier_name}
            if not scoped:
                continue
            worst = max(scoped, key=lambda k: scoped[k])
            capacity = tier.capacity_bytes
            if TIER_SCOPE.get(tier_name) == "instance" and tier_name == "peer_hbm":
                # peer_hbm is the whole instance's device memory pooled; its capacity is
                # already the 1,536 GiB total.
                pass
            out[tier_name] = (worst, scoped[worst], capacity)
        return out

    def violations(self, hardware: Hardware) -> list[str]:
        """Every over-capacity scope, as readable messages. Empty means the plan fits."""
        problems: list[str] = []
        for tier_name, (scope_key, nbytes, capacity) in sorted(self.high_water(hardware).items()):
            if nbytes > capacity:
                culprits = sorted(
                    {e.module for e in self.entries
                     if e.tier == tier_name and e.scope_key == scope_key},
                )[:4]
                detail = ", ".join(culprits)
                if len(culprits) == 4:
                    detail += ", ..."
                problems.append(
                    f"{tier_name} at {scope_key}: {format_bytes(nbytes)} exceeds "
                    f"{format_bytes(capacity)} by {format_bytes(nbytes - capacity)} "
                    f"({detail})"
                )
        return problems

    def report(self, hardware: Hardware) -> str:
        """The memory table the ranking report reads."""
        lines = ["tier           worst scope        used         capacity     fill"]
        for tier_name, (scope_key, nbytes, capacity) in sorted(self.high_water(hardware).items()):
            fill = (nbytes / capacity * 100) if capacity else 0.0
            lines.append(
                f"{tier_name:<14} {scope_key:<18} {format_bytes(nbytes):>11}  "
                f"{format_bytes(capacity):>11}  {fill:5.1f}%"
            )
        kinds = self.by_kind()
        lines.append("")
        lines.append(
            "totals: " + ", ".join(
                f"{kind} {format_bytes(nbytes)}" for kind, nbytes in sorted(kinds.items())
            )
        )
        return "\n".join(lines)


def scope_key_for(tier: str, unit: Address) -> str:
    """Which scope a placement on ``unit`` charges its bytes to in ``tier``."""
    scope = TIER_SCOPE.get(tier, "instance")
    if scope == "logical_nc":
        return str(unit.logical())
    if scope == "device":
        return f"d{unit.device}"
    return "instance"
