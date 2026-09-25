# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The contract an agent-written cost model is written against.

The boundary this file draws is the whole design of the simulator, so it is worth being
explicit about which side of it the agent is on.

**The cost model decides what work exists.** For one shard of one module in one workload it
says: a matmul of these dimensions in this dtype, then an elementwise pass over this many
bytes, then a gather of this many rows, then an allreduce of this tensor. That is a
statement about the *model* — about what DeepSeek V4.1 Flash's attention actually computes
at a 8192-token context — and it is exactly the thing that has to be derived from the
module's source rather than bootstrapped from anywhere.

**The framework decides what that costs.** ``matmul_seconds`` is here, not in the cost
model, and it reads its peak rate from the system YAML and its achieved fraction from the
probe overlay. A cost model cannot make a matmul cheaper by believing in a better GPU; it
can only misstate the matmul's shape, which is arithmetic a reviewer checks against the
module's source in a minute.

That is also why the helpers take shapes rather than seconds. A model that wants to hand
the framework a raw duration has to go around this interface, and the invariant suite looks
for exactly that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from floorplan.parser import Hardware, ceil_div
from floorplan.schema import Address, Residency, Split
from floorplan.sim import collectives
from floorplan.sim.engine import Op, Schedule
from floorplan.sim.memory import MemoryLedger, scope_key_for

#: Bytes per element, by the dtype names the model config and partition graph use.
DTYPE_BYTES: dict[str, int] = {
    "float8_e4m3fn": 1, "float8_e5m2": 1, "float8_e8m0fnu": 1, "fp8": 1, "fp4": 1,
    "bfloat16": 2, "float16": 2, "bf16": 2, "fp16": 2,
    "float32": 4, "fp32": 4, "int32": 4, "int64": 8, "complex64": 8,
}

#: Which entry of ``device.compute`` a dtype's matmuls are rated at. fp4 deliberately maps
#: to the fp8 rate: the datasheet does not quote fp4, and inventing a 2x would flatter every
#: scheme that uses it. See the ``fp4_note`` in the system YAML.
DTYPE_PEAK: dict[str, str] = {
    "fp8": "fp8_flops", "float8_e4m3fn": "fp8_flops", "float8_e5m2": "fp8_flops",
    "fp4": "fp8_flops",
    "bf16": "bf16_flops", "bfloat16": "bf16_flops",
    "fp16": "fp16_flops", "float16": "fp16_flops",
    "fp32": "fp32_flops", "float32": "fp32_flops",
}


@dataclass(frozen=True)
class Workload:
    """One of the four points the loop is measured at."""

    name: str
    phase: str            # "prefill" or "decode"
    context_tokens: int   # tokens already in the KV cache (decode) or being ingested (prefill)
    new_tokens: int       # tokens produced this step: the whole prompt, or 1
    batch: int = 1

    def is_prefill(self) -> bool:
        return self.phase == "prefill"


@dataclass(frozen=True)
class Shard:
    """One shard of one module on one unit — the granularity a cost model is called at."""

    module: str
    kind: str
    unit: Address
    shard_index: int
    shard_count: int
    splits: tuple[Split, ...]
    fraction: float
    #: This shard's share of the module's parameters, after splitting and fraction.
    param_bytes: int
    #: The whole module's activation figure from the partition graph, unsplit.
    module_activation_bytes: int
    residency: Residency
    stage: int
    overlap_collectives: bool
    #: The module's entry in ``plan/partition_graph.yaml``.
    graph_entry: dict[str, Any]
    #: All units the module's shards landed on, in shard order — the collective's group.
    group: tuple[Address, ...]

    def factor(self, dim: str) -> int:
        """How many ways this module was split along ``dim``; 1 if it was not."""
        for split in self.splits:
            if split.dim == dim:
                return split.factor
        return 1

    def collective_for(self, dim: str) -> str:
        for split in self.splits:
            if split.dim == dim:
                return split.collective
        return "none"

    def layer_indices(self) -> list[int]:
        return list(self.graph_entry.get("layer_indices") or [])


class CostModel(Protocol):
    """What an agent-written module cost model must provide.

    ``matches`` is asked for every module in the graph and the first model to claim it owns
    it, so a model can serve one module id, a whole partition group, or an archetype. A
    module no model claims is a hard error: the coverage check will not let a scheme deploy
    a module that nothing can cost.
    """

    #: Human-readable owner name, for traces and error messages.
    name: str

    def matches(self, module_id: str, entry: dict[str, Any]) -> bool:
        ...

    def emit(self, ctx: "Context") -> list[int]:
        """Submit this shard's ops and return the indices downstream modules wait on."""
        ...


_REGISTRY: list[CostModel] = []


def register(model: CostModel) -> CostModel:
    """Register a cost model. Call once per model, at import time.

    Usable as a decorator on a zero-argument class, or directly on an instance.
    """
    instance = model() if isinstance(model, type) else model
    _REGISTRY.append(instance)
    return model


def registry() -> list[CostModel]:
    return list(_REGISTRY)


def clear_registry() -> None:
    """For tests, and for the runner when it reloads models between targets."""
    _REGISTRY.clear()


def resolve(module_id: str, entry: dict[str, Any]) -> CostModel:
    """The model that owns this module, or an error naming what is missing."""
    for model in _REGISTRY:
        try:
            if model.matches(module_id, entry):
                return model
        except Exception as exc:  # a broken matcher must not look like an absent model
            raise CostModelError(
                f"cost model '{getattr(model, 'name', model)}' raised in matches() for "
                f"'{module_id}': {exc}"
            ) from exc
    raise CostModelError(
        f"no cost model claims module '{module_id}' (kind '{entry.get('kind')}'). "
        f"Every module in the floorplan needs one — add a model under sim/modules/ whose "
        f"matches() returns True for it. Registered: "
        f"{', '.join(getattr(m, 'name', str(m)) for m in _REGISTRY) or 'none'}"
    )


class CostModelError(RuntimeError):
    """A cost model is missing, broken, or described work the framework cannot price."""


@dataclass
class Context:
    """Everything a cost model may read, and the only ways it may record work.

    The helpers convert shapes to seconds using the system YAML's rates; the ``op*`` methods
    put the result on the timeline. A cost model that computes its own durations has left
    the interface, and ``invariants.py`` checks for it.
    """

    hardware: Hardware
    schedule: Schedule
    ledger: MemoryLedger
    workload: Workload
    shard: Shard
    #: Op indices this shard must wait for: its producers' completions.
    deps: tuple[int, ...]
    #: The model's ``config.json``, for shapes the graph does not carry.
    config: dict[str, Any]
    #: Scratch space shared across one workload's run, for a model that needs to remember
    #: something between shards (a router's expert assignment, say). Cleared per workload.
    scratch: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Costing helpers: shapes in, seconds out, at rates from the system YAML.
    # ------------------------------------------------------------------
    def matmul_seconds(self, m: int, n: int, k: int, dtype: str = "bf16") -> float:
        """A dense ``(m, k) @ (k, n)`` matmul on one logical NeuronCore.

        Charged at the device peak for ``dtype`` divided by the number of logical cores,
        times the probed efficiency. Small-K matmuls — K below the tensor engine's
        stationary limit of 128, which is the MoE and LoRA case — are charged at
        ``matmul_small_k`` instead, because a GEMM that cannot fill the systolic array
        does not reach the same fraction of peak and pretending otherwise would make
        fine-grained expert parallelism look free.
        """
        if min(m, n, k) <= 0:
            return 0.0
        peak_key = DTYPE_PEAK.get(dtype)
        if peak_key is None:
            raise CostModelError(
                f"no peak rate for dtype '{dtype}'. Known: {', '.join(sorted(DTYPE_PEAK))}"
            )
        device_peak = self.hardware.compute.get(peak_key)
        if device_peak is None:
            raise CostModelError(
                f"the system YAML has no device.compute.{peak_key} "
                f"(needed for a {dtype} matmul)"
            )
        per_unit_peak = device_peak / len(self.hardware.devices[0].logical_ncs)
        stationary = (
            self.hardware.devices[0].logical_ncs[0].engines.get("tensor").stationary_max or 128
        )
        efficiency = self.hardware.require_efficiency(
            "matmul_small_k" if k < stationary
            else ("matmul_fp8" if peak_key == "fp8_flops" else "matmul_bf16")
        )
        return (2.0 * m * n * k) / (per_unit_peak * efficiency)

    def elementwise_seconds(self, nbytes: int, engine: str = "vector") -> float:
        """A pass over ``nbytes`` on the vector or scalar engine, bandwidth-bound in SBUF."""
        if nbytes <= 0:
            return 0.0
        key = "vector_elementwise" if engine == "vector" else "scalar_activation"
        efficiency = self.hardware.require_efficiency(key)
        unit = self.hardware.logical_nc(str(self.shard.unit))
        return nbytes / (unit.hbm_bank_bandwidth_bytes_per_s * efficiency)

    def dma_seconds(self, nbytes: int, tier: str = "hbm_bank", accesses: int = 1) -> float:
        """Move ``nbytes`` from ``tier`` into SBUF, in ``accesses`` transfers.

        Tiers reached only *through* another (NVMe through host DRAM) are charged both
        legs, which is text constraint 6.
        """
        if nbytes <= 0:
            return 0.0
        seconds = 0.0
        chain = list(self.hardware.tiers[tier].via) + [tier]
        for name in chain:
            if name not in self.hardware.tiers:
                raise CostModelError(f"no tier '{name}' on target '{self.hardware.name}'")
            seconds += self.hardware.tiers[name].transfer_seconds(nbytes, accesses)
        strided = self.hardware.require_efficiency(
            "dma_small_strided" if accesses > 1 else "dma_large_contiguous"
        )
        return seconds / strided

    def gather_seconds(self, rows: int, row_bytes: int, tier: str = "hbm_bank") -> float:
        """A gather of ``rows`` scattered rows of ``row_bytes`` each, on GPSIMD.

        The Engram and MoE primitive. Two costs, and the larger wins: fetching the rows
        from ``tier`` as random accesses, and GPSIMD's own throughput assembling them.
        """
        if rows <= 0 or row_bytes <= 0:
            return 0.0
        fetch = self.dma_seconds(rows * row_bytes, tier=tier, accesses=rows)
        efficiency = self.hardware.require_efficiency("gpsimd_gather")
        unit = self.hardware.logical_nc(str(self.shard.unit))
        assemble = (rows * row_bytes) / (unit.hbm_bank_bandwidth_bytes_per_s * efficiency)
        return max(fetch, assemble)

    def collective_cost(
        self, kind: str, bytes_per_participant: int, group: tuple[Address, ...] | None = None,
    ) -> collectives.CollectiveCost:
        """Price a collective over this shard's group (or an explicit one)."""
        units = [str(u) for u in (group if group is not None else self.shard.group)]
        return collectives.cost(self.hardware, kind, bytes_per_participant, units)

    # ------------------------------------------------------------------
    # Recording work.
    # ------------------------------------------------------------------
    def op(
        self,
        name: str,
        engine: str,
        seconds: float,
        deps: tuple[int, ...] | list[int] | None = None,
        holds: frozenset[str] | set[str] = frozenset(),
        bytes_moved: int = 0,
    ) -> int:
        """Put one op on the timeline. Returns its index, to chain the next one onto.

        ``deps`` defaults to this shard's producers, which is what a first op wants;
        pass the previous op's index to chain within a module.
        """
        return self.schedule.submit(Op(
            name=name,
            unit=str(self.shard.unit),
            engine=engine,
            seconds=float(seconds),
            module=self.shard.module,
            deps=tuple(self.deps if deps is None else deps),
            holds=frozenset(holds),
            bytes_moved=int(bytes_moved),
        ))

    def op_matmul(
        self, name: str, m: int, n: int, k: int, dtype: str = "bf16",
        deps: tuple[int, ...] | list[int] | None = None,
    ) -> int:
        """A matmul on the tensor engine, holding SBUF.

        Holding SBUF is what makes text constraint 1 bite: a GPSIMD gather that also holds
        SBUF cannot run concurrently on the same logical core.
        """
        return self.op(
            name, "tensor", self.matmul_seconds(m, n, k, dtype), deps=deps, holds={"sbuf"},
        )

    def op_gather(
        self, name: str, rows: int, row_bytes: int, tier: str = "hbm_bank",
        deps: tuple[int, ...] | list[int] | None = None,
    ) -> int:
        """A gather on GPSIMD, holding SBUF. See ``op_matmul`` on why that matters."""
        return self.op(
            name, "gpsimd", self.gather_seconds(rows, row_bytes, tier), deps=deps,
            holds={"sbuf"}, bytes_moved=rows * row_bytes,
        )

    def op_collective(
        self, name: str, kind: str, bytes_per_participant: int,
        deps: tuple[int, ...] | list[int] | None = None,
        group: tuple[Address, ...] | None = None,
    ) -> int:
        """A collective on the CC cores.

        Overlapped when the placement asked for it, which frees the unit's compute engines
        for the duration but still makes dependents wait — text constraint 4.
        """
        priced = self.collective_cost(kind, bytes_per_participant, group)
        if priced.seconds <= 0:
            # Nothing on the wire (single participant, or a group inside one logical core).
            # Return the last dependency so the caller can chain onto something real.
            deps_tuple = tuple(self.deps if deps is None else deps)
            return deps_tuple[-1] if deps_tuple else self.op(f"{name}:noop", "cc", 0.0, deps=deps)
        self.schedule.record_bytes(priced.link_class, priced.bytes_on_wire)
        return self.schedule.submit(Op(
            name=f"{name}:{kind}",
            unit=str(self.shard.unit),
            engine="cc",
            seconds=priced.seconds,
            module=self.shard.module,
            deps=tuple(self.deps if deps is None else deps),
            bytes_moved=priced.bytes_on_wire,
            participants=tuple(str(u) for u in (group or self.shard.group)),
            overlapped=self.shard.overlap_collectives,
        ))

    # ------------------------------------------------------------------
    # Recording memory.
    # ------------------------------------------------------------------
    def charge_kv(self, nbytes: int, tier: str | None = None) -> None:
        """Charge KV-cache bytes for this shard to the tier its weights live in."""
        chosen = tier or self.shard.residency.tier
        self.ledger.add_kv(
            chosen, scope_key_for(chosen, self.shard.unit), nbytes, self.shard.module,
        )

    def charge_activation(self, nbytes: int, tier: str | None = None) -> None:
        """Charge this shard's peak activation working set."""
        chosen = tier or self.shard.residency.tier
        self.ledger.add_activation(
            chosen, scope_key_for(chosen, self.shard.unit), nbytes, self.shard.module,
        )

    # ------------------------------------------------------------------
    # Conveniences a cost model would otherwise re-derive.
    # ------------------------------------------------------------------
    def dtype_bytes(self, dtype: str) -> int:
        width = DTYPE_BYTES.get(dtype)
        if width is None:
            raise CostModelError(
                f"unknown dtype '{dtype}'. Known: {', '.join(sorted(DTYPE_BYTES))}"
            )
        return width

    def shard_of(self, total: int) -> int:
        """``total`` divided by this shard's split, rounded up.

        Up, not nearest: the largest shard is what a capacity check and a critical path
        both care about, and 384 experts over 5 units is 77 somewhere.
        """
        return ceil_div(total, self.shard.shard_count) if self.shard.shard_count > 1 else total

    def tokens(self) -> int:
        """Tokens this step processes: the prompt in prefill, or the batch in decode."""
        workload = self.workload
        return (
            workload.new_tokens * workload.batch if workload.is_prefill()
            else workload.batch
        )
