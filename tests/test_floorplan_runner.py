# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The runner: the DAG walk, prefill chunking, and the cost-model interface.

Uses a synthetic cost model rather than an agent-written one, so what is under test is the
machinery an agent writes against. The behaviours checked here are the ones `invariants.py`
will later demand of real cost models — split scaling, context sensitivity, chunked prefill —
and the framework has to make them expressible before a cost model can be faulted for missing
them.
"""

from __future__ import annotations

from typing import Any

import pytest

from floorplan.parser import Hardware, load_system
from floorplan.schema import Floorplan
from floorplan.sim import api
from floorplan.sim.api import Workload
from floorplan.sim.runner import (
    HEAVIEST,
    WORKLOADS,
    charge_weights,
    deployable_modules,
    simulate_workload,
    topological_order,
    workload,
)

pytestmark = pytest.mark.floorplan


@pytest.fixture
def hardware() -> Hardware:
    system = load_system("trn2-16device", apply_probes=False)
    system["efficiency"] = {
        "matmul_fp8": 0.19, "matmul_bf16": 0.376, "matmul_small_k": 0.047,
        "vector_elementwise": 2.2, "scalar_activation": 4.8, "gpsimd_gather": 0.049,
        "dma_large_contiguous": 0.277, "dma_small_strided": 0.0031,
        "collective_allreduce": 1.0, "collective_all_to_all": 1.0,
    }
    system["links"]["intra_device"].update(
        {"bandwidth_bytes_per_s": 8.04e11, "latency_us": 1.0},
    )
    for tier in system["tiers"]:
        if tier["name"] == "host_dram":
            tier.update({"bandwidth_bytes_per_s": 1.29e10, "latency_us": 2.0})
        if tier["name"] == "nvme":
            tier.update({
                "bandwidth_bytes_per_s": 9.8e9, "latency_us": 30.0,
                "random_read_iops": 132_000,
            })
    return Hardware.from_system(system)


@pytest.fixture
def graph() -> dict[str, dict[str, Any]]:
    """embed -> attention -> ffn -> lm_head, plus an off-path vision module."""
    return {
        "embed": {
            "kind": "embed", "inputs": ["tokens"], "outputs": ["h.0"],
            "param_bytes": 1 << 30, "activation_bytes": 1 << 24,
        },
        "layers.0.attention": {
            "kind": "attention", "inputs": ["h.0"], "outputs": ["h.0.attn"],
            "param_bytes": 1 << 27, "activation_bytes": 1 << 24, "layer_indices": [0],
        },
        "layers.0.ffn": {
            "kind": "mlp", "inputs": ["h.0.attn"], "outputs": ["h.1"],
            "param_bytes": 1 << 32, "activation_bytes": 1 << 24, "layer_indices": [0],
        },
        "lm_head": {
            "kind": "lm_head", "inputs": ["h.1"], "outputs": ["logits"],
            "param_bytes": 1 << 30, "activation_bytes": 1 << 24,
        },
        "vision": {
            "kind": "vision", "inputs": ["pixels"], "outputs": ["v"],
            "param_bytes": 1 << 28, "activation_bytes": 0,
        },
    }


@pytest.fixture
def config() -> dict[str, Any]:
    return {"dim": 5120, "n_heads": 64, "moe_inter_dim": 2304, "n_activated_experts": 6}


class Synthetic:
    """A cost model in the shape a real one takes: shapes in, ops out, nothing invented."""

    name = "synthetic"

    def matches(self, module_id: str, entry: dict[str, Any]) -> bool:
        return True

    def emit(self, ctx: api.Context) -> list[int]:
        dim = int(ctx.config["dim"])
        tokens = ctx.tokens()
        # Per-shard work, so a wider split costs less each — the scaling invariant.
        width = ctx.shard_of(dim)
        last = ctx.op_matmul("proj", tokens, width, dim)
        if ctx.shard.kind == "attention":
            # Quadratic in context during prefill, linear during decode: the context
            # invariant, and the reason both matter to a floorplan.
            context = ctx.workload.context_tokens
            scores = tokens * context if ctx.workload.is_prefill() else context
            last = ctx.op("scores", "vector", ctx.elementwise_seconds(scores * 2), deps=[last])
            # `kv_tokens()` is context x this shard's share of the batch, which is what makes
            # batch a capacity axis and not only a throughput one.
            ctx.charge_kv(ctx.kv_tokens() * dim * 2 * 2)
        if ctx.shard.kind == "mlp":
            last = ctx.op_gather(
                "experts", rows=6, row_bytes=width * 2,
                tier=ctx.shard.residency.tier, deps=[last],
            )
        collective = ctx.shard.collective_for(
            next((s.dim for s in ctx.shard.splits), "hidden"),
        )
        if collective != "none":
            last = ctx.op_collective("rejoin", collective, tokens * width * 2, deps=[last])
        ctx.charge_activation(ctx.shard.module_activation_bytes // max(ctx.shard.shard_count, 1))
        return [last]


@pytest.fixture(autouse=True)
def registered():
    api.clear_registry()
    api.register(Synthetic())
    yield
    api.clear_registry()


def _plan(**overrides) -> Floorplan:
    data = {
        "version": 1,
        "target": "trn2-16device",
        "placements": [
            {"module": "embed", "units": ["d0.l0"], "stage": 0},
            {
                "module": "layers.0.attention",
                "units": ["d0.l0", "d0.l1", "d0.l2", "d0.l3"],
                "splits": [{"dim": "head", "factor": 4, "collective": "allreduce"}],
                "stage": 1,
            },
            {
                "module": "layers.0.ffn",
                "units": ["d1.l0", "d1.l1", "d1.l2", "d1.l3"],
                "splits": [{"dim": "expert", "factor": 4, "collective": "all_to_all"}],
                "stage": 2,
            },
            {"module": "lm_head", "units": ["d15.l0"], "stage": 3},
        ],
    }
    data.update(overrides)
    return Floorplan.from_dict(data)


# ---------------------------------------------------------------------------------------
def test_all_four_workloads_produce_a_positive_latency(hardware, graph, config):
    plan = _plan()
    order = topological_order(graph)
    for point in WORKLOADS:
        result = simulate_workload(plan, hardware, graph, config, point, order)
        assert result.seconds > 0, point.name
        assert result.trace.scheduled


def test_simulation_is_deterministic(hardware, graph, config):
    plan = _plan()
    order = topological_order(graph)
    first = simulate_workload(plan, hardware, graph, config, workload("prefill_8192_b1"), order)
    second = simulate_workload(plan, hardware, graph, config, workload("prefill_8192_b1"), order)
    assert first.seconds == second.seconds
    assert first.trace.by_module() == second.trace.by_module()


def test_every_placed_module_contributes_work(hardware, graph, config):
    plan = _plan()
    result = simulate_workload(
        plan, hardware, graph, config, workload("prefill_8192_b1"), topological_order(graph),
    )
    assert set(result.trace.by_module()) == plan.modules()
    assert all(seconds > 0 for seconds in result.trace.by_module().values())


def test_the_vision_module_is_neither_required_nor_run(hardware, graph, config):
    assert "vision" not in deployable_modules(graph)
    result = simulate_workload(
        _plan(), hardware, graph, config, workload("prefill_8192_b1"), topological_order(graph),
    )
    assert "vision" not in result.trace.by_module()


def test_dependencies_order_the_dag(hardware, graph, config):
    """lm_head cannot start before the ffn that produces its input has finished."""
    result = simulate_workload(
        _plan(), hardware, graph, config, workload("prefill_128_b1"), topological_order(graph),
    )
    ffn_finish = max(
        entry.finish for entry in result.trace.scheduled
        if entry.op.module == "layers.0.ffn"
    )
    head_start = min(
        entry.start for entry in result.trace.scheduled if entry.op.module == "lm_head"
    )
    assert head_start >= ffn_finish - 1e-12


def test_a_wider_split_costs_less_per_shard(hardware, graph, config):
    """The scaling invariant, from the framework's side."""
    order = topological_order(graph)

    def per_shard(factor: int) -> float:
        plan = _plan(placements=[
            {"module": "embed", "units": ["d0.l0"], "stage": 0},
            {
                "module": "layers.0.attention",
                "units": [f"d0.l{index}" for index in range(factor)],
                "splits": [{"dim": "head", "factor": factor, "collective": "allreduce"}],
                "stage": 1,
            },
            {"module": "layers.0.ffn", "units": ["d1.l0"], "stage": 2},
            {"module": "lm_head", "units": ["d15.l0"], "stage": 3},
        ])
        result = simulate_workload(plan, hardware, graph, config, workload("prefill_8192_b1"), order)
        tensor = sum(
            entry.op.seconds for entry in result.trace.scheduled
            if entry.op.module == "layers.0.attention" and entry.op.engine == "tensor"
        )
        return tensor / factor

    assert per_shard(2) / per_shard(4) == pytest.approx(2.0, rel=0.2)


def test_longer_context_costs_more_in_both_phases(hardware, graph, config):
    plan = _plan()
    order = topological_order(graph)
    results = {
        point.name: simulate_workload(plan, hardware, graph, config, point, order)
        for point in WORKLOADS
    }
    assert results["prefill_8192_b1"].seconds > results["prefill_128_b1"].seconds * 8
    assert results["decode_8192_b1"].seconds > results["decode_128_b1"].seconds


def test_prefill_is_chunked_and_decode_is_not(hardware, graph, config):
    """Where pipeline parallelism earns its keep, and where it cannot."""
    plan = _plan(runtime={"prefill_chunk_tokens": 2048, "pipeline_chunks": True})
    order = topological_order(graph)
    long_prefill = simulate_workload(plan, hardware, graph, config, workload("prefill_8192_b1"), order)
    short_prefill = simulate_workload(plan, hardware, graph, config, workload("prefill_128_b1"), order)
    decode = simulate_workload(plan, hardware, graph, config, workload("decode_8192_b1"), order)

    assert long_prefill.chunks == 4          # 8192 / 2048
    assert short_prefill.chunks == 1         # 128 is one short chunk
    assert decode.chunks == 1                # one token in flight, nothing to pipeline


def test_chunking_can_be_turned_off(hardware, graph, config):
    plan = _plan(runtime={"prefill_chunk_tokens": 2048, "pipeline_chunks": False})
    result = simulate_workload(
        plan, hardware, graph, config, workload("prefill_8192_b1"), topological_order(graph),
    )
    assert result.chunks == 1


def test_kv_is_charged_once_despite_four_chunks(hardware, graph, config):
    """A chunked prefill calls the cost model per chunk; the ledger keeps the peak."""
    plan = _plan()
    result = simulate_workload(
        plan, hardware, graph, config, workload("prefill_8192_b1"), topological_order(graph),
    )
    kv_entries = [e for e in result.ledger.entries if e.kind == "kv"]
    # One entry per (tier, scope, module), not one per chunk.
    assert len(kv_entries) == len({(e.tier, e.scope_key, e.module) for e in kv_entries})


def test_weights_are_charged_per_shard_not_per_module(hardware, graph):
    from floorplan.sim.memory import MemoryLedger

    plan = _plan()
    ledger = MemoryLedger()
    charge_weights(plan, graph, ledger)
    ffn = [e for e in ledger.entries if e.module == "layers.0.ffn"]
    assert len(ffn) == 4
    assert all(entry.nbytes == (1 << 32) // 4 for entry in ffn)


def test_tiered_weights_charge_the_backing_tier_and_the_cache(hardware, graph):
    """A cache is a copy: 10% in HBM does not remove 10% from host DRAM."""
    from floorplan.sim.memory import MemoryLedger

    plan = _plan(placements=[
        {"module": "embed", "units": ["d0.l0"]},
        {"module": "layers.0.attention", "units": ["d0.l1"]},
        {
            "module": "layers.0.ffn", "units": ["d1.l0"],
            "weights": {
                "tier": "host_dram", "resident_fraction": 0.1,
                "cache_tier": "hbm_bank", "hit_rate": 0.8,
            },
        },
        {"module": "lm_head", "units": ["d15.l0"]},
    ])
    ledger = MemoryLedger()
    charge_weights(plan, graph, ledger)
    backing = [e for e in ledger.entries if e.tier == "host_dram"]
    cached = [e for e in ledger.entries if "(cache)" in e.module]
    assert backing[0].nbytes == 1 << 32
    assert cached[0].nbytes == pytest.approx((1 << 32) * 0.1, rel=1e-6)


def test_an_oversized_shard_is_reported_as_a_capacity_violation(hardware, graph, config):
    """A 30 GiB module on one bank is not slow, it is impossible."""
    from floorplan.sim.memory import MemoryLedger

    graph = dict(graph)
    graph["layers.0.ffn"] = {**graph["layers.0.ffn"], "param_bytes": 30 * 1024 ** 3}
    plan = _plan(placements=[
        {"module": "embed", "units": ["d0.l0"]},
        {"module": "layers.0.attention", "units": ["d0.l1"]},
        {"module": "layers.0.ffn", "units": ["d1.l0"]},
        {"module": "lm_head", "units": ["d15.l0"]},
    ])
    ledger = MemoryLedger()
    charge_weights(plan, graph, ledger)
    problems = ledger.violations(hardware)
    assert problems
    assert "hbm_bank at d1.l0" in problems[0]


def test_a_missing_cost_model_is_a_clear_error(hardware, graph, config):
    api.clear_registry()

    class OnlyEmbed:
        name = "only-embed"

        def matches(self, module_id, entry):
            return module_id == "embed"

        def emit(self, ctx):
            return [ctx.op("x", "tensor", ctx.matmul_seconds(8, 8, 8))]

    api.register(OnlyEmbed())
    with pytest.raises(api.CostModelError, match="no cost model claims"):
        simulate_workload(
            _plan(), hardware, graph, config, workload("prefill_128_b1"), topological_order(graph),
        )


def test_small_k_matmuls_are_charged_at_the_measured_small_k_rate(hardware, graph, config):
    """A K below the stationary limit of 128 cannot fill the array. 8x on this hardware."""
    plan = _plan()
    result = simulate_workload(
        plan, hardware, graph, config, workload("prefill_128_b1"), topological_order(graph),
    )
    context = next(
        api.Context(
            hardware=hardware, schedule=result.trace and __import__(
                "floorplan.sim.engine", fromlist=["Schedule"],
            ).Schedule(),
            ledger=result.ledger, workload=Workload("t", "prefill", 128, 128),
            shard=entry, deps=(), config=config,
        )
        for entry in [_any_shard(hardware, graph)]
    )
    big_k = context.matmul_seconds(128, 512, 512)
    small_k = context.matmul_seconds(128, 512, 32)
    # Same FLOPs per unit of K, but the small-K rate is ~8x worse, so per-FLOP cost rises.
    assert (small_k / 32) / (big_k / 512) == pytest.approx(0.376 / 0.047, rel=0.05)


def _any_shard(hardware, graph):
    from floorplan.schema import Address, Residency

    return api.Shard(
        module="embed", kind="embed", unit=Address(0, 0), shard_index=0, shard_count=1,
        splits=(), fraction=1.0, param_bytes=1 << 20, module_activation_bytes=1 << 20,
        residency=Residency(), stage=0, overlap_collectives=False,
        graph_entry=graph["embed"], group=(Address(0, 0),),
    )


# ---------------------------------------------------------------------------------------
# `stage` is a real ordering constraint (PR #5 review, comment 3)
# ---------------------------------------------------------------------------------------
@pytest.fixture
def parallel_graph() -> dict[str, dict[str, Any]]:
    """Two independent branches joining at a sink.

    Needed because a linear chain cannot show what `stage` does: the data dependencies already
    force the order, so stage ordering is redundant there. `left` and `right` both consume
    `tokens` and could run concurrently, which is exactly the case a pipeline boundary changes.
    """
    return {
        "left": {"kind": "mlp", "inputs": ["tokens"], "outputs": ["l"],
                 "param_bytes": 1 << 26, "activation_bytes": 1 << 20, "layer_indices": [0]},
        "right": {"kind": "mlp", "inputs": ["tokens"], "outputs": ["r"],
                  "param_bytes": 1 << 26, "activation_bytes": 1 << 20, "layer_indices": [1]},
        "sink": {"kind": "lm_head", "inputs": ["l", "r"], "outputs": ["logits"],
                 "param_bytes": 1 << 20, "activation_bytes": 1 << 20},
    }


def _branch_plan(left_stage: int, right_stage: int) -> Floorplan:
    return Floorplan.from_dict({
        "version": 1, "target": "trn2-16device",
        "placements": [
            {"module": "left", "units": ["d0.l0"], "stage": left_stage},
            {"module": "right", "units": ["d1.l0"], "stage": right_stage},
            {"module": "sink", "units": ["d2.l0"], "stage": max(left_stage, right_stage) + 1},
        ],
    })


def test_stage_changes_the_timeline(hardware, parallel_graph, config):
    """Pipeline depth is the documented control, so it has to move a metric.

    It used to be inert: `stage` was copied into `Shard` and never consulted, so ordering came
    only from data dependencies and unit contention. Two plans differing only in their stage
    numbers produced byte-identical timelines, and the pipeline-depth question — the one both
    decode metrics are most sensitive to — was unmeasurable.
    """
    order = topological_order(parallel_graph)
    concurrent = _branch_plan(0, 0)
    pipelined = _branch_plan(0, 1)

    together = simulate_workload(
        concurrent, hardware, parallel_graph, config, workload("decode_128_b1"), order)
    apart = simulate_workload(
        pipelined, hardware, parallel_graph, config, workload("decode_128_b1"), order)
    assert apart.seconds > together.seconds, (
        "putting two independent branches in different stages must serialize them"
    )


def test_a_later_stage_waits_for_an_earlier_one(hardware, parallel_graph, config):
    """Even where no tensor connects them — that is what a pipeline boundary means."""
    plan = _branch_plan(0, 1)
    result = simulate_workload(
        plan, hardware, parallel_graph, config, workload("decode_128_b1"), topological_order(parallel_graph),
    )
    left_finish = max(e.finish for e in result.trace.scheduled if e.op.module == "left")
    right_start = min(e.start for e in result.trace.scheduled if e.op.module == "right")
    assert right_start >= left_finish - 1e-12


# ---------------------------------------------------------------------------------------
# HBM efficiency is not reapplied to probed tiers (comment 13)
# ---------------------------------------------------------------------------------------
def _context(hardware, graph, config, tier="hbm_bank"):
    from floorplan.schema import Address, Residency
    from floorplan.sim.engine import Schedule as S
    from floorplan.sim.memory import MemoryLedger as L

    shard = api.Shard(
        module="layers.1.engram", kind="other", unit=Address(0, 0), shard_index=0,
        shard_count=1, splits=(), fraction=1.0, param_bytes=1 << 20,
        module_activation_bytes=1 << 20, residency=Residency(tier=tier), stage=0,
        overlap_collectives=False, graph_entry=graph["embed"], group=(Address(0, 0),),
    )
    return api.Context(
        hardware=hardware, schedule=S(), ledger=L(),
        workload=Workload("t", "decode", 8192, 1), shard=shard, deps=(), config=config,
    )


def test_probed_tiers_are_not_scaled_by_the_hbm_dma_coefficient(hardware, graph, config):
    """`host_dram` and `nvme` rates are measured achieved values, not fractions of peak.

    Dividing them by `dma_small_strided` (~0.003) again inflated an already-measured 30 us
    NVMe read by over 300x, which is what made every off-HBM residency look unusable.
    """
    ctx = _context(hardware, graph, config, tier="nvme")
    rows, row_bytes = 32, 256
    modelled = ctx.dma_seconds(rows * row_bytes, tier="nvme", accesses=rows)

    nvme = hardware.tiers["nvme"]
    host = hardware.tiers["host_dram"]
    raw = (nvme.transfer_seconds(rows * row_bytes, rows)
           + host.transfer_seconds(rows * row_bytes, rows))
    assert modelled == pytest.approx(raw, rel=1e-9)
    # And it is nowhere near the 300x the old double-count produced.
    assert modelled < raw * 2


def test_hbm_tiers_still_get_the_coefficient(hardware, graph, config):
    ctx = _context(hardware, graph, config)
    nbytes = 1 << 20
    modelled = ctx.dma_seconds(nbytes, tier="hbm_bank", accesses=1)
    raw = hardware.tiers["hbm_bank"].transfer_seconds(nbytes, 1)
    assert modelled == pytest.approx(raw / 0.277, rel=1e-6)


def test_nvme_is_charged_both_legs(hardware, graph, config):
    """Text constraint 6: no direct accelerator-to-NVMe path."""
    ctx = _context(hardware, graph, config, tier="nvme")
    nbytes, accesses = 1 << 20, 16
    through = ctx.dma_seconds(nbytes, tier="nvme", accesses=accesses)
    host_only = ctx.dma_seconds(nbytes, tier="host_dram", accesses=accesses)
    assert through > host_only


# ---------------------------------------------------------------------------------------
# peer_hbm pays its torus distance (comment 14)
# ---------------------------------------------------------------------------------------
def test_peer_hbm_cost_depends_on_hop_count(hardware, graph, config):
    """A remote read from a neighbour and from the far corner must not cost the same."""
    from floorplan.schema import Address, Residency
    from floorplan.sim.engine import Schedule as S
    from floorplan.sim.memory import MemoryLedger as L

    def seconds(owner: int) -> float:
        shard = api.Shard(
            module="m", kind="mlp", unit=Address(0, 0), shard_index=0, shard_count=1,
            splits=(), fraction=1.0, param_bytes=1 << 20, module_activation_bytes=0,
            residency=Residency(tier="peer_hbm", backing_device=owner), stage=0,
            overlap_collectives=False, graph_entry=graph["embed"], group=(Address(0, 0),),
        )
        ctx = api.Context(
            hardware=hardware, schedule=S(), ledger=L(),
            workload=Workload("t", "decode", 128, 1), shard=shard, deps=(), config=config,
        )
        return ctx.dma_seconds(1 << 16, tier="peer_hbm", accesses=8)

    assert hardware.hops(0, 1) == 1 and hardware.hops(0, 15) == 2
    assert seconds(15) > seconds(1)
    assert seconds(0) < seconds(1)          # same device, no hop


# ---------------------------------------------------------------------------------------
# Batch as a metric axis
# ---------------------------------------------------------------------------------------
def test_the_grid_spans_phase_context_and_batch():
    from floorplan.schema import BATCH_SIZES, CONTEXT_LENGTHS

    assert len(WORKLOADS) == 2 * len(CONTEXT_LENGTHS) * len(BATCH_SIZES) == 16
    assert {w.batch for w in WORKLOADS} == set(BATCH_SIZES)
    assert {w.context_tokens for w in WORKLOADS} == set(CONTEXT_LENGTHS)
    assert HEAVIEST == "prefill_8192_b32"


def test_a_larger_batch_costs_more_in_both_phases(hardware, graph, config):
    order = topological_order(graph)
    plan = _plan()
    for phase in ("prefill", "decode"):
        series = [
            simulate_workload(
                plan, hardware, graph, config, workload(f"{phase}_8192_b{batch}"), order,
            ).seconds
            for batch in (1, 4, 8, 32)
        ]
        assert series == sorted(series), f"{phase} is not monotone in batch: {series}"
        assert series[-1] > series[0]


def test_prefill_scales_roughly_with_batch(hardware, graph, config):
    """32x the samples is 32x the tokens, so prefill must scale close to linearly."""
    order = topological_order(graph)
    plan = _plan()
    one = simulate_workload(
        plan, hardware, graph, config, workload("prefill_128_b1"), order).seconds
    many = simulate_workload(
        plan, hardware, graph, config, workload("prefill_128_b32"), order).seconds
    assert many / one > 16


def test_kv_grows_with_batch_and_pressures_capacity(hardware, graph, config):
    """Why batch is a capacity axis: 8192 tokens at batch 32 holds 32x the KV."""
    order = topological_order(graph)
    plan = _plan()
    small = simulate_workload(
        plan, hardware, graph, config, workload("decode_8192_b1"), order)
    large = simulate_workload(
        plan, hardware, graph, config, workload("decode_8192_b32"), order)
    assert large.ledger.by_kind()["kv"] > small.ledger.by_kind()["kv"] * 8


def test_a_batch_split_divides_the_samples_not_the_weights(hardware, graph, config):
    """`batch_per_shard` is the samples; `weight_divisor` stays 1 for a batch split."""
    from floorplan.schema import Address, Placement, Residency, Split
    from floorplan.sim.engine import Schedule as S
    from floorplan.sim.memory import MemoryLedger as L
    from floorplan.sim.runner import weight_divisor

    placement = Placement(
        module="m", units=[Address(0, i) for i in range(4)],
        splits=[Split("batch", 4, "none")],
    )
    assert weight_divisor(placement) == 1

    def samples(batch: int, shard_index: int) -> int:
        shard = api.Shard(
            module="m", kind="mlp", unit=Address(0, shard_index), shard_index=shard_index,
            shard_count=4, splits=(Split("batch", 4, "none"),), fraction=1.0,
            param_bytes=1 << 20, module_activation_bytes=0, residency=Residency(),
            stage=0, overlap_collectives=False, graph_entry=graph["embed"],
            group=tuple(Address(0, i) for i in range(4)),
        )
        ctx = api.Context(
            hardware=hardware, schedule=S(), ledger=L(),
            workload=Workload("t", "decode", 128, 1, batch=batch), shard=shard,
            deps=(), config=config,
        )
        return ctx.batch_per_shard()

    # Batch 32 over 4 shards: 8 each.
    assert [samples(32, i) for i in range(4)] == [8, 8, 8, 8]
    # Batch 1 over 4 shards: one shard does the work, three idle. Data parallelism wider than
    # the batch leaves units idle rather than making the work four times cheaper.
    assert sorted(samples(1, i) for i in range(4)) == [0, 0, 0, 1]
    # Batch 4 over 4 shards divides evenly.
    assert [samples(4, i) for i in range(4)] == [1, 1, 1, 1]
