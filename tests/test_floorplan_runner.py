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
    WORKLOADS,
    charge_weights,
    deployable_modules,
    simulate_workload,
    topological_order,
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
            ctx.charge_kv(context * dim * 2 * 2)
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
    for workload in WORKLOADS:
        result = simulate_workload(plan, hardware, graph, config, workload, order)
        assert result.seconds > 0, workload.name
        assert result.trace.scheduled


def test_simulation_is_deterministic(hardware, graph, config):
    plan = _plan()
    order = topological_order(graph)
    first = simulate_workload(plan, hardware, graph, config, WORKLOADS[1], order)
    second = simulate_workload(plan, hardware, graph, config, WORKLOADS[1], order)
    assert first.seconds == second.seconds
    assert first.trace.by_module() == second.trace.by_module()


def test_every_placed_module_contributes_work(hardware, graph, config):
    plan = _plan()
    result = simulate_workload(
        plan, hardware, graph, config, WORKLOADS[1], topological_order(graph),
    )
    assert set(result.trace.by_module()) == plan.modules()
    assert all(seconds > 0 for seconds in result.trace.by_module().values())


def test_the_vision_module_is_neither_required_nor_run(hardware, graph, config):
    assert "vision" not in deployable_modules(graph)
    result = simulate_workload(
        _plan(), hardware, graph, config, WORKLOADS[1], topological_order(graph),
    )
    assert "vision" not in result.trace.by_module()


def test_dependencies_order_the_dag(hardware, graph, config):
    """lm_head cannot start before the ffn that produces its input has finished."""
    result = simulate_workload(
        _plan(), hardware, graph, config, WORKLOADS[0], topological_order(graph),
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
        result = simulate_workload(plan, hardware, graph, config, WORKLOADS[1], order)
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
        workload.name: simulate_workload(plan, hardware, graph, config, workload, order)
        for workload in WORKLOADS
    }
    assert results["prefill_8192"].seconds > results["prefill_128"].seconds * 8
    assert results["decode_8192"].seconds > results["decode_128"].seconds


def test_prefill_is_chunked_and_decode_is_not(hardware, graph, config):
    """Where pipeline parallelism earns its keep, and where it cannot."""
    plan = _plan(runtime={"prefill_chunk_tokens": 2048, "pipeline_chunks": True})
    order = topological_order(graph)
    long_prefill = simulate_workload(plan, hardware, graph, config, WORKLOADS[1], order)
    short_prefill = simulate_workload(plan, hardware, graph, config, WORKLOADS[0], order)
    decode = simulate_workload(plan, hardware, graph, config, WORKLOADS[3], order)

    assert long_prefill.chunks == 4          # 8192 / 2048
    assert short_prefill.chunks == 1         # 128 is one short chunk
    assert decode.chunks == 1                # one token in flight, nothing to pipeline


def test_chunking_can_be_turned_off(hardware, graph, config):
    plan = _plan(runtime={"prefill_chunk_tokens": 2048, "pipeline_chunks": False})
    result = simulate_workload(
        plan, hardware, graph, config, WORKLOADS[1], topological_order(graph),
    )
    assert result.chunks == 1


def test_kv_is_charged_once_despite_four_chunks(hardware, graph, config):
    """A chunked prefill calls the cost model per chunk; the ledger keeps the peak."""
    plan = _plan()
    result = simulate_workload(
        plan, hardware, graph, config, WORKLOADS[1], topological_order(graph),
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
            _plan(), hardware, graph, config, WORKLOADS[0], topological_order(graph),
        )


def test_small_k_matmuls_are_charged_at_the_measured_small_k_rate(hardware, graph, config):
    """A K below the stationary limit of 128 cannot fill the array. 8x on this hardware."""
    plan = _plan()
    result = simulate_workload(
        plan, hardware, graph, config, WORKLOADS[0], topological_order(graph),
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
