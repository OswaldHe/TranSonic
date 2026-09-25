# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The simulation framework: the timeline, the collective model, the memory ledger.

Framework only — no agent-written cost model is involved, and none of these tests touch a
device or an artifact. What is being checked is that the machinery a cost model plugs into
behaves the way `sim/api.py` promises, because a cost model written against a promise the
framework does not keep would be wrong in a way no review would catch.
"""

from __future__ import annotations

import pytest

from floorplan.parser import Hardware, Tier, load_system
from floorplan.schema import Address
from floorplan.sim import collectives
from floorplan.sim.engine import ENGINES, Op, Schedule, ScheduleError
from floorplan.sim.memory import MemoryLedger, scope_key_for

pytestmark = pytest.mark.floorplan


@pytest.fixture
def hardware() -> Hardware:
    """The 16-device target with plausible probed values filled in.

    Filled in here rather than read from `probed.yaml` so these tests do not depend on a probe
    having run, and so a probe that changes a coefficient cannot silently change what they
    assert. The values are the shape of the real measurements, not placeholders.
    """
    system = load_system("trn2-16device", apply_probes=False)
    system["efficiency"] = {
        "matmul_fp8": 0.40, "matmul_bf16": 0.41, "matmul_small_k": 0.056,
        "vector_elementwise": 2.4, "scalar_activation": 5.1, "gpsimd_gather": 0.049,
        "dma_large_contiguous": 0.287, "dma_small_strided": 0.0034,
        "collective_allreduce": 1.0, "collective_all_to_all": 1.0,
    }
    system["links"]["intra_device"].update(
        {"bandwidth_bytes_per_s": 8.3e11, "latency_us": 0.7},
    )
    for tier in system["tiers"]:
        if tier["name"] == "host_dram":
            tier.update({"bandwidth_bytes_per_s": 1.29e10, "latency_us": 0.7})
        if tier["name"] == "nvme":
            tier.update({
                "bandwidth_bytes_per_s": 2.45e9, "latency_us": 30.0,
                "random_read_iops": 33_000,
            })
    return Hardware.from_system(system)


# ---------------------------------------------------------------------------------------
# The timeline
# ---------------------------------------------------------------------------------------
def test_same_engine_serializes():
    schedule = Schedule()
    first = schedule.submit(Op("a", "d0.l0", "tensor", 1.0))
    second = schedule.submit(Op("b", "d0.l0", "tensor", 1.0))
    assert schedule.finish_of(first) == 1.0
    assert schedule.finish_of(second) == 2.0
    assert schedule.trace.makespan() == 2.0


def test_different_engines_overlap():
    schedule = Schedule()
    schedule.submit(Op("a", "d0.l0", "tensor", 1.0))
    schedule.submit(Op("b", "d0.l0", "vector", 1.0))
    assert schedule.trace.makespan() == 1.0


def test_different_units_overlap():
    schedule = Schedule()
    schedule.submit(Op("a", "d0.l0", "tensor", 1.0))
    schedule.submit(Op("b", "d0.l1", "tensor", 1.0))
    assert schedule.trace.makespan() == 1.0


def test_dependencies_are_honoured():
    schedule = Schedule()
    first = schedule.submit(Op("a", "d0.l0", "tensor", 1.0))
    second = schedule.submit(Op("b", "d0.l1", "vector", 1.0, deps=(first,)))
    assert schedule.finish_of(second) == 2.0


def test_sbuf_exclusion_serializes_gpsimd_against_tensor():
    """Text constraint 1: GPSIMD and the tensor engine cannot both be in SBUF.

    Different engines, so without the resource they would overlap. This is the single
    constraint most likely to change a floorplan's ranking, since it lands on exactly the
    modules that gather — MoE and Engram.
    """
    schedule = Schedule()
    schedule.submit(Op("matmul", "d0.l0", "tensor", 1.0, holds=frozenset({"sbuf"})))
    schedule.submit(Op("gather", "d0.l0", "gpsimd", 1.0, holds=frozenset({"sbuf"})))
    assert schedule.trace.makespan() == 2.0

    # On different units they do not contend.
    other = Schedule()
    other.submit(Op("matmul", "d0.l0", "tensor", 1.0, holds=frozenset({"sbuf"})))
    other.submit(Op("gather", "d0.l1", "gpsimd", 1.0, holds=frozenset({"sbuf"})))
    assert other.trace.makespan() == 1.0


def test_overlapped_collective_frees_the_engine_but_not_dependents():
    """Text constraint 4: overlap is real, but a collective and its consumer cannot overlap."""
    schedule = Schedule()
    collective = schedule.submit(
        Op("ar", "d0.l0", "cc", 1.0, overlapped=True),
    )
    # Unrelated compute on the same unit is not blocked.
    unrelated = schedule.submit(Op("other", "d0.l0", "cc", 1.0))
    assert schedule.finish_of(unrelated) == 1.0
    # A dependent still waits.
    dependent = schedule.submit(Op("use", "d0.l0", "tensor", 1.0, deps=(collective,)))
    assert schedule.finish_of(dependent) == 2.0


def test_forward_dependency_is_rejected():
    schedule = Schedule()
    with pytest.raises(ScheduleError, match="has not been submitted"):
        schedule.submit(Op("a", "d0.l0", "tensor", 1.0, deps=(5,)))


def test_unknown_engine_and_resource_are_rejected():
    schedule = Schedule()
    with pytest.raises(ScheduleError, match="unknown engine"):
        schedule.submit(Op("a", "d0.l0", "quantum", 1.0))
    with pytest.raises(ScheduleError, match="unknown resource"):
        schedule.submit(Op("a", "d0.l0", "tensor", 1.0, holds=frozenset({"cache"})))


def test_negative_duration_is_rejected():
    schedule = Schedule()
    with pytest.raises(ScheduleError, match="negative duration"):
        schedule.submit(Op("a", "d0.l0", "tensor", -1.0))


def test_critical_path_follows_the_latest_chain():
    schedule = Schedule()
    fast = schedule.submit(Op("fast", "d0.l0", "tensor", 0.1))
    slow = schedule.submit(Op("slow", "d0.l1", "tensor", 5.0))
    schedule.submit(Op("join", "d0.l2", "vector", 1.0, deps=(fast, slow)))
    names = [entry.op.name for entry in schedule.trace.critical_path()]
    assert names == ["slow", "join"]


def test_utilization_is_per_unit_averaged():
    schedule = Schedule()
    schedule.submit(Op("a", "d0.l0", "tensor", 1.0))
    schedule.submit(Op("b", "d0.l1", "tensor", 0.5))
    # Makespan 1.0, two units, 1.5 busy seconds -> 75%.
    assert schedule.trace.utilization()["tensor"] == pytest.approx(0.75)


def test_busiest_unit_finds_the_imbalance():
    schedule = Schedule()
    schedule.submit(Op("a", "d0.l0", "tensor", 3.0))
    schedule.submit(Op("b", "d0.l1", "tensor", 1.0))
    unit, busy = schedule.trace.busiest_unit()
    assert (unit, busy) == ("d0.l0", 3.0)


def test_engine_vocabulary_matches_nki():
    assert ENGINES == {"tensor", "vector", "scalar", "gpsimd", "dma", "cc"}


# ---------------------------------------------------------------------------------------
# Collectives
# ---------------------------------------------------------------------------------------
def test_group_inside_one_logical_core_is_free(hardware):
    """Text constraint 2: the two physical cores at LNC=2 share one address space."""
    cost = collectives.cost(hardware, "allreduce", 1 << 20, ["d0.l0.p0", "d0.l0.p1"])
    assert cost.seconds == 0.0
    assert cost.bytes_on_wire == 0
    assert cost.link_class == "none"


def test_intra_device_beats_inter_device(hardware):
    """The question a floorplan keeps asking, and the model has to answer it consistently."""
    inside = collectives.cost(
        hardware, "allreduce", 1 << 24, ["d0.l0", "d0.l1", "d0.l2", "d0.l3"],
    )
    across = collectives.cost(
        hardware, "allreduce", 1 << 24, ["d0.l0", "d1.l0", "d2.l0", "d3.l0"],
    )
    assert inside.link_class == "intra_device"
    assert across.link_class == "inter_device"
    assert inside.bytes_on_wire == across.bytes_on_wire
    assert inside.seconds < across.seconds


def test_hop_count_costs_latency_not_bandwidth(hardware):
    """A far group pays more, and the extra is latency — constraint 5 made arithmetic."""
    near = collectives.cost(hardware, "allreduce", 1 << 12, ["d0.l0", "d1.l0"])
    far = collectives.cost(hardware, "allreduce", 1 << 12, ["d0.l0", "d15.l0"])
    assert near.hops == 1 and far.hops == 2
    assert far.seconds > near.seconds
    assert far.bytes_on_wire == near.bytes_on_wire


def test_allreduce_moves_twice_allgather(hardware):
    group = ["d0.l0", "d0.l1", "d0.l2", "d0.l3"]
    allreduce = collectives.cost(hardware, "allreduce", 1 << 20, group)
    allgather = collectives.cost(hardware, "allgather", 1 << 20, group)
    assert allreduce.bytes_on_wire == 2 * allgather.bytes_on_wire


def test_wire_volume_follows_the_ring_formula(hardware):
    group = [f"d0.l{index}" for index in range(4)]
    cost = collectives.cost(hardware, "allgather", 1000, group)
    assert cost.bytes_on_wire == int(1000 * (4 - 1) / 4)


def test_wider_split_communicates_more(hardware):
    """The invariant the whole search depends on: parallelism is not free."""
    narrow = collectives.cost(hardware, "allreduce", 1 << 20, ["d0.l0", "d0.l1"])
    wide = collectives.cost(hardware, "allreduce", 1 << 20, [f"d0.l{i}" for i in range(4)])
    assert wide.bytes_on_wire > narrow.bytes_on_wire


def test_single_participant_is_free(hardware):
    assert collectives.cost(hardware, "allreduce", 1 << 20, ["d0.l0"]).seconds == 0.0


def test_none_is_free_at_any_size(hardware):
    assert collectives.cost(hardware, "none", 1 << 30, [f"d{i}.l0" for i in range(8)]).seconds == 0


def test_unknown_collective_is_rejected(hardware):
    with pytest.raises(ValueError, match="unknown collective"):
        collectives.cost(hardware, "gossip", 1 << 10, ["d0.l0", "d0.l1"])


def test_small_tensors_are_latency_bound(hardware):
    """Decode moves small tensors; if the model were bandwidth-only it would miss this."""
    tiny = collectives.cost(hardware, "allreduce", 64, [f"d{i}.l0" for i in range(16)])
    bandwidth_term = tiny.bytes_on_wire / 8.3e11
    assert tiny.seconds > bandwidth_term * 100


# ---------------------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------------------
def test_bank_pressure_rolls_up_into_the_device(hardware):
    """Four full banks must make the device full, not empty.

    Accounting them separately would let a plan put 24 GiB on each of a device's four banks
    and report the device as holding nothing.
    """
    ledger = MemoryLedger()
    bank = 24 * 1024 ** 3
    for index in range(4):
        ledger.add_weights("hbm_bank", f"d0.l{index}", bank, f"m{index}")
    high_water = ledger.high_water(hardware)
    assert high_water["hbm_bank"][1] == bank
    assert high_water["device_hbm"][1] == 4 * bank
    assert not ledger.violations(hardware)     # exactly at capacity, not over


def test_over_capacity_bank_is_reported_with_the_culprit(hardware):
    ledger = MemoryLedger()
    ledger.add_weights("hbm_bank", "d0.l0", 25 * 1024 ** 3, "layers.1.engram")
    problems = ledger.violations(hardware)
    assert len(problems) == 1
    assert "hbm_bank at d0.l0" in problems[0]
    assert "layers.1.engram" in problems[0]


def test_kv_is_a_peak_not_a_sum():
    """A chunked prefill charges KV once per chunk; the ledger must keep the largest.

    Summing would report four KV caches for a four-chunk prompt and make every long-context
    plan look infeasible.
    """
    ledger = MemoryLedger()
    for nbytes in (1 << 20, 4 << 20, 2 << 20):
        ledger.add_kv("hbm_bank", "d0.l0", nbytes, "layers.0.attention")
    assert ledger.by_kind()["kv"] == 4 << 20


def test_weights_accumulate_across_modules():
    ledger = MemoryLedger()
    ledger.add_weights("hbm_bank", "d0.l0", 100, "a")
    ledger.add_weights("hbm_bank", "d0.l0", 200, "b")
    assert ledger.totals()[("hbm_bank", "d0.l0")] == 300


def test_negative_bytes_are_rejected():
    ledger = MemoryLedger()
    with pytest.raises(ValueError, match="negative"):
        ledger.add_weights("hbm_bank", "d0.l0", -1, "m")


@pytest.mark.parametrize("tier,expected", [
    ("hbm_bank", "d3.l2"),
    ("device_hbm", "d3"),
    ("host_dram", "instance"),
    ("nvme", "instance"),
])
def test_scope_keys(tier, expected):
    assert scope_key_for(tier, Address(3, 2)) == expected


def test_nvme_is_charged_through_host_dram(hardware):
    """Text constraint 6: there is no direct accelerator-to-NVMe path."""
    assert hardware.tiers["nvme"].via == ("host_dram",)
