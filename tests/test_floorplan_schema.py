# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The floorplan schema and the hardware parser. No device, no network, no artifact."""

from __future__ import annotations

import pytest
import yaml

from floorplan.parser import Hardware, SystemError_, ceil_div, load_system, shard_bytes
from floorplan.schema import (
    Address,
    Floorplan,
    FloorplanError,
    Placement,
    Residency,
    Split,
    legal_dims,
)

pytestmark = pytest.mark.floorplan


# ---------------------------------------------------------------------------------------
# Addresses
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("d0.l0", (0, 0, None)),
    ("d15.l3", (15, 3, None)),
    ("d7.l2.p1", (7, 2, 1)),
])
def test_address_parses(text, expected):
    address = Address.parse(text)
    assert (address.device, address.logical_nc, address.physical_nc) == expected
    assert str(address) == text


@pytest.mark.parametrize("text", ["d0", "l0", "0.0", "d0.l", "dx.l0", "d0.l0.p", ""])
def test_address_rejects_nonsense(text):
    with pytest.raises(FloorplanError):
        Address.parse(text)


def test_logical_drops_physical_suffix():
    assert Address.parse("d3.l2.p1").logical() == Address(3, 2)


# ---------------------------------------------------------------------------------------
# Placements
# ---------------------------------------------------------------------------------------
def _placement(**kwargs) -> Placement:
    defaults = dict(module="layers.0.attention", units=[Address(0, 0)])
    defaults.update(kwargs)
    return Placement(**defaults)


def test_units_must_match_shard_count():
    placement = _placement(
        units=[Address(0, 0), Address(0, 1)],
        splits=[Split("head", 4)],
    )
    with pytest.raises(FloorplanError, match="2 unit.* for 4 shard"):
        placement.validate("p")


def test_composed_splits_multiply():
    placement = _placement(
        units=[Address(0, index) for index in range(4)],
        splits=[Split("head", 2), Split("seq", 2)],
    )
    placement.validate("p")
    assert placement.shard_count() == 4


def test_repeated_unit_is_rejected():
    placement = _placement(units=[Address(0, 0), Address(0, 0)], splits=[Split("head", 2)])
    with pytest.raises(FloorplanError, match="repeats unit"):
        placement.validate("p")


def test_same_dim_split_twice_is_rejected():
    placement = _placement(
        units=[Address(0, index) for index in range(4)],
        splits=[Split("head", 2), Split("head", 2)],
    )
    with pytest.raises(FloorplanError, match="splits the 'head' dim twice"):
        placement.validate("p")


def test_unknown_dim_and_collective_are_rejected():
    with pytest.raises(FloorplanError, match="unknown partition dim"):
        Split("sideways", 2).validate("p")
    with pytest.raises(FloorplanError, match="unknown collective"):
        Split("head", 2, "telepathy").validate("p")


# ---------------------------------------------------------------------------------------
# Residency
# ---------------------------------------------------------------------------------------
def test_tiering_requires_cache_tier_and_hit_rate():
    with pytest.raises(FloorplanError, match="needs a cache_tier"):
        Residency(tier="host_dram", resident_fraction=0.1).validate("p")
    with pytest.raises(FloorplanError, match="needs an explicit hit_rate"):
        Residency(
            tier="host_dram", resident_fraction=0.1, cache_tier="hbm_bank",
        ).validate("p")
    # Complete is fine.
    Residency(
        tier="host_dram", resident_fraction=0.1, cache_tier="hbm_bank", hit_rate=0.8,
    ).validate("p")


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5])
def test_resident_fraction_bounds(fraction):
    with pytest.raises(FloorplanError):
        Residency(resident_fraction=fraction).validate("p")


# ---------------------------------------------------------------------------------------
# Whole floorplans
# ---------------------------------------------------------------------------------------
def _minimal(**overrides) -> dict:
    data = {
        "version": 1,
        "target": "trn2-16device",
        "placements": [
            {"module": "embed", "units": ["d0.l0"], "weights": {"tier": "hbm_bank"}},
        ],
    }
    data.update(overrides)
    return data


def test_fractions_must_sum_to_one():
    data = _minimal(placements=[
        {"module": "embed", "units": ["d0.l0"], "fraction": 0.5},
        {"module": "embed", "units": ["d0.l1"], "fraction": 0.4},
    ])
    with pytest.raises(FloorplanError, match="sum to 0.9, not 1"):
        Floorplan.from_dict(data)


def test_fractions_summing_to_one_across_placements_is_fine():
    plan = Floorplan.from_dict(_minimal(placements=[
        {"module": "embed", "units": ["d0.l0"], "fraction": 0.5},
        {"module": "embed", "units": ["d0.l1"], "fraction": 0.5},
    ]))
    assert len(plan.placements) == 2


def test_thirds_are_within_tolerance():
    third = 1.0 / 3.0
    plan = Floorplan.from_dict(_minimal(placements=[
        {"module": "embed", "units": [f"d0.l{i}"], "fraction": third} for i in range(3)
    ]))
    assert len(plan.placements) == 3


def test_unknown_keys_are_rejected():
    with pytest.raises(FloorplanError, match="unknown top-level key"):
        Floorplan.from_dict(_minimal(strategy="aggressive"))
    with pytest.raises(FloorplanError, match="unknown runtime key"):
        Floorplan.from_dict(_minimal(runtime={"chunk": 512}))
    with pytest.raises(FloorplanError, match="unknown key"):
        Floorplan.from_dict(_minimal(placements=[
            {"module": "embed", "units": ["d0.l0"], "priority": 3},
        ]))


def test_round_trip_is_stable(tmp_path):
    plan = Floorplan.from_dict(_minimal(placements=[
        {
            "module": "layers.0.attention",
            "units": ["d0.l0", "d0.l1"],
            "splits": [{"dim": "head", "factor": 2, "collective": "allreduce"}],
            "weights": {"tier": "hbm_bank"},
            "stage": 3,
            "overlap_collectives": True,
        },
    ]))
    path = tmp_path / "fp.yaml"
    plan.dump(path, "# header")
    again = Floorplan.load(path)
    assert again.to_dict() == plan.to_dict()
    assert again.placements[0].stage == 3
    assert again.placements[0].overlap_collectives is True


# ---------------------------------------------------------------------------------------
# Against real hardware
# ---------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def hardware() -> Hardware:
    return Hardware.from_system(load_system("trn2-16device", apply_probes=False))


def test_sixteen_devices_four_cores_each(hardware):
    assert len(hardware.devices) == 16
    assert hardware.unit_count() == 64
    assert hardware.lnc == 2
    assert all(len(d.logical_ncs) == 4 for d in hardware.devices)


def test_bank_is_a_quarter_of_the_device(hardware):
    unit = hardware.devices[0].logical_ncs[0]
    assert unit.hbm_bank_bytes == 24 * 1024 ** 3
    assert unit.hbm_bank_bytes * 4 == hardware.devices[0].hbm_bytes


def test_torus_wraps(hardware):
    # 4x4 with wrap: the far corner is 2 hops, not 6.
    assert hardware.hops(0, 0) == 0
    assert hardware.hops(0, 1) == 1
    assert hardware.hops(0, 3) == 1        # wraps along the row
    assert hardware.hops(0, 12) == 1       # wraps along the column
    assert hardware.hops(0, 15) == 2       # the diagonal opposite
    assert max(
        hardware.hops(a, b) for a in range(16) for b in range(16)
    ) == 4


def test_out_of_range_addresses_are_rejected(hardware):
    system = load_system("trn2-16device", apply_probes=False)
    plan = Floorplan.from_dict(_minimal(placements=[
        {"module": "embed", "units": ["d16.l0"]},
    ]))
    with pytest.raises(FloorplanError, match="has 16"):
        plan.validate_against(system)
    plan = Floorplan.from_dict(_minimal(placements=[
        {"module": "embed", "units": ["d0.l4"]},
    ]))
    with pytest.raises(FloorplanError, match="a device has 4"):
        plan.validate_against(system)


def test_load_time_only_tiers_are_refused(hardware):
    """EBS and the network are declared by the system YAML but are not weight tiers.

    They fail at construction rather than at `validate_against`, because `WEIGHT_TIERS` does
    not contain them at all — the earliest possible rejection, and the right one: a floorplan
    that reads EBS during a forward pass is not a slow plan, it is a nonsensical one.
    """
    for tier in ("ebs", "network"):
        with pytest.raises(FloorplanError, match=f"unknown weight tier '{tier}'"):
            Floorplan.from_dict(_minimal(placements=[
                {"module": "embed", "units": ["d0.l0"], "weights": {"tier": tier}},
            ]))

    # And a tier that is not declared by the target at all is caught against the hardware.
    system = load_system("trn2-1device", apply_probes=False)
    plan = Floorplan.from_dict(_minimal(target="trn2-1device", placements=[
        {"module": "embed", "units": ["d0.l0"], "weights": {"tier": "peer_hbm"}},
    ]))
    with pytest.raises(FloorplanError, match="not declared by target"):
        plan.validate_against(system)


def test_one_device_target_rejects_other_devices():
    system = load_system("trn2-1device", apply_probes=False)
    plan = Floorplan.from_dict(_minimal(target="trn2-1device", placements=[
        {"module": "embed", "units": ["d1.l0"]},
    ]))
    with pytest.raises(FloorplanError, match="has 1"):
        plan.validate_against(system)


# ---------------------------------------------------------------------------------------
# Inheritance and provenance
# ---------------------------------------------------------------------------------------
def test_one_device_inherits_the_silicon():
    single = load_system("trn2-1device", apply_probes=False)
    multi = load_system("trn2-16device", apply_probes=False)
    assert single["device"] == multi["device"]
    assert single["efficiency"] == multi["efficiency"]


def test_unknown_system_names_its_alternatives():
    with pytest.raises(SystemError_, match="trn2-16device"):
        load_system("trn2-imaginary")


def test_simulator_refuses_unmeasured_values():
    hardware = Hardware.from_system(load_system("trn2-16device", apply_probes=False))
    unresolved = hardware.unresolved()
    assert unresolved, "the datasheet alone should leave efficiencies unmeasured"
    assert any(name.startswith("efficiency.") for name in unresolved)
    with pytest.raises(SystemError_, match="probe"):
        hardware.require_efficiency("matmul_bf16")


def test_probe_required_tier_raises_with_guidance():
    hardware = Hardware.from_system(load_system("trn2-16device", apply_probes=False))
    with pytest.raises(SystemError_, match="probe_required"):
        hardware.tiers["nvme"].transfer_seconds(1 << 20)


# ---------------------------------------------------------------------------------------
# Tier cost model
# ---------------------------------------------------------------------------------------
def test_small_random_reads_are_iops_bound_not_bandwidth_bound():
    from floorplan.parser import Tier

    # Numbers in the shape of the probed instance store: fast in bytes, slow in operations.
    tier = Tier(
        name="nvme", scope="instance", capacity_bytes=1 << 40,
        bandwidth_bytes_per_s=2.45e9, latency_us=30.0,
        random_read_iops=33_000, min_transfer_granularity_bytes=4096,
    )
    # 1000 scattered 256-byte rows: 256 KB of payload, which bandwidth alone would call
    # instant. IOPS and latency are what actually bound it.
    by_bandwidth = 1000 * 256 / 2.45e9
    actual = tier.transfer_seconds(1000 * 256, accesses=1000)
    assert actual > by_bandwidth * 100
    assert actual == pytest.approx(max(1000 / 33_000, 1000 * 30e-6), rel=0.01)


def test_granularity_floors_the_byte_count():
    from floorplan.parser import Tier

    tier = Tier(
        name="nvme", scope="instance", capacity_bytes=1 << 40,
        bandwidth_bytes_per_s=1e9, latency_us=0.0, min_transfer_granularity_bytes=4096,
    )
    # Ten 256-byte reads still move ten 4 KiB blocks off the device.
    assert tier.transfer_seconds(10 * 256, accesses=10) == pytest.approx(10 * 4096 / 1e9)


# ---------------------------------------------------------------------------------------
# Dimension legality
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("module_id,kind,dim,legal", [
    ("layers.0.attention", "attention", "head", True),
    ("layers.0.attention", "attention", "expert", False),
    ("layers.0.ffn", "mlp", "expert", True),
    ("layers.0.ffn", "mlp", "head", False),
    ("layers.1.engram", "other", "ngram", True),
    ("layers.0.hc_attn_in", "other", "hidden", True),
    ("layers.0.hc_attn_in", "other", "ngram", False),
    ("lm_head", "lm_head", "vocab", True),
    ("embed", "embed", "vocab", True),
])
def test_legal_dims(module_id, kind, dim, legal):
    assert (dim in legal_dims(module_id, {"kind": kind})) is legal


def test_layer_split_needs_a_multi_layer_module():
    system = load_system("trn2-16device", apply_probes=False)
    modules = {
        "layers.0.attention": {
            "kind": "attention", "layer_indices": [0], "param_bytes": 1,
            "inputs": [], "outputs": [],
        },
    }
    plan = Floorplan.from_dict(_minimal(placements=[
        {
            "module": "layers.0.attention", "units": ["d0.l0", "d0.l1"],
            "splits": [{"dim": "layer", "factor": 2}],
        },
    ]))
    with pytest.raises(FloorplanError, match="spanning two or more layers"):
        plan.validate_against(system, modules)


def test_shard_bytes_rounds_up():
    assert ceil_div(384, 5) == 77
    assert shard_bytes(384, 5) == 77
    assert shard_bytes(384, 1) == 384


# ---------------------------------------------------------------------------------------
# YAML scalar coercion
# ---------------------------------------------------------------------------------------
def test_unsigned_exponents_are_still_numbers():
    """PyYAML parses `6.67e14` as a *string*, not a float.

    YAML 1.1's float grammar wants a signed exponent, so `6.67e+14` is a number and
    `6.67e14` is text. Every compute peak in the system YAML is written the second way, and an
    `isinstance(value, float)` filter silently dropped all of them — leaving `device.compute`
    empty and every matmul un-costable, with no error until a cost model asked for a rate.
    """
    import yaml as _yaml

    from floorplan.parser import as_float

    parsed = _yaml.safe_load("unsigned: 6.67e14\nsigned: 6.67e+14\n")
    assert isinstance(parsed["unsigned"], str)      # the trap
    assert isinstance(parsed["signed"], float)
    assert as_float(parsed["unsigned"]) == as_float(parsed["signed"]) == 6.67e14


@pytest.mark.parametrize("value,expected", [
    (None, None), (True, None), (False, None), ("", None), ("abc", None),
    (5, 5.0), (5.5, 5.5), ("2.9e12", 2.9e12), ("  1e3  ", 1000.0), ("-1.5", -1.5),
])
def test_as_float_coercion(value, expected):
    from floorplan.parser import as_float

    assert as_float(value) == expected


def test_every_compute_peak_reaches_the_hardware_model():
    """The regression the bug above caused: peaks present in the YAML but absent in the model."""
    system = load_system("trn2-16device", apply_probes=False)
    hardware = Hardware.from_system(system)
    for key in ("fp8_flops", "bf16_flops", "fp32_flops"):
        assert hardware.compute.get(key), f"device.compute.{key} did not reach the model"
    assert hardware.compute["bf16_flops"] == pytest.approx(6.67e14)
    assert hardware.compute["fp8_flops"] == pytest.approx(1.299e15)
    # fp4 is explicitly null in the YAML and must stay absent rather than defaulting.
    assert hardware.compute.get("fp4_flops") is None
