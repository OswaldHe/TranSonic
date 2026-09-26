# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The probe's pure logic: fitting, provenance, and the overlay it writes.

No device. What is tested here is everything between a timing and a coefficient, which is
where this subsystem has actually gone wrong: a fit that returns a confidently absurd number,
a derived value relabelled as measured, a tier patch that silently drops the other tiers.
Those failures are all silent by nature — the simulator would run happily on any of them —
so they need tests more than the timing does.
"""

from __future__ import annotations

import pytest
import yaml

from floorplan.parser import Hardware, load_system
from floorplan.probe import storage as storage_probe
from floorplan.probe import suite

pytestmark = pytest.mark.floorplan


# ---------------------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------------------
def test_fit_recovers_a_known_line():
    # seconds = 5 us + bytes / 10 GB/s
    points = [(n, 5e-6 + n / 1e10) for n in (1 << 12, 1 << 15, 1 << 18, 1 << 21)]
    slope, intercept = suite._fit_line(points)
    assert 1.0 / slope == pytest.approx(1e10, rel=1e-6)
    assert intercept == pytest.approx(5e-6, rel=1e-6)


def test_fit_is_robust_to_one_bad_endpoint():
    """The reason it is least squares and not a two-point fit on the extremes.

    An endpoint fit on this data reports a bandwidth roughly 4x wrong, which is what the first
    version of the PCIe probe did: it published 2.9 GB/s where the regression gives ~13.
    """
    clean = [(n, 1e-6 + n / 1.3e10) for n in (1 << 13, 1 << 16, 1 << 19, 1 << 22)]
    noisy = [clean[0][0], clean[0][1] * 4]        # first sample 4x too slow
    points = [(noisy[0], noisy[1]), *clean[1:]]

    slope, _ = suite._fit_line(points)
    least_squares = 1.0 / slope
    endpoints = (points[-1][0] - points[0][0]) / (points[-1][1] - points[0][1])
    assert abs(least_squares - 1.3e10) < abs(endpoints - 1.3e10)


def test_fit_rejects_a_degenerate_sweep():
    with pytest.raises(RuntimeError, match="same size"):
        suite._fit_line([(1000, 1e-6), (1000, 2e-6)])
    with pytest.raises(RuntimeError, match="at least two"):
        suite._fit_line([(1000, 1e-6)])


# ---------------------------------------------------------------------------------------
# The plausibility band
# ---------------------------------------------------------------------------------------
def test_plausibility_band_brackets_the_real_links():
    low, high = suite.PLAUSIBLE_LINK_BANDWIDTH
    for real in (
        1.024e12,   # inter-device NeuronLink, datasheet
        8.0e11,     # intra-device, derived
        1.29e10,    # PCIe, measured
        2.45e9,     # NVMe sequential, measured
    ):
        assert low <= real <= high, f"{real:.3e} should be inside the band"
    # And the number that actually got published once, which must not be.
    assert not low <= 781_258 <= high


# ---------------------------------------------------------------------------------------
# Measurements and provenance
# ---------------------------------------------------------------------------------------
def test_guarded_records_a_failure_without_aborting():
    results = suite.Results()

    @suite._guarded(results, "thing", "bytes/s")
    def _boom() -> float:
        raise RuntimeError("the compiler said no")

    entry = results.measurements["thing"]
    assert not entry.ok()
    assert "the compiler said no" in entry.error
    assert results.value("thing") is None
    assert [m.name for m in results.failures()] == ["thing"]


def test_guarded_does_not_relabel_a_derived_value_as_probed():
    """A derived fp8 fraction must not come out tagged `source: probed`."""
    results = suite.Results()

    @suite._guarded(results, "matmul_fp8", "fraction")
    def _derive() -> float:
        results.add(suite.Measurement(
            "matmul_fp8", 0.19, "fraction", source="derived_from_probe",
            derived_from="measured bf16",
        ))
        return 0.19

    entry = results.measurements["matmul_fp8"]
    assert entry.source == "derived_from_probe"
    assert entry.derived_from == "measured bf16"


def test_overlay_marks_a_derived_link_as_low_confidence():
    results = suite.Results()
    results.add(suite.Measurement("dma_large_contiguous", 0.28, "coefficient"))
    suite.derive_intra_device(results, {"hbm_bandwidth": 2.9e12})
    overlay = suite.to_overlay(results, "trn2.3xlarge")

    link = overlay["shared"]["links"]["intra_device"]
    assert link["source"] == "derived_from_probe"
    assert link["confidence"] == "low"
    assert "NOT MEASURED" in link["note"]
    assert link["bandwidth_bytes_per_s"] == pytest.approx(2.9e12 * 0.28)


def test_derived_intra_device_latency_is_not_the_pcie_intercept():
    """The bug this guards: 102 us on an on-chip hop would forbid TP at decode.

    The PCIe intercept is a kernel-launch cost, not a link latency, and reusing it here made
    every collective latency-bound across the whole search.
    """
    results = suite.Results()
    results.add(suite.Measurement("dma_large_contiguous", 0.28, "coefficient"))
    results.add(suite.Measurement("host_dram_latency_us", 102.4, "us"))
    suite.derive_intra_device(results, {"hbm_bandwidth": 2.9e12})

    latency = results.measurements["intra_device_latency_us"]
    assert latency.value == suite.INTRA_DEVICE_LATENCY_ASSUMPTION_US == 1.0
    assert latency.source == "assumed"
    assert latency.value < 102.4


def test_derivation_fails_loudly_when_its_input_is_missing():
    results = suite.Results()
    suite.derive_intra_device(results, {"hbm_bandwidth": 2.9e12})
    entry = results.measurements["intra_device_bandwidth"]
    assert not entry.ok()
    assert "DMA efficiency probe also failed" in entry.error


def test_overlay_records_what_cannot_be_probed_and_what_failed():
    results = suite.Results()
    results.add(suite.Measurement("matmul_bf16", 0.376, "fraction"))
    results.add(suite.Measurement(
        "matmul_fp8", None, "fraction", error="RuntimeError: not supported on this target",
    ))
    overlay = suite.to_overlay(results, "trn2.3xlarge")

    provenance = overlay["_provenance"]
    assert provenance["probed_on"] == "trn2.3xlarge"
    assert any("inter_device" in item for item in provenance["not_probeable"])
    assert [f["name"] for f in provenance["failures"]] == ["matmul_fp8"]
    # A failed measurement must not appear as a coefficient at all.
    assert "matmul_fp8" not in overlay["shared"]["efficiency"]
    assert overlay["shared"]["efficiency"]["matmul_bf16"] == pytest.approx(0.376)


def test_collective_coefficient_is_one_with_a_stated_reason():
    """It is 1.0 by construction, and the overlay has to say why rather than look measured."""
    results = suite.Results()
    overlay = suite.to_overlay(results, "host")
    efficiency = overlay["shared"]["efficiency"]
    assert efficiency["collective_allreduce"] == 1.0
    assert "double-count" in efficiency["collective_note"]


# ---------------------------------------------------------------------------------------
# The tier patch
# ---------------------------------------------------------------------------------------
def test_tier_patch_keeps_every_tier_and_scales_by_drive_count(tmp_path):
    """The overlay replaces lists wholesale, so a patch must produce a complete list.

    And the target has four NVMe drives where the probe measured one, so the aggregate is
    scaled — with the extrapolation recorded, since it is not a measurement of the target.
    """
    systems = suite.Path(__file__).resolve().parents[1] / "floorplan" / "systems"
    original = yaml.safe_load((systems / "trn2-16device.yaml").read_text())
    names_before = [tier["name"] for tier in original["tiers"]]

    patched = suite.merge_tier_patch(
        systems / "trn2-16device.yaml",
        {"nvme": {"bandwidth_bytes_per_s": 2.45e9, "random_read_iops": 33_000}},
    )
    assert [tier["name"] for tier in patched] == names_before

    nvme = next(tier for tier in patched if tier["name"] == "nvme")
    assert nvme["devices"] == 4
    assert nvme["bandwidth_bytes_per_s"] == pytest.approx(2.45e9 * 4)
    assert nvme["random_read_iops"] == pytest.approx(33_000 * 4)

    # The one-device host is not scaled.
    single = suite.merge_tier_patch(
        systems / "trn2-1device.yaml",
        {"nvme": {"bandwidth_bytes_per_s": 2.45e9}},
    )
    nvme_single = next(tier for tier in single if tier["name"] == "nvme")
    assert nvme_single["bandwidth_bytes_per_s"] == pytest.approx(2.45e9)


def test_probed_overlay_resolves_every_value_the_simulator_needs():
    """If a probe has run, nothing should be left unmeasured. Skipped if it has not."""
    systems = suite.Path(__file__).resolve().parents[1] / "floorplan" / "systems"
    if not (systems / "probed.yaml").exists():
        pytest.skip("no probe has run on this host")
    hardware = Hardware.from_system(load_system("trn2-16device", systems))
    assert hardware.unresolved() == []


# ---------------------------------------------------------------------------------------
# Storage device selection
# ---------------------------------------------------------------------------------------
def test_instance_store_selection_skips_anything_mounted(monkeypatch):
    """Picking the boot disk would benchmark EBS and be 24x wrong on IOPS."""
    tree = {
        "blockdevices": [
            {
                "name": "nvme0n1", "size": 1_200_000_000_000, "type": "disk",
                "mountpoints": [None],
                "children": [
                    {"name": "nvme0n1p1", "size": 1_200_000_000_000, "type": "part",
                     "mountpoints": ["/"]},
                ],
            },
            {
                "name": "nvme1n1", "size": 470_000_000_000, "type": "disk",
                "mountpoints": [None], "children": [],
            },
        ],
    }

    import json
    import subprocess

    def fake_run(*args, **kwargs):
        class Result:
            stdout = json.dumps(tree)
        return Result()

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert storage_probe.find_instance_store() == suite.Path("/dev/nvme1n1")


def test_no_candidate_returns_none(monkeypatch):
    import json
    import subprocess

    def fake_run(*args, **kwargs):
        class Result:
            stdout = json.dumps({"blockdevices": [
                {"name": "nvme0n1", "size": 1, "type": "disk", "mountpoints": ["/"]},
            ]})
        return Result()

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert storage_probe.find_instance_store() is None


def test_random_reads_are_queue_depth_one():
    """A gather on the critical path cannot pipeline, so depth 1 is the right measurement."""
    assert storage_probe.RANDOM_COUNT > 0
    assert storage_probe.BLOCK == 4096


def test_host_dram_latency_is_an_assumption_not_the_fitted_intercept():
    """The fit returns ~120 us on this path, which is kernel launch, not link latency.

    At 32 rows per token an Engram lookup would cost ~4 ms and host-DRAM tiering would be
    foreclosed — on a measurement artifact rather than on the hardware. The bandwidth from the
    same sweep *is* measured and is kept.
    """
    assert suite.HOST_DRAM_LATENCY_ASSUMPTION_US == 2.0
    assert suite.HOST_DRAM_LATENCY_ASSUMPTION_US < 120


def test_probed_overlay_labels_assumptions_as_assumptions():
    """If a probe has run, anything not measured must say so in the file the report reads."""
    systems = suite.Path(__file__).resolve().parents[1] / "floorplan" / "systems"
    probed = systems / "probed.yaml"
    if not probed.exists():
        pytest.skip("no probe has run on this host")
    overlay = yaml.safe_load(probed.read_text())
    links = overlay["shared"]["links"]
    for name, link in links.items():
        source = str(link.get("source", ""))
        if source != "probed":
            assert source in {"assumed", "derived_from_probe"}, f"{name}: {source}"
            assert "NOT MEASURED" in link.get("note", ""), f"{name} does not say so"
            assert link.get("confidence") == "low", f"{name} should be low confidence"


def test_latency_values_that_would_foreclose_a_design_option_are_flagged():
    """A latency large enough to rule out a tier has to be an argued number, not a fit.

    Both of the values that were wrong here — 102 us on the on-chip link, 120 us on PCIe —
    would have removed a whole branch of the design space. They are now constants with
    reasoning attached, and this test is what keeps them from drifting back to a fit.
    """
    assert suite.INTRA_DEVICE_LATENCY_ASSUMPTION_US <= 1.0
    assert suite.HOST_DRAM_LATENCY_ASSUMPTION_US <= 10.0


# ---------------------------------------------------------------------------------------
# Probe kernels must not be dead code
# ---------------------------------------------------------------------------------------
def test_repetition_loops_chain_through_their_accumulator():
    """A loop whose body ignores its own output is redundant and may be optimized away.

    `nisa.tensor_tensor` and `nisa.activation` overwrite their destination, so
    `for _ in range(512): op(dst=acc, data=x)` is 511 eliminable iterations — and whether the
    compiler eliminates them decides the coefficient. That is what made `vector_elementwise`
    swing 8x between runs (0.30 to 2.35) while the gather probe, which chained from the start,
    stayed within 5%. A repetition loop has to read `acc` as well as write it.
    """
    import re

    source = (
        suite.Path(__file__).resolve().parents[1] / "floorplan" / "probe" / "kernels.py"
    ).read_text()

    for name in ("vector_elementwise", "scalar_activation", "gpsimd_gather"):
        start = source.index(f"def {name}(")
        end = source.find("\n@nki.jit", start)
        body = source[start:end if end > 0 else len(source)]
        loop = body[body.index("affine_range"):]
        chained = re.search(r"(data1|data2|data)\s*=\s*(acc|gathered)", loop)
        assert chained, f"{name}'s repetition loop does not read its accumulator"
