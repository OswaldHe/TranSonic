# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Projecting a floorplan placement onto one device.

The rule these pin down is the one the whole pipeline's fidelity rests on. It decides that a rank
is 1/4 of an MoE rather than 1/8, which decides what the submodule is, which decides whether the
"+10% over the submodule" bound is even arithmetically reachable. Getting the *dropping order*
wrong (a weight dim before an activation dim) would silently double per-core weight residency
beyond what the projection already costs.
"""

from __future__ import annotations

import textwrap

import pytest

from optimization import PROJECTION_TARGET_UNITS
from optimization.projection import (
    WEIGHT_PARTITION_DIMS,
    Factor,
    Projection,
    ProjectionError,
    project,
    project_module,
)

pytestmark = pytest.mark.optimization


def test_weight_dims_match_the_floorplan_schema():
    """The duplicated vocabulary has to stay equal to the floorplan's.

    `projection.py` duplicates `WEIGHT_PARTITION_DIMS` so a projection can be computed from a
    scheme file alone, without the floorplan package or its probe data. A divergence would make the
    projection drop the wrong factor.
    """
    from floorplan.schema import WEIGHT_PARTITION_DIMS as upstream

    assert WEIGHT_PARTITION_DIMS == upstream


def test_moe_narrows_the_expert_split():
    """`expert x8` across two devices becomes `expert x4` on one."""
    result = project("layers.1.ffn", [{"dim": "expert", "factor": 8, "collective": "all_to_all"}],
                     [f"d{d}.l{l}" for d in (0, 1) for l in range(4)])
    assert result.projected_units == 4
    assert [(f.dim, f.factor) for f in result.projected] == [("expert", 4)]
    assert result.diverges
    assert result.shard_fraction == "1/4"
    # The number the floorplan's own argument turns on: a projected core holds twice the weights.
    assert result.weight_residency_ratio() == pytest.approx(2.0)
    # The collective survives the narrowing — how the shards rejoin does not change.
    assert result.projected[0].collective == "all_to_all"


def test_attention_drops_the_batch_factor_not_the_head_factor():
    """`head x4 * batch x2` becomes `head x4`, not `head x2 * batch x2`.

    A batch split divides activations and leaves every participant holding the module's full
    weights, so dropping it costs no per-core capacity. Narrowing the head factor instead would
    double the weight each core streams — the opposite of what the plan was optimizing for.
    """
    result = project(
        "layers.10.attention",
        [{"dim": "head", "factor": 4, "collective": "allreduce"},
         {"dim": "batch", "factor": 2, "collective": "none"}],
        [f"d{d}.l{l}" for d in (0, 1) for l in range(4)],
    )
    assert [(f.dim, f.factor) for f in result.projected] == [("head", 4)]
    assert [f.dim for f in result.dropped] == ["batch"]
    assert result.weight_residency_ratio() == pytest.approx(1.0)


def test_engram_sixteen_units_across_four_devices():
    """`ngram x8 * batch x2` on 16 units becomes `ngram x4`."""
    result = project(
        "layers.1.engram",
        [{"dim": "ngram", "factor": 8, "collective": "allgather"},
         {"dim": "batch", "factor": 2, "collective": "none"}],
        [f"d{d}.l{l}" for d in range(4) for l in range(4)],
    )
    assert result.planned_units == 16
    assert result.planned_devices == 4
    assert [(f.dim, f.factor) for f in result.projected] == [("ngram", 4)]


def test_a_single_device_placement_is_passed_through():
    """Already four units: used unchanged, and reported as not diverging.

    Projecting *up* to fill the device would invent parallelism the plan did not ask for.
    """
    result = project("layers.1.hc_attn_in",
                     [{"dim": "hidden", "factor": 4, "collective": "allreduce"}],
                     [f"d0.l{l}" for l in range(4)])
    assert [(f.dim, f.factor) for f in result.projected] == [("hidden", 4)]
    assert not result.diverges
    assert result.dropped == [] and result.scaled == []


def test_a_narrower_placement_is_not_widened():
    result = project("m", [{"dim": "hidden", "factor": 2}], ["d0.l0", "d0.l1"])
    assert result.projected_units == 2
    assert not result.diverges


def test_an_unsplit_placement_is_left_alone():
    result = project("m", [], ["d0.l0"])
    assert result.projected == []
    assert result.projected_units == 1
    assert not result.diverges


def test_a_one_way_split_is_dropped_rather_than_kept():
    """Narrowing to factor 1 removes the split: a 1-way split is not a split."""
    result = project("m", [{"dim": "hidden", "factor": 8}], [f"d0.l{i}" for i in range(8)],
                     target_units=1)
    assert result.projected == []
    assert [f.dim for f in result.dropped] == ["hidden"]


def test_a_factor_with_no_small_enough_divisor_raises():
    """A prime factor wider than the target cannot be projected, and says so."""
    with pytest.raises(ProjectionError, match="discard all of its parallelism"):
        project("m", [{"dim": "hidden", "factor": 7}], [f"d0.l{i}" for i in range(7)],
                target_units=4)


def test_target_units_must_be_positive():
    with pytest.raises(ProjectionError, match="target_units"):
        project("m", [], ["d0.l0"], target_units=0)


def test_describe_names_the_divergence_and_its_cost():
    """The prose the submodule repo carries has to say what was given up, not just what was chosen.

    An agent told only "split the experts four ways" has no way to know it is working against the
    runner-up plan, and neither does a reader of the report.
    """
    text = project("layers.1.ffn", [{"dim": "expert", "factor": 8, "collective": "all_to_all"}],
                   [f"d{d}.l{l}" for d in (0, 1) for l in range(4)],
                   target="trn2-16device").describe()
    assert "8 logical NeuronCore(s) spanning 2 device(s)" in text
    assert "expertx4" in text
    assert "1/4" in text
    assert "2x the weight bytes" in text
    assert "ranked second" in text


def test_describe_says_nothing_was_given_up_when_nothing_was():
    text = project("m", [{"dim": "hidden", "factor": 4}], [f"d0.l{i}" for i in range(4)]).describe()
    assert "nothing was given up" in text
    assert "gave up" not in text.split("nothing was given up")[0]


def test_project_module_reads_a_scheme(tmp_path):
    scheme = tmp_path / "rank1.yaml"
    scheme.write_text(textwrap.dedent("""
        version: 1
        target: trn2-16device
        placements:
          - module: layers.1.ffn
            units: [d0.l0, d1.l0, d0.l1, d1.l1, d0.l2, d1.l2, d0.l3, d1.l3]
            splits:
              - {dim: expert, factor: 8, collective: all_to_all}
          - module: embed
            units: [d0.l0, d0.l1, d0.l2, d0.l3]
            splits:
              - {dim: vocab, factor: 4, collective: allreduce}
    """))
    result = project_module(scheme, "layers.1.ffn")
    assert result.target == "trn2-16device"
    assert result.projected_units == PROJECTION_TARGET_UNITS
    assert not project_module(scheme, "embed").diverges


def test_project_module_names_the_available_modules_when_one_is_missing(tmp_path):
    scheme = tmp_path / "rank1.yaml"
    scheme.write_text(
        "version: 1\ntarget: t\nplacements:\n  - module: embed\n    units: [d0.l0]\n"
    )
    with pytest.raises(ProjectionError, match="embed"):
        project_module(scheme, "layers.99.ffn")


def test_a_file_that_is_not_a_scheme_is_rejected(tmp_path):
    path = tmp_path / "notascheme.yaml"
    path.write_text("hello: world\n")
    with pytest.raises(ProjectionError, match="floorplan scheme"):
        project_module(path, "anything")


def test_the_real_scheme_projects_every_module_it_places():
    """Every placement in the shipped DeepSeek scheme has to be projectable.

    The check that matters for usability: if any of the 271 placements raised, the pipeline would be
    unusable for that module and the failure would only surface at `init`.
    """
    scheme = (
        "/home/ubuntu/workspace/floorplan-deepseek/schemes/rank1.yaml"
    )
    import pathlib

    if not pathlib.Path(scheme).is_file():
        pytest.skip("the local floorplan project is not present")

    import yaml

    data = yaml.safe_load(pathlib.Path(scheme).read_text())
    modules = sorted({p["module"] for p in data["placements"]})
    for module in modules:
        result = project_module(scheme, module)
        assert 1 <= result.projected_units <= PROJECTION_TARGET_UNITS, module


# ======================================================================================
# the review of #6: reading a recorded projection back instead of recomputing it
# ======================================================================================


def test_a_recorded_projection_round_trips_exactly():
    """Later stages and the report read this back rather than re-planning from the scheme."""
    original = Projection(
        module="layers.1.ffn",
        planned=[Factor("expert", 8, "all_to_all")], planned_units=8, planned_devices=2,
        projected=[Factor("expert", 4, "all_to_all")], projected_units=4,
        scaled=[(Factor("expert", 8, "all_to_all"), Factor("expert", 4, "all_to_all"))],
        target="trn2-16device",
    )
    assert Projection.from_dict(original.to_dict()).to_dict() == original.to_dict()


def test_a_dropped_factor_survives_the_round_trip():
    original = Projection(
        module="m", planned=[Factor("head", 4), Factor("batch", 2)], planned_units=8,
        planned_devices=2, projected=[Factor("head", 4)], projected_units=4,
        dropped=[Factor("batch", 2)],
    )
    assert Projection.from_dict(original.to_dict()).dropped == [Factor("batch", 2)]


def test_a_same_width_different_dimension_is_a_difference():
    """The old drift guard compared only the unit count, so `expert x4` -> `head x4` passed it —
    four ranks either way, and an entirely different kernel."""
    was = Projection(module="m", planned=[], planned_units=4, planned_devices=1,
                     projected=[Factor("expert", 4)], projected_units=4)
    now = Projection(module="m", planned=[], planned_units=4, planned_devices=1,
                     projected=[Factor("head", 4)], projected_units=4)
    assert was.differences(now)


def test_a_changed_collective_is_a_difference():
    was = Projection(module="m", planned=[], planned_units=4, planned_devices=1,
                     projected=[Factor("expert", 4, "all_to_all")], projected_units=4)
    now = Projection(module="m", planned=[], planned_units=4, planned_devices=1,
                     projected=[Factor("expert", 4, "allreduce")], projected_units=4)
    assert was.differences(now)


def test_an_identical_projection_reports_no_difference():
    one = Projection(module="m", planned=[Factor("expert", 8)], planned_units=8, planned_devices=2,
                     projected=[Factor("expert", 4)], projected_units=4)
    assert one.differences(Projection.from_dict(one.to_dict())) == []


def test_an_unreadable_factor_label_is_refused():
    with pytest.raises(ProjectionError):
        Factor.parse("not a factor")
