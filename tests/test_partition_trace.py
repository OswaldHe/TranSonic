# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for hook-based tracing, argument encoding, and replay."""

import pytest

from model_partition.runtime.module_runner import (
    ReplayError,
    TraceBundle,
    decode_call,
    expected_output,
    first_tensor,
    replay_record,
)
from model_partition.storage import DumpPolicy
from model_partition.trace import (
    TENSOR_KEY,
    UNSUPPORTED_KEY,
    CallRecord,
    Tracer,
    decode_value,
    forward_no_cache,
    slice_for_dump,
)
from model_partition.verify.numerics import compare

torch = pytest.importorskip("torch")


def decoder_module(run) -> str:
    """Id of a decoder module in this run's plan (grouping depends on budget)."""
    return next(m.id for m in run.graph.partitioned_modules if m.kind == "decoder_layers")


# -- argument encoding -------------------------------------------------------


def test_every_module_produces_records(tiny_run):
    traced = {r.module_id for r in tiny_run.bundle.records}
    planned = {m.id for m in tiny_run.graph.partitioned_modules}
    assert traced == planned


def test_records_cover_every_sample(tiny_run):
    assert tiny_run.bundle.sample_ids() == ["s0", "s1"]
    for sample_id in tiny_run.sample_ids:
        assert tiny_run.bundle.select(sample_id=sample_id)


def test_no_unresolved_submodules(tiny_run):
    tracer = Tracer(tiny_run.build_model(), tiny_run.graph, tiny_run.bundle.store)
    assert tracer.unresolved_submodules() == []


def test_tensor_args_are_dumped_by_reference(tiny_run):
    record = tiny_run.bundle.select(module_id=decoder_module(tiny_run), sample_id="s0")[0]
    assert record.args and TENSOR_KEY in record.args[0]
    assert record.tensor_names()


def test_records_round_trip_through_yaml(tiny_run):
    reloaded = TraceBundle.load(tiny_run.layout.trace_dir)
    assert len(reloaded.records) == len(tiny_run.bundle.records)
    assert reloaded.module_ids() == tiny_run.bundle.module_ids()
    assert reloaded.weights.keys() == tiny_run.bundle.weights.keys()


def test_decode_value_rebuilds_nested_structures():
    encoded = {"__list__": [1, {"__tuple__": [{TENSOR_KEY: "t"}, None]}, {"k": True}]}
    result = decode_value(encoded, lambda name: f"<{name}>")
    assert result == [1, ("<t>", None), {"k": True}]


def test_decode_value_turns_unsupported_into_none():
    assert decode_value({UNSUPPORTED_KEY: "DynamicCache"}, lambda n: n) is None


def test_unsupported_marker_blocks_replay_instead_of_faking_it(tiny_run):
    """Regression: a cache object must fail loudly, not be silently replaced."""
    record = tiny_run.bundle.select(module_id=decoder_module(tiny_run), sample_id="s0")[0]
    record.kwargs = {"past_key_values": {UNSUPPORTED_KEY: "HybridCache"}}
    assert record.has_unsupported()
    with pytest.raises(ReplayError, match="unserializable value"):
        replay_record(tiny_run.build_model(), record, tiny_run.bundle.store)


def test_real_trace_has_no_unsupported_values(tiny_run):
    """Tracing disables caching, so nothing unserializable should be recorded."""
    assert not any(r.has_unsupported() for r in tiny_run.bundle.records)


def test_forward_no_cache_falls_back_when_unsupported(tiny_run):
    """The toy model's forward takes no use_cache kwarg."""
    model = tiny_run.build_model()
    out = forward_no_cache(model, torch.zeros(1, 4, dtype=torch.long))
    assert out.shape[0] == 1


# -- replay ------------------------------------------------------------------


def test_replay_reproduces_every_module_exactly(tiny_run):
    model = tiny_run.build_model()
    for record in tiny_run.bundle.records:
        actual = first_tensor(replay_record(model, record, tiny_run.bundle.store))
        reference = first_tensor(expected_output(record, tiny_run.bundle.store))
        assert compare(actual, reference, record.module_id).passed


def test_decode_call_returns_positional_args_not_dict_keys(tiny_run):
    """Regression: args were double-wrapped, so decoding yielded dict keys."""
    record = tiny_run.bundle.select(module_id=decoder_module(tiny_run), sample_id="s0")[0]
    args, kwargs = decode_call(record, tiny_run.bundle.store)
    assert isinstance(args, tuple) and args
    assert torch.is_tensor(args[0])
    assert isinstance(kwargs, dict)


def test_replay_of_absent_submodule_raises(tiny_run):
    record = tiny_run.bundle.select(module_id=decoder_module(tiny_run), sample_id="s0")[0]
    record.submodule = "model.layers.999"
    with pytest.raises(ReplayError, match="absent from this model"):
        replay_record(tiny_run.build_model(), record, tiny_run.bundle.store)


def test_missing_tensor_in_manifest_raises(tiny_run):
    record = tiny_run.bundle.select(module_id=decoder_module(tiny_run), sample_id="s0")[0]
    record.args = [{TENSOR_KEY: "not-in-manifest"}]
    with pytest.raises(ReplayError, match="not in the manifest"):
        replay_record(tiny_run.build_model(), record, tiny_run.bundle.store)


# -- weight dumps ------------------------------------------------------------


def test_weights_dumped_for_every_module(tiny_run):
    for module in tiny_run.graph.partitioned_modules:
        assert tiny_run.bundle.weights.get(module.id), module.id


def test_weight_blobs_verify_against_their_digests(tiny_run):
    store = tiny_run.bundle.store
    for entry in store.find(role="weight"):
        store.verify(entry)


def test_moe_layers_dump_more_weights_than_dense(tiny_moe_run):
    """Odd layers carry experts, so their dumps are larger."""
    weights = tiny_moe_run.bundle.weights
    assert len(weights["layers.1"]) > len(weights["layers.0"])


# -- long-context slicing ----------------------------------------------------


def test_slice_for_dump_leaves_short_tensors_alone():
    tensor = torch.zeros(1, 128, 8)
    kept, info = slice_for_dump(tensor, 128, DumpPolicy())
    assert info is None and kept.shape == tensor.shape


def test_slice_for_dump_cuts_the_sequence_axis():
    tensor = torch.arange(16384 * 4, dtype=torch.float32).reshape(1, 16384, 4)
    kept, info = slice_for_dump(tensor, 16384, DumpPolicy())
    assert info is not None
    assert info.axis == 1 and info.head == 128 and info.tail == 128
    assert info.original_length == 16384
    assert kept.shape == (1, 256, 4)
    # Head and tail windows, not a resample.
    assert torch.equal(kept[0, :128], tensor[0, :128])
    assert torch.equal(kept[0, 128:], tensor[0, -128:])


def test_slice_for_dump_ignores_axes_that_merely_match_by_accident():
    """Only an axis equal to the sequence length is sliced."""
    tensor = torch.zeros(2048, 4)
    kept, info = slice_for_dump(tensor, 2048, DumpPolicy())
    assert info is not None and info.axis == 0
    weights = torch.zeros(64, 64)
    kept2, info2 = slice_for_dump(weights, 2048, DumpPolicy())
    assert info2 is None and kept2.shape == (64, 64)


def test_full_dumps_disable_slicing():
    tensor = torch.zeros(1, 4096, 4)
    kept, info = slice_for_dump(tensor, 4096, DumpPolicy(full_dumps=True))
    assert info is None and kept.shape == tensor.shape


def test_slicing_is_recorded_in_the_manifest(tiny_run):
    """A sliced dump must be marked so a later comparison can align to it."""
    from model_partition.tensorstore import SliceInfo

    store = tiny_run.bundle.store
    meta = store.write("sliced", torch.zeros(1, 256, 4), role="output",
                       module_id="layers.0", sample_id="s0",
                       slice_info=SliceInfo(axis=1, head=128, tail=128, original_length=16384))
    assert meta.slice_info["original_length"] == 16384


# -- bundle selection --------------------------------------------------------


def test_select_filters_and_orders(tiny_run):
    records = tiny_run.bundle.select(sample_id="s0")
    assert [r.order for r in records] == sorted(r.order for r in records)
    assert all(r.sample_id == "s0" for r in records)


def test_select_by_step(tiny_run):
    assert tiny_run.bundle.select(step=0)
    assert tiny_run.bundle.select(step=99) == []


def test_call_record_dict_round_trip():
    record = CallRecord(module_id="m", submodule="s", sample_id="x", step=2,
                        args=[{TENSOR_KEY: "a"}], kwargs={"flag": True},
                        output={TENSOR_KEY: "b"}, order=7)
    assert CallRecord.from_dict(record.to_dict()) == record
