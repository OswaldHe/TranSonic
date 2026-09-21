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
    kept, info = slice_for_dump(tensor, 16384, DumpPolicy(slice_long=True))
    assert info is not None
    assert info.axes == [1] and info.head == 128 and info.tail == 128
    assert info.original_length == 16384
    assert kept.shape == (1, 256, 4)
    # Head and tail windows, not a resample.
    assert torch.equal(kept[0, :128], tensor[0, :128])
    assert torch.equal(kept[0, 128:], tensor[0, -128:])


def test_slice_for_dump_only_touches_sequence_axes():
    """Only an axis equal to the sequence length is windowed."""
    policy = DumpPolicy(slice_long=True)
    kept, info = slice_for_dump(torch.zeros(2048, 4), 2048, policy)
    assert info is not None and info.axes == [0] and kept.shape == (256, 4)
    weights = torch.zeros(64, 64)
    kept2, info2 = slice_for_dump(weights, 2048, policy)
    assert info2 is None and kept2.shape == (64, 64)


def test_slice_for_dump_windows_every_sequence_axis():
    """A mask square in the sequence must stay square, or it describes nothing."""
    mask = torch.zeros(1, 1, 2048, 2048)
    kept, info = slice_for_dump(mask, 2048, DumpPolicy(slice_long=True))
    assert info.axes == [2, 3]
    assert kept.shape == (1, 1, 256, 256)


def test_windowing_is_off_by_default():
    tensor = torch.zeros(1, 4096, 4)
    kept, info = slice_for_dump(tensor, 4096, DumpPolicy())
    assert info is None and kept.shape == tensor.shape


def test_slicing_is_recorded_in_the_manifest(tiny_run):
    """A sliced dump must be marked so a later comparison can align to it."""
    from model_partition.tensorstore import SliceInfo

    store = tiny_run.bundle.store
    meta = store.write("sliced", torch.zeros(1, 256, 4), role="output",
                       module_id="layers.0", sample_id="s0",
                       slice_info=SliceInfo(axes=[1], head=128, tail=128,
                                            original_length=16384))
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


def test_dump_budget_is_enforced_not_just_declared(tiny_run):
    """Regression: max_total_bytes was documented as a hard abort but ignored."""
    from model_partition.tensorstore import TensorStore
    from model_partition.trace import DumpBudgetExceeded

    store = TensorStore(tiny_run.root / "capped")
    tracer = Tracer(tiny_run.build_model(), tiny_run.graph, store,
                    policy=DumpPolicy(max_total_bytes=1024))
    with pytest.raises(DumpBudgetExceeded, match="max_total_bytes"):
        tracer.dump_weights()
    assert tracer.bytes_written > 1024


def test_bytes_written_is_tracked_without_a_ceiling(tiny_run):
    from model_partition.tensorstore import TensorStore

    store = TensorStore(tiny_run.root / "counted")
    tracer = Tracer(tiny_run.build_model(), tiny_run.graph, store)
    tracer.dump_weights()
    assert tracer.bytes_written == sum(e.nbytes for e in store.entries)


def test_a_derived_buffer_is_recorded_as_the_forward_saw_it(tiny_run):
    """The copy a module hands out afterwards is not always the one that ran.

    An offloaded model returns a bfloat16 view of a rotary `inv_freq` that executed in
    float32, and a module rebuilt from that view drifts as positions grow. So the
    buffer is captured during the traced call, not read back later.
    """
    import torch

    from model_partition.tensorstore import TensorStore

    from model_partition.trace import _lookup

    model = tiny_run.build_model()
    target = next(m for m in tiny_run.graph.partitioned_modules if m.submodules)
    owner = _lookup(model, target.submodules[0])
    executed = torch.full((4,), 0.6042963862, dtype=torch.float32)
    owner.register_buffer("derived_probe", executed.clone(), persistent=False)

    store = TensorStore(tiny_run.root / "derived")
    tracer = Tracer(model, tiny_run.graph, store)
    from tests.fixtures.tiny_llm import sample_inputs

    tracer.trace_sample("s", sample_inputs(1, 8))
    # Whatever the module hands out now is a different, coarser copy.
    owner.derived_probe = executed.to(torch.bfloat16).float()

    # dump_weights leaves derived buffers alone; dump_derived takes the captured ones.
    assert not any("derived_probe" in n
                   for n in tracer.dump_weights(module_ids=[target.id])[target.id])
    names = tracer.dump_derived()[target.id]
    probe = next(n for n in names if n.endswith("derived_probe"))
    entry = next(e for e in store.entries if e.name == probe)
    recorded = store.read_torch(entry)
    assert recorded.dtype is torch.float32
    assert recorded[0].item() == pytest.approx(0.6042963862, abs=1e-9)


# -- entry points the main forward does not reach -----------------------------


def _drafting_model():
    """A backbone plus a draft stack its own ``forward`` never calls.

    Both DeepSeek V4 models are shaped this way: the MTP/DSpark blocks are in the
    checkpoint and in the module tree, and `Transformer.forward` runs the backbone
    alone. One is driven by a second method on the model, the other by calling the
    block with the backbone's last hidden state, so both conventions are here.
    """
    import torch

    class Draft(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(4, 4)

        def forward(self, hidden, start_pos, input_ids):
            return self.proj(hidden) + start_pos + input_ids.sum()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(16, 4)
            self.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(3)])
            self.mtp = torch.nn.ModuleList([Draft()])

        def forward(self, input_ids, start_pos=0):
            hidden = self.embed(input_ids)
            for layer in self.layers:
                hidden = layer(hidden)
            return hidden.sum(-1), hidden

        def forward_spec(self, input_ids, last_hidden, start_pos):
            return self.mtp[0](last_hidden, start_pos, input_ids)

    return Model()


def _drafting_graph():
    from model_partition.planner.graph import ModuleNode, PartitionGraph

    return PartitionGraph(model="drafting", modules=[
        ModuleNode(id="00-backbone", kind="decoder_layers",
                   submodules=[f"layers.{i}" for i in range(3)]),
        ModuleNode(id="01-draft", kind="mtp", submodules=["mtp.0"]),
    ])


def _trace_spec(**kwargs):
    from model_partition.spec import parse_spec

    return parse_spec({"source": "/tmp/x", "name": "t", "trace": kwargs}).trace


def test_a_module_the_forward_never_calls_records_nothing(tiny_run):
    """The gap the extra passes exist to close, stated as a fact about hooks."""
    from model_partition.tensorstore import TensorStore

    tracer = Tracer(_drafting_model(), _drafting_graph(),
                    TensorStore(tiny_run.root / "unreached"))
    tracer.trace_sample("s", torch.zeros(1, 5, dtype=torch.long))
    assert {r.module_id for r in tracer.records} == {"00-backbone"}


def test_an_extra_pass_reaches_the_draft_stack(tiny_run):
    """A second entry point, driven with what the first returned."""
    from model_partition.tensorstore import TensorStore

    store = TensorStore(tiny_run.root / "extra")
    tracer = Tracer(_drafting_model(), _drafting_graph(), store)
    spec = _trace_spec(returns=["logits", "last_hidden"], extra_passes=[
        {"entry": "forward_spec", "args": ["input_ids", "last_hidden", "start_pos"],
         "decode": True},
    ])
    input_ids = torch.zeros(1, 5, dtype=torch.long)
    tracer.trace_sample("s", input_ids)
    produced = tracer.trace_extra_passes("s", input_ids, spec, only={"mtp.0"})

    assert [r.module_id for r in produced] == ["01-draft"]
    # A decode step: one position, at the one after the prompt, and its own step
    # number so nothing confuses it with the prefill's records.
    assert produced[0].step == 1
    recorded = store.read_torch(next(e for e in store.entries
                                     if e.name == produced[0].output[TENSOR_KEY]))
    assert recorded.shape == (1, 1, 4)


def test_an_extra_pass_records_only_what_it_was_asked_for(tiny_run):
    """The backbone runs again to reach the draft stack; it is already traced."""
    from model_partition.tensorstore import TensorStore

    tracer = Tracer(_drafting_model(), _drafting_graph(),
                   TensorStore(tiny_run.root / "only"))
    spec = _trace_spec(returns=["logits", "last_hidden"], extra_passes=[
        {"entry": "forward_spec", "args": ["input_ids", "last_hidden", "start_pos"],
         "decode": True},
    ])
    input_ids = torch.zeros(1, 5, dtype=torch.long)
    tracer.trace_sample("s", input_ids)
    before = len(tracer.records)
    tracer.trace_extra_passes("s", input_ids, spec, only={"mtp.0"})
    assert len(tracer.records) == before + 1


def test_an_extra_pass_can_take_the_stack_hidden_state(tiny_run):
    """What no forward returns: the hidden state the last layer produced."""
    from model_partition.tensorstore import TensorStore

    tracer = Tracer(_drafting_model(), _drafting_graph(),
                    TensorStore(tiny_run.root / "hidden"))
    spec = _trace_spec(extra_passes=[
        {"entry": "mtp.0", "args": ["hidden", "start_pos", "input_ids"], "decode": True},
    ])
    input_ids = torch.zeros(1, 5, dtype=torch.long)
    tracer.trace_sample("s", input_ids)
    produced = tracer.trace_extra_passes("s", input_ids, spec, only={"mtp.0"})
    assert [r.module_id for r in produced] == ["01-draft"]


def test_an_extra_pass_naming_nothing_on_the_model_says_so(tiny_run):
    from model_partition.tensorstore import TensorStore

    from model_partition.trace import TraceError

    tracer = Tracer(_drafting_model(), _drafting_graph(),
                    TensorStore(tiny_run.root / "absent"))
    spec = _trace_spec(extra_passes=[{"entry": "forward_draft", "args": ["input_ids"]}])
    with pytest.raises(TraceError, match="no callable"):
        tracer.trace_extra_passes("s", torch.zeros(1, 5, dtype=torch.long), spec)


def test_an_argument_the_forward_does_not_produce_is_reported(tiny_run):
    """Naming the returns wrongly is a spec mistake, and it says which name."""
    from model_partition.tensorstore import TensorStore

    from model_partition.trace import TraceError

    tracer = Tracer(_drafting_model(), _drafting_graph(),
                    TensorStore(tiny_run.root / "wrong"))
    # The forward returns two values, so a third name binds to nothing.
    spec = _trace_spec(returns=["logits", "last_hidden", "draft_hidden"], extra_passes=[
        {"entry": "forward_spec", "args": ["input_ids", "draft_hidden", "start_pos"]},
    ])
    with pytest.raises(TraceError, match="draft_hidden"):
        tracer.trace_extra_passes("s", torch.zeros(1, 5, dtype=torch.long), spec)
