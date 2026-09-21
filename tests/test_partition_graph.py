# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the partition graph: validation, ordering, dedup, serialization."""

import pytest

from model_partition.planner.graph import (
    GraphError,
    ModuleNode,
    PartitionGraph,
    TensorRef,
)


def chain_graph(budget: int = 1 << 30) -> PartitionGraph:
    """embed -> layers.0-1 -> layers.2-3 -> head, 4 layers."""
    tensors = {
        name: TensorRef(name=name, shape=["batch", "seq", 128])
        for name in ("tokens", "h.0", "h.2", "h.4", "logits")
    }
    modules = [
        ModuleNode(id="embed", kind="embed", inputs=["tokens"], outputs=["h.0"],
                   submodules=["model.embed_tokens"], param_bytes=1024),
        ModuleNode(id="layers.0-1", kind="decoder_layers", inputs=["h.0"], outputs=["h.2"],
                   layer_indices=[0, 1], submodules=["model.layers.0", "model.layers.1"],
                   param_bytes=2048, code_signature="decoder"),
        ModuleNode(id="layers.2-3", kind="decoder_layers", inputs=["h.2"], outputs=["h.4"],
                   layer_indices=[2, 3], submodules=["model.layers.2", "model.layers.3"],
                   param_bytes=2048, code_signature="decoder"),
        ModuleNode(id="head", kind="lm_head", inputs=["h.4"], outputs=["logits"],
                   submodules=["lm_head"], param_bytes=512),
    ]
    return PartitionGraph(
        model="toy", modules=modules, tensors=tensors, entry_tensors=["tokens"],
        output_tensors=["logits"], budget_bytes=budget, num_layers=4,
    )


def test_valid_chain_passes_validation():
    graph = chain_graph()
    assert graph.validate() == []
    assert graph.topological_order() == ["embed", "layers.0-1", "layers.2-3", "head"]


def test_long_range_edge_orders_after_its_producer():
    """A kv_source_layer_ids-style skip edge must still order correctly."""
    graph = chain_graph()
    graph.tensors["kv.0"] = TensorRef(name="kv.0", kind="kv", shape=["batch", "seq", 64])
    graph.by_id("layers.0-1").outputs.append("kv.0")
    graph.by_id("head").inputs.append("kv.0")
    assert graph.validate() == []
    order = graph.topological_order()
    assert order.index("layers.0-1") < order.index("head")


def test_cycle_is_rejected():
    graph = chain_graph()
    graph.by_id("embed").inputs.append("h.4")
    with pytest.raises(GraphError, match="Cycle detected"):
        graph.validate()


def test_undefined_input_tensor_is_rejected():
    graph = chain_graph()
    graph.by_id("head").inputs.append("nowhere")
    with pytest.raises(GraphError, match="undefined tensor 'nowhere'"):
        graph.validate()


def test_duplicate_module_id_is_rejected():
    graph = chain_graph()
    graph.modules.append(ModuleNode(id="head", kind="norm"))
    with pytest.raises(GraphError, match="Duplicate module id"):
        graph.validate()


def test_tensor_with_two_producers_is_rejected():
    graph = chain_graph()
    graph.by_id("head").outputs.append("h.4")
    with pytest.raises(GraphError, match="produced by multiple modules"):
        graph.validate()


def test_module_over_budget_is_rejected():
    graph = chain_graph(budget=1024)
    with pytest.raises(GraphError, match="over the 1024-byte budget"):
        graph.validate()


def test_resident_bytes_sums_params_activations_kv():
    node = ModuleNode(id="m", kind="attention", param_bytes=10, activation_bytes=20, kv_bytes=30)
    assert node.resident_bytes == 60


def test_uncovered_layer_is_a_warning_not_an_error():
    graph = chain_graph()
    graph.num_layers = 6
    warnings = graph.validate()
    assert any("[4, 5]" in w for w in warnings)


def test_layer_claimed_by_two_decoder_modules_is_rejected():
    graph = chain_graph()
    graph.by_id("layers.2-3").layer_indices = [1, 2, 3]
    with pytest.raises(GraphError, match="assigned to multiple decoder modules"):
        graph.validate()


def test_split_layer_across_kinds_is_allowed():
    """attention | router | experts may share a layer index."""
    graph = chain_graph()
    graph.tensors["h.mid"] = TensorRef(name="h.mid")
    graph.by_id("layers.0-1").layer_indices = [0]
    graph.by_id("layers.0-1").outputs = ["h.mid"]
    graph.modules.append(ModuleNode(
        id="layers.1.experts", kind="moe_experts", inputs=["h.mid"], outputs=["h.2"],
        layer_indices=[1], param_bytes=64, composition="parallel",
        submodules=["model.layers.1.mlp.experts.0", "model.layers.1.mlp.experts.1"],
    ))
    graph.modules.append(ModuleNode(
        id="layers.1.attn", kind="attention", inputs=["h.mid"], outputs=["attn.1"],
        layer_indices=[1], param_bytes=64, submodules=["model.layers.1.self_attn"],
    ))
    graph.tensors["attn.1"] = TensorRef(name="attn.1")
    assert graph.validate() == []


def test_signature_groups_deduplicate_identical_layers():
    groups = chain_graph().signature_groups()
    assert groups["decoder"] == ["layers.0-1", "layers.2-3"]
    assert groups["embed"] == ["embed"]


def test_unpartitioned_nodes_skip_budget_and_coverage():
    graph = chain_graph(budget=4096)
    graph.modules.append(ModuleNode(
        id="vision", kind="vision", inputs=["tokens"], outputs=["vis"],
        param_bytes=1 << 40, partitioned=False,
    ))
    graph.tensors["vis"] = TensorRef(name="vis")
    assert graph.validate() == []
    assert "vision" not in [m.id for m in graph.partitioned_modules]


def test_yaml_round_trip_preserves_structure(tmp_path):
    graph = chain_graph()
    graph.metadata = {"planner": "auto"}
    path = graph.save(tmp_path / "plan" / "graph.yaml")
    reloaded = PartitionGraph.load(path)
    assert reloaded.validate() == []
    assert reloaded.to_dict() == graph.to_dict()
    assert reloaded.metadata == {"planner": "auto"}


def test_unknown_kind_is_rejected():
    with pytest.raises(GraphError, match="unknown kind"):
        ModuleNode.from_dict({"id": "m", "kind": "quantum"})


def test_missing_required_key_is_rejected():
    with pytest.raises(GraphError, match="missing required key"):
        ModuleNode.from_dict({"id": "m"})


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(GraphError, match="not found"):
        PartitionGraph.load(tmp_path / "absent.yaml")


def test_tensor_bytes_per_token_ignores_symbolic_dims():
    ref = TensorRef(name="h", dtype="bfloat16", shape=["batch", "seq", 5120])
    assert ref.bytes_per_token() == 5120 * 2


# -- self-dependencies, composition, functional nodes ------------------------


def test_a_module_consuming_its_own_output_is_a_cycle():
    """A self-edge cannot be scheduled, so dropping it would hide the mistake."""
    graph = chain_graph()
    graph.by_id("layers.0-1").inputs.append("h.2")
    with pytest.raises(GraphError, match="Cycle detected"):
        graph.topological_order()
    with pytest.raises(GraphError, match="Cycle detected"):
        graph.validate()


def test_composition_defaults_to_sequential_and_round_trips():
    graph = chain_graph()
    assert graph.by_id("layers.0-1").composition == "sequential"
    graph.by_id("layers.0-1").composition = "parallel"
    revived = PartitionGraph.from_dict(graph.to_dict())
    assert revived.by_id("layers.0-1").composition == "parallel"
    # The default stays out of the serialized form.
    assert "composition" not in graph.by_id("head").to_dict()


def test_an_unknown_composition_is_rejected():
    with pytest.raises(GraphError, match="unknown composition"):
        ModuleNode.from_dict({"id": "m", "kind": "mlp", "composition": "fanout"})


def test_is_parallel_needs_more_than_one_branch():
    """One submodule behaves the same either way, so it is not a parallel group."""
    single = ModuleNode(id="m", kind="moe_experts", composition="parallel",
                        submodules=["model.layers.0.mlp.experts.0"])
    assert not single.is_parallel
    single.submodules.append("model.layers.0.mlp.experts.1")
    assert single.is_parallel


def test_a_module_with_no_submodule_is_functional():
    combine = ModuleNode(id="layers.0.combine", kind="mlp")
    assert combine.functional
    combine.submodules = ["model.layers.0.mlp"]
    assert not combine.functional
    # Out-of-scope nodes are not functional; they are simply not this run's work.
    assert not ModuleNode(id="vision", kind="vision", partitioned=False).functional


def test_validate_reports_functional_modules_as_unverifiable():
    graph = chain_graph()
    graph.modules.append(ModuleNode(
        id="layers.3.combine", kind="mlp", inputs=["h.4"], outputs=[], layer_indices=[3],
    ))
    warnings = graph.validate()
    assert any("layers.3.combine" in w and "no numeric verification" in w for w in warnings)
