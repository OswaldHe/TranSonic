# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the deterministic seed planner."""

import json

import pytest

from model_partition.hardware import GIB
from model_partition.planner.graph import GraphError
from model_partition.planner.auto import (
    PlanOptions,
    classify_global,
    component_role,
    layer_parts,
    moe_parts,
    plan,
)
from model_partition.sizing import CostModel, ModelInventory
from model_partition.weights_index import TensorEntry, WeightIndex
from tests.fixtures.tiny_llm import TinyConfig, write_tiny_repo

pytest.importorskip("torch")


def tiny_inventory(tmp_path, **config_kwargs) -> ModelInventory:
    repo = write_tiny_repo(tmp_path / "t", TinyConfig(**config_kwargs))
    config = json.loads((repo / "config.json").read_text())
    return ModelInventory.build(WeightIndex.from_local(repo), config)


def synthetic_inventory(n_layers: int, layer_bytes: int, signature_of=lambda i: "attn") -> ModelInventory:
    entries = [TensorEntry("model.embed_tokens.weight", "bfloat16", (8, 8), 64, "s0"),
               TensorEntry("model.norm.weight", "bfloat16", (8,), 16, "s0"),
               TensorEntry("lm_head.weight", "bfloat16", (8, 8), 64, "s0")]
    for i in range(n_layers):
        entries.append(TensorEntry(
            f"model.layers.{i}.self_attn.{signature_of(i)}.weight",
            "bfloat16", (8, 8), layer_bytes, "s0",
        ))
    return ModelInventory.build(
        WeightIndex(entries=entries),
        {"hidden_size": 8, "num_hidden_layers": n_layers, "vocab_size": 8},
    )


# -- structure ---------------------------------------------------------------


def test_plan_covers_embed_stack_norm_and_head(tmp_path):
    graph = plan(tiny_inventory(tmp_path), budget_bytes=GIB, options=PlanOptions(seq_len=8))
    assert graph.validate() == []
    ids = [m.id for m in graph.modules]
    assert ids[0] == "embed" and ids[-1] == "lm_head"
    assert "final_norm" in ids
    assert graph.entry_tensors == ["tokens"]
    assert graph.output_tensors == ["logits"]


def test_plan_is_a_connected_chain_for_a_dense_model(tmp_path):
    graph = plan(tiny_inventory(tmp_path), budget_bytes=GIB, options=PlanOptions(seq_len=8))
    order = graph.topological_order()
    assert order == [m.id for m in graph.modules]
    # Each module's input is the previous module's output.
    for earlier, later in zip(graph.modules, graph.modules[1:]):
        assert earlier.outputs[0] in later.inputs


def test_dense_layers_group_together_under_a_large_budget(tmp_path):
    graph = plan(tiny_inventory(tmp_path), budget_bytes=GIB, options=PlanOptions(seq_len=8))
    decoders = [m for m in graph.modules if m.kind == "decoder_layers"]
    assert len(decoders) == 1
    assert decoders[0].layer_indices == [0, 1, 2, 3]


def test_one_layer_per_module_override(tmp_path):
    graph = plan(tiny_inventory(tmp_path), budget_bytes=GIB,
                 options=PlanOptions(seq_len=8, one_layer_per_module=True))
    decoders = [m for m in graph.modules if m.kind == "decoder_layers"]
    assert [m.layer_indices for m in decoders] == [[0], [1], [2], [3]]


def test_max_layers_per_module_override(tmp_path):
    graph = plan(tiny_inventory(tmp_path), budget_bytes=GIB,
                 options=PlanOptions(seq_len=8, max_layers_per_module=2))
    decoders = [m for m in graph.modules if m.kind == "decoder_layers"]
    assert [m.layer_indices for m in decoders] == [[0, 1], [2, 3]]


# -- hybrid stacks -----------------------------------------------------------


def test_groups_never_mix_layer_signatures(tmp_path):
    """Qwen3.5/3.8 alternate linear and full attention; kernels must not share a module."""
    inventory = tiny_inventory(tmp_path, n_experts=4)
    graph = plan(inventory, budget_bytes=GIB, options=PlanOptions(seq_len=8))
    decoders = [m for m in graph.modules if m.kind == "decoder_layers"]
    assert [m.layer_indices for m in decoders] == [[0], [1], [2], [3]]
    assert len({m.code_signature for m in decoders}) == 2


def test_hybrid_stack_groups_runs_of_like_layers():
    """3 linear + 1 full, repeating: runs group, the odd one out stays alone."""
    inventory = synthetic_inventory(12, 1024, lambda i: "full" if i % 4 == 3 else "linear")
    graph = plan(inventory, budget_bytes=GIB, options=PlanOptions(seq_len=8))
    decoders = [m for m in graph.modules if m.kind == "decoder_layers"]
    assert [m.layer_indices for m in decoders] == [
        [0, 1, 2], [3], [4, 5, 6], [7], [8, 9, 10], [11],
    ]


def test_dedup_groups_are_far_fewer_than_modules():
    inventory = synthetic_inventory(64, 1024, lambda i: "full" if i % 4 == 3 else "linear")
    graph = plan(inventory, budget_bytes=GIB, options=PlanOptions(seq_len=8))
    assert len(graph.partitioned_modules) > 30
    assert len(graph.signature_groups()) == 5  # embed, 2 layer kinds, norm, head


def test_homogeneous_grouping_can_be_disabled():
    inventory = synthetic_inventory(8, 1024, lambda i: "full" if i % 4 == 3 else "linear")
    graph = plan(inventory, budget_bytes=GIB,
                 options=PlanOptions(seq_len=8, homogeneous_groups=False, max_layers_per_module=4))
    decoders = [m for m in graph.modules if m.kind == "decoder_layers"]
    assert [m.layer_indices for m in decoders] == [[0, 1, 2, 3], [4, 5, 6, 7]]


# -- budget-driven splitting -------------------------------------------------


def test_tight_budget_splits_the_stack_into_more_modules():
    inventory = synthetic_inventory(8, 4096)
    loose = plan(inventory, budget_bytes=GIB, options=PlanOptions(seq_len=8))
    tight = plan(inventory, budget_bytes=20_000, options=PlanOptions(seq_len=8))
    assert len(tight.partitioned_modules) > len(loose.partitioned_modules)
    assert tight.validate() == []


def test_oversized_moe_layer_splits_into_attention_router_experts_combine(tmp_path):
    inventory = tiny_inventory(tmp_path, n_experts=8)
    graph = plan(inventory, budget_bytes=300_000, options=PlanOptions(seq_len=8))
    # The combine step is functional, which validate() reports rather than rejects.
    assert all("Functional module" in w for w in graph.validate())
    kinds = {m.kind for m in graph.modules}
    assert {"attention", "moe_router", "moe_experts"} <= kinds

    experts = [m for m in graph.modules if m.kind == "moe_experts" and m.layer_indices == [1]]
    assert len(experts) > 1
    # Experts are alternatives, not a pipeline.
    assert all(e.composition == "parallel" for e in experts)
    # Fan-out from the router, fan-in to the combine module.
    combine = graph.by_id("layers.1.combine")
    assert all(e.outputs[0] in combine.inputs for e in experts)
    assert combine.outputs == ["h.2"]


def test_expert_groups_stay_within_budget(tmp_path):
    inventory = tiny_inventory(tmp_path, n_experts=8)
    budget = 300_000
    graph = plan(inventory, budget_bytes=budget, options=PlanOptions(seq_len=8))
    for module in graph.partitioned_modules:
        assert module.resident_bytes <= budget


def test_explicit_experts_per_group_is_honoured(tmp_path):
    inventory = tiny_inventory(tmp_path, n_experts=8)
    graph = plan(inventory, budget_bytes=10 * GIB,
                 options=PlanOptions(seq_len=8, experts_per_group=2, max_layers_per_module=1))
    # A large budget means no split is needed, so experts stay inside the layer.
    assert graph.validate() == []


def test_expert_ranges_partition_every_expert(tmp_path):
    inventory = tiny_inventory(tmp_path, n_experts=8)
    graph = plan(inventory, budget_bytes=300_000, options=PlanOptions(seq_len=8))
    covered = []
    for module in graph.modules:
        if module.kind == "moe_experts" and module.layer_indices == [1]:
            start, end = module.expert_range
            covered.extend(range(start, end + 1))
    assert sorted(covered) == list(range(8))


# -- accounting --------------------------------------------------------------


def test_param_bytes_are_conserved_across_modules(tmp_path):
    inventory = tiny_inventory(tmp_path)
    graph = plan(inventory, budget_bytes=GIB, options=PlanOptions(seq_len=8))
    planned = sum(m.param_bytes for m in graph.partitioned_modules)
    assert planned == inventory.total_param_bytes()


def test_excluded_subtrees_appear_as_unpartitioned_nodes():
    index = WeightIndex(entries=[
        TensorEntry("model.layers.0.self_attn.q.weight", "bfloat16", (8, 8), 128, "s0"),
        TensorEntry("model.embed_tokens.weight", "bfloat16", (8, 8), 64, "s0"),
        TensorEntry("model.visual.blocks.0.attn.qkv.weight", "bfloat16", (8, 8), 900, "s0"),
    ])
    inventory = ModelInventory.build(index, {"hidden_size": 8, "num_hidden_layers": 1, "vocab_size": 8})
    graph = plan(inventory, budget_bytes=GIB, options=PlanOptions(seq_len=8))
    vision = graph.by_id("vision")
    assert vision.partitioned is False
    assert vision.param_bytes == 900
    assert "vision" not in [m.id for m in graph.partitioned_modules]


def test_tied_head_names_the_head_module_not_the_embedding(tmp_path):
    """A tied head owns no tensor but is still its own module.

    Falling back to the embedding would hook the embedding, so the head's recorded
    "output" would be the embedding's and verification would compare the wrong
    tensor while appearing to pass.
    """
    inventory = tiny_inventory(tmp_path, tie_word_embeddings=True)
    graph = plan(inventory, budget_bytes=GIB, options=PlanOptions(seq_len=8))
    head = graph.by_id("lm_head")
    assert "tied" in head.notes
    assert head.submodules == ["lm_head"]
    assert head.param_bytes == graph.by_id("embed").param_bytes


def test_kv_bytes_recorded_for_decoder_modules():
    inventory = synthetic_inventory(4, 1024)
    cost = CostModel(hidden_size=8, num_kv_heads=2, head_dim=16, seq_len=128)
    graph = plan(inventory, budget_bytes=GIB, options=PlanOptions(seq_len=128), cost=cost)
    decoder = next(m for m in graph.modules if m.kind == "decoder_layers")
    assert decoder.kv_bytes == cost.kv_bytes(len(decoder.layer_indices), 128)


def test_metadata_records_planner_and_seq_len():
    graph = plan(synthetic_inventory(2, 128), budget_bytes=GIB, options=PlanOptions(seq_len=4096))
    assert graph.metadata == {"planner": "auto", "seq_len": 4096}


def test_empty_stack_yields_embed_and_head_only():
    inventory = synthetic_inventory(0, 0)
    graph = plan(inventory, budget_bytes=GIB, options=PlanOptions(seq_len=8))
    assert [m.kind for m in graph.partitioned_modules] == ["embed", "norm", "lm_head"]
    assert graph.validate() == []


# -- helpers -----------------------------------------------------------------


@pytest.mark.parametrize("name,expected", [
    ("model.embed_tokens.weight", ("embed", "embed")),
    ("transformer.wte.weight", ("embed", "embed")),
    ("lm_head.weight", ("lm_head", "lm_head")),
    ("model.norm.weight", ("norm", "final_norm")),
    ("transformer.ln_f.bias", ("norm", "final_norm")),
    ("model.something_else", None),
])
def test_classify_global(name, expected):
    assert classify_global(name) == expected


@pytest.mark.parametrize("tensor_name,expected", [
    ("model.embed_tokens.weight", "model.embed_tokens"),
    ("model.layers.0.self_attn.q_proj.bias", "model.layers.0.self_attn.q_proj"),
    ("model.layers.0.mlp.experts.3.up_proj.weight_scale_inv", "model.layers.0.mlp.experts.3.up_proj"),
    ("model.layers.0.linear_attn.A_log", "model.layers.0.linear_attn"),
    ("model.norm", "model.norm"),
])
def test_module_path_of_strips_parameter_suffix(tensor_name, expected):
    """Hooks attach to modules, so plan submodules must name modules."""
    from model_partition.planner.auto import module_path_of
    assert module_path_of(tensor_name) == expected


# -- role discovery ----------------------------------------------------------


@pytest.mark.parametrize("component,role", [
    ("self_attn", "attention"),
    ("linear_attn", "attention"),
    ("attn", "attention"),
    ("attn_norm", "attention"),
    ("input_layernorm", "attention"),
    ("mlp", "ffn"),
    ("ffn", "ffn"),
    ("ffn_norm", "ffn"),
    ("post_attention_layernorm", "ffn"),
    ("pre_feedforward_layernorm", "ffn"),
    ("indexer", "other"),
])
def test_component_role_buckets_by_side_of_the_residual_stream(component, role):
    assert component_role(component) == role


def _moe_layer_inventory():
    index = WeightIndex(entries=[
        TensorEntry("model.layers.0.self_attn.q_proj.weight", "bfloat16", (8,), 10, "s0"),
        TensorEntry("model.layers.0.input_layernorm.weight", "bfloat16", (8,), 20, "s0"),
        TensorEntry("model.layers.0.post_attention_layernorm.weight", "bfloat16", (8,), 5, "s0"),
        TensorEntry("model.layers.0.mlp.gate.weight", "bfloat16", (8,), 40, "s0"),
        TensorEntry("model.layers.0.mlp.experts.0.up_proj.weight", "bfloat16", (8,), 80, "s0"),
        TensorEntry("model.layers.0.mlp.experts.1.up_proj.weight", "bfloat16", (8,), 80, "s0"),
        TensorEntry("model.layers.0.mlp.shared_experts.up_proj.weight", "bfloat16", (8,), 7, "s0"),
        TensorEntry("model.layers.0.hc_attn_base", "bfloat16", (8,), 1, "s0"),
    ])
    return ModelInventory.build(index, {"hidden_size": 8, "num_hidden_layers": 1})


def test_layer_parts_splits_a_layer_across_its_children():
    parts = layer_parts(_moe_layer_inventory(), 0)
    # Norms lead: submodules run in listed order.
    assert parts.attention == ["model.layers.0.input_layernorm", "model.layers.0.self_attn"]
    assert parts.ffn == ["model.layers.0.post_attention_layernorm", "model.layers.0.mlp"]
    assert parts.bytes_for("attention") == 30
    assert parts.bytes_for("ffn") == 5 + 40 + 80 + 80 + 7


def test_layer_parts_reports_tensors_the_layer_owns_directly():
    """A parameter of the layer module itself belongs to no child module."""
    parts = layer_parts(_moe_layer_inventory(), 0)
    assert parts.own_params == ["model.layers.0.hc_attn_base"]
    assert not any("hc_attn_base" in path for path in parts.attention + parts.ffn)


def test_moe_parts_separates_router_experts_and_shared():
    router, experts, shared, nbytes = moe_parts(_moe_layer_inventory(), "model.layers.0.mlp")
    assert router == ["model.layers.0.mlp.gate"]
    assert experts == {0: "model.layers.0.mlp.experts.0", 1: "model.layers.0.mlp.experts.1"}
    assert shared == ["model.layers.0.mlp.shared_experts"]
    assert nbytes == {"router": 40, "experts": 160, "shared": 7}


# -- attention / FFN separation ----------------------------------------------


def test_split_attention_ffn_gives_each_side_its_own_module(tmp_path):
    """Requested by instruction, not by capacity: it applies whatever the budget."""
    inventory = tiny_inventory(tmp_path)
    graph = plan(inventory, budget_bytes=10 * GIB,
                 options=PlanOptions(seq_len=8, split_attention_ffn=True))

    kinds = {m.kind for m in graph.partitioned_modules}
    assert "decoder_layers" not in kinds
    assert {"attention", "mlp"} <= kinds
    for index in range(len(inventory.layers)):
        attention = graph.by_id(f"layers.{index}.attention")
        ffn = graph.by_id(f"layers.{index}.ffn")
        assert ffn.inputs == attention.outputs
        assert all("attn" in s or "norm" in s for s in attention.submodules)
        assert any("mlp" in s or "ffn" in s for s in ffn.submodules)


def test_split_attention_ffn_conserves_every_parameter(tmp_path):
    inventory = tiny_inventory(tmp_path, n_experts=4)
    graph = plan(inventory, budget_bytes=10 * GIB,
                 options=PlanOptions(seq_len=8, split_attention_ffn=True))
    planned = sum(m.param_bytes for m in graph.partitioned_modules)
    assert planned == inventory.total_param_bytes()


def test_split_attention_ffn_chains_the_residual_stream(tmp_path):
    inventory = tiny_inventory(tmp_path)
    graph = plan(inventory, budget_bytes=10 * GIB,
                 options=PlanOptions(seq_len=8, split_attention_ffn=True))
    assert graph.validate() == []
    order = graph.topological_order()
    for index in range(len(inventory.layers)):
        assert order.index(f"layers.{index}.attention") < order.index(f"layers.{index}.ffn")


def test_an_over_budget_moe_ffn_splits_into_router_experts_and_combine(tmp_path):
    inventory = tiny_inventory(tmp_path, n_experts=8)
    graph = plan(inventory, budget_bytes=150_000,
                 options=PlanOptions(seq_len=8, split_attention_ffn=True,
                                     experts_per_group=2))
    experts = [m for m in graph.partitioned_modules
               if m.kind == "moe_experts" and m.layer_indices == [1]]
    assert len(experts) == 4
    assert all(len(m.submodules) == 2 and m.is_parallel for m in experts)
    router = graph.by_id("layers.1.router")
    assert any("gate" in s for s in router.submodules)
    assert graph.by_id("layers.1.combine").functional


def test_a_dense_layer_keeps_one_ffn_module(tmp_path):
    """Nothing to route, so there is nothing to split the FFN into."""
    inventory = tiny_inventory(tmp_path, n_experts=8)
    graph = plan(inventory, budget_bytes=150_000,
                 options=PlanOptions(seq_len=8, split_attention_ffn=True,
                                     experts_per_group=2))
    # The toy model puts experts on odd layers only.
    dense = graph.by_id("layers.0.ffn")
    assert dense.kind == "mlp" and not dense.functional
    assert "layers.0.router" not in [m.id for m in graph.modules]


def test_a_dense_ffn_over_budget_is_reported_not_carved_up(tmp_path):
    """Cutting a single matmul chain apart is not the seed planner's call."""
    inventory = tiny_inventory(tmp_path)
    graph = plan(inventory, budget_bytes=60_000,
                 options=PlanOptions(seq_len=8, split_attention_ffn=True))
    ffn = graph.by_id("layers.0.ffn")
    assert "over the per-module budget" in ffn.notes
    with pytest.raises(GraphError, match="over the .* budget"):
        graph.validate(60_000)


def test_norms_are_listed_before_the_computation_they_normalize_for(tmp_path):
    """Submodules run in listed order, and checkpoint order says nothing about it."""
    inventory = tiny_inventory(tmp_path)
    parts = layer_parts(inventory, 0)
    assert "norm" in parts.attention[0]
    assert "norm" in parts.ffn[0]
