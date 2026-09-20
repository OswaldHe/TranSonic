# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for structural inventory, quantization sizing, and the cost model."""

import json

import pytest

from model_partition.hardware import GIB
from model_partition.sizing import (
    CostModel,
    ModelInventory,
    QuantProfile,
    config_get,
)
from model_partition.weights_index import TensorEntry, WeightIndex
from tests.fixtures.tiny_llm import TinyConfig, write_tiny_repo

pytest.importorskip("torch")


def entry(name, dtype="bfloat16", shape=(4, 4), nbytes=32, shard="s0"):
    return TensorEntry(name, dtype, shape, nbytes, shard)


#: Shaped after deepseek_v41: the language model lives under text_config.
DEEPSEEK_LIKE = {
    "model_type": "deepseek_v41",
    "quantization_config": {
        "quant_method": "fp8", "weight_block_size": [32, 32],
        "scale_fmt": "ue8m0", "expert_dtype": "fp4",
    },
    "text_config": {
        "hidden_size": 5120, "num_hidden_layers": 40, "num_attention_heads": 64,
        "num_key_value_heads": 1, "head_dim": 512, "vocab_size": 129280,
        "n_routed_experts": 384, "num_experts_per_tok": 6,
    },
    "vision_config": {"hidden_size": 1024, "num_hidden_layers": 32},
}


def test_config_get_falls_through_to_text_config():
    assert config_get(DEEPSEEK_LIKE, "hidden_size") == 5120
    assert config_get(DEEPSEEK_LIKE, "model_type") == "deepseek_v41"
    assert config_get(DEEPSEEK_LIKE, "absent", "fallback") == "fallback"


def test_quant_profile_parsed_from_config():
    quant = QuantProfile.from_config(DEEPSEEK_LIKE)
    assert quant.quantized and quant.method == "fp8"
    assert quant.block_size == (32, 32)
    assert quant.scale_fmt == "ue8m0"
    assert quant.expert_dtype == "fp4"


def test_unquantized_config_yields_empty_profile():
    assert QuantProfile.from_config({"hidden_size": 8}).quantized is False


def test_packed_fp4_experts_hold_two_values_per_byte():
    quant = QuantProfile.from_config(DEEPSEEK_LIKE)
    packed = entry("model.layers.3.mlp.experts.7.up_proj.weight", dtype="int8", nbytes=1000)
    plain = entry("model.layers.3.self_attn.q_proj.weight", dtype="int8", nbytes=1000)
    assert quant.elements_per_byte(packed) == 2
    assert quant.elements_per_byte(plain) == 1
    # Dequantizing packed fp4 to bf16 is a 4x expansion in bytes.
    assert quant.dequant_bytes(packed) == 4000
    assert quant.dequant_bytes(plain) == 0


def test_fp8_dequant_to_bf16_doubles_bytes():
    quant = QuantProfile.from_config(DEEPSEEK_LIKE)
    fp8 = entry("model.layers.0.self_attn.q_proj.weight", dtype="float8_e4m3fn", nbytes=1024)
    assert quant.dequant_bytes(fp8) == 2048


def test_bf16_weights_need_no_dequant():
    assert QuantProfile().dequant_bytes(entry("w", dtype="bfloat16", nbytes=64)) == 0


# -- inventory ---------------------------------------------------------------


@pytest.fixture
def tiny_inventory(tmp_path):
    repo = write_tiny_repo(tmp_path / "tiny", TinyConfig(n_experts=4))
    config = json.loads((repo / "config.json").read_text())
    return ModelInventory.build(WeightIndex.from_local(repo), config)


def test_inventory_groups_layers_and_globals(tiny_inventory):
    assert tiny_inventory.num_layers == 4
    assert len(tiny_inventory.layers) == 4
    assert tiny_inventory.hidden_size == 64
    assert tiny_inventory.vocab_size == 128
    assert tiny_inventory.global_bytes > 0
    assert set(tiny_inventory.global_tensor_names) == {
        "model.embed_tokens.weight", "model.norm.weight", "lm_head.weight",
    }


def test_inventory_detects_experts_on_moe_layers_only(tiny_inventory):
    dense = tiny_inventory.layers[0]
    moe = tiny_inventory.layers[1]
    assert dense.has_experts is False and dense.n_experts == 0
    assert moe.has_experts is True and moe.n_experts == 4
    assert moe.expert_bytes > 0
    assert moe.non_expert_bytes == moe.param_bytes - moe.expert_bytes


def test_signature_groups_collapse_identical_layers(tiny_inventory):
    groups = tiny_inventory.signature_groups()
    assert sorted(sorted(v) for v in groups.values()) == [[0, 2], [1, 3]]


def test_total_param_bytes_matches_checkpoint(tiny_inventory):
    assert tiny_inventory.total_param_bytes() == tiny_inventory.index.total_bytes


def test_excluded_subtree_bytes_leave_the_layer_budget(tmp_path):
    """Vision tensors must not inflate module sizing when vision is out of scope."""
    index = WeightIndex(entries=[
        entry("model.layers.0.self_attn.q_proj.weight", nbytes=100),
        entry("model.embed_tokens.weight", nbytes=50),
        entry("model.visual.blocks.0.attn.qkv.weight", nbytes=900),
        entry("model.mtp.layers.0.weight", nbytes=70),
        entry("model.engram.embed.weight", nbytes=30),
    ])
    inventory = ModelInventory.build(index, {"hidden_size": 8, "num_hidden_layers": 1})
    assert inventory.subtree_bytes == {"vision": 900, "mtp": 70, "engram": 30}
    assert inventory.layers[0].param_bytes == 100
    assert inventory.global_bytes == 50
    assert inventory.total_param_bytes(include_excluded=False) == 150


def test_included_second_stack_keeps_its_own_index_space():
    """A vision tower's block 0 must not merge into text layer 0."""
    index = WeightIndex(entries=[
        entry("model.layers.0.self_attn.q_proj.weight", nbytes=100),
        entry("model.layers.1.self_attn.q_proj.weight", nbytes=100),
        entry("model.visual.blocks.0.attn.qkv.weight", nbytes=900),
    ])
    inventory = ModelInventory.build(index, {"hidden_size": 8}, include=("vision",))
    assert "vision" not in inventory.subtree_bytes
    assert inventory.stack_prefix == "model.layers."
    assert [layer.param_bytes for layer in inventory.layers] == [100, 100]
    assert inventory.aux_stack_bytes == {"model.visual.blocks.": 900}
    assert inventory.total_param_bytes() == 1100


def test_main_stack_is_the_one_with_most_layers():
    index = WeightIndex(entries=[
        entry("model.visual.blocks.0.w", nbytes=10),
        entry("model.visual.blocks.1.w", nbytes=10),
        entry("model.visual.blocks.2.w", nbytes=10),
        entry("model.layers.0.w", nbytes=10),
    ])
    assert index.main_stack_prefix() == "model.visual.blocks."
    assert index.stack_prefixes() == {"model.visual.blocks.": 3, "model.layers.": 1}


def test_layer_types_surface_hybrid_stacks():
    config = {"hidden_size": 8, "text_config": {"layer_types": ["linear_attention", "full_attention"]}}
    inventory = ModelInventory.build(WeightIndex(), config)
    assert inventory.layer_types == ["linear_attention", "full_attention"]


# -- representative layer selection (post-loop retention) --------------------


def build_inventory(n_layers: int, signature_of=lambda i: "same") -> ModelInventory:
    entries = [
        entry(f"model.layers.{i}.{signature_of(i)}.weight", nbytes=100)
        for i in range(n_layers)
    ]
    return ModelInventory.build(WeightIndex(entries=entries),
                                {"hidden_size": 8, "num_hidden_layers": n_layers})


def test_representative_layers_keep_requested_plus_mid_and_last():
    keep = build_inventory(64).representative_layers()
    assert keep == [0, 1, 5, 32, 63]


def test_representative_layers_cover_every_distinct_signature():
    """A hybrid stack must not lose the only copy of a kernel variant."""
    inventory = build_inventory(24, lambda i: "full_attn" if i % 4 == 3 else "linear_attn")
    keep = inventory.representative_layers()
    kept_signatures = {inventory.layers[i].signature for i in keep}
    assert kept_signatures == set(inventory.signature_groups())


def test_representative_layers_handles_short_stacks():
    assert build_inventory(2).representative_layers() == [0, 1]
    assert build_inventory(1).representative_layers() == [0]
    assert build_inventory(0).representative_layers() == []


# -- cost model --------------------------------------------------------------


def test_cost_model_from_config_derives_head_dim():
    model = CostModel.from_config({"hidden_size": 512, "num_attention_heads": 8})
    assert model.head_dim == 64
    explicit = CostModel.from_config(DEEPSEEK_LIKE)
    assert explicit.head_dim == 512 and explicit.num_kv_heads == 1


def test_activation_bytes_scale_with_sequence_length():
    model = CostModel(hidden_size=5120, activation_multiplier=6.0)
    assert model.activation_bytes(128) == 128 * 5120 * 2 * 6
    assert model.activation_bytes(16384) == 128 * model.activation_bytes(128)


def test_kv_bytes_zero_without_head_shape():
    assert CostModel(hidden_size=64).kv_bytes(4) == 0


def test_kv_bytes_scale_with_layers():
    model = CostModel(hidden_size=5120, num_kv_heads=8, head_dim=128, seq_len=1024)
    assert model.kv_bytes(2) == 2 * model.kv_bytes(1)
    assert model.kv_bytes(1) == 2 * 1024 * 8 * 128 * 2


def test_max_layers_per_module_respects_budget():
    """Qwen3.8-27B: 64 layers, ~0.85 GiB each, against a 28.6 GiB budget."""
    model = CostModel(hidden_size=5120, seq_len=16384, num_kv_heads=4, head_dim=256)
    layer_bytes = int(0.85 * GIB)
    count = model.max_layers_per_module(layer_bytes, int(28.6 * GIB))
    assert 1 <= count <= 64
    assert model.resident_bytes(layer_bytes * count, count) <= 28.6 * GIB
    assert model.resident_bytes(layer_bytes * (count + 1), count + 1) > 28.6 * GIB


def test_max_layers_is_at_least_one_for_zero_sized_layers():
    assert CostModel(hidden_size=8).max_layers_per_module(0, 1024) == 1


def test_oversized_layer_yields_zero_layers_per_module():
    """Signals the planner to split within a layer instead of grouping layers."""
    model = CostModel(hidden_size=5120, seq_len=16384)
    assert model.max_layers_per_module(100 * GIB, GIB) == 0
