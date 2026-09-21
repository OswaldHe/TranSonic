# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the checkpoint tensor inventory."""

import struct

import pytest

from model_partition.weights_index import (
    TensorEntry,
    WeightIndex,
    WeightIndexError,
    read_safetensors_header,
)
from tests.fixtures.tiny_llm import TinyConfig, write_tiny_repo

pytest.importorskip("torch")
pytest.importorskip("safetensors")


@pytest.fixture
def repo(tmp_path):
    return write_tiny_repo(tmp_path / "tiny", TinyConfig(n_experts=4))


def entry(name, dtype="bfloat16", shape=(4, 4), nbytes=32, shard="s0"):
    return TensorEntry(name, dtype, shape, nbytes, shard)


def test_layer_index_extracted_from_name():
    assert entry("model.layers.12.self_attn.q_proj.weight").layer_index == 12
    assert entry("model.embed_tokens.weight").layer_index is None
    assert entry("transformer.h.3.attn.weight").layer_index == 3


def test_expert_index_extracted_from_name():
    assert entry("model.layers.1.mlp.experts.37.up_proj.weight").expert_index == 37
    assert entry("model.layers.1.mlp.gate.weight").expert_index is None


def test_normalized_name_collapses_layer_and_expert_indices():
    a = entry("model.layers.3.mlp.experts.7.up_proj.weight")
    b = entry("model.layers.9.mlp.experts.2.up_proj.weight")
    assert a.normalized() == b.normalized() == "model.layers.{i}.mlp.experts.{e}.up_proj.weight"


def test_index_from_local_reads_real_safetensors(repo):
    index = WeightIndex.from_local(repo)
    assert index.num_layers == 4
    assert index.total_bytes > 0
    assert index.bytes_by_dtype() == {"float32": index.total_bytes}
    names = {e.name for e in index.entries}
    assert "model.embed_tokens.weight" in names
    assert "lm_head.weight" in names


def test_globals_exclude_stack_tensors(repo):
    index = WeightIndex.from_local(repo)
    global_names = {e.name for e in index.global_tensors()}
    assert global_names == {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}


def test_layer_grouping_partitions_all_stack_tensors(repo):
    index = WeightIndex.from_local(repo)
    by_layer = index.by_layer()
    assert sorted(by_layer) == [0, 1, 2, 3]
    grouped = sum(len(v) for v in by_layer.values())
    assert grouped + len(index.global_tensors()) == len(index.entries)


def test_identical_layers_share_a_signature_hybrid_ones_do_not(repo):
    """Even layers are dense, odd layers are MoE: two signatures, not one."""
    index = WeightIndex.from_local(repo)
    assert index.layer_signature(0) == index.layer_signature(2)
    assert index.layer_signature(1) == index.layer_signature(3)
    assert index.layer_signature(0) != index.layer_signature(1)


def test_expert_indices_discovered(repo):
    assert WeightIndex.from_local(repo).expert_indices() == [0, 1, 2, 3]


def test_layer_bytes_and_shards(repo):
    index = WeightIndex.from_local(repo)
    assert index.layer_bytes(1) > index.layer_bytes(0)  # MoE layer is bigger
    assert index.shards_for(index.by_layer()[0]) == ["model.safetensors"]
    assert index.largest_shard_bytes == index.total_bytes


def test_matching_and_excluding(repo):
    index = WeightIndex.from_local(repo)
    experts = index.matching(".experts.")
    assert experts and all(".experts." in e.name for e in experts)
    assert len(index.excluding(".experts.")) == len(index.entries) - len(experts)


def test_tied_embeddings_omit_lm_head(tmp_path):
    repo = write_tiny_repo(tmp_path / "tied", TinyConfig(tie_word_embeddings=True))
    names = {e.name for e in WeightIndex.from_local(repo).entries}
    assert "lm_head.weight" not in names


def test_header_parsing_returns_dtype_and_shape(repo):
    header = read_safetensors_header(repo / "model.safetensors")
    info = header["model.embed_tokens.weight"]
    assert info["dtype"] == "F32"
    assert info["shape"] == [128, 64]


def test_missing_safetensors_directory_raises(tmp_path):
    with pytest.raises(WeightIndexError, match="No .safetensors files"):
        WeightIndex.from_local(tmp_path)


def test_truncated_file_raises(tmp_path):
    bad = tmp_path / "model.safetensors"
    bad.write_bytes(b"\x01\x02")
    with pytest.raises(WeightIndexError, match="too short"):
        read_safetensors_header(bad)


def test_implausible_header_length_raises(tmp_path):
    bad = tmp_path / "model.safetensors"
    bad.write_bytes(struct.pack("<Q", 1 << 40))
    with pytest.raises(WeightIndexError, match="implausible header length"):
        read_safetensors_header(bad)


def test_corrupt_header_json_raises(tmp_path):
    bad = tmp_path / "model.safetensors"
    payload = b"{not json"
    bad.write_bytes(struct.pack("<Q", len(payload)) + payload)
    with pytest.raises(WeightIndexError, match="corrupt safetensors header"):
        read_safetensors_header(bad)


@pytest.mark.slow
def test_index_from_hub_without_downloading_weights():
    """Metadata-only inventory of a real repo."""
    try:
        index = WeightIndex.from_hub("Qwen/Qwen3.5-0.8B")
    except WeightIndexError as exc:
        # This one test really does talk to the Hub, so a Hub that is down says
        # nothing about the code. Everything else about indexing is covered locally.
        if any(code in str(exc) for code in ("503", "502", "504", "Connection")):
            pytest.skip(f"the Hub is unavailable: {exc}")
        raise
    assert index.num_layers == 24
    assert index.total_bytes > 1_500_000_000
    assert "bfloat16" in index.bytes_by_dtype()
