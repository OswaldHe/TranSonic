# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for reading module weights out of the checkpoint.

The path ``cache_weights: false`` depends on: without it, skipping the weight
dumps would leave verification with no parameter values at all.
"""

from pathlib import Path

import pytest

from model_partition.weights_source import CheckpointWeights, WeightSourceError

pytest.importorskip("torch")
pytest.importorskip("safetensors")


@pytest.fixture
def shard(tmp_path):
    import torch
    from safetensors.torch import save_file

    save_file({
        "model.layers.0.self_attn.q_proj.weight": torch.arange(4.0).reshape(2, 2),
        "model.layers.0.mlp.down_proj.weight": torch.ones(2, 2),
    }, str(tmp_path / "model.safetensors"))
    return CheckpointWeights(root=tmp_path, shard_of={
        "model.layers.0.self_attn.q_proj.weight": "model.safetensors",
        "model.layers.0.mlp.down_proj.weight": "model.safetensors",
    })


def test_exact_names_resolve_without_remapping(shard):
    assert shard.resolve("model.layers.0.mlp.down_proj.weight") == \
        "model.layers.0.mlp.down_proj.weight"
    assert shard.remapped == {}


def test_a_vendor_name_resolves_by_unique_suffix(shard):
    """Vendor code names its modules differently from the checkpoint."""
    assert shard.resolve("layers.0.mlp.down_proj.weight") == \
        "model.layers.0.mlp.down_proj.weight"
    assert shard.remapped["layers.0.mlp.down_proj.weight"].startswith("model.")


def test_an_ambiguous_suffix_does_not_resolve():
    source = CheckpointWeights(root=Path("."), shard_of={
        "a.block.weight": "s", "b.block.weight": "s",
    })
    assert source.resolve("block.weight") is None


def test_an_unknown_name_does_not_resolve(shard):
    assert shard.resolve("model.layers.9.self_attn.q_proj.weight") is None


def test_load_returns_tensors_keyed_by_the_requested_name(shard):
    loaded = shard.load(["model.layers.0.mlp.down_proj.weight"])
    assert list(loaded) == ["model.layers.0.mlp.down_proj.weight"]
    assert tuple(loaded["model.layers.0.mlp.down_proj.weight"].shape) == (2, 2)


def test_load_reports_every_missing_name_at_once(shard):
    with pytest.raises(WeightSourceError, match="not in the checkpoint"):
        shard.load(["nope.weight", "also.missing.weight"])


def test_load_reports_a_shard_that_was_never_fetched(tmp_path):
    source = CheckpointWeights(root=tmp_path, shard_of={"w": "absent.safetensors"})
    with pytest.raises(WeightSourceError, match="fetch the weights first"):
        source.load(["w"])


def test_from_ingest_indexes_every_tensor(tmp_path):
    from model_partition.ingest import ingest
    from model_partition.spec import parse_spec
    from tests.fixtures.tiny_llm import TinyConfig, write_tiny_repo

    repo = write_tiny_repo(tmp_path / "repo", TinyConfig())
    result = ingest(parse_spec({"source": str(repo), "name": "t"}))
    source = CheckpointWeights.from_ingest(result)
    assert source.root == repo
    assert len(source.shard_of) == len(result.index.entries)


# -- integration with the trace bundle ---------------------------------------


def test_a_bundle_without_dumps_reads_from_the_checkpoint(tiny_run):
    """The whole point: no dumped weights, real values anyway."""
    from model_partition.runtime.module_runner import load_named_weights

    bundle = tiny_run.bundle
    module_id = next(m.id for m in tiny_run.graph.partitioned_modules
                     if bundle.weight_params.get(m.id))
    dumped = load_named_weights(bundle, module_id)

    bundle.weights = {}
    bundle.checkpoint = CheckpointWeights.from_ingest(tiny_run.result)
    from_checkpoint = load_named_weights(bundle, module_id)

    assert set(from_checkpoint) == set(dumped)
    import torch

    for name, tensor in dumped.items():
        assert torch.equal(from_checkpoint[name].float(), tensor.float())


def test_no_dumps_and_no_checkpoint_yields_nothing(tiny_run):
    """Reported as empty rather than silently substituting anything."""
    from model_partition.runtime.module_runner import load_named_weights

    bundle = tiny_run.bundle
    bundle.weights = {}
    bundle.checkpoint = None
    assert load_named_weights(bundle, tiny_run.graph.partitioned_modules[0].id) == {}
