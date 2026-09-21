# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ingest, the loader paths, and checkpoint-free standalone replay."""

import json

import pytest

from model_partition.ingest import (
    IngestError,
    ensure_shards,
    ensure_weights,
    ingest,
    list_repo_files,
    load_config,
    shards_for_tensors,
)
from model_partition.loaders import LoaderError, build_loader
from model_partition.spec import parse_spec
from tests.fixtures.tiny_llm import TinyConfig, write_tiny_repo

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")


def local_spec(repo, **overrides):
    payload = {
        "source": str(repo), "name": "tiny", "dtype": "float32",
        "trust_remote_code": True,
    }
    payload.update(overrides)
    return parse_spec(payload)


@pytest.fixture
def repo(tmp_path):
    return write_tiny_repo(tmp_path / "repo", TinyConfig())


# -- ingest ------------------------------------------------------------------


def test_ingest_resolves_a_local_repo(repo):
    result = ingest(local_spec(repo))
    assert result.revision is None
    assert result.root == repo.resolve()
    assert result.config["model_type"] == "tiny_llm"
    assert result.index.num_layers == 4
    assert result.local_shards() == ["model.safetensors"]
    assert result.missing_shards() == []


def test_a_cached_remote_snapshot_is_indexed_without_the_hub(repo, monkeypatch):
    """A Hub hiccup must not fail a run whose checkpoint is already on disk."""
    from model_partition import ingest as ingest_module
    from model_partition.weights_index import WeightIndex

    def unavailable(*args, **kwargs):
        raise AssertionError("the Hub was asked for metadata it did not need")

    monkeypatch.setattr(WeightIndex, "from_hub", classmethod(unavailable))
    index = ingest_module._remote_index(
        local_spec(repo), repo, "abc123", ["config.json", "model.safetensors"], None)
    assert index.num_layers == 4


def test_an_incomplete_snapshot_still_asks_the_hub(repo, monkeypatch):
    """Indexing half a checkpoint would under-count it and mis-size the plan."""
    from model_partition import ingest as ingest_module
    from model_partition.weights_index import WeightIndex

    asked = []
    monkeypatch.setattr(WeightIndex, "from_hub",
                        classmethod(lambda cls, *a, **k: asked.append(a) or WeightIndex()))
    ingest_module._remote_index(
        local_spec(repo), repo, "abc123",
        ["model.safetensors", "model-00002-of-00002.safetensors"], None)
    assert asked


def test_vendor_code_is_preferred_and_its_entry_detected(repo):
    """A repo shipping inference code needs no hand-written entry."""
    result = ingest(local_spec(repo))
    assert result.loader == "repo_code"
    assert result.entry == "inference/model.py"
    assert result.code_paths == ["inference"]


def test_explicit_loader_overrides_detection(repo):
    result = ingest(local_spec(repo, loader="transformers"))
    assert result.loader == "transformers"


def test_repo_without_vendor_code_uses_transformers(tmp_path):
    plain = write_tiny_repo(tmp_path / "plain", TinyConfig(), with_vendor_code=False)
    assert ingest(local_spec(plain)).loader == "transformers"


def test_missing_entry_is_reported(repo):
    (repo / "inference" / "model.py").unlink()
    with pytest.raises(Exception, match="needs an 'entry' present in the repo"):
        ingest(local_spec(repo, loader="repo_code", entry="inference/model.py"))


def test_missing_directory_is_reported(tmp_path):
    with pytest.raises(IngestError, match="Local model directory not found"):
        list_repo_files(parse_spec({"source": str(tmp_path / "absent")}))


def test_missing_config_is_reported(repo):
    (repo / "config.json").unlink()
    with pytest.raises(IngestError, match="Model config not found"):
        ingest(local_spec(repo))


def test_malformed_config_is_reported(repo):
    (repo / "config.json").write_text("{not json")
    with pytest.raises(IngestError, match="Invalid JSON"):
        load_config(repo)


def test_config_must_be_an_object(repo):
    (repo / "config.json").write_text("[1, 2]")
    with pytest.raises(IngestError, match="must contain a JSON object"):
        load_config(repo)


def test_repo_files_are_listed_relative(repo):
    files = list_repo_files(local_spec(repo))
    assert "config.json" in files
    assert "inference/model.py" in files


def test_shards_for_tensors_maps_back_to_files(repo):
    result = ingest(local_spec(repo))
    names = [e.name for e in result.index.entries[:3]]
    assert shards_for_tensors(result, names) == ["model.safetensors"]


def test_ensure_weights_is_a_no_op_for_a_local_repo(repo):
    result = ingest(local_spec(repo))
    paths = ensure_weights(result)
    assert [p.name for p in paths] == ["model.safetensors"]


def test_ensure_shards_reports_a_missing_local_shard(repo):
    result = ingest(local_spec(repo))
    (repo / "model.safetensors").unlink()
    with pytest.raises(IngestError, match="Missing local shards"):
        ensure_shards(result, ["model.safetensors"])


# -- repo_code loader --------------------------------------------------------


def test_repo_code_requires_trust_remote_code(repo):
    """Importing vendor code must be opt-in."""
    result = ingest(local_spec(repo, trust_remote_code=False))
    with pytest.raises(LoaderError, match="trust_remote_code: true"):
        build_loader(result).build_meta()


def test_repo_code_builds_meta_structure_without_weights(repo):
    loaded = build_loader(ingest(local_spec(repo))).build_meta()
    assert loaded.meta and loaded.device == "meta"
    name, container = loaded.layer_container()
    assert name.endswith("layers") and len(container) == 4


def test_repo_code_builds_a_real_model(repo):
    from safetensors.torch import load_file

    loader = build_loader(ingest(local_spec(repo)))
    state = load_file(str(repo / "model.safetensors"))
    loaded = loader.build(state)
    out = loaded.model(torch.zeros(1, 4, dtype=torch.long))
    assert out.shape == (1, 4, 128)
    assert not loaded.model.training


def test_repo_code_build_config_only_initializes_buffers(repo):
    """Unlike meta + to_empty, from-config init leaves no garbage behind."""
    loaded = build_loader(ingest(local_spec(repo))).build_config_only()
    assert not loaded.meta
    for _, tensor in loaded.model.named_parameters():
        assert torch.isfinite(tensor).all()


def test_repo_code_reports_a_factory_that_is_absent(repo):
    (repo / "inference" / "model.py").write_text("VALUE = 1\n")
    with pytest.raises(LoaderError, match="exposes none of"):
        build_loader(ingest(local_spec(repo))).build_meta()


def test_repo_code_reports_an_import_failure(repo):
    (repo / "inference" / "model.py").write_text("raise RuntimeError('boom')\n")
    with pytest.raises(LoaderError, match="Failed importing"):
        build_loader(ingest(local_spec(repo))).build_meta()


def test_repo_code_reports_a_failing_factory(repo):
    (repo / "inference" / "model.py").write_text(
        "def build_model(config, state_dict=None):\n    raise ValueError('nope')\n"
    )
    with pytest.raises(LoaderError, match="factory failed"):
        build_loader(ingest(local_spec(repo))).build_meta()


def test_submodule_lookup_walks_module_lists(repo):
    from safetensors.torch import load_file

    loader = build_loader(ingest(local_spec(repo)))
    loaded = loader.build(load_file(str(repo / "model.safetensors")))
    assert type(loaded.submodule("model.layers.2")).__name__ == "TinyDecoderLayer"


def test_unknown_loader_is_reported(repo):
    result = ingest(local_spec(repo))
    result.loader = "telepathy"
    with pytest.raises(LoaderError, match="Unknown loader"):
        build_loader(result)


# -- transformers loader -----------------------------------------------------


def test_transformers_loader_reports_an_unreadable_config(repo):
    """The error names the repo_code alternative rather than just failing."""
    result = ingest(local_spec(repo, loader="transformers"))
    with pytest.raises(LoaderError, match="loader: repo_code"):
        build_loader(result).build_meta()


# -- standalone replay -------------------------------------------------------


def _with_modules(run):
    """Extract into the run so replay finds an implementation to launch."""
    from model_partition.extract import extract

    extract(run.graph, run.build_model(), run.layout.modules_dir,
            run_root=run.layout.root, sample_ids=run.sample_ids,
            weight_tensors=run.bundle.weights)
    return next(m.id for m in run.graph.partitioned_modules if m.kind == "decoder_layers")


def test_standalone_replay_needs_no_checkpoint(tiny_run):
    from model_partition.runtime.standalone import replay_module

    module_id = _with_modules(tiny_run)
    # Remove the checkpoint: the module directory and the trace are the whole input.
    (tiny_run.repo / "model.safetensors").unlink()
    comparisons = replay_module(tiny_run.layout.root, module_id)
    assert comparisons and all(c.passed for c in comparisons)


def test_standalone_replay_reports_an_unknown_module(tiny_run):
    from model_partition.runtime.standalone import StandaloneError, replay_module

    with pytest.raises(StandaloneError, match="not in the plan"):
        replay_module(tiny_run.layout.root, "nope")


def test_standalone_replay_reports_a_missing_sample(tiny_run):
    from model_partition.runtime.standalone import StandaloneError, replay_module

    module_id = _with_modules(tiny_run)
    with pytest.raises(StandaloneError, match="No trace records"):
        replay_module(tiny_run.layout.root, module_id, sample_id="absent")


def test_standalone_replay_says_so_when_nothing_was_extracted(tiny_run):
    """Without a module directory there is no implementation to replay."""
    from model_partition.runtime.standalone import StandaloneError, replay_module

    module_id = next(m.id for m in tiny_run.graph.partitioned_modules
                     if m.kind == "decoder_layers")
    with pytest.raises(StandaloneError, match="No extracted implementation"):
        replay_module(tiny_run.layout.root, module_id)


def test_load_run_requires_a_plan(tiny_run):
    from model_partition.runtime.standalone import StandaloneError, load_run

    tiny_run.layout.graph_path.unlink()
    with pytest.raises(StandaloneError, match="No partition graph"):
        load_run(tiny_run.layout.root)


def test_loaded_run_exposes_the_spec(tiny_run):
    from model_partition.runtime.standalone import load_run

    run = load_run(tiny_run.layout.root)
    assert run.spec.source == str(tiny_run.repo)
    assert run.graph.partitioned_modules
    assert run.bundle.records


def test_run_manifest_records_the_resolved_spec(tiny_run):
    payload = json.loads(json.dumps(tiny_run.layout.read_run()))
    assert payload["spec"]["source"] == str(tiny_run.repo)
    assert payload["loader"] == "repo_code"


def test_evict_shards_is_a_no_op_for_local_weights(repo):
    """A local model's weights are the user's own files and must not be deleted."""
    from model_partition.ingest import evict_shards

    result = ingest(local_spec(repo))
    assert evict_shards(result, ["model.safetensors"]) == 0
    assert (repo / "model.safetensors").is_file()


def test_repo_code_build_reads_the_checkpoint_without_being_handed_one(repo):
    """Regression: build() with no state_dict silently ran on fresh weights."""
    from safetensors.torch import load_file

    expected = load_file(str(repo / "model.safetensors"))
    loaded = build_loader(ingest(local_spec(repo))).build()
    actual = dict(loaded.model.state_dict())
    assert torch.allclose(actual["model.layers.0.self_attn.q_proj.weight"],
                          expected["model.layers.0.self_attn.q_proj.weight"])


def test_two_builds_agree_because_both_read_the_checkpoint(repo):
    """Tracing dumps weights from one instance and activations from another."""
    loader = build_loader(ingest(local_spec(repo)))
    first = loader.build().model.state_dict()
    second = loader.build().model.state_dict()
    for key in first:
        assert torch.equal(first[key], second[key]), key


def test_build_config_only_does_not_read_the_checkpoint(repo):
    from safetensors.torch import load_file

    expected = load_file(str(repo / "model.safetensors"))
    fresh = build_loader(ingest(local_spec(repo))).build_config_only().model.state_dict()
    key = "model.layers.0.self_attn.q_proj.weight"
    assert not torch.allclose(fresh[key], expected[key])


def test_tied_checkpoint_loads_despite_the_missing_head(tmp_path):
    """A tied-embedding checkpoint omits lm_head.weight; a strict load would fail."""
    tied = write_tiny_repo(tmp_path / "tied", TinyConfig(tie_word_embeddings=True))
    loaded = build_loader(ingest(local_spec(tied))).build()
    out = loaded.model(torch.zeros(1, 4, dtype=torch.long))
    assert out.shape == (1, 4, 128)


# -- shard eviction ----------------------------------------------------------
#
# The Hub cache keeps one blob per digest and symlinks every snapshot entry at it,
# so a blob can belong to another revision or another run.


def _fake_cache(tmp_path, revisions=("rev-a",)):
    """A hub-style cache: one blob, one snapshot symlink per revision."""
    repo = tmp_path / "models--org--model"
    blobs = repo / "blobs"
    blobs.mkdir(parents=True)
    blob = blobs / "deadbeef"
    blob.write_bytes(b"\x00" * 2048)
    snapshots = []
    for revision in revisions:
        snapshot = repo / "snapshots" / revision
        snapshot.mkdir(parents=True)
        (snapshot / "model.safetensors").symlink_to(blob)
        snapshots.append(snapshot)
    return blob, snapshots


def _remote_result(snapshot, revision):
    from model_partition.ingest import IngestResult
    from model_partition.spec import parse_spec
    from model_partition.weights_index import TensorEntry, WeightIndex

    index = WeightIndex(entries=[
        TensorEntry("w", "bfloat16", (2, 2), 2048, "model.safetensors"),
    ])
    return IngestResult(
        spec=parse_spec({"source": "hf:org/model", "name": "m"}),
        root=snapshot, config={}, loader="transformers", index=index, revision=revision,
    )


def test_evicting_the_only_reference_frees_the_blob(tmp_path):
    from model_partition.ingest import evict_shards

    blob, (snapshot,) = _fake_cache(tmp_path)
    result = _remote_result(snapshot, "rev-a")
    assert evict_shards(result, ["model.safetensors"]) == 2048
    assert not blob.exists()


def test_a_blob_another_revision_still_uses_is_left_alone(tmp_path):
    """Deleting it would corrupt the cache for every other user of those bytes."""
    from model_partition.ingest import evict_shards

    blob, (mine, theirs) = _fake_cache(tmp_path, revisions=("rev-a", "rev-b"))
    result = _remote_result(mine, "rev-a")
    freed = evict_shards(result, ["model.safetensors"])

    assert freed == 0, "shared bytes were not reclaimed, so none should be reported"
    assert blob.exists()
    assert (theirs / "model.safetensors").resolve() == blob
    assert not (mine / "model.safetensors").exists()


def test_evicting_an_absent_shard_is_a_no_op(tmp_path):
    from model_partition.ingest import evict_shards

    _, (snapshot,) = _fake_cache(tmp_path)
    result = _remote_result(snapshot, "rev-a")
    assert evict_shards(result, ["missing.safetensors"]) == 0


def test_a_downloaded_shard_is_visible_where_the_run_looks(tmp_path, monkeypatch):
    """ensure_shards must leave the shard at result.shard_path, not only in a cache."""
    from model_partition import ingest as ingest_module

    elsewhere = tmp_path / "hub" / "blobs"
    elsewhere.mkdir(parents=True)
    downloaded = elsewhere / "abc123"
    downloaded.write_bytes(b"\x01" * 16)

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    result = _remote_result(snapshot, "rev-a")

    monkeypatch.setattr("huggingface_hub.hf_hub_download",
                        lambda *a, **k: str(downloaded))
    paths = ingest_module.ensure_shards(result, ["model.safetensors"])

    assert paths == [snapshot / "model.safetensors"]
    assert result.missing_shards() == []
    assert (snapshot / "model.safetensors").read_bytes() == b"\x01" * 16
