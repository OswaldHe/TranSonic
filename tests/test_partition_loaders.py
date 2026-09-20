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


def test_standalone_replay_needs_no_checkpoint(tiny_run):
    from model_partition.runtime.standalone import replay_module

    module_id = next(m.id for m in tiny_run.graph.partitioned_modules
                     if m.kind == "decoder_layers")
    # Remove the weights the model would otherwise be loaded from.
    (tiny_run.repo / "model.safetensors").unlink()
    comparisons = replay_module(tiny_run.layout.root, module_id)
    assert comparisons and all(c.passed for c in comparisons)


def test_standalone_replay_reports_an_unknown_module(tiny_run):
    from model_partition.runtime.standalone import StandaloneError, replay_module

    with pytest.raises(StandaloneError, match="not in the plan"):
        replay_module(tiny_run.layout.root, "nope")


def test_standalone_replay_reports_a_missing_sample(tiny_run):
    from model_partition.runtime.standalone import StandaloneError, replay_module

    module_id = next(m.id for m in tiny_run.graph.partitioned_modules
                     if m.kind == "decoder_layers")
    with pytest.raises(StandaloneError, match="No trace records"):
        replay_module(tiny_run.layout.root, module_id, sample_id="absent")


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
