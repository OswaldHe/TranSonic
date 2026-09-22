# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for model specification parsing and loader resolution."""

import pytest
import yaml

from model_partition.spec import SpecError, load_spec, parse_spec, slugify


def test_minimal_spec_defaults():
    spec = parse_spec({"source": "hf:Qwen/Qwen3.8-27B"})
    assert spec.repo_id == "Qwen/Qwen3.8-27B"
    assert spec.loader == "auto"
    assert spec.dtype == "bfloat16"
    assert spec.enabled is True
    # Vision, MTP and engram are out of scope until explicitly enabled.
    assert spec.scope.excluded() == ["vision", "mtp", "engram"]
    assert spec.name == "qwen-qwen3.8-27b"


def test_slugify_is_filesystem_safe():
    assert slugify("deepseek-ai/DeepSeek-V4.1-Flash") == "deepseek-ai-deepseek-v4.1-flash"
    assert slugify("  Spaces And CAPS  ") == "spaces-and-caps"


def test_loader_auto_prefers_vendor_inference_code():
    spec = parse_spec({"source": "hf:deepseek-ai/DeepSeek-V4-Flash"})
    assert spec.resolve_loader(["config.json", "inference/model.py"]) == "repo_code"
    assert spec.resolve_loader(["config.json", "tokenizer.json"]) == "transformers"


def test_explicit_loader_overrides_auto_detection():
    spec = parse_spec({"source": "hf:x/y", "loader": "transformers"})
    assert spec.resolve_loader(["inference/model.py"]) == "transformers"


def test_repo_code_loader_requires_entry():
    with pytest.raises(SpecError, match="requires 'entry'"):
        parse_spec({"source": "hf:x/y", "loader": "repo_code"})
    spec = parse_spec({
        "source": "hf:x/y", "loader": "repo_code",
        "entry": "inference/model.py", "code_paths": ["inference/"],
    })
    assert spec.entry == "inference/model.py"


def test_local_source_detection(tmp_path):
    spec = parse_spec({"source": str(tmp_path)})
    assert spec.is_local is True
    assert spec.repo_id is None


@pytest.mark.parametrize("payload,match", [
    ({}, "requires a non-empty string 'source'"),
    ({"source": "hf:x/y", "loader": "nope"}, "loader must be one of"),
    ({"source": "hf:x/y", "bogus": 1}, "Unknown spec key"),
    ({"source": "hf:x/y", "scope": {"audio": True}}, "Unknown scope key"),
    ({"source": "hf:x/y", "inputs": {"nope": 1}}, "Unknown inputs key"),
    ({"source": "hf:x/y", "inputs": {"long_token_budgets": [0]}}, "positive ints"),
    ({"source": "hf:x/y", "code_paths": "inference/"}, "list of strings"),
    ({"source": "hf:x/y", "trace": {"passes": []}}, "Unknown trace key"),
    ({"source": "hf:x/y", "trace": {"extra_passes": [{"args": ["input_ids"]}]}},
     "requires 'entry'"),
    # Caught here rather than an hour into loading a 475 GiB model.
    ({"source": "hf:x/y", "trace": {"extra_passes": [{"entry": "f", "args": ["hidde"]}]}},
     "asks for hidde"),
])
def test_invalid_specs_rejected(payload, match):
    with pytest.raises(SpecError, match=match):
        parse_spec(payload)


def test_round_trip_through_yaml(tmp_path):
    original = parse_spec({
        "source": "hf:Qwen/Qwen3.5-0.8B",
        "revision": "abc123",
        "scope": {"mtp": True},
        "trace": {"returns": ["logits", "hidden_states"],
                  "extra_passes": [{"entry": "forward_spec",
                                    "args": ["input_ids", "hidden_states"],
                                    "decode": True}]},
        "inputs": {"short": "short.jsonl", "long_token_budgets": [2048]},
        "overrides": {"max_new_tokens": 8},
    })
    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(original.to_dict()))
    reloaded = load_spec(path)
    assert reloaded.revision == "abc123"
    assert reloaded.scope.mtp is True
    assert reloaded.trace.extra_passes[0].entry == "forward_spec"
    assert reloaded.trace.extra_passes[0].decode is True
    assert reloaded.inputs.long_token_budgets == [2048]
    assert reloaded.overrides == {"max_new_tokens": 8}


def test_inputs_resolve_relative_to_spec_dir(tmp_path):
    (tmp_path / "short.jsonl").write_text("")
    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump({"source": "hf:x/y", "inputs": {"short": "short.jsonl"}}))
    spec = load_spec(path)
    short, long = spec.inputs.resolve(spec.base_dir())
    assert short == tmp_path / "short.jsonl"
    assert long is None


def test_missing_spec_file_raises(tmp_path):
    with pytest.raises(SpecError, match="not found"):
        load_spec(tmp_path / "absent.yaml")
