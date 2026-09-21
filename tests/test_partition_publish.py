# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for publishing a run's artifacts."""

import json

import pytest

from model_partition.publish import PublishError, collect, publish_run


def _run(tmp_path, passed=True, blob_bytes=1024):
    root = tmp_path / "run"
    (root / "plan").mkdir(parents=True)
    (root / "modules" / "00-attn").mkdir(parents=True)
    (root / "trace" / "activations").mkdir(parents=True)
    (root / "logs").mkdir()
    (root / "run.yaml").write_text("spec:\n  source: hf:org/model\n  loader: repo_code\n")
    (root / "plan" / "partition_graph.yaml").write_text("modules: []\n")
    (root / "modules" / "index.yaml").write_text("n_groups: 1\n")
    (root / "modules" / "00-attn" / "source.py").write_text("class Attention:\n    pass\n")
    (root / "trace" / "activations" / "big.bin").write_bytes(b"\x00" * blob_bytes)
    (root / "logs" / "agent-plan-1.log").write_bytes(b"x" * 4096)
    (root / "state.json").write_text(json.dumps({
        "passed": passed, "iteration": 1,
        "stages": {"verify_modules": {"status": "ok", "detail": "396 checks passed"}},
    }))
    return root


def test_collect_leaves_out_logs_and_caches(tmp_path):
    root = _run(tmp_path)
    (root / "modules" / "00-attn" / "__pycache__").mkdir()
    (root / "modules" / "00-attn" / "__pycache__" / "x.pyc").write_bytes(b"junk")

    files, total = collect(root)
    names = {p.relative_to(root).as_posix() for p in files}
    assert "trace/activations/big.bin" in names
    assert not any(name.startswith("logs/") for name in names)
    assert not any("__pycache__" in name for name in names)
    assert total == sum(p.stat().st_size for p in files)


def test_a_directory_that_is_not_a_run_is_refused(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(PublishError, match="does not look like a finished run"):
        collect(tmp_path / "empty")
    with pytest.raises(PublishError, match="No such run directory"):
        collect(tmp_path / "absent")


def test_a_run_over_the_ceiling_is_stopped_before_uploading(tmp_path):
    root = _run(tmp_path, blob_bytes=4096)
    with pytest.raises(PublishError, match="exceeds the"):
        publish_run(root, "org/artifacts", max_bytes=1024)


def test_a_run_that_did_not_pass_is_not_published_by_default(tmp_path):
    root = _run(tmp_path, passed=False)
    with pytest.raises(PublishError, match="did not pass"):
        publish_run(root, "org/artifacts")


def test_a_dry_run_reports_the_size_without_credentials(tmp_path):
    """The size is knowable before the first byte moves, and so is the cost."""
    root = _run(tmp_path)
    lines: list[str] = []
    result = publish_run(root, "org/artifacts", dry_run=True, report=lines.append)
    assert result.dry_run and result.files >= 4
    assert result.total_bytes > 0
    assert not result.url
    assert "would upload" in result.summary()
    assert any("big.bin" in line for line in lines)


def test_publishing_without_a_token_says_so(tmp_path, monkeypatch):
    """And it says so before contacting the Hub, not after."""
    from model_partition import publish

    monkeypatch.setattr(publish, "find_token", lambda: None)
    with pytest.raises(PublishError, match="no HuggingFace token"):
        publish_run(_run(tmp_path), "org/artifacts")


def test_a_token_beside_the_checkout_is_used(tmp_path, monkeypatch):
    """Where an operator can leave one without git ever seeing it."""
    from model_partition import publish

    checkout = tmp_path / "checkout"
    (checkout / ".git").mkdir(parents=True)
    (tmp_path / publish.TOKEN_FILE).write_text("hf_example\n")
    monkeypatch.chdir(checkout)
    monkeypatch.setattr("huggingface_hub.get_token", lambda: None)
    assert publish.find_token() == "hf_example"

    (tmp_path / publish.TOKEN_FILE).unlink()
    assert publish.find_token() is None


def test_the_upload_carries_a_readme_describing_the_artifacts(tmp_path, monkeypatch):
    """Someone who finds the repo should be able to use it without this repo."""
    from model_partition import publish

    root = _run(tmp_path)
    seen = {}

    class FakeApi:
        def create_repo(self, repo_id, **kwargs):
            seen["repo"] = (repo_id, kwargs)

        def upload_large_folder(self, **kwargs):
            seen["upload"] = kwargs
            seen["readme"] = (root / "README.md").read_text()

    monkeypatch.setattr(publish, "_api", lambda: (FakeApi(), "token"))
    result = publish_run(root, "org/artifacts", private=True)

    assert seen["repo"] == ("org/artifacts", {"repo_type": "dataset", "private": True,
                                              "exist_ok": True})
    assert "logs/**" in seen["upload"]["ignore_patterns"]
    assert "modules/<group>/" in seen["readme"]
    assert "396 checks passed" in seen["readme"]
    assert result.url.endswith("org/artifacts")
    # Written for the upload and not left behind in the run.
    assert not (root / "README.md").exists()
