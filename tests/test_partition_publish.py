# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for publishing a run's artifacts."""

import json
from pathlib import Path

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


def test_a_dry_run_does_not_materialize_a_selection(tmp_path, monkeypatch):
    """Copying a selection's weights out of a 475 GiB checkpoint is not a size estimate."""
    from model_partition import publish

    root = _run(tmp_path)
    called: list[object] = []
    monkeypatch.setattr(publish, "materialize_weights",
                        lambda *a, **k: called.append(a) or 0)
    lines: list[str] = []
    publish_run(root, "org/artifacts", dry_run=True, module_ids=["layers.0"],
                skip_check=True, report=lines.append)
    assert not called
    assert any("would materialize" in line for line in lines)


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
    # And how to get the weights. Most of a module's parameters stay in the checkpoint —
    # this set is published without a second copy of it — so a reader who is not told
    # about `fetch_weights.py` has module directories that cannot reproduce their own
    # reference and no stated reason why.
    assert "fetch_weights.py" in seen["readme"]
    assert result.url.endswith("org/artifacts")
    # Written for the upload and not left behind in the run.
    assert not (root / "README.md").exists()


def test_a_selection_carries_only_its_own_modules(tiny_run, tmp_path):
    """A whole run of a 475 GiB model is not a useful thing to hand anybody. A few
    modules of each kind, each verifiable on its own, is."""
    from model_partition.extract import extract
    from model_partition.publish import collect, representative_modules

    extract(tiny_run.graph, tiny_run.build_model(), tiny_run.layout.modules_dir,
            run_root=tiny_run.layout.root, sample_ids=tiny_run.sample_ids,
            weight_tensors=tiny_run.bundle.weights)
    root = tiny_run.layout.root

    every, _ = collect(root)
    chosen = representative_modules(root)
    assert chosen, "no representative modules found"
    some, _ = collect(root, module_ids=chosen[:1])
    assert len(some) < len(every)

    names = {p.relative_to(root).as_posix() for p in some}
    # Its own activations, and nobody else's.
    assert any(f"trace/activations/{chosen[0]}/" in n for n in names)
    others = [m for m in representative_modules(root) if m != chosen[0]]
    for other in others:
        assert not any(f"trace/activations/{other}/" in n for n in names), other
    # And the things a module needs whatever it is.
    assert "plan/partition_graph.yaml" in names
    assert "modules/index.yaml" in names


def test_the_selection_prefers_a_module_the_trace_still_holds(tiny_deep_run):
    """Retention deletes all but a few layers, and the group's first module is often one
    of the ones it dropped. Publishing that one materializes weights for a module whose
    reference output is no longer there."""
    from model_partition.extract import extract
    from model_partition.publish import representative_modules

    run = tiny_deep_run
    extract(run.graph, run.build_model(), run.layout.modules_dir,
            run_root=run.layout.root, sample_ids=run.sample_ids,
            weight_tensors=run.bundle.weights, bundle=run.bundle)
    root = run.layout.root
    chosen = representative_modules(root)
    assert chosen

    # The decoder group holds 12 modules. Drop the chosen one's records, as retention
    # would, and a sibling of the same group is picked in its place.
    group = next(ids for ids in run.graph.signature_groups().values() if len(ids) > 1)
    dropped = next(m for m in chosen if m in group)
    bundle = run.bundle
    bundle.records = [r for r in bundle.records if r.module_id != dropped]
    bundle.save()

    again = representative_modules(root)
    assert dropped not in again
    assert len(again) == len(chosen)
    assert set(again) & set(group)


def test_publishing_a_subset_writes_its_weights_into_the_run(tiny_run, tmp_path):
    """A run that read its weights from the checkpoint cannot be verified away from it,
    so the selection's weights are dumped beside the activations they are checked
    against before anything is uploaded."""
    from model_partition.publish import materialize_weights
    from model_partition.runtime.module_runner import TraceBundle

    root = tiny_run.layout.root
    bundle = TraceBundle.load(root / "trace")
    module_id = next(m.id for m in tiny_run.graph.partitioned_modules
                     if bundle.weight_params.get(m.id))

    # Leave the store as `cache_weights: false` leaves it: activations and an index of
    # which tensors each module owns, with the values still in the checkpoint.
    kept = [e for e in bundle.store.entries
            if not (e.role == "weight" and e.module_id == module_id)]
    dropped = len(bundle.store.entries) - len(kept)
    assert dropped, "the fixture dumped no weights for this module"
    bundle.store.entries = kept
    bundle.weights.pop(module_id, None)
    bundle.save()
    assert materialize_weights(root, [module_id]) > 0
    filled = TraceBundle.load(root / "trace")
    entries = len(filled.store.entries)

    # And again adds nothing: a module's weights are not read twice, which for a 94 GiB
    # table is the difference between a publish and an afternoon.
    materialize_weights(root, [module_id])
    assert len(TraceBundle.load(root / "trace").store.entries) == entries
    added = dropped
    assert added > 0
    again = TraceBundle.load(root / "trace")
    names = {(e.extra or {}).get("param") for e in again.store.entries
             if e.module_id == module_id}
    assert set(bundle.weight_params[module_id]) <= names


def test_a_run_that_did_not_pass_is_refused_before_its_weights_are_read(
        tmp_path, monkeypatch):
    """Both refusals come before anything is written.

    Materializing a selection reads tens or hundreds of gigabytes out of the checkpoint,
    so reaching "this run did not pass" afterwards costs hours and a disk for nothing.
    """
    from model_partition import publish

    root = _run(tmp_path, passed=False)
    called: list[object] = []
    monkeypatch.setattr(publish, "materialize_weights",
                        lambda *a, **k: called.append(a) or 0)
    with pytest.raises(PublishError, match="did not pass"):
        publish_run(root, "org/artifacts", module_ids=["layers.0"], skip_check=True)
    assert not called


def test_a_selection_over_the_ceiling_is_refused_before_its_weights_are_read(
        tmp_path, monkeypatch):
    """What is on disk already is a floor for what will be there once the weights are in."""
    from model_partition import publish

    root = _run(tmp_path, blob_bytes=4096)
    called: list[object] = []
    monkeypatch.setattr(publish, "materialize_weights",
                        lambda *a, **k: called.append(a) or 0)
    with pytest.raises(PublishError, match="exceeds the"):
        # `upload_all` sends the whole run while the selection says which modules also
        # carry their weights, so the ceiling applies to everything on disk.
        publish_run(root, "org/artifacts", max_bytes=1024, upload_all=True,
                    module_ids=["layers.0"], skip_check=True)
    assert not called


def test_materialization_reads_the_revision_the_trace_was_taken_from(tmp_path, monkeypatch):
    """A spec may name a moving branch, or name nothing at all.

    Re-ingesting it would read weights from wherever that points now, and those do not
    belong beside these activations — so the resolved revision in `run.yaml` wins.
    """
    from model_partition import ingest as ingest_module
    from model_partition.publish import _weight_source

    root = tmp_path / "run"
    root.mkdir()
    (root / "run.yaml").write_text(
        "spec:\n  source: hf:org/model\n  loader: transformers\n  revision: main\n"
        "revision: dba1be0a40aa45a94ad051997016db3960a90277\n"
    )
    seen: list[str] = []

    def fake_ingest(spec, *a, **k):
        seen.append(spec.revision or "")
        raise RuntimeError("stop here")

    monkeypatch.setattr(ingest_module, "ingest", fake_ingest)
    with pytest.raises(PublishError, match="cannot be reached"):
        _weight_source(root)
    assert seen == ["dba1be0a40aa45a94ad051997016db3960a90277"]


def test_a_fetched_checkpoint_is_never_republished(tmp_path):
    """`hf/` is what `fetch_weights.py` downloads on the reader's machine.

    Uploading it back would publish a copy of the model, which is the one thing an
    artifact set exists to avoid.
    """
    from model_partition.publish import DEFAULT_EXCLUDE

    root = _run(tmp_path)
    (root / "hf").mkdir()
    (root / "hf" / "model-00001-of-00002.safetensors").write_bytes(b"\x00" * 2048)

    # And the Hub client's own resume bookkeeping, which it writes into the folder it is
    # uploading: thousands of files, none of them content.
    (root / ".cache" / "huggingface").mkdir(parents=True)
    (root / ".cache" / "huggingface" / "upload.json").write_text("{}")

    files, _total = collect(root)
    names = {p.relative_to(root).as_posix() for p in files}
    assert not any(name.startswith("hf/") for name in names)
    assert not any(name.startswith(".cache/") for name in names)
    assert "hf/**" in DEFAULT_EXCLUDE and ".cache/**" in DEFAULT_EXCLUDE


def test_the_cli_default_exclusion_matches_the_library_one():
    """The CLI spells the default out so it need not import `publish` eagerly."""
    import click

    from model_partition.cli import upload
    from model_partition.publish import DEFAULT_EXCLUDE

    option = next(p for p in upload.params
                  if isinstance(p, click.Option) and p.name == "exclude")
    assert tuple(option.default) == tuple(DEFAULT_EXCLUDE)


def test_the_self_containment_check_brings_the_fetched_checkpoint_along(tmp_path, monkeypatch):
    """`hf/` is not uploaded — republishing the model is what an artifact set avoids — but a
    reader runs `fetch_weights.py` and has it. Checking without it asked whether the modules
    work having skipped a documented step, so a `cache_weights: false` run failed here for
    every group and the only way to publish was `--skip-check`."""
    from model_partition import publish

    root = _run(tmp_path)
    (root / "hf").mkdir()
    (root / "hf" / "model-00001-of-00002.safetensors").write_bytes(b"\x00" * 64)
    # What `fetch_weights.py` leaves behind beside the shards. A directory cannot be
    # hardlinked and cannot be copied as a file either, so including it crashed the check.
    (root / "hf" / ".cache" / "huggingface").mkdir(parents=True)
    (root / "hf" / ".cache" / "huggingface" / "download.json").write_text("{}")

    files, _total = collect(root)
    assert not any(p.name.endswith(".safetensors") for p in files), "hf/ is uploaded"

    linked: list[str] = []

    def fake_groups(run_dir, selected):
        # Record what the check copied, then claim there is nothing to run.
        for path in sorted(Path(run_dir).rglob("*")):
            if path.is_file():
                linked.append(path.relative_to(run_dir).as_posix())
        return {}

    monkeypatch.setattr(publish, "_published_groups", fake_groups)
    assert publish.check_self_contained(root, files, ["layers.0"]) == []
    assert "hf/model-00001-of-00002.safetensors" in linked
    assert "modules/00-attn/source.py" in linked


def test_the_upload_limits_its_own_concurrency(tmp_path, monkeypatch):
    """A published run is thousands of small feature maps, so the Hub's request-per-minute
    limit binds long before bandwidth does. The client's default worker count scales with
    the core count, and on 16 cores it burst through a free account's 1000 requests per 5
    minutes and aborted a third of the way through 16 000 files."""
    from model_partition import publish

    root = _run(tmp_path)
    seen = {}

    class FakeApi:
        def create_repo(self, repo_id, **kwargs):
            pass

        def upload_large_folder(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(publish, "_api", lambda: (FakeApi(), "token"))
    publish_run(root, "org/artifacts")
    assert seen["num_workers"] == publish.UPLOAD_WORKERS <= 4

    publish_run(root, "org/artifacts", workers=8)
    assert seen["num_workers"] == 8
