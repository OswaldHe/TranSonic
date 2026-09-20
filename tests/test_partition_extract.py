# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for per-module code extraction and deduplication."""

import pytest
import yaml

from model_partition.extract import collect_sources, extract

pytest.importorskip("torch")


def test_one_directory_per_signature_group(tiny_run, tmp_path):
    out = tmp_path / "modules"
    groups = extract(tiny_run.graph, tiny_run.build_model(), out,
                     run_root=tiny_run.layout.root)
    assert len(groups) == len(tiny_run.graph.signature_groups())
    for group in groups:
        assert group.directory and group.directory.is_dir()
        assert (group.directory / "inference.py").is_file()
        assert (group.directory / "verify.py").is_file()
        assert (group.directory / "source.py").is_file()
        assert (group.directory / "meta.yaml").is_file()


def test_repeated_layers_share_one_implementation(tiny_deep_run, tmp_path):
    """12 layers with identical structure must not yield 12 copies."""
    run = tiny_deep_run
    groups = extract(run.graph, run.build_model(), tmp_path / "modules",
                     run_root=run.layout.root)
    decoder = [g for g in groups if len(g.module_ids) > 1]
    assert decoder, "expected the repeated decoder layers to be deduplicated"
    assert len(decoder[0].module_ids) == 12
    assert len(groups) < len(run.graph.partitioned_modules)


def test_hybrid_stack_yields_one_implementation_per_variant(tiny_moe_run, tmp_path):
    run = tiny_moe_run
    groups = extract(run.graph, run.build_model(), tmp_path / "modules",
                     run_root=run.layout.root)
    decoder_groups = [g for g in groups if any("layers" in mid for mid in g.module_ids)]
    assert len(decoder_groups) == 2  # dense layers and MoE layers
    classes = {g.class_name for g in decoder_groups}
    assert classes == {"TinyDecoderLayer"}


def test_index_summarizes_every_group(tiny_run, tmp_path):
    out = tmp_path / "modules"
    groups = extract(tiny_run.graph, tiny_run.build_model(), out,
                     run_root=tiny_run.layout.root)
    payload = yaml.safe_load((out / "index.yaml").read_text())
    assert payload["n_groups"] == len(groups)
    assert payload["n_modules"] == len(tiny_run.graph.partitioned_modules)
    assert len(payload["groups"]) == len(groups)
    assert all(entry["directory"] for entry in payload["groups"])


def test_meta_records_module_ids_and_layers(tiny_deep_run, tmp_path):
    run = tiny_deep_run
    groups = extract(run.graph, run.build_model(), tmp_path / "modules",
                     run_root=run.layout.root)
    decoder = next(g for g in groups if len(g.module_ids) > 1)
    meta = yaml.safe_load((decoder.directory / "meta.yaml").read_text())
    assert meta["module_ids"] == decoder.module_ids
    assert meta["layer_indices"] == list(range(12))
    assert meta["kind"] == "decoder_layers"
    assert meta["submodules"]


def test_source_contains_the_real_implementation(tiny_run, tmp_path):
    groups = extract(tiny_run.graph, tiny_run.build_model(), tmp_path / "modules",
                     run_root=tiny_run.layout.root)
    decoder = next(g for g in groups if g.class_name == "TinyDecoderLayer")
    source = (decoder.directory / "source.py").read_text()
    assert "class TinyDecoderLayer" in source
    assert "def forward" in source
    # Children's classes are collected too, so the module reads end to end.
    assert "class TinyAttention" in source
    assert decoder.source_lines > 20


def test_source_header_records_provenance(tiny_run, tmp_path):
    """Each extracted group names the files its source came from."""
    groups = extract(tiny_run.graph, tiny_run.build_model(), tmp_path / "modules",
                     run_root=tiny_run.layout.root)
    decoder = next(g for g in groups if g.class_name == "TinyDecoderLayer")
    source = (decoder.directory / "source.py").read_text()
    assert "Provenance:" in source
    assert "tiny_llm.py" in source


def _extract(run, tmp_path):
    return extract(run.graph, run.build_model(), tmp_path / "modules",
                   run_root=run.layout.root, sample_ids=run.sample_ids,
                   weight_tensors=run.bundle.weights)


def _script(run, tmp_path, name, *args, group_class="TinyDecoderLayer"):
    import subprocess
    import sys

    groups = _extract(run, tmp_path)
    group = next(g for g in groups if g.class_name == group_class)
    return subprocess.run(
        [sys.executable, str(group.directory / name), "--run", str(run.layout.root), *args],
        capture_output=True, text=True, timeout=300,
    )


def test_generated_scripts_are_valid_python_and_name_their_modules(tiny_run, tmp_path):
    import ast

    for group in _extract(tiny_run, tmp_path):
        for name in ("inference.py", "verify.py"):
            text = (group.directory / name).read_text()
            ast.parse(text)  # raises on a broken template render
            assert str(tiny_run.layout.root) in text
            for module_id in group.module_ids:
                assert module_id in text


def test_inference_script_runs_the_module_from_its_dumps(tiny_run, tmp_path):
    """The inference code must actually run, not just parse."""
    completed = _script(tiny_run, tmp_path, "inference.py")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "output shape=" in completed.stdout
    assert "dumped tensors" in completed.stdout


def test_inference_script_can_save_its_output(tiny_run, tmp_path):
    out = tmp_path / "out.bin"
    completed = _script(tiny_run, tmp_path, "inference.py", "--save", str(out))
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert out.is_file() and out.stat().st_size > 0


def test_verify_script_passes_against_intact_artifacts(tiny_run, tmp_path):
    completed = _script(tiny_run, tmp_path, "verify.py")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "PASS" in completed.stdout
    assert "match the dumped reference" in completed.stdout


def test_verify_script_fails_when_the_reference_disagrees(tiny_run, tmp_path):
    """The verifier is a gate: it must report a wrong result, not wave it through."""
    store = tiny_run.bundle.store
    target = next(m.id for m in tiny_run.graph.partitioned_modules
                  if m.kind == "decoder_layers")
    outputs = store.find(role="output", module_id=target)
    assert outputs
    blob = store.blob_path(outputs[0])
    blob.write_bytes(b"\x7f" * outputs[0].nbytes)

    completed = _script(tiny_run, tmp_path, "verify.py")
    assert completed.returncode == 1, completed.stdout + completed.stderr
    assert "FAIL" in completed.stdout


def test_verify_script_reports_unusable_artifacts(tiny_run, tmp_path):
    completed = _script(tiny_run, tmp_path, "verify.py", "--sample", "no-such-sample")
    assert completed.returncode == 2
    assert "No trace records" in completed.stdout + completed.stderr


def test_verify_script_covers_every_module_and_sample(tiny_run, tmp_path):
    completed = _script(tiny_run, tmp_path, "verify.py", "--all-modules", "--all-samples")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    for sample_id in tiny_run.sample_ids:
        assert f"@{sample_id}" in completed.stdout


def test_verify_script_tolerance_can_be_loosened(tiny_run, tmp_path):
    completed = _script(tiny_run, tmp_path, "verify.py", "--rtol", "0.5", "--atol", "0.5")
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_group_directory_names_are_readable_and_unique(tiny_moe_run, tmp_path):
    groups = extract(tiny_moe_run.graph, tiny_moe_run.build_model(), tmp_path / "modules",
                     run_root=tiny_moe_run.layout.root)
    names = [g.directory.name for g in groups]
    assert len(set(names)) == len(names)
    assert any("decoder_layers" in name for name in names)
    assert all(name[:2].isdigit() for name in names)


def test_missing_submodule_is_recorded_not_fatal(tiny_run, tmp_path):
    graph = tiny_run.graph
    for module in graph.partitioned_modules:
        module.submodules = ["model.nonexistent"]
    groups = extract(graph, tiny_run.build_model(), tmp_path / "modules",
                     run_root=tiny_run.layout.root)
    assert all(g.class_name == "(not instantiated)" for g in groups)
    assert all(g.source_lines == 0 for g in groups)


def test_collect_sources_deduplicates_repeated_child_classes(tiny_run):
    model = tiny_run.build_model()
    sources, names, provenance = collect_sources(model)
    assert len(names) == len(set(names))
    assert "tiny_llm" in provenance
    assert any("TinyCausalLM" in name for name in names)


def test_collect_sources_respects_the_class_cap(tiny_run):
    _, names, _ = collect_sources(tiny_run.build_model(), max_classes=2)
    assert len(names) <= 2
