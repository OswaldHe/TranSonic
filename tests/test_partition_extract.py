# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for per-module code extraction and deduplication."""

from pathlib import Path

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


def test_source_imports_on_its_own(tiny_run, tmp_path):
    """The launcher imports it, so a file that cannot import is useless."""
    from model_partition.runtime.launcher import load_source

    groups = extract(tiny_run.graph, tiny_run.build_model(), tmp_path / "modules",
                     run_root=tiny_run.layout.root)
    decoder = next(g for g in groups if g.class_name == "TinyDecoderLayer")
    module = load_source(decoder.directory, decoder.source_module)
    assert hasattr(module, decoder.class_name)


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


def test_source_is_the_implementation_and_names_where_it_came_from(tiny_run, tmp_path):
    """source.py is the code that runs, so it has to be the real file, importable."""
    groups = extract(tiny_run.graph, tiny_run.build_model(), tmp_path / "modules",
                     run_root=tiny_run.layout.root)
    decoder = next(g for g in groups if g.class_name == "TinyDecoderLayer")
    source = (decoder.directory / "source.py").read_text()
    assert decoder.launchable
    assert "copied verbatim from" in source
    assert "tiny_llm.py" in source
    assert "class TinyDecoderLayer" in source
    # A verbatim file keeps its imports, which is what makes it importable.
    assert "import torch" in source


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
    # The module's reference is its *last* submodule's output.
    for entry in outputs:
        store.blob_path(entry).write_bytes(b"\x7f" * entry.nbytes)

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
    sources, names, files = collect_sources(model)
    assert len(names) == len(set(names))
    assert len(files) == len(set(files))
    assert any("TinyCausalLM" in name for name in names)
    assert any(name.startswith("tests.fixtures.tiny_llm") or "tiny_llm" in name
               for name in names), names


def test_the_originating_files_are_copied_verbatim(tiny_run, tmp_path):
    """The extracted source.py does not import; the real file does."""
    from model_partition.extract import SOURCE_FILES_DIR, extract

    groups = extract(tiny_run.graph, tiny_run.build_model(), tmp_path / "modules",
                     run_root=tiny_run.layout.root, sample_ids=tiny_run.sample_ids)
    copies = tmp_path / "modules" / SOURCE_FILES_DIR
    assert copies.is_dir()

    wanted = {Path(path) for group in groups for path in group.source_files}
    assert wanted, "source_files should not be empty"
    for origin in wanted:
        assert (copies / origin.name).read_bytes() == origin.read_bytes()


def test_meta_records_where_the_implementation_came_from(tiny_run, tmp_path):
    import yaml

    from model_partition.extract import extract

    groups = extract(tiny_run.graph, tiny_run.build_model(), tmp_path / "modules",
                     run_root=tiny_run.layout.root, sample_ids=tiny_run.sample_ids)
    for group in groups:
        meta = yaml.safe_load((group.directory / "meta.yaml").read_text())
        assert meta["source_files"], f"{group.directory.name} records no provenance"
        assert (group.directory / "source.py").read_text().count(
            meta["source_files"][0]) >= 1


def test_collect_sources_respects_the_class_cap(tiny_run):
    _, names, _ = collect_sources(tiny_run.build_model(), max_classes=2)
    assert len(names) <= 2


# -- the per-module README ---------------------------------------------------


def _readme(run, tmp_path):
    from model_partition.extract import extract

    groups = extract(run.graph, run.build_model(), tmp_path / "modules",
                     run_root=run.layout.root, sample_ids=run.sample_ids,
                     weight_tensors=run.bundle.weights,
                     param_names=run.bundle.weight_params)
    return groups, {g.signature: (g.directory / "README.md").read_text() for g in groups}


def test_every_group_gets_a_readme(tiny_run, tmp_path):
    groups, readmes = _readme(tiny_run, tmp_path)
    assert len(readmes) == len(groups)
    assert all(text.strip() for text in readmes.values())


def test_the_readme_states_purpose_preconditions_and_postconditions(tiny_run, tmp_path):
    _, readmes = _readme(tiny_run, tmp_path)
    for text in readmes.values():
        assert "## What this module does" in text
        assert "## Pre-conditions" in text
        assert "## Post-conditions" in text
        assert "python verify.py" in text


def test_the_readme_names_the_modules_and_dataflow(tiny_run, tmp_path):
    groups, readmes = _readme(tiny_run, tmp_path)
    for group in groups:
        text = readmes[group.signature]
        for module_id in group.module_ids:
            assert f"`{module_id}`" in text
        module = tiny_run.graph.by_id(group.module_ids[0])
        for tensor in module.inputs + module.outputs:
            assert f"`{tensor}`" in text


def test_the_readme_records_how_many_weights_the_module_needs(tiny_run, tmp_path):
    groups, readmes = _readme(tiny_run, tmp_path)
    group = next(g for g in groups if tiny_run.bundle.weight_params.get(g.module_ids[0]))
    count = len(tiny_run.bundle.weight_params[group.module_ids[0]])
    assert f"{count} tensor(s)" in readmes[group.signature]


def test_the_readme_flags_a_module_that_cannot_be_verified(tiny_run, tmp_path):
    """A functional module has no reference, and its README has to say so."""
    from model_partition.planner.graph import ModuleNode

    tiny_run.graph.modules.append(ModuleNode(
        id="layers.0.combine", kind="mlp", layer_indices=[0], code_signature="combine",
    ))
    _, readmes = _readme(tiny_run, tmp_path)
    assert "functional" in readmes["combine"]
    assert "cannot check it" in readmes["combine"]


def test_the_readme_explains_a_parallel_group(tiny_split_run, tmp_path):
    graph = tiny_split_run.graph
    parallel = [m for m in graph.partitioned_modules if m.is_parallel]
    assert parallel, "fixture should produce a parallel expert group"
    _, readmes = _readme(tiny_split_run, tmp_path)
    text = readmes[parallel[0].code_signature]
    assert "parallel" in text and "submodule=" in text


def test_the_readme_is_regenerated_unlike_inference_py(tiny_run, tmp_path):
    """It belongs to the harness, so a stale copy must not survive."""
    groups, _ = _readme(tiny_run, tmp_path)
    directory = groups[0].directory
    (directory / "README.md").write_text("stale\n")
    (directory / "inference.py").write_text("# agent's fix\n")

    _readme(tiny_run, tmp_path)
    assert (directory / "README.md").read_text() != "stale\n"
    assert (directory / "inference.py").read_text() == "# agent's fix\n"


# -- the reference covers the whole group -------------------------------------


def test_source_covers_every_submodule_of_a_group(tiny_run, tmp_path):
    """A group spans a norm and the computation it feeds; both belong in source.py.

    Taking only the first submodule's source left the attention groups documented by
    an RMSNorm, which is no use to anyone porting them.
    """
    from model_partition.extract import extract

    graph = tiny_run.graph
    group_module = next((m for m in graph.partitioned_modules if len(m.submodules) > 1), None)
    if group_module is None:
        pytest.skip("this plan has no multi-submodule group")

    groups = extract(graph, tiny_run.build_model(), tmp_path / "modules",
                     run_root=tiny_run.layout.root, sample_ids=tiny_run.sample_ids)
    group = next(g for g in groups if group_module.id in g.module_ids)
    source = (group.directory / "source.py").read_text()
    assert len(group.classes) > 1, group.classes
    for class_name in group.classes:
        assert class_name.split(".")[-1] in source


def test_the_group_is_named_after_what_it_computes(tiny_run, tmp_path):
    """Not after the normalization that happens to be listed first."""
    from model_partition.extract import extract
    from model_partition.trace import _lookup

    graph = tiny_run.graph
    group_module = next((m for m in graph.partitioned_modules if len(m.submodules) > 1), None)
    if group_module is None:
        pytest.skip("this plan has no multi-submodule group")

    model = tiny_run.build_model()
    groups = extract(graph, model, tmp_path / "modules",
                     run_root=tiny_run.layout.root, sample_ids=tiny_run.sample_ids)
    group = next(g for g in groups if group_module.id in g.module_ids)

    heaviest = max((_lookup(model, name) for name in group_module.submodules),
                   key=lambda m: sum(p.numel() for p in m.parameters()))
    assert group.class_name == type(heaviest).__name__


def test_the_principal_submodule_is_the_one_with_the_parameters():
    from model_partition.extract import _principal

    class Fake:
        def __init__(self, n):
            self._n = n

        def parameters(self):
            import torch

            return [torch.zeros(self._n)]

    small, large = Fake(4), Fake(400)
    assert _principal([small, large]) is large
    assert _principal([large, small]) is large
