# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the partition CLI and its registration under autohelix."""

import json

import pytest
import yaml
from click.testing import CliRunner

from model_partition.cli import (
    MODELS_DIR,
    build_options,
    partition,
    resolve_spec,
)

pytest.importorskip("torch")


@pytest.fixture
def runner():
    return CliRunner()


# -- registration ------------------------------------------------------------


def test_partition_is_registered_under_autohelix():
    from autohelix.cli import main

    assert "partition" in main.commands
    assert "Partition" in (main.commands["partition"].help or "")


def test_autohelix_partition_lists_subcommands(runner):
    from autohelix.cli import main

    result = runner.invoke(main, ["partition", "--help"])
    assert result.exit_code == 0
    for command in ("run", "plan", "inspect", "replay", "report", "tokens", "list"):
        assert command in result.output


def test_autohelix_help_does_not_import_torch():
    """The lazy group keeps heavyweight imports out of unrelated invocations."""
    import subprocess
    import sys

    code = (
        "import sys; from autohelix.cli import main; "
        "print('torch' in sys.modules)"
    )
    completed = subprocess.run([sys.executable, "-c", code],
                               capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "False"


# -- spec resolution ---------------------------------------------------------


def test_bundled_specs_resolve_by_name():
    for path in MODELS_DIR.glob("*.yaml"):
        spec = resolve_spec(path.stem)
        assert spec.source


def test_every_bundled_spec_is_valid_yaml_and_parses():
    for path in MODELS_DIR.glob("*.yaml"):
        payload = yaml.safe_load(path.read_text())
        assert payload["source"].startswith("hf:")
        assert "inputs" in payload


def test_deepseek_specs_say_what_is_reachable_on_one_machine():
    """Both exceed this machine's memory, and the notes have to be honest about it."""
    for name in ("deepseek-v4-flash", "deepseek-v4.1-flash"):
        notes = resolve_spec(name).notes
        assert "inspect" in notes and "plan" in notes
        assert "trace" in notes


def test_deepseek_v41_uses_the_vendor_inference_code():
    """transformers does not recognize deepseek_v41, so vendor code is the only path."""
    spec = resolve_spec("deepseek-v4.1-flash")
    assert spec.loader == "repo_code"
    assert spec.entry == "inference/model.py"
    assert spec.trust_remote_code is True


def test_deepseek_v4_uses_the_vendor_inference_code():
    """transformers names its deepseek_v4 modules the HF way, so it loads nothing."""
    spec = resolve_spec("deepseek-v4-flash")
    assert spec.loader == "repo_code"
    assert spec.entry == "inference/model.py"
    assert spec.trust_remote_code is True


def test_every_bundled_spec_states_how_it_wants_to_be_partitioned():
    from model_partition.cli import MODELS_DIR
    from model_partition.spec import load_spec

    for path in MODELS_DIR.glob("*.yaml"):
        spec = load_spec(path)
        assert spec.partition.split_attention_ffn, path.name
        assert "attention" in spec.partition.prompt.lower(), path.name


def test_qwen_specs_leave_the_loader_on_auto():
    for name in ("qwen3.5-0.8b", "qwen3.8-27b"):
        assert resolve_spec(name).loader == "auto"


def test_spec_resolves_from_a_path(tmp_path):
    path = tmp_path / "custom.yaml"
    path.write_text(yaml.safe_dump({"source": "hf:a/b"}))
    assert resolve_spec(str(path)).repo_id == "a/b"


def test_spec_resolves_from_a_repo_id():
    assert resolve_spec("hf:org/model").repo_id == "org/model"
    assert resolve_spec("org/model").repo_id == "org/model"


def test_spec_resolves_from_a_local_directory(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    assert resolve_spec(str(tmp_path)).is_local


def test_unknown_spec_lists_the_bundled_names():
    with pytest.raises(Exception) as excinfo:
        resolve_spec("not-a-model")
    message = str(excinfo.value)
    assert "Unknown spec" in message
    assert "qwen3.8-27b" in message


# -- options -----------------------------------------------------------------


def test_defaults_file_populates_options():
    options = build_options()
    assert options.judge_kind == "claude"
    assert options.min_judge_score == 4
    assert options.retention_layers == (1, 5)
    assert options.headroom == 0.80


def test_cli_overrides_beat_defaults():
    options = build_options(None, max_iterations=9, judge_kind="stub")
    assert options.max_iterations == 9
    assert options.judge_kind == "stub"


def test_none_overrides_are_ignored():
    assert build_options(None, max_iterations=None).max_iterations == build_options().max_iterations


def test_unknown_keys_are_dropped(tmp_path):
    path = tmp_path / "defaults.yaml"
    path.write_text(yaml.safe_dump({"loop": {"max_iterations": 2, "nonsense": True}}))
    options = build_options(path)
    assert options.max_iterations == 2
    assert not hasattr(options, "nonsense")


def test_retention_layers_from_yaml_become_a_tuple(tmp_path):
    path = tmp_path / "defaults.yaml"
    path.write_text(yaml.safe_dump({"loop": {"retention_layers": [2, 4, 6]}}))
    assert build_options(path).retention_layers == (2, 4, 6)


# -- commands ----------------------------------------------------------------


def test_list_shows_bundled_specs(runner):
    result = runner.invoke(partition, ["list"])
    assert result.exit_code == 0
    assert "qwen3.8-27b" in result.output
    assert "deepseek-v4.1-flash" in result.output


def test_a_disabled_spec_refuses_to_run(runner, tmp_path):
    path = tmp_path / "off.yaml"
    path.write_text(yaml.safe_dump({
        "source": "hf:a/b", "name": "off", "enabled": False,
        "notes": "not ready",
    }))
    result = runner.invoke(partition, ["run", str(path)])
    assert result.exit_code != 0
    combined = str(result.output) + str(result.exception)
    assert "enabled: false" in combined and "not ready" in combined


def test_report_renders_stage_status(runner, tmp_path):
    from model_partition.layout import RunLayout
    from model_partition.loop.state import OK, LoopState

    layout = RunLayout.create("m", tmp_path).ensure()
    state = LoopState(slug="m", iteration=2)
    state.mark("ingest", OK, "h", "did the thing")
    state.save(layout.state_file)

    result = runner.invoke(partition, ["report", str(layout.root)])
    assert result.exit_code == 0
    assert "iteration : 2" in result.output
    assert "did the thing" in result.output


def test_report_json_mode(runner, tmp_path):
    from model_partition.layout import RunLayout
    from model_partition.loop.state import OK, LoopState

    layout = RunLayout.create("m", tmp_path).ensure()
    LoopState(slug="m").mark("ingest", OK) or None
    state = LoopState(slug="m")
    state.mark("ingest", OK)
    state.save(layout.state_file)

    result = runner.invoke(partition, ["report", str(layout.root), "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output)["slug"] == "m"


def test_tokens_requires_a_completed_run(runner, tmp_path):
    from model_partition.layout import RunLayout

    layout = RunLayout.create("m", tmp_path).ensure()
    result = runner.invoke(partition, ["tokens", str(layout.root)])
    assert result.exit_code != 0
    assert "run the loop first" in str(result.output) + str(result.exception)


def test_tokens_prints_the_transcript(runner, tmp_path):
    from model_partition.layout import RunLayout

    layout = RunLayout.create("m", tmp_path).ensure()
    layout.tokens_file.write_text("sample: s0\ntext  : hello\n")
    result = runner.invoke(partition, ["tokens", str(layout.root)])
    assert result.exit_code == 0
    assert "hello" in result.output


def test_plan_command_writes_a_graph(runner, tiny_run, tmp_path):
    result = runner.invoke(partition, [
        "plan", str(tiny_run.repo),
        "--artifact-root", str(tmp_path / "artifacts"),
        "--gpu-memory-gib", "1", "--device", "cpu",
    ])
    assert result.exit_code == 0, result.output + str(result.exception)
    assert "deduplicated implementation group" in result.output
    graphs = list((tmp_path / "artifacts").rglob("partition_graph.yaml"))
    assert graphs


def test_replay_reports_a_bad_module_id(runner, tiny_run):
    result = runner.invoke(partition, ["replay", str(tiny_run.layout.root), "no-such-module"])
    assert result.exit_code != 0
    assert "not in the plan" in str(result.output) + str(result.exception)


def test_replay_verifies_a_module_from_artifacts_alone(runner, tiny_run):
    """No checkpoint and no model: the module directory and the dumps are the input."""
    from model_partition.extract import extract

    module_id = next(m.id for m in tiny_run.graph.partitioned_modules
                     if m.kind == "decoder_layers")
    extract(tiny_run.graph, tiny_run.build_model(), tiny_run.layout.modules_dir,
            run_root=tiny_run.layout.root, sample_ids=tiny_run.sample_ids,
            weight_tensors=tiny_run.bundle.weights)
    (tiny_run.repo / "model.safetensors").unlink()
    result = runner.invoke(partition, ["replay", str(tiny_run.layout.root), module_id])
    assert result.exit_code == 0, result.output + str(result.exception)
    assert ": ok" in result.output


# -- the partition instruction on the command line ---------------------------


def test_partition_prompt_reaches_the_options():
    options = build_options(None, partition_prompt="split the layers")
    assert options.partition_prompt == "split the layers"


def test_partition_prompt_can_come_from_a_file(tmp_path):
    path = tmp_path / "how.md"
    path.write_text("Split attention from the FFN.\n")
    assert build_options(None, partition_prompt_file=path).partition_prompt == \
        "Split attention from the FFN.\n"


def test_an_explicit_prompt_wins_over_the_file(tmp_path):
    path = tmp_path / "how.md"
    path.write_text("from the file\n")
    options = build_options(None, partition_prompt="inline", partition_prompt_file=path)
    assert options.partition_prompt == "inline"


def test_the_prompt_file_is_not_mistaken_for_an_option():
    """partition_prompt_file is a CLI convenience, not a LoopOptions field."""
    from dataclasses import fields

    from model_partition.loop.stages import LoopOptions

    assert "partition_prompt_file" not in {f.name for f in fields(LoopOptions)}


def test_split_attention_ffn_is_a_flag():
    assert build_options(None, split_attention_ffn=True).split_attention_ffn is True
    assert build_options(None).split_attention_ffn is False


# -- bundled resources -------------------------------------------------------


def test_bundled_resources_resolve_in_the_source_tree():
    from model_partition.cli import CONFIG_DIR, DEFAULTS_FILE, MODELS_DIR

    assert DEFAULTS_FILE.is_file()
    assert CONFIG_DIR.is_dir() and MODELS_DIR.is_dir()
    assert list(MODELS_DIR.glob("*.yaml"))


def test_a_spec_finds_its_input_set_relative_to_itself():
    """The relation the wheel's force-include has to preserve."""
    from model_partition.cli import MODELS_DIR
    from model_partition.spec import load_spec

    spec = load_spec(MODELS_DIR / "qwen3.5-0.8b.yaml")
    short, long = spec.inputs.resolve(spec.base_dir())
    assert short.is_file() and long.is_file()


def test_resource_lookup_prefers_a_directory_beside_the_package(tmp_path, monkeypatch):
    """Inside an installed wheel the data sits next to the package, not above it."""
    from model_partition import cli

    package = tmp_path / "model_partition"
    (package / "config").mkdir(parents=True)
    (tmp_path / "config").mkdir()
    monkeypatch.setattr(cli, "__file__", str(package / "cli.py"))
    assert cli._resource_dir("config") == package / "config"


def test_resource_lookup_falls_back_to_the_source_layout(tmp_path, monkeypatch):
    from model_partition import cli

    package = tmp_path / "model_partition"
    package.mkdir(parents=True)
    (tmp_path / "config").mkdir()
    monkeypatch.setattr(cli, "__file__", str(package / "cli.py"))
    assert cli._resource_dir("config") == tmp_path / "config"


# -- spec overrides ----------------------------------------------------------


def test_spec_overrides_sit_above_the_shared_defaults():
    """A model that cannot afford a dequant mirror has to be able to say so."""
    spec = resolve_spec("deepseek-v4.1-flash")
    assert spec.overrides, "this spec should carry loop overrides"
    options = build_options(None, spec=spec)
    assert options.cache_dequant is False
    assert options.cache_weights is False


def test_the_command_line_wins_over_a_spec_override():
    spec = resolve_spec("deepseek-v4.1-flash")
    assert build_options(None, spec=spec, cache_weights=True).cache_weights is True


def test_a_spec_overriding_an_unknown_option_is_rejected(tmp_path):
    """Silently ignoring it is how `cache_dequant: false` stopped meaning anything."""
    from model_partition.spec import parse_spec

    spec = parse_spec({"source": "hf:a/b", "name": "x", "overrides": {"cahce_weights": False}})
    import click

    with pytest.raises(click.ClickException, match="unknown loop option"):
        build_options(None, spec=spec)


def test_every_bundled_spec_overrides_only_real_options():
    from model_partition.cli import MODELS_DIR
    from model_partition.spec import load_spec

    for path in MODELS_DIR.glob("*.yaml"):
        build_options(None, spec=load_spec(path))
