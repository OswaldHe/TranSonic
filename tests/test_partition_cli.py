# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the partition CLI and its registration under autohelix."""

import json
from pathlib import Path

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
    assert "Trainium" in (main.commands["partition"].help or "")


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


def test_deepseek_v41_ships_disabled_with_a_reason():
    """The ultimate target is documented but must not run yet."""
    spec = resolve_spec("deepseek-v4.1-flash")
    assert spec.enabled is False
    assert "765" in spec.notes or "exceeds local disk" in spec.notes


def test_deepseek_specs_use_the_vendor_inference_code():
    for name in ("deepseek-v4-flash", "deepseek-v4.1-flash"):
        spec = resolve_spec(name)
        assert spec.loader == "repo_code"
        assert spec.entry == "inference/model.py"
        assert spec.trust_remote_code is True


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
    assert options.headroom == 0.60


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


def test_list_shows_bundled_specs_and_marks_disabled(runner):
    result = runner.invoke(partition, ["list"])
    assert result.exit_code == 0
    assert "qwen3.8-27b" in result.output
    assert "(disabled)" in result.output


def test_disabled_spec_refuses_to_run(runner):
    result = runner.invoke(partition, ["run", "deepseek-v4.1-flash"])
    assert result.exit_code != 0
    assert "enabled: false" in str(result.output) + str(result.exception)


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
    """No checkpoint is read: structure from config, weights from the dumps."""
    module_id = next(m.id for m in tiny_run.graph.partitioned_modules
                     if m.kind == "decoder_layers")
    result = runner.invoke(partition, ["replay", str(tiny_run.layout.root), module_id])
    assert result.exit_code == 0, result.output + str(result.exception)
    assert ": ok" in result.output
