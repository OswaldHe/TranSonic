# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The memory directory: who reads it, what reaches the prompt, and what cannot be written."""

from __future__ import annotations

import pytest

from optimization.config import ConfigError, PipelineConfig
from optimization.memory import (
    EVERY,
    SEEDED_REL,
    MemoryError_,
    MemorySpec,
    describe_for_prompt,
    entry_names,
    seed,
)

pytestmark = pytest.mark.optimization


@pytest.fixture
def store(tmp_path):
    """A memory directory with the shape the MoE one has."""
    root = tmp_path / "memory"
    (root / "submodule" / "notes").mkdir(parents=True)
    (root / "module").mkdir(parents=True)
    (root / "README.md").write_text("# index\n")
    (root / "submodule" / "source.py").write_text("# the rank kernel\n")
    (root / "submodule" / "notes" / "iter-1.md").write_text("tried X\n")
    (root / "module" / "source.py").write_text("# the 4-rank kernel\n")
    return root


# -- the selector ----------------------------------------------------------------------


def test_absent_block_is_disabled():
    spec = MemorySpec.from_config(None, max_iterations=10)
    assert not spec.enabled
    assert not spec.reads_at(1)


def test_path_with_no_selector_reads_every_iteration(store):
    spec = MemorySpec.from_config({"path": str(store)}, max_iterations=5)
    assert spec.enabled and spec.every
    assert all(spec.reads_at(i) for i in range(1, 6))


@pytest.mark.parametrize("section,reads,skips", [
    ({"iterations": EVERY}, [1, 5], []),
    ({"iterations": [1, 3]}, [1, 3], [2, 4]),
    ({"from": 2, "to": 4}, [2, 3, 4], [1, 5]),
    ({"at": 3}, [3], [1, 2, 4]),
])
def test_selector_grammar_matches_the_schedule(store, section, reads, skips):
    spec = MemorySpec.from_config({"path": str(store), **section}, max_iterations=5)
    assert [spec.reads_at(i) for i in reads] == [True] * len(reads)
    assert [spec.reads_at(i) for i in skips] == [False] * len(skips)


def test_selector_past_the_budget_is_refused(store):
    with pytest.raises(MemoryError_, match="budget is 5"):
        MemorySpec.from_config({"path": str(store), "at": 9}, max_iterations=5)


def test_iteration_zero_is_refused(store):
    with pytest.raises(MemoryError_, match="1-based"):
        MemorySpec.from_config({"path": str(store), "at": 0}, max_iterations=5)


def test_a_prompt_with_no_path_is_refused():
    with pytest.raises(MemoryError_, match="no 'path:'"):
        MemorySpec.from_config({"prompt": "read the notes first"}, max_iterations=5)


def test_unknown_key_is_refused(store):
    with pytest.raises(MemoryError_, match="unknown key"):
        MemorySpec.from_config({"path": str(store), "readme": "x"}, max_iterations=5)


def test_iterations_as_a_stray_string_is_refused(store):
    with pytest.raises(MemoryError_, match="must be 'all'"):
        MemorySpec.from_config({"path": str(store), "iterations": "every"}, max_iterations=5)


def test_relative_path_resolves_against_the_config_not_the_cwd(tmp_path, store):
    spec = MemorySpec.from_config(
        {"path": "memory"}, max_iterations=5, base_dir=tmp_path,
    )
    assert spec.path == store.resolve()


# -- seeding ---------------------------------------------------------------------------


def test_seed_copies_read_only_under_autohelix(tmp_path, store):
    spec = MemorySpec.from_config({"path": str(store)}, max_iterations=5)
    worktree = tmp_path / "wt"
    worktree.mkdir()

    count = seed(spec, worktree)

    landed = worktree / SEEDED_REL
    assert count == 4
    assert (landed / "submodule" / "notes" / "iter-1.md").read_text() == "tried X\n"
    # Under `.autohelix/`, which is gitignored and outside the editable scope — so the agent
    # cannot commit a copy of an old kernel as this iteration's work.
    assert SEEDED_REL.parts[0] == ".autohelix"
    for path in landed.rglob("*"):
        if path.is_file():
            assert not path.stat().st_mode & 0o222, f"{path} is writable"


def test_seed_is_a_snapshot_not_a_pointer(tmp_path, store):
    spec = MemorySpec.from_config({"path": str(store)}, max_iterations=5)
    worktree = tmp_path / "wt"
    worktree.mkdir()
    seed(spec, worktree)

    (store / "module" / "source.py").write_text("# changed under the iteration\n")

    assert (worktree / SEEDED_REL / "module" / "source.py").read_text() == "# the 4-rank kernel\n"


def test_seed_of_a_missing_directory_is_zero_not_a_crash(tmp_path):
    spec = MemorySpec(path=tmp_path / "absent", every=True)
    worktree = tmp_path / "wt"
    worktree.mkdir()
    assert seed(spec, worktree) == 0


def test_reseeding_replaces_rather_than_merges(tmp_path, store):
    spec = MemorySpec.from_config({"path": str(store)}, max_iterations=5)
    worktree = tmp_path / "wt"
    worktree.mkdir()
    seed(spec, worktree)
    (store / "submodule" / "source.py").unlink()

    seed(spec, worktree)

    assert not (worktree / SEEDED_REL / "submodule" / "source.py").exists()


# -- the prompt ------------------------------------------------------------------------


def test_prompt_block_carries_the_operator_prose_and_the_path(store):
    spec = MemorySpec.from_config(
        {"path": str(store), "prompt": "Open submodule/notes first."}, max_iterations=5,
    )
    block = describe_for_prompt(spec, 1, seeded=4)
    assert "Open submodule/notes first." in block
    assert str(SEEDED_REL) in block
    assert "`submodule`" in block and "`module`" in block


def test_prompt_block_is_empty_for_an_iteration_that_does_not_read(store):
    spec = MemorySpec.from_config({"path": str(store), "at": 1}, max_iterations=5)
    assert describe_for_prompt(spec, 2, seeded=4) == ""


def test_prompt_block_is_empty_when_nothing_was_seeded(store):
    spec = MemorySpec.from_config({"path": str(store)}, max_iterations=5)
    assert describe_for_prompt(spec, 1, seeded=0) == ""


def test_entry_names_lists_the_index_first(store):
    """The prompt tells the agent to open the index first, so the listing has to show it."""
    assert entry_names(store) == ["README.md", "module", "submodule"]


# -- warnings --------------------------------------------------------------------------


def test_missing_directory_warns_rather_than_refusing(tmp_path):
    spec = MemorySpec(path=tmp_path / "absent", every=True, prompt="x")
    warnings = spec.validate()
    assert any("does not exist" in w for w in warnings)


def test_path_nobody_reads_warns(store):
    spec = MemorySpec(path=store, iterations=(), every=False, prompt="x")
    assert any("no iteration reads it" in w for w in spec.validate())


def test_empty_prompt_warns(store):
    spec = MemorySpec(path=store, every=True, prompt="")
    assert any("memory.prompt is empty" in w for w in spec.validate())


# -- the round trip through the derived loop config -------------------------------------


def _pipeline(tmp_path, memory_block):
    for name in ("bootstrap", "artifact"):
        (tmp_path / name).mkdir(exist_ok=True)
    (tmp_path / "scheme.yaml").write_text("{}\n")
    return {
        "module": {"id": "layers.1.ffn", "bootstrap_repo": str(tmp_path / "bootstrap"),
                   "artifact": str(tmp_path / "artifact")},
        "floorplan": {"scheme": str(tmp_path / "scheme.yaml"), "target_units": 4},
        "workspace": {"root": str(tmp_path / "runs")},
        "submodule": {"goal": "go faster", "budget": {"iterations": 5}, "memory": memory_block},
        "full": {"goal": "go faster still", "budget": {"iterations": 5}},
    }


def test_memory_survives_the_derived_config_round_trip(tmp_path, store):
    block = {"path": str(store), "iterations": [1, 4], "prompt": "Start with module/."}
    config = PipelineConfig.from_dict(_pipeline(tmp_path, block))

    payload = config.derive_loop_config("submodule")

    # The loop re-parses exactly this, so what it reads back has to say what the operator wrote.
    reparsed = MemorySpec.from_config(payload["memory"], max_iterations=5)
    assert reparsed.path == store.resolve()
    assert reparsed.reads_at(1) and reparsed.reads_at(4)
    assert not reparsed.reads_at(2)
    assert reparsed.prompt == "Start with module/."


def test_every_survives_the_round_trip_as_all(tmp_path, store):
    config = PipelineConfig.from_dict(_pipeline(tmp_path, {"path": str(store), "prompt": "x"}))
    payload = config.derive_loop_config("submodule")
    assert payload["memory"]["iterations"] == EVERY
    assert MemorySpec.from_config(payload["memory"], max_iterations=5).every


def test_no_memory_block_emits_no_memory_key(tmp_path):
    data = _pipeline(tmp_path, None)
    data["submodule"].pop("memory")
    payload = PipelineConfig.from_dict(data).derive_loop_config("submodule")
    assert "memory" not in payload


def test_a_bad_memory_block_fails_the_pipeline_config(tmp_path, store):
    data = _pipeline(tmp_path, {"path": str(store), "at": 99})
    with pytest.raises(ConfigError, match="budget is 5"):
        PipelineConfig.from_dict(data)


def test_stage_warnings_reach_the_operator(tmp_path):
    """A memory directory that is not there yet is a warning, never a refusal.

    The pipeline config is written before the memory is assembled, so a stage that refused to
    start over a directory it will not read until iteration 4 would be the worse failure. It
    belongs in `warnings()`, which the CLI prints, and not in `validate()`, which stops the run.
    """
    data = _pipeline(tmp_path, {"path": str(tmp_path / "absent"), "prompt": "x"})
    config = PipelineConfig.from_dict(data)
    assert any("does not exist" in w for w in config.warnings())
    assert not any("memory" in p for p in config.validate())


# -- the shared top-level block ---------------------------------------------------------


def test_shared_path_is_written_once(tmp_path, store):
    """The redundancy this removes: both stages read the same directory, so `path:` is one line."""
    data = _pipeline(tmp_path, {"iterations": [1, 2]})
    data["memory"] = {"path": str(store), "prompt": "shared prose"}
    data["full"]["memory"] = {"at": 1}
    config = PipelineConfig.from_dict(data)

    assert config.submodule.memory.path == store.resolve()
    assert config.full.memory.path == store.resolve()
    assert config.submodule.memory.prompt == "shared prose"
    assert [i for i in range(1, 6) if config.submodule.memory.reads_at(i)] == [1, 2]
    assert [i for i in range(1, 6) if config.full.memory.reads_at(i)] == [1]


def test_stage_selector_replaces_the_shared_one_wholesale(tmp_path, store):
    """`iterations: all` at the top must not outrank `at: 3` in a stage."""
    data = _pipeline(tmp_path, {"at": 3})
    data["memory"] = {"path": str(store), "iterations": EVERY, "prompt": "x"}
    config = PipelineConfig.from_dict(data)
    assert not config.submodule.memory.every
    assert [i for i in range(1, 6) if config.submodule.memory.reads_at(i)] == [3]


def test_stage_prompt_overrides_the_shared_one(tmp_path, store):
    data = _pipeline(tmp_path, {"prompt": "stage prose"})
    data["memory"] = {"path": str(store), "prompt": "shared prose"}
    config = PipelineConfig.from_dict(data)
    assert config.submodule.memory.prompt == "stage prose"
    assert config.full.memory.prompt == "shared prose"


def test_shared_block_alone_enables_both_stages(tmp_path, store):
    data = _pipeline(tmp_path, None)
    data["submodule"].pop("memory")
    data["memory"] = {"path": str(store), "prompt": "x"}
    config = PipelineConfig.from_dict(data)
    assert config.submodule.memory.every and config.full.memory.every


def test_no_memory_anywhere_stays_disabled(tmp_path):
    data = _pipeline(tmp_path, None)
    data["submodule"].pop("memory")
    config = PipelineConfig.from_dict(data)
    assert not config.submodule.memory.enabled and not config.full.memory.enabled


def test_unknown_top_level_memory_is_still_reported(tmp_path, store):
    data = _pipeline(tmp_path, None)
    data["submodule"].pop("memory")
    data["memroy"] = {"path": str(store)}
    with pytest.raises(ConfigError, match="unknown top-level key"):
        PipelineConfig.from_dict(data)
