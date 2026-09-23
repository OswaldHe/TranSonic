# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""What carries between module repos, and what must not.

Each module is its own git repo, so the memory is the only path between them. That makes two
things worth pinning: that an entry survives the round trip, and that the agent cannot write
back through it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bootstrap import memory as mem
from bootstrap import preset

pytestmark = pytest.mark.bootstrap


def _repo(tmp_path: Path, name: str = "02-Attention-long8192", *, passed: bool = True) -> Path:
    repo = tmp_path / "bootstrap-runs" / name
    (repo / ".autohelix" / "bootstrap").mkdir(parents=True)
    (repo / ".autohelix" / "notes").mkdir(parents=True)
    (repo / ".autohelix" / "bootstrap" / "manifest.json").write_text(json.dumps({
        "artifact_group": "02-Attention",
        "module_id": "layers.2.attention",
        "sample_id": "long-needle-8192-0",
        "submodules": ["layers.2.attn_norm", "layers.2.attn"],
        "composition": "sequential",
        "state_included": True,
        "tolerance": {"MIN_COSINE": 0.9995},
        "tensors": [
            {"name": "input", "role": "input", "dtype": "bfloat16", "shape": [1, 8192, 5120]},
            {"name": "reference", "role": "golden", "dtype": "bfloat16", "shape": [1, 8192, 5120]},
        ],
    }))
    (repo / "source.py").write_text("# the kernel\n")
    (repo / "inference.py").write_text("# the validator\n")
    (repo / ".autohelix" / "notes" / "iter-1.md").write_text("what the compiler said\n")
    return repo


def test_memory_lives_beside_the_repos_not_inside_one(tmp_path: Path) -> None:
    """It outlives any single repo and is read by all of them, so it cannot be a child."""
    repo = _repo(tmp_path)
    store = mem.location(repo)
    assert store.name == mem.MEMORY_DIRNAME
    assert store.parent == repo.parent
    assert store not in repo.parents


def test_an_entry_round_trips_with_its_kernel_and_notes(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    entry = mem.record(repo, passed=True, iterations=5, summary="gate: all 6 checks pass")
    assert entry is not None
    assert entry.name == "02-Attention__long-needle-8192-0"

    assert (entry / "source.py").read_text() == "# the kernel\n"
    assert (entry / "inference.py").read_text() == "# the validator\n"
    assert (entry / "notes" / "iter-1.md").is_file()

    listed = mem.entries(mem.location(repo))
    assert [e.module_id for e in listed] == ["layers.2.attention"]
    assert listed[0].passed
    assert listed[0].input_shape == "bfloat16(1, 8192, 5120)"

    index = (mem.location(repo) / mem.INDEX_NAME).read_text()
    assert "layers.2.attention" in index
    assert "passed in 5 iter" in index
    # The index has to warn, because it is the one file the agent is told to read first.
    assert "do not paste" in index


def test_a_failed_module_is_recorded_and_labelled(tmp_path: Path) -> None:
    """Its notes are where the dead ends are written down; the label is what lets them weigh it."""
    repo = _repo(tmp_path)
    mem.record(repo, passed=False, iterations=5, summary="gate: 5/6 passing, failing e")
    listed = mem.entries(mem.location(repo))
    assert listed and not listed[0].passed
    assert "failed after 5 iter" in (mem.location(repo) / mem.INDEX_NAME).read_text()


def test_re_recording_a_module_replaces_rather_than_duplicates(tmp_path: Path) -> None:
    """Accumulation is across modules; a module re-run is the same module."""
    repo = _repo(tmp_path)
    mem.record(repo, passed=False, iterations=1, summary="first")
    mem.record(repo, passed=True, iterations=2, summary="second")
    listed = mem.entries(mem.location(repo))
    assert len(listed) == 1
    assert listed[0].passed and listed[0].iterations == 2


def test_the_seeded_copy_cannot_be_written_back_through(tmp_path: Path) -> None:
    """One iteration must not be able to plant a precedent for the next to find."""
    repo = _repo(tmp_path)
    mem.record(repo, passed=True, iterations=2, summary="ok")
    worktree = tmp_path / "wt"
    (worktree / ".autohelix" / "notes").mkdir(parents=True)

    assert mem.seed(repo, worktree) == 1
    planted = worktree / mem.SEEDED_REL / "02-Attention__long-needle-8192-0" / "source.py"
    assert planted.is_file()
    with pytest.raises(PermissionError):
        planted.write_text("tampered")

    # And the next iteration must still be able to replace the read-only copy.
    assert mem.seed(repo, worktree) == 1
    assert (worktree / ".autohelix" / "notes").is_dir()


def test_seeding_is_quiet_when_there_is_nothing_to_seed(tmp_path: Path) -> None:
    """The first module bootstrapped anywhere has no memory, which is not an error."""
    repo = _repo(tmp_path)
    worktree = tmp_path / "wt"
    worktree.mkdir()
    assert mem.seed(repo, worktree) == 0
    assert not (worktree / mem.SEEDED_REL).exists()


def test_clearing_empties_it_and_accumulation_resumes(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    mem.record(repo, passed=True, iterations=2, summary="ok")
    store = mem.location(repo)
    assert mem.clear(store) == 1
    assert mem.entries(store) == []
    # Not a terminal state: the next module starts it again.
    mem.record(repo, passed=True, iterations=1, summary="ok")
    assert len(mem.entries(store)) == 1


def test_recording_never_fails_a_run(tmp_path: Path) -> None:
    """A run that produced a kernel must not be reported as failed over the memory."""
    assert mem.record(tmp_path / "not-a-repo", passed=True, iterations=1, summary="ok") is None


def test_the_goal_points_at_the_path_the_code_seeds(tmp_path: Path) -> None:
    """The preset is a fixed file, so nothing substitutes this path in — it has to match.

    A rename on either side would leave the agent being told to read a directory that is not
    there, and it would look like the memory simply being empty.
    """
    goal = preset.render_goal()
    assert mem.SEEDED_REL in goal
    assert mem.INDEX_NAME in goal
    # And it has to say what the limits are, not just where the directory is.
    assert "read-only" in goal
    assert "reference_torch.py" in goal
