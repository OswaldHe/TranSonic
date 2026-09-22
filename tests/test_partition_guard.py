# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the harness guard.

The prompt is not a boundary: an agent runs with the whole run directory
writable. These check that the files deciding whether a module is correct come
back exactly as they were, and that the ones the agent is meant to edit do not.
"""

from model_partition.loop.guard import HarnessGuard


def populated(root):
    (root / "modules" / "00-decoder").mkdir(parents=True)
    (root / "trace" / "activations" / "layers.0").mkdir(parents=True)
    (root / "reports").mkdir(parents=True)
    (root / "plan").mkdir(parents=True)
    (root / "run.yaml").write_text("spec: {}\n")
    (root / "modules" / "index.yaml").write_text("n_groups: 1\n")
    (root / "modules" / "00-decoder" / "verify.py").write_text("assert tolerance\n")
    (root / "modules" / "00-decoder" / "inference.py").write_text("baseline\n")
    (root / "modules" / "00-decoder" / "README.md").write_text("# module\n")
    (root / "trace" / "records.yaml").write_text("records: []\n")
    (root / "trace" / "activations" / "layers.0" / "h.bin").write_bytes(b"\x00" * 16)
    (root / "reports" / "verify.json").write_text("{}\n")
    (root / "plan" / "partition_graph.yaml").write_text("model: t\n")
    return root


def test_an_untouched_run_reports_clean(tmp_path):
    guard = HarnessGuard.capture(populated(tmp_path))
    assert guard.restore().clean


def test_an_edited_verifier_is_put_back(tmp_path):
    root = populated(tmp_path)
    guard = HarnessGuard.capture(root)
    verifier = root / "modules" / "00-decoder" / "verify.py"
    verifier.write_text("return 0  # always pass\n")

    report = guard.restore()
    assert "modules/00-decoder/verify.py" in report.restored
    assert verifier.read_text() == "assert tolerance\n"


def test_a_deleted_verifier_is_put_back(tmp_path):
    root = populated(tmp_path)
    guard = HarnessGuard.capture(root)
    (root / "modules" / "00-decoder" / "verify.py").unlink()
    assert guard.restore().restored == ["modules/00-decoder/verify.py"]
    assert (root / "modules" / "00-decoder" / "verify.py").is_file()


def test_the_editable_surface_is_left_alone(tmp_path):
    """The two files the loop owns must survive the guard untouched."""
    root = populated(tmp_path)
    guard = HarnessGuard.capture(root)
    (root / "modules" / "00-decoder" / "inference.py").write_text("fixed\n")
    (root / "plan" / "partition_graph.yaml").write_text("model: revised\n")

    assert guard.restore().clean
    assert (root / "modules" / "00-decoder" / "inference.py").read_text() == "fixed\n"
    assert (root / "plan" / "partition_graph.yaml").read_text() == "model: revised\n"


def test_a_rewritten_reference_tensor_is_reported_not_repaired(tmp_path):
    """Reference tensors are too large to snapshot, so tampering fails the iteration."""
    root = populated(tmp_path)
    guard = HarnessGuard.capture(root)
    blob = root / "trace" / "activations" / "layers.0" / "h.bin"
    blob.write_bytes(b"\x01" * 16)

    report = guard.restore()
    assert report.tampered == ["trace/activations/layers.0/h.bin"]
    assert not report.clean
    assert "reference tensor" in report.summary()


def test_a_deleted_reference_tensor_is_reported(tmp_path):
    root = populated(tmp_path)
    guard = HarnessGuard.capture(root)
    (root / "trace" / "activations" / "layers.0" / "h.bin").unlink()
    assert guard.restore().tampered == ["trace/activations/layers.0/h.bin"]


def test_a_new_file_is_not_the_guard_s_business(tmp_path):
    """Notes and rationale the agent writes are allowed to appear."""
    root = populated(tmp_path)
    guard = HarnessGuard.capture(root)
    (root / "plan" / "rationale.md").write_text("why\n")
    assert guard.restore().clean


def test_capture_on_an_empty_directory_is_harmless(tmp_path):
    assert HarnessGuard.capture(tmp_path).restore().clean


def test_a_rewritten_reference_is_caught_even_with_its_mtime_restored(tmp_path):
    """The trace is the acceptance reference, and mtime is settable by anyone."""
    import os

    from model_partition.loop.guard import HarnessGuard

    root = tmp_path / "run"
    (root / "trace" / "activations").mkdir(parents=True)
    blob = root / "trace" / "activations" / "reference.bin"
    blob.write_bytes(b"\x01" * 64)
    before = blob.stat()

    guard = HarnessGuard.capture(root)
    # Same length, same mtime, different numbers: what an agent would do to make its
    # own output look correct.
    blob.write_bytes(b"\x02" * 64)
    os.utime(blob, ns=(before.st_atime_ns, before.st_mtime_ns))

    report = guard.restore()
    assert "trace/activations/reference.bin" in report.tampered
