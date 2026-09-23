# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The gate has to refuse the obvious ways around it.

A constraint the agent cannot see is only as good as its refusals, so these are mostly
negative tests: each one is a way to satisfy the letter of a check without doing the work,
and each one must come back failing. The positive test is that the generated stub scores
exactly the three checks it is supposed to score, so a green check means something.

No device and no network: every test writes two small files and runs the static half.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bootstrap import nki_checker as chk

pytestmark = pytest.mark.bootstrap


GOOD_SOURCE = '''\
"""A kernel. Read reference_torch.py for the specification."""

import nki
import nki.language as nl


@nki.jit
def kernel(hidden_states, weight):
    return nl.add(hidden_states, weight)
'''

GOOD_INFERENCE = '''\
"""Validate the kernel. Prints ##autohelix[latency_ms=...] and ##autohelix[passed=...]."""

import subprocess

import torch
import torch_neuronx

from source import kernel

RTOL = 0.02
ATOL = 0.02
MIN_COSINE = 0.9999
MIN_PASS_FRACTION = 0.999

NEURON_EXPLORER = "neuron-explorer"
FIELD = "total_exec_time"


def load():
    x = torch.frombuffer(open("tensors/input.bin", "rb").read(), dtype=torch.bfloat16)
    ref = torch.frombuffer(open("tensors/reference.bin", "rb").read(), dtype=torch.bfloat16)
    return x, ref


def main():
    x, ref = load()
    traced = torch_neuronx.trace(kernel, (x,))
    return 0
'''


def _manifest(tmp_path: Path, files: dict[str, bytes]) -> Path:
    """Write the tensors and a manifest describing them."""
    tensors = []
    for name, payload in files.items():
        path = tmp_path / "tensors" / f"{name}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        tensors.append({
            "name": name,
            "role": "input" if name == "input" else ("golden" if name == "reference" else "weight"),
            "file": f"tensors/{name}.bin",
            "dtype": "bfloat16",
            "shape": [len(payload) // 2],
            "nbytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "required": name in ("input", "reference"),
        })
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"tensors": tensors}))
    return path


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repo with valid tensors and a valid pair of files, ready to be sabotaged."""
    _manifest(tmp_path, {"input": b"\x00\x01" * 8, "reference": b"\x02\x03" * 8})
    (tmp_path / "source.py").write_text(GOOD_SOURCE)
    (tmp_path / "inference.py").write_text(GOOD_INFERENCE)
    return tmp_path


def _static(repo: Path) -> dict[str, chk.CheckResult]:
    """The static checks only — (d) and (e) need a device run to conclude anything."""
    manifest = json.loads((repo / "manifest.json").read_text())
    allowed = {e["file"] for e in manifest["tensors"]}
    source = chk._parse(repo / chk.SOURCE_FILE)
    inference = chk._parse(repo / chk.INFERENCE_FILE)
    source_text = (repo / chk.SOURCE_FILE).read_text()
    run = chk.RunOutcome(ran=False, detail="not run")
    return {
        "a": chk.check_kernel_formalization(source, inference),
        "b": chk.check_self_containment(source, inference, allowed),
        "c": chk.check_nki_only(source, source_text),
        "e": chk.check_pass_test(inference, run),
        "f": chk.check_provenance(inference, repo, manifest),
    }


def test_valid_pair_passes_the_static_checks(repo: Path) -> None:
    results = _static(repo)
    for key in ("a", "b", "c", "f"):
        assert results[key].passed, f"({key}) should pass: {results[key].findings}"


def test_prose_mentioning_a_reference_is_not_a_path_escape(repo: Path) -> None:
    """The stub's own docstring names reference_torch.py; that must not fail (b)."""
    assert results_ok(repo, "b")


def results_ok(repo: Path, key: str) -> bool:
    return _static(repo)[key].passed


# -- (a) ------------------------------------------------------------------------------


def test_kernel_without_the_jit_decorator_fails(repo: Path) -> None:
    (repo / "source.py").write_text(GOOD_SOURCE.replace("@nki.jit\n", ""))
    assert not _static(repo)["a"].passed


def test_kernel_under_another_name_fails(repo: Path) -> None:
    (repo / "source.py").write_text(GOOD_SOURCE.replace("def kernel(", "def attention("))
    assert not _static(repo)["a"].passed


def test_inference_not_tracing_fails(repo: Path) -> None:
    (repo / "inference.py").write_text(
        GOOD_INFERENCE.replace("traced = torch_neuronx.trace(kernel, (x,))", "traced = kernel(x)")
    )
    assert not _static(repo)["a"].passed


# -- (b) ------------------------------------------------------------------------------


def test_source_importing_the_frozen_reference_fails(repo: Path) -> None:
    (repo / "source.py").write_text(GOOD_SOURCE + "\nimport reference_torch\n")
    assert not _static(repo)["b"].passed


def test_nki_load_is_not_file_io(repo: Path) -> None:
    """`nl.load` moves a tensor into on-chip memory and is in essentially every kernel.

    Flagging it as file I/O would fail every correct kernel, which is a far worse failure
    than missing a trick the reviewer is also looking for.
    """
    (repo / "source.py").write_text(GOOD_SOURCE.replace(
        "return nl.add(hidden_states, weight)",
        "a = nl.load(hidden_states)\n    b = nl.load(weight)\n    return nl.add(a, b)",
    ))
    result = _static(repo)["b"]
    assert result.passed, result.findings


def test_source_reading_the_reference_tensor_fails(repo: Path) -> None:
    """The kernel returning the answer it was meant to compute.

    `tensors/reference.bin` is a legitimate file in the manifest, so the allowlist alone
    permits this — it is refused because the kernel may not do file I/O at all.
    """
    (repo / "source.py").write_text(
        GOOD_SOURCE + '\n\ndef cheat():\n    return open("tensors/reference.bin", "rb").read()\n'
    )
    result = _static(repo)["b"]
    assert not result.passed
    assert any("must not read files" in f or "must not reference files" in f for f in result.findings)


def test_reading_a_bin_outside_the_manifest_fails(repo: Path) -> None:
    (repo / "inference.py").write_text(
        GOOD_INFERENCE.replace("tensors/input.bin", "tensors/smuggled.bin")
    )
    assert not _static(repo)["b"].passed


def test_escaping_the_repo_fails(repo: Path) -> None:
    (repo / "inference.py").write_text(
        GOOD_INFERENCE.replace("tensors/input.bin", "../../trace/activations/x.bin")
    )
    assert not _static(repo)["b"].passed


def test_dynamic_import_fails(repo: Path) -> None:
    (repo / "source.py").write_text(GOOD_SOURCE + '\nimportlib.import_module("torch")\n')
    assert not _static(repo)["b"].passed


# -- (c) ------------------------------------------------------------------------------


def test_torch_in_the_kernel_fails(repo: Path) -> None:
    (repo / "source.py").write_text(GOOD_SOURCE.replace("import nki\n", "import nki\nimport torch\n"))
    assert not _static(repo)["c"].passed


def test_torch_used_without_importing_it_fails(repo: Path) -> None:
    (repo / "source.py").write_text(GOOD_SOURCE.replace("nl.add(", "torch.add("))
    assert not _static(repo)["c"].passed


def test_numpy_in_the_kernel_fails(repo: Path) -> None:
    (repo / "source.py").write_text(
        GOOD_SOURCE.replace("import nki\n", "import nki\nimport numpy as np\n")
    )
    assert not _static(repo)["c"].passed


def test_torch_in_a_comment_is_reported_but_not_fatal(repo: Path) -> None:
    (repo / "source.py").write_text(GOOD_SOURCE + "\n# translated from the torch version\n")
    result = _static(repo)["c"]
    assert result.passed
    assert any("comment mentions" in f for f in result.findings)


# -- (e) ------------------------------------------------------------------------------


def _pass_test_with_clean_run(repo: Path) -> chk.CheckResult:
    """(e) against a run that succeeded, so the only thing it can fail on is the bar.

    Going through `_static` would not do: its run never happened, so (e) would fail
    whatever the tolerances say and every test below would pass for the wrong reason.
    """
    inference = chk._parse(repo / chk.INFERENCE_FILE)
    run = chk.RunOutcome(ran=True, return_code=0, output="##autohelix[passed=1]")
    return chk.check_pass_test(inference, run)


def test_the_clean_run_baseline_passes(repo: Path) -> None:
    """Guards the helper above: without sabotage, (e) must pass."""
    assert _pass_test_with_clean_run(repo).passed


@pytest.mark.parametrize("name", sorted(chk.PINNED_TOLERANCE))
def test_loosening_any_tolerance_fails(repo: Path, name: str) -> None:
    original = f"{name} = {chk.PINNED_TOLERANCE[name]:g}"
    loosened = f"{name} = {chk.PINNED_TOLERANCE[name] * 10:g}"
    text = (repo / "inference.py").read_text()
    assert original in text, original
    (repo / "inference.py").write_text(text.replace(original, loosened))
    result = _pass_test_with_clean_run(repo)
    assert not result.passed
    assert any(name in f and "may not be loosened" in f for f in result.findings)


@pytest.mark.parametrize("name", sorted(chk.PINNED_TOLERANCE))
def test_tightening_a_tolerance_also_fails(repo: Path, name: str) -> None:
    """The bar is pinned, not bounded: it is the tolerance the reference was recorded at,
    and a repo that quietly moved it is not reporting against that reference."""
    original = f"{name} = {chk.PINNED_TOLERANCE[name]:g}"
    text = (repo / "inference.py").read_text()
    (repo / "inference.py").write_text(
        text.replace(original, f"{name} = {chk.PINNED_TOLERANCE[name] / 10:g}")
    )
    assert not _pass_test_with_clean_run(repo).passed


@pytest.mark.parametrize("name", sorted(chk.PINNED_TOLERANCE))
def test_deleting_a_tolerance_fails(repo: Path, name: str) -> None:
    original = f"{name} = {chk.PINNED_TOLERANCE[name]:g}\n"
    text = (repo / "inference.py").read_text()
    assert original in text, original
    (repo / "inference.py").write_text(text.replace(original, ""))
    result = _pass_test_with_clean_run(repo)
    assert not result.passed
    assert any(name in f and "does not declare" in f for f in result.findings)


def test_computing_a_tolerance_instead_of_declaring_it_fails(repo: Path) -> None:
    """`RTOL = 2e-2 * 1` evaluates to the right number but is not legible as the bar.

    Allowing an expression would allow `RTOL = 2e-2 * FUDGE`, so only a bare literal counts.
    """
    text = (repo / "inference.py").read_text()
    assert "RTOL = 0.02" in text
    (repo / "inference.py").write_text(text.replace("RTOL = 0.02", "RTOL = 2e-2 * 1"))
    assert not _pass_test_with_clean_run(repo).passed


def test_the_ceiling_is_checked_against_the_run_not_just_pinned(repo: Path) -> None:
    """A declared ceiling the candidate then ignores would be no ceiling at all.

    This closes the gap a pass fraction leaves: at 0.999, 0.1% of elements may be wrong by
    any amount, and cosine notices one wild value but not a few dozen merely-bad ones.
    """
    text = (repo / "inference.py").read_text().replace(
        "MIN_PASS_FRACTION = 0.999", "MIN_PASS_FRACTION = 0.999\nMAX_ABS_ERR = 0.5")
    (repo / "inference.py").write_text(text)
    inference = chk._parse(repo / chk.INFERENCE_FILE)
    bar = {**chk.PINNED_TOLERANCE, chk.CEILING_NAME: 0.5}

    within = chk.RunOutcome(ran=True, return_code=0,
                            output="##autohelix[max_abs_err=0.4]\n##autohelix[passed=1]")
    assert chk.check_pass_test(inference, within, bar).passed

    over = chk.RunOutcome(ran=True, return_code=0,
                          output="##autohelix[max_abs_err=0.9]\n##autohelix[passed=1]")
    result = chk.check_pass_test(inference, over, bar)
    assert not result.passed
    assert any("ceiling" in f for f in result.findings)

    silent = chk.RunOutcome(ran=True, return_code=0, output="##autohelix[passed=1]")
    result = chk.check_pass_test(inference, silent, bar)
    assert not result.passed
    assert any(chk.MAX_ABS_ERR_MARKER in f for f in result.findings)


def test_the_ceiling_literal_is_pinned_like_the_others(repo: Path) -> None:
    inference = chk._parse(repo / chk.INFERENCE_FILE)
    bar = {**chk.PINNED_TOLERANCE, chk.CEILING_NAME: 0.5}
    run = chk.RunOutcome(ran=True, return_code=0,
                         output="##autohelix[max_abs_err=0.4]\n##autohelix[passed=1]")
    # Absent from inference.py entirely.
    result = chk.check_pass_test(inference, run, bar)
    assert not result.passed
    assert any(chk.CEILING_NAME in f and "does not declare" in f for f in result.findings)


def test_a_manifest_without_a_ceiling_does_not_require_one(repo: Path) -> None:
    """There is no default for a number derived from this module's own recorded output."""
    assert chk.CEILING_NAME not in chk.expected_tolerance({})
    assert chk.expected_tolerance({"tolerance": {chk.CEILING_NAME: 1.5}})[chk.CEILING_NAME] == 1.5


def test_nonzero_exit_fails(repo: Path) -> None:
    inference = chk._parse(repo / chk.INFERENCE_FILE)
    run = chk.RunOutcome(ran=True, return_code=1, output="##autohelix[passed=1]")
    assert not chk.check_pass_test(inference, run).passed


def test_claiming_a_pass_without_one_fails(repo: Path) -> None:
    inference = chk._parse(repo / chk.INFERENCE_FILE)
    run = chk.RunOutcome(ran=True, return_code=0, output="##autohelix[passed=0]")
    assert not chk.check_pass_test(inference, run).passed


def test_exit_zero_and_a_pass_marker_is_accepted(repo: Path) -> None:
    inference = chk._parse(repo / chk.INFERENCE_FILE)
    run = chk.RunOutcome(ran=True, return_code=0, output="##autohelix[passed=1]")
    assert chk.check_pass_test(inference, run).passed


# -- (d) ------------------------------------------------------------------------------


def test_missing_profile_artifacts_fail(repo: Path) -> None:
    inference = chk._parse(repo / chk.INFERENCE_FILE)
    run = chk.RunOutcome(ran=True, return_code=0, output="##autohelix[latency_ms=1.5]")
    result = chk.check_measurement(inference, repo, run)
    assert not result.passed
    assert any(".neff" in f for f in result.findings)


def test_a_stale_profile_does_not_count(repo: Path) -> None:
    """Files from an earlier run are not evidence that this run measured anything."""
    import os
    import time

    for extension in (".neff", ".ntff"):
        stale = repo / f"old{extension}"
        stale.write_bytes(b"stale")
        old = time.time() - 3600
        os.utime(stale, (old, old))
    inference = chk._parse(repo / chk.INFERENCE_FILE)
    run = chk.RunOutcome(ran=True, return_code=0, output="##autohelix[latency_ms=1.5]",
                         started_at=time.time())
    result = chk.check_measurement(inference, repo, run)
    assert not result.passed


def test_a_fresh_profile_and_a_latency_pass(repo: Path) -> None:
    import time

    started = time.time()
    for extension in (".neff", ".ntff"):
        (repo / f"fresh{extension}").write_bytes(b"fresh")
    inference = chk._parse(repo / chk.INFERENCE_FILE)
    run = chk.RunOutcome(ran=True, return_code=0, output="##autohelix[latency_ms=1.5]",
                         started_at=started)
    assert chk.check_measurement(inference, repo, run).passed


def test_a_nonpositive_latency_fails(repo: Path) -> None:
    import time

    started = time.time()
    for extension in (".neff", ".ntff"):
        (repo / f"fresh{extension}").write_bytes(b"fresh")
    inference = chk._parse(repo / chk.INFERENCE_FILE)
    run = chk.RunOutcome(ran=True, return_code=0, output="##autohelix[latency_ms=0]",
                         started_at=started)
    assert not chk.check_measurement(inference, repo, run).passed


def test_not_invoking_the_profiler_fails(repo: Path) -> None:
    (repo / "inference.py").write_text(GOOD_INFERENCE.replace('"neuron-explorer"', '"timeit"'))
    inference = chk._parse(repo / chk.INFERENCE_FILE)
    run = chk.RunOutcome(ran=True, return_code=0, output="##autohelix[latency_ms=1.5]")
    result = chk.check_measurement(inference, repo, run)
    assert not result.passed
    assert any("neuron-explorer" in f for f in result.findings)


# -- (f) ------------------------------------------------------------------------------


def test_editing_a_tensor_fails(repo: Path) -> None:
    (repo / "tensors" / "reference.bin").write_bytes(b"\x00" * 16)
    result = _static(repo)["f"]
    assert not result.passed
    assert any("has been modified" in f for f in result.findings)


def test_not_reading_the_reference_fails(repo: Path) -> None:
    text = (repo / "inference.py").read_text().replace('"tensors/reference.bin"', '"tensors/input.bin"')
    (repo / "inference.py").write_text(text)
    result = _static(repo)["f"]
    assert not result.passed
    assert any("reference.bin" in f for f in result.findings)


@pytest.mark.parametrize("call", [
    "torch.randn(16)", "torch.ones(16)", "torch.full((16,), 1.0)", "torch.arange(16)",
    "torch.eye(4)", "torch.rand(16)", "numpy.random.rand(16)",
])
def test_fabricating_a_tensor_fails(repo: Path, call: str) -> None:
    (repo / "inference.py").write_text(GOOD_INFERENCE + f"\nfake = {call}\n")
    result = _static(repo)["f"]
    assert not result.passed
    assert any("must come from the recorded" in f or "generated values" in f
               for f in result.findings)


def test_filling_a_tensor_in_place_fails(repo: Path) -> None:
    (repo / "inference.py").write_text(GOOD_INFERENCE + "\nx.fill_(1.0)\n")
    assert not _static(repo)["f"].passed


@pytest.mark.parametrize("call", ["torch.zeros(16)", "torch.empty(16)", "torch.zeros_like(x)"])
def test_allocating_an_output_buffer_is_allowed(repo: Path, call: str) -> None:
    (repo / "inference.py").write_text(GOOD_INFERENCE + f"\nout = {call}\n")
    assert _static(repo)["f"].passed


# -- the whole gate -------------------------------------------------------------------


def test_a_missing_file_is_reported_as_every_check_failing(tmp_path: Path) -> None:
    """An unusable repo gets a verdict that still names all six checks, not a traceback."""
    manifest = _manifest(tmp_path, {"input": b"\x00\x01", "reference": b"\x02\x03"})
    (tmp_path / "source.py").write_text(GOOD_SOURCE)
    exit_code = chk.main(["--repo", str(tmp_path), "--manifest", str(manifest)])
    assert exit_code == 1


def test_a_missing_manifest_is_reported_not_raised(tmp_path: Path) -> None:
    (tmp_path / "source.py").write_text(GOOD_SOURCE)
    (tmp_path / "inference.py").write_text(GOOD_INFERENCE)
    exit_code = chk.main(["--repo", str(tmp_path), "--manifest", str(tmp_path / "nope.json")])
    assert exit_code == 1


def test_the_report_names_every_check(repo: Path) -> None:
    results = list(_static(repo).values())
    report = chk.format_report(results, chk.RunOutcome(ran=False))
    for result in results:
        assert result.title in report
