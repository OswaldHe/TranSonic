# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The stage-2 gate: is this submodule repo a usable starting point for an optimization loop?

Deliberately *not* a check that the cut is right. How to divide a module into per-rank work is a
judgement about that module's semantics — which tensors are replicated, which are sharded, what a
partial result even is — and encoding it in a script would make this pipeline work for MoE and
nothing else. The agent decides the cut; this gate asks the module-agnostic questions:

    a  shape          `source.py` defines the declared entry point, and `inference.py` drives it
    b  self-contained  neither file imports or opens anything outside its allowlist and this repo
    c  measured        the run left a fresh profile and reported a real latency
    d  passes          the baseline exits 0 and clears the bar it declares
    e  provenance      tensors are the recorded bytes, not fabricated ones
    f  declared        `submodule.json` states the cut, and its reassembly recipe reproduces the
                       whole module's golden from the per-rank goldens
    g  one core        the run really used a single NeuronCore

(f) is the one that earns the rest. A wrong cut cannot be caught by reading `source.py`, but it
*can* be caught arithmetically: if the agent declares "rank r computes this, and the module is the
sum of the four ranks plus the shared path", then applying that recipe to the four dumped goldens
either reproduces the module's recorded output or it does not. That is a host-side numpy check over
bytes already on disk, it costs a second, and it is what makes a bad cut fail here instead of two
stages later — while staying entirely ignorant of what the module computes.

(d) requires the baseline to *pass*, which is the opposite of `bootstrap`'s red-by-construction
gate. An optimization loop measures a metric from iteration 0, and there is no metric to measure
until something works; a red baseline would make the 5% regression gate compare against nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from bootstrap.nki_checker import CheckResult
from optimization import candidate
from optimization.candidate import (
    CEILING_NAME,
    DECLARATION_FILE,
    INFERENCE_FILE,
    LATENCY_MARKER,
    MAX_ABS_ERR_MARKER,
    PASSED_MARKER,
    SOURCE_FILE,
    TOLERANCE_NAMES,
    CheckerError,
    RunOutcome,
)

CHECK_TITLES: dict[str, str] = {
    "a": "shape",
    "b": "self-containment",
    "c": "measurement",
    "d": "baseline passes",
    "e": "data provenance",
    "f": "declared cut",
    "g": "single core",
}

#: Import roots `source.py` may have. Unlike bootstrap's, torch is *allowed*: from iteration 4 the
#: constraint schedule permits a torch implementation, and a gate that forbade it would contradict
#: the schedule. Keeping NKI-only is the schedule's job, checked per iteration by a script the
#: agent cannot see.
SOURCE_ALLOWED_IMPORTS = frozenset({"nki", "neuronxcc", "torch", "torch_neuronx", "torch_xla", "numpy"})

#: What `inference.py` may have on top. It loads `.bin` bytes and drives the device.
INFERENCE_ALLOWED_IMPORTS = SOURCE_ALLOWED_IMPORTS | {"source"}

#: Where `optimize submodule` records what it built. Found by walking up, so a worktree at
#: `<repo>/.autohelix/worktrees/iter-N` still finds the repo's manifest above it.
MANIFEST_REL = ".autohelix/optimization/submodule.json"

#: How many NeuronCores a submodule run may use. One: the point of the stage is a kernel for a
#: single logical core, and a candidate that quietly used four would report a latency the
#: assembled module can never reproduce.
SUBMODULE_CORES = "1"


def find_manifest(repo: Path) -> Path:
    """The manifest for this repo, from the repo or any ancestor.

    Walking up is what lets the gate's command line be a fixed string with no per-repo path in
    it — so the command never names anything the agent could read to learn what is checked.
    """
    for candidate in [repo, *repo.parents]:
        path = candidate / MANIFEST_REL
        if path.is_file():
            return path
    raise CheckerError(
        f"no {MANIFEST_REL} at or above {repo} — was this repo made by `autohelix optimize`?"
    )


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckerError(f"{path} is unreadable: {exc}") from exc
    if not isinstance(data, dict):
        raise CheckerError(f"{path} is not a JSON object")
    return data


def expected_tolerance(manifest: dict[str, Any]) -> dict[str, float]:
    """The bar this repo is pinned to, from the manifest.

    No defaults. Unlike bootstrap — which can fall back to the bfloat16 row because it knows the
    reference's dtype — a submodule's golden is an intermediate the agent chose, so its bar was
    derived when the repo was built or it does not exist. Guessing one would let a repo with no
    recorded bar pass at a bar nobody chose.
    """
    recorded = manifest.get("tolerance") or {}
    missing = [n for n in (*TOLERANCE_NAMES, CEILING_NAME) if n not in recorded]
    if missing:
        raise CheckerError(
            f"the manifest records no {', '.join(missing)} — the submodule's numerical bar has to "
            f"be derived and written down when the repo is built"
        )
    return {n: float(recorded[n]) for n in (*TOLERANCE_NAMES, CEILING_NAME)}


# --------------------------------------------------------------------------------------
# the checks
# --------------------------------------------------------------------------------------


def check_shape(repo: Path, manifest: dict[str, Any]) -> CheckResult:
    """(a) The declared entry point exists and the validator reaches it."""
    entry = str(manifest.get("entry_point") or "kernel")
    findings: list[str] = []

    source = candidate._parse(repo / SOURCE_FILE)
    # Any top-level binding, not only a `def`. `kernel = _impl[2]` binds the NKI launch grid at
    # module scope, which is how a kernel reaches both physical cores of an LNC=2 pair while the
    # frozen validator still calls `kernel(*args)` with no subscript. Requiring a `FunctionDef`
    # rejected that, and it is the shape the 2x dual-core split needs.
    names = candidate.top_level_names(source)
    if entry not in names:
        findings.append(
            f"{SOURCE_FILE} defines no top-level '{entry}'. "
            f"Found: {', '.join(sorted(names)) or 'nothing'}"
        )

    inference = candidate._parse(repo / INFERENCE_FILE)
    imported = candidate._import_roots(inference)
    if "source" not in imported:
        findings.append(f"{INFERENCE_FILE} never imports {SOURCE_FILE}")
    if not candidate.invokes_entry_point(inference, entry):
        findings.append(
            f"{INFERENCE_FILE} neither traces nor calls '{entry}' — nothing connects the "
            f"validator to the kernel it is supposed to measure"
        )
    kinds = candidate.traced_entry_points(source)
    detail = f"'{entry}' present" + (f", @nki.jit on {', '.join(kinds)}" if kinds else "")
    return CheckResult("a", CHECK_TITLES["a"], not findings,
                       detail if not findings else f"{len(findings)} problem(s)", findings)


def check_self_contained(repo: Path) -> CheckResult:
    """(b) Neither file reaches outside its allowlist or outside this repository."""
    findings: list[str] = []
    for filename, allowed in ((SOURCE_FILE, SOURCE_ALLOWED_IMPORTS),
                              (INFERENCE_FILE, INFERENCE_ALLOWED_IMPORTS)):
        tree = candidate._parse(repo / filename)
        findings += candidate.import_findings(tree, allowed, filename)
        findings += candidate.path_findings(tree, filename)
    # The kernel may not open files at all: one that can read `tensors/` can read the golden and
    # hand it back, and the .bin allowlist on the validator's side cannot distinguish the two.
    source = candidate._parse(repo / SOURCE_FILE)
    for dotted, line in sorted(candidate.called_attributes(source).items(), key=lambda kv: kv[1]):
        if dotted.rsplit(".", 1)[-1] in candidate.FILE_IO_NAMES:
            findings.append(
                f"{SOURCE_FILE}:{line} calls '{dotted}' — the kernel receives its tensors as "
                f"arguments and may not read files"
            )
    return CheckResult("b", CHECK_TITLES["b"], not findings,
                       "both files are self-contained" if not findings
                       else f"{len(findings)} problem(s)", findings)


def check_measurement(run: RunOutcome) -> CheckResult:
    """(c) A fresh profile, and a latency read out of it."""
    findings: list[str] = []
    for kind in ("neff", "ntff"):
        if not run.artifacts.get(kind):
            findings.append(f"the run left no fresh .{kind} behind")
    latency = candidate.marker_value(run.output, LATENCY_MARKER)
    if latency is None:
        findings.append(f"the run printed no ##autohelix[{LATENCY_MARKER}=...] line")
    elif latency <= 0:
        findings.append(f"the reported latency is {latency:g} ms, which is not a measurement")
    summary = f"latency {latency:g} ms from a fresh profile" if not findings else \
        f"{len(findings)} problem(s)"
    return CheckResult("c", CHECK_TITLES["c"], not findings, summary, findings)


def check_baseline(run: RunOutcome, bar: dict[str, float], repo: Path) -> CheckResult:
    """(d) The baseline exits 0, declares the pinned bar, and clears it."""
    findings: list[str] = []
    if not run.ran:
        findings.append(f"the validator did not start: {run.detail}")
    else:
        if run.detail:
            findings.append(run.detail)
        if run.return_code != 0:
            tail = "\n".join(run.output.strip().splitlines()[-12:])
            findings.append(f"{INFERENCE_FILE} exited {run.return_code}\nlast output:\n{tail}")

    findings += candidate.pinned_constants(candidate._parse(repo / INFERENCE_FILE), bar, INFERENCE_FILE)

    passed = candidate.marker_value(run.output, PASSED_MARKER)
    if passed is None:
        findings.append(f"the run printed no ##autohelix[{PASSED_MARKER}=...] line")
    elif passed != 1:
        findings.append(f"the run reported {PASSED_MARKER}={passed:g}: the output does not match")

    worst = candidate.marker_value(run.output, MAX_ABS_ERR_MARKER)
    ceiling = bar[CEILING_NAME]
    if worst is None:
        findings.append(
            f"the run printed no ##autohelix[{MAX_ABS_ERR_MARKER}=...] line, so the "
            f"{CEILING_NAME} ceiling could not be checked"
        )
    elif worst > ceiling:
        findings.append(
            f"worst element is off by {worst:g}, over the {CEILING_NAME} ceiling of {ceiling:g}"
        )
    return CheckResult("d", CHECK_TITLES["d"], not findings,
                       "baseline runs clean at the pinned bar" if not findings
                       else f"{len(findings)} problem(s)", findings)


def check_provenance(repo: Path, manifest: dict[str, Any]) -> CheckResult:
    """(e) The tensors fed in are the recorded bytes, unedited and unfabricated."""
    findings: list[str] = []
    inference = candidate._parse(repo / INFERENCE_FILE)
    findings += candidate.fabrication_findings(inference, INFERENCE_FILE)

    recorded, hashed = candidate.provenance_findings(repo, manifest)
    findings += recorded
    total = len(candidate.recorded_tensors(manifest))
    return CheckResult("e", CHECK_TITLES["e"], not findings,
                       f"{total} tensor(s) present, {hashed} hash-checked" if not findings
                       else f"{len(findings)} problem(s)", findings)


def check_declaration(repo: Path, manifest: dict[str, Any]) -> CheckResult:
    """(f) The cut is declared, and the declared reassembly reproduces the module's golden.

    The arithmetic is done by `assemble.verify_recipe`, which knows nothing about what the module
    computes — it applies the declared recipe to the declared per-rank goldens and compares the
    result against the module's recorded output at the module's own bar.
    """
    from optimization.recipe import RecipeError, verify_recipe

    findings: list[str] = []
    path = repo / DECLARATION_FILE
    if not path.is_file():
        return CheckResult(
            "f", CHECK_TITLES["f"], False, "no declaration",
            [f"{DECLARATION_FILE} is missing — stage 4 reads it to know how to reassemble "
             f"these ranks into the whole module, and cannot proceed without it"],
        )
    try:
        declaration = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return CheckResult("f", CHECK_TITLES["f"], False, "unparseable declaration",
                           [f"{DECLARATION_FILE} is not valid JSON: {exc}"])

    required = ("module", "dim", "factor", "shard", "inputs", "outputs", "reassembly")
    for key in required:
        if key not in declaration:
            findings.append(f"{DECLARATION_FILE} does not state '{key}'")
    if findings:
        return CheckResult("f", CHECK_TITLES["f"], False,
                           f"{len(findings)} field(s) missing", findings)

    factor = int(declaration.get("factor") or 0)
    projected = candidate.projected_units(manifest)
    if projected and factor != projected:
        findings.append(
            f"{DECLARATION_FILE} declares a {factor}-way cut, but the floorplan projection for "
            f"this module is {projected}-way. The assembly runs {projected} ranks, so a "
            f"{factor}-way cut cannot be reassembled into the whole module"
        )

    # The dimension too, not only its width. Checking the factor alone let a declaration claim
    # `head x4` against an `expert x4` projection: the recipe verifier only asks whether the shards
    # recombine to the golden, and the module gate only asks for correctness and a collective, so a
    # valid kernel for a *different* four-way placement passed while the report attributed its
    # measurements to the recorded floorplan. Only checked when the projection splits one dimension,
    # because a multi-dimension projection has no single dimension the declaration must name.
    dims = candidate.projected_dims(manifest)
    declared_dim = str(declaration.get("dim") or "").strip()
    if len(dims) == 1 and declared_dim and declared_dim.lower() != dims[0].lower():
        findings.append(
            f"{DECLARATION_FILE} declares the cut along '{declared_dim}', but the floorplan "
            f"projection for this module splits '{dims[0]}'. A {factor}-way cut along another "
            f"dimension can still reproduce the golden, so nothing downstream catches this — and "
            f"the report would then credit this kernel to a placement it does not implement"
        )

    try:
        outcome = verify_recipe(repo, declaration, manifest)
    except RecipeError as exc:
        findings.append(str(exc))
    else:
        if not outcome.reproduces:
            findings.append(
                f"applying the declared reassembly to the {factor} per-rank goldens does not "
                f"reproduce the module's recorded output: {outcome.detail}"
            )
    return CheckResult("f", CHECK_TITLES["f"], not findings,
                       f"{factor}-way cut along '{declaration.get('dim')}', reassembly verified"
                       if not findings else f"{len(findings)} problem(s)", findings)


def check_single_core(repo: Path) -> CheckResult:
    """(g) The run used one NeuronCore.

    The gate sets `NEURON_RT_NUM_CORES=1` in the child's environment, so the only way to use more
    is for the candidate to overwrite it. Checked as a static read of the validator rather than by
    inspecting the profile: a candidate that sets the variable is stating an intent, and the
    intent is what the check is about.
    """
    inference = candidate._parse(repo / INFERENCE_FILE)
    findings = candidate.core_allocation_findings(inference, INFERENCE_FILE, SUBMODULE_CORES)
    if "torchrun" in (repo / INFERENCE_FILE).read_text():
        findings.append(
            f"{INFERENCE_FILE} mentions torchrun: a submodule is a single-rank kernel, and a "
            f"multi-process run here would measure something the assembly cannot reproduce"
        )
    return CheckResult("g", CHECK_TITLES["g"], not findings,
                       f"ran with NEURON_RT_NUM_CORES={SUBMODULE_CORES}" if not findings
                       else f"{len(findings)} problem(s)", findings)


def evaluate(repo: Path, manifest: dict[str, Any],
             timeout: int) -> tuple[list[CheckResult], RunOutcome]:
    """Run the candidate once, then answer all seven checks from that one execution."""
    bar = expected_tolerance(manifest)
    run = candidate.run_candidate(
        repo, [sys.executable, INFERENCE_FILE], timeout=timeout,
        env_overrides={"NEURON_RT_NUM_CORES": SUBMODULE_CORES},
    )
    results = [
        check_shape(repo, manifest),
        check_self_contained(repo),
        check_measurement(run),
        check_baseline(run, bar, repo),
        check_provenance(repo, manifest),
        check_declaration(repo, manifest),
        check_single_core(repo),
    ]
    return results, run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default=".", help="the candidate repository")
    parser.add_argument("--json", default=None, help="where to write the machine-readable verdict")
    parser.add_argument("--timeout", type=int, default=candidate.DEFAULT_RUN_TIMEOUT)
    args = parser.parse_args(argv)

    repo = Path(args.repo).resolve()
    try:
        manifest = load_manifest(find_manifest(repo))
        results, run = evaluate(repo, manifest, args.timeout)
    except CheckerError as exc:
        report = f"\nsubmodule gate\n\n  [FAIL] the repo is unusable — {exc}\n"
        print(report)
        if args.json:
            candidate.write_verdict([], None, "submodule gate", Path(args.json),
                               extra={"passed": False, "error": str(exc), "report": report})
        return 2

    # The latency goes into the verdict, not just the run's stdout: `optimization.readback` is
    # AutoHelix's metric command and reads it from here, so the gate stays the only thing that
    # executes the candidate. Two runs could disagree about which code was measured.
    extra: dict[str, Any] = {}
    latency = candidate.marker_value(run.output, LATENCY_MARKER)
    if latency is not None:
        extra[LATENCY_MARKER] = latency

    verdict = candidate.write_verdict(
        results, run, "submodule gate", Path(args.json) if args.json else None, extra=extra,
    )
    print(verdict.report)
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
