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
import ast
import json
import sys
from pathlib import Path
from typing import Any

from bootstrap.nki_checker import CheckResult
from optimization import gate
from optimization.gate import (
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

    source = gate._parse(repo / SOURCE_FILE)
    functions = gate.top_level_functions(source)
    if entry not in functions:
        findings.append(
            f"{SOURCE_FILE} defines no top-level '{entry}'. "
            f"Found: {', '.join(sorted(functions)) or 'nothing'}"
        )

    inference = gate._parse(repo / INFERENCE_FILE)
    imported = gate._import_roots(inference)
    if "source" not in imported:
        findings.append(f"{INFERENCE_FILE} never imports {SOURCE_FILE}")
    calls = gate.called_attributes(inference)
    traced = [d for d in calls if d.endswith("trace") or d.endswith("nki_jit")]
    if not traced and entry not in calls and f"source.{entry}" not in calls:
        findings.append(
            f"{INFERENCE_FILE} neither traces nor calls '{entry}' — nothing connects the "
            f"validator to the kernel it is supposed to measure"
        )
    kinds = gate.traced_entry_points(source)
    detail = f"'{entry}' present" + (f", @nki.jit on {', '.join(kinds)}" if kinds else "")
    return CheckResult("a", CHECK_TITLES["a"], not findings,
                       detail if not findings else f"{len(findings)} problem(s)", findings)


def check_self_contained(repo: Path) -> CheckResult:
    """(b) Neither file reaches outside its allowlist or outside this repository."""
    findings: list[str] = []
    for filename, allowed in ((SOURCE_FILE, SOURCE_ALLOWED_IMPORTS),
                              (INFERENCE_FILE, INFERENCE_ALLOWED_IMPORTS)):
        tree = gate._parse(repo / filename)
        findings += gate.import_findings(tree, allowed, filename)
        findings += gate.path_findings(tree, filename)
    # The kernel may not open files at all: one that can read `tensors/` can read the golden and
    # hand it back, and the .bin allowlist on the validator's side cannot distinguish the two.
    source = gate._parse(repo / SOURCE_FILE)
    for dotted, line in sorted(gate.called_attributes(source).items(), key=lambda kv: kv[1]):
        if dotted.rsplit(".", 1)[-1] in gate.FILE_IO_NAMES:
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
    latency = gate.marker_value(run.output, LATENCY_MARKER)
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

    findings += gate.pinned_constants(gate._parse(repo / INFERENCE_FILE), bar, INFERENCE_FILE)

    passed = gate.marker_value(run.output, PASSED_MARKER)
    if passed is None:
        findings.append(f"the run printed no ##autohelix[{PASSED_MARKER}=...] line")
    elif passed != 1:
        findings.append(f"the run reported {PASSED_MARKER}={passed:g}: the output does not match")

    worst = gate.marker_value(run.output, MAX_ABS_ERR_MARKER)
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
    inference = gate._parse(repo / INFERENCE_FILE)
    findings += gate.fabrication_findings(inference, INFERENCE_FILE)

    recorded = manifest.get("tensors") or {}
    if not recorded:
        findings.append("the manifest lists no tensors, so provenance cannot be established")
    for name, entry in sorted(recorded.items()):
        rel = str(entry.get("file") or "")
        path = repo / rel
        if not path.is_file():
            findings.append(f"{rel} is missing from the repo")
            continue
        want = entry.get("sha256")
        if want and gate_sha256(path) != want:
            findings.append(
                f"{rel} has been edited since the repo was built — a golden changed to agree "
                f"with a wrong kernel is the one failure no numerical check can catch"
            )
        size = entry.get("bytes")
        if size is not None and path.stat().st_size != int(size):
            findings.append(
                f"{rel} is {path.stat().st_size} bytes, recorded as {size}"
            )
    return CheckResult("e", CHECK_TITLES["e"], not findings,
                       f"{len(recorded)} tensor(s) match the record" if not findings
                       else f"{len(findings)} problem(s)", findings)


def gate_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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
    projected = int((manifest.get("projection") or {}).get("projected_units") or 0)
    if projected and factor != projected:
        findings.append(
            f"{DECLARATION_FILE} declares a {factor}-way cut, but the floorplan projection for "
            f"this module is {projected}-way. The assembly runs {projected} ranks, so a "
            f"{factor}-way cut cannot be reassembled into the whole module"
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


def check_single_core(run: RunOutcome, repo: Path) -> CheckResult:
    """(g) The run used one NeuronCore.

    The gate sets `NEURON_RT_NUM_CORES=1` in the child's environment, so the only way to use more
    is for the candidate to overwrite it. Checked as a static read of the validator rather than by
    inspecting the profile: a candidate that sets the variable is stating an intent, and the
    intent is what the check is about.
    """
    findings: list[str] = []
    inference = gate._parse(repo / INFERENCE_FILE)
    for text, line in gate._string_literals(inference):
        if text.strip() in {"NEURON_RT_NUM_CORES", "NEURON_RT_VISIBLE_CORES"}:
            assigned = _environ_assignments(inference)
            if text.strip() in assigned:
                findings.append(
                    f"{INFERENCE_FILE}:{line} sets {text.strip()} itself. How much of the device "
                    f"a submodule may use is not the validator's to choose — it is one core"
                )
    if "torchrun" in (repo / INFERENCE_FILE).read_text():
        findings.append(
            f"{INFERENCE_FILE} mentions torchrun: a submodule is a single-rank kernel, and a "
            f"multi-process run here would measure something the assembly cannot reproduce"
        )
    return CheckResult("g", CHECK_TITLES["g"], not findings,
                       f"ran with NEURON_RT_NUM_CORES={SUBMODULE_CORES}" if not findings
                       else f"{len(findings)} problem(s)", findings)


def _environ_assignments(tree: ast.Module) -> set[str]:
    """Environment variables the file assigns, as in `os.environ["X"] = ...`."""
    assigned: set[str] = set()
    for node in ast.walk(tree):
        targets: list[Any] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.Call):
            dotted = gate.called_attributes(ast.Module(body=[ast.Expr(node)], type_ignores=[]))
            if any(d.endswith("setdefault") or d.endswith("putenv") for d in dotted):
                for arg in node.args[:1]:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        assigned.add(arg.value)
        for target in targets:
            if isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant):
                if isinstance(target.slice.value, str):
                    assigned.add(target.slice.value)
    return assigned


# --------------------------------------------------------------------------------------
# driving
# --------------------------------------------------------------------------------------


def evaluate(repo: Path, manifest: dict[str, Any],
             timeout: int) -> tuple[list[CheckResult], RunOutcome]:
    """Run the candidate once, then answer all seven checks from that one execution."""
    bar = expected_tolerance(manifest)
    run = gate.run_candidate(
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
        check_single_core(run, repo),
    ]
    return results, run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default=".", help="the candidate repository")
    parser.add_argument("--json", default=None, help="where to write the machine-readable verdict")
    parser.add_argument("--timeout", type=int, default=gate.DEFAULT_RUN_TIMEOUT)
    args = parser.parse_args(argv)

    repo = Path(args.repo).resolve()
    try:
        manifest = load_manifest(find_manifest(repo))
        results, run = evaluate(repo, manifest, args.timeout)
    except CheckerError as exc:
        report = f"\nsubmodule gate\n\n  [FAIL] the repo is unusable — {exc}\n"
        print(report)
        if args.json:
            gate.write_verdict([], None, "submodule gate", Path(args.json),
                               extra={"passed": False, "error": str(exc), "report": report})
        return 2

    # The latency goes into the verdict, not just the run's stdout: `optimization.readback` is
    # AutoHelix's metric command and reads it from here, so the gate stays the only thing that
    # executes the candidate. Two runs could disagree about which code was measured.
    extra: dict[str, Any] = {}
    latency = gate.marker_value(run.output, LATENCY_MARKER)
    if latency is not None:
        extra[LATENCY_MARKER] = latency

    verdict = gate.write_verdict(
        results, run, "submodule gate", Path(args.json) if args.json else None, extra=extra,
    )
    print(verdict.report)
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
