# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Trainium NKI bootstrap constraint: six checks, one exit code.

The gate `autohelix bootstrap` runs after every iteration, answering one question — is this
repo a real NKI kernel with a real validator yet — in six parts: (a) kernel formalization,
(b) self-containment, (c) NKI-only source, (d) metric measurement, (e) pass-test,
(f) data provenance. `bootstrap/README.md` spells each out.

**The agent never sees this file**; `preset.yaml`'s goal states all six requirements in
prose. So if the two drift the goal is wrong, not the agent, and every message here has to
be actionable by someone who has only read the goal.

Exits 0 only when all six pass. Per-check verdicts go to stdout as a table and to `--json`
as a record, so a failing run says what is left rather than only that something is wrong.
Standard library only, so whatever the repo's own inference.py needs installed cannot
perturb it.

    python -m bootstrap.nki_checker --repo .
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: The two files the agent owns. Everything else in a module repo is frozen.
SOURCE_FILE = "source.py"
INFERENCE_FILE = "inference.py"

#: The kernel entry point, by name. A fixed name is what lets check (a) be a structural
#: question rather than a guess about which function is "the" kernel.
KERNEL_FUNCTION = "kernel"

#: The four constants inference.py must declare, and the fallback for a manifest recording
#: no bar of its own. The real one is per-repo — the tolerance the artifact accepted *this*
#: reference at, which follows its dtype — so `bootstrap init` derives it into the manifest.
#: These defaults are the bfloat16 row.
TOLERANCE_NAMES = ("RTOL", "ATOL", "MIN_COSINE", "MIN_PASS_FRACTION")
PINNED_TOLERANCE: dict[str, float] = {
    "RTOL": 2e-2,
    "ATOL": 2e-2,
    "MIN_COSINE": 0.9999,
    "MIN_PASS_FRACTION": 0.999,
}

#: A fifth constant, and the only one with no sensible default: it is `atol + rtol *
#: max|reference|` for *this* module's recorded output, so it exists only when the manifest
#: records it. Unlike the other four it is also checked against what the run reports — a
#: declared ceiling the candidate then ignores would be no ceiling at all.
CEILING_NAME = "MAX_ABS_ERR"
MAX_ABS_ERR_MARKER = "max_abs_err"

#: Import roots source.py may have. NKI and the standard library, and that is all.
SOURCE_ALLOWED_IMPORTS = frozenset({"nki", "neuronxcc"})

#: Import roots inference.py may have on top of source.py's. It has to build tensors and
#: drive the device, so torch is legitimate here even though (c) forbids it in the kernel.
INFERENCE_EXTRA_IMPORTS = frozenset({"torch", "torch_neuronx", "torch_xla", "numpy", "source"})

#: Names whose mere appearance in source.py means the kernel is not NKI-only.
BANNED_IN_SOURCE = frozenset({"torch", "numpy", "np", "scipy", "sp", "torch_neuronx", "torch_xla"})

#: Escape hatches that would let either file reach outside its declared imports.
DYNAMIC_IMPORT_NAMES = frozenset({"__import__", "importlib", "exec", "eval", "compile"})

#: Ways to open a file, banned in source.py: a kernel that can open files can open
#: `tensors/reference.bin` and return the answer, which the .bin allowlist alone permits.
#:
#: Narrow on purpose. `nl.load` moves a tensor into on-chip memory and is in every kernel,
#: and `frombuffer` reinterprets bytes already in hand; flagging either would fail correct
#: work, which is worse than missing a trick the reviewer also looks for.
FILE_IO_NAMES = frozenset({
    "open", "fromfile", "read_bytes", "read_text", "loadtxt", "memmap",
})

#: Substrings that mean a path is reaching back into the artifact or the checkpoint
#: instead of using the repo's own copies. Tested only against path-shaped literals —
#: prose mentioning `reference_torch.py` is documentation, not an escape.
FORBIDDEN_PATH_MARKERS = (
    "..",
    "partition-artifact",
    "model_partition",
    "safetensors",
    # Every frozen reference: reference_*.py, vendor_*.py, compat_*.py. They are carried in
    # to be read, not opened at runtime. `vendor` also catches the artifact's own directory.
    "reference_",
    "vendor",
    "compat",
    "/trace/",
    "/hf/",
)

#: Extensions that make a bare literal a path even without a separator in it.
PATH_SUFFIXES = (
    ".bin", ".py", ".safetensors", ".json", ".yaml", ".yml", ".pt", ".pth", ".npy", ".npz",
    ".neff", ".ntff", ".csv", ".txt",
)

#: Tensor constructors that can fabricate data. `zeros`/`empty` and their `_like` forms
#: are deliberately absent: allocating an output buffer is ordinary, and a zeroed weight
#: cannot pass check (e) anyway.
SYNTHETIC_CONSTRUCTORS = frozenset({
    "randn", "randn_like", "rand", "rand_like", "randint", "randint_like", "randperm",
    "normal", "uniform", "full", "full_like", "ones", "ones_like", "arange", "linspace",
    "logspace", "eye", "bernoulli", "multinomial", "poisson", "manual_seed",
})

#: In-place fills, the same problem spelled differently.
SYNTHETIC_INPLACE = frozenset({
    "normal_", "uniform_", "random_", "fill_", "exponential_", "cauchy_",
    "log_normal_", "geometric_", "bernoulli_",
})

#: Markers inference.py must print. The latency one is informational — the bootstrap loop
#: targets no metric — but (d) is the check that the measurement path actually works, so
#: the number has to be real and present.
LATENCY_MARKER = "latency_ms"
PASSED_MARKER = "passed"

#: How long inference.py gets. Comfortably inside the 1200s the constraint itself is
#: given, so a hung device run is reported as a failed check rather than a dead harness.
DEFAULT_RUN_TIMEOUT = 900

CHECK_TITLES: dict[str, str] = {
    "a": "kernel formalization",
    "b": "self-containment",
    "c": "nki-only",
    "d": "metric measurement",
    "e": "pass-test",
    "f": "data provenance",
}


def expected_tolerance(manifest: dict[str, Any]) -> dict[str, float]:
    """The bar this repo is held to: the manifest's, or the bfloat16 default.

    `MAX_ABS_ERR` joins the four when the manifest carries it, and is absent otherwise —
    there is no default for a number derived from this module's own recorded output.
    """
    recorded = manifest.get("tolerance") or {}
    bar = {name: float(recorded.get(name, PINNED_TOLERANCE[name])) for name in TOLERANCE_NAMES}
    if CEILING_NAME in recorded:
        bar[CEILING_NAME] = float(recorded[CEILING_NAME])
    return bar


@dataclass
class CheckResult:
    """One of the six verdicts."""

    key: str
    title: str
    passed: bool
    summary: str = ""
    findings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "check": self.key,
            "title": self.title,
            "passed": self.passed,
            "summary": self.summary,
            "findings": self.findings,
        }


class CheckerError(RuntimeError):
    """The repo or the manifest is unusable, so no verdict can be reached."""


# --------------------------------------------------------------------------------------
# parsing helpers
# --------------------------------------------------------------------------------------


def _parse(path: Path) -> ast.Module:
    """Parse a file the agent owns, reporting a syntax error as a finding not a crash."""
    try:
        return ast.parse(path.read_text(), filename=str(path))
    except FileNotFoundError as exc:
        raise CheckerError(f"{path.name} is missing") from exc
    except SyntaxError as exc:
        raise CheckerError(f"{path.name} does not parse: line {exc.lineno}: {exc.msg}") from exc


def _import_roots(tree: ast.Module) -> dict[str, int]:
    """Top-level package of every import, mapped to the line it appears on."""
    roots: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.setdefault(alias.name.split(".")[0], node.lineno)
        elif isinstance(node, ast.ImportFrom):
            # `from . import x` has no module; a relative import is out of bounds anyway
            # and is reported under its own name.
            root = (node.module or "").split(".")[0] if node.level == 0 else "<relative>"
            roots.setdefault(root or "<relative>", node.lineno)
    return roots


def _docstring_nodes(tree: ast.Module) -> set[int]:
    """`id()` of every docstring constant in the tree.

    Docstrings are prose and are excluded from the path rules: this file's own generated
    stubs legitimately write `reference_torch.py` and `##autohelix[latency_ms=...]` in
    theirs, and neither is a path. Excluding them hides nothing, because a docstring cannot
    execute and the two ways to turn a string into code — `exec` and `importlib` — are
    refused outright.
    """
    marked: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
            if isinstance(first.value.value, str):
                marked.add(id(first.value))
    return marked


def _string_literals(tree: ast.Module) -> list[tuple[str, int]]:
    """Every non-docstring string constant, with its line."""
    docstrings = _docstring_nodes(tree)
    out: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in docstrings:
                out.append((node.value, node.lineno))
    return out


def _path_like(text: str) -> bool:
    """Whether a literal is plausibly a filesystem path rather than a sentence.

    A path has no whitespace and either a separator or a file extension. Without this the
    rules below fire on error messages — "load the tensors from tensors/*.bin" is advice,
    not a file that must be in the manifest.
    """
    if not text or len(text) > 512 or any(c.isspace() for c in text):
        return False
    return "/" in text or text.endswith(PATH_SUFFIXES)


def _decorator_names(func: ast.FunctionDef) -> list[str]:
    """Dotted names of a function's decorators, `@nki.jit` and `@nki.jit(...)` alike."""
    names: list[str] = []
    for dec in func.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        parts: list[str] = []
        while isinstance(target, ast.Attribute):
            parts.append(target.attr)
            target = target.value
        if isinstance(target, ast.Name):
            parts.append(target.id)
        if parts:
            names.append(".".join(reversed(parts)))
    return names


def _module_level_numbers(tree: ast.Module) -> dict[str, tuple[Any, int]]:
    """Module-level `NAME = <number literal>` assignments.

    Only bare literals are collected. An expression — `RTOL = 2e-2 * 5` — is absent from
    the result and so reads as a missing constant, which is the intent: the pinned bar has
    to be legible at a glance, not computed.
    """
    found: dict[str, tuple[Any, int]] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not (isinstance(node.value, ast.Constant) and isinstance(node.value.value, (int, float))):
            continue
        if isinstance(node.value.value, bool):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                found[target.id] = (node.value.value, node.lineno)
    return found


def _attribute_name(node: ast.Attribute) -> str:
    """`torch.nn.functional` for a chain of attributes, innermost name first."""
    parts: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------------------
# the checks
# --------------------------------------------------------------------------------------


def check_kernel_formalization(source: ast.Module, inference: ast.Module) -> CheckResult:
    """(a) A `@nki.jit` kernel in source.py, reached through torch_neuronx.trace."""
    findings: list[str] = []

    kernels = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == KERNEL_FUNCTION]
    if not kernels:
        defined = [n.name for n in source.body if isinstance(n, ast.FunctionDef)]
        findings.append(
            f"{SOURCE_FILE} defines no top-level function named '{KERNEL_FUNCTION}'"
            + (f" (found: {', '.join(defined)})" if defined else "")
        )
    else:
        decorators = _decorator_names(kernels[0])
        if not any(d == "nki.jit" or d.endswith(".nki.jit") or d == "jit" for d in decorators):
            findings.append(
                f"{SOURCE_FILE}:{kernels[0].lineno}: '{KERNEL_FUNCTION}' is not decorated with "
                f"@nki.jit (decorators: {', '.join(decorators) or 'none'})"
            )

    # inference.py has to reach the kernel by name, and drive it through a trace.
    imports_kernel = False
    for node in ast.walk(inference):
        if isinstance(node, ast.ImportFrom) and (node.module or "") == "source":
            if any(a.name == KERNEL_FUNCTION for a in node.names):
                imports_kernel = True
        elif isinstance(node, ast.Attribute) and _attribute_name(node) == f"source.{KERNEL_FUNCTION}":
            imports_kernel = True
    if not imports_kernel:
        findings.append(
            f"{INFERENCE_FILE} never references source.{KERNEL_FUNCTION} — the kernel must be "
            f"its entry point, not a reimplementation"
        )

    traced = False
    for node in ast.walk(inference):
        if isinstance(node, ast.Attribute) and _attribute_name(node).endswith("torch_neuronx.trace"):
            traced = True
        elif isinstance(node, ast.ImportFrom) and (node.module or "") == "torch_neuronx":
            if any(a.name == "trace" for a in node.names):
                traced = True
    if not traced:
        findings.append(f"{INFERENCE_FILE} never calls torch_neuronx.trace")

    return CheckResult(
        key="a",
        title=CHECK_TITLES["a"],
        passed=not findings,
        summary="kernel is formalized and traced" if not findings else f"{len(findings)} problem(s)",
        findings=findings,
    )


def check_self_containment(
    source: ast.Module, inference: ast.Module, allowed_bins: set[str],
) -> CheckResult:
    """(b) Nothing imported or opened beyond the pair, the stdlib, NKI and the .bin set."""
    findings: list[str] = []
    stdlib = set(sys.stdlib_module_names)

    for name, tree, allowed in (
        (SOURCE_FILE, source, SOURCE_ALLOWED_IMPORTS),
        (INFERENCE_FILE, inference, SOURCE_ALLOWED_IMPORTS | INFERENCE_EXTRA_IMPORTS),
    ):
        for root, line in sorted(_import_roots(tree).items()):
            if root in stdlib or root in allowed:
                continue
            findings.append(
                f"{name}:{line}: imports '{root}', which is outside the allowed set "
                f"(standard library, {', '.join(sorted(allowed))})"
            )
        for node in ast.walk(tree):
            found = None
            if isinstance(node, ast.Name) and node.id in DYNAMIC_IMPORT_NAMES:
                found = node.id
            elif isinstance(node, ast.Attribute) and node.attr in DYNAMIC_IMPORT_NAMES:
                found = node.attr
            if found:
                findings.append(
                    f"{name}:{node.lineno}: uses '{found}', which can load code the import "
                    f"list does not declare"
                )

        if name == SOURCE_FILE:
            # The kernel takes tensors as arguments; it has no business touching the
            # filesystem, and the one file it could usefully open is the reference.
            for node in ast.walk(tree):
                opener = None
                if isinstance(node, ast.Name) and node.id in FILE_IO_NAMES:
                    opener = node.id
                elif isinstance(node, ast.Attribute) and node.attr in FILE_IO_NAMES:
                    opener = node.attr
                if opener:
                    findings.append(
                        f"{name}:{node.lineno}: uses '{opener}' — the kernel receives its "
                        f"tensors as arguments and must not read files"
                    )
            for text, line in _string_literals(tree):
                if _path_like(text):
                    findings.append(
                        f"{name}:{line}: names the path '{text}' — the kernel must not "
                        f"reference files at all"
                    )

        for text, line in _string_literals(tree):
            if not _path_like(text):
                continue
            for marker in FORBIDDEN_PATH_MARKERS:
                if marker in text:
                    findings.append(
                        f"{name}:{line}: path '{text}' contains '{marker}' — this repo must "
                        f"not reach outside itself for code or data"
                    )
                    break
            if text.endswith(".bin") and text not in allowed_bins:
                findings.append(
                    f"{name}:{line}: reads '{text}', which is not one of this repo's tensors"
                )

    return CheckResult(
        key="b",
        title=CHECK_TITLES["b"],
        passed=not findings,
        summary="both files are self-contained" if not findings else f"{len(findings)} problem(s)",
        findings=findings,
    )


def check_nki_only(source: ast.Module, source_text: str) -> CheckResult:
    """(c) source.py is NKI, not torch with NKI spelling."""
    findings: list[str] = []
    seen: set[tuple[str, int]] = set()

    for root, line in sorted(_import_roots(source).items()):
        if root in BANNED_IN_SOURCE:
            findings.append(f"{SOURCE_FILE}:{line}: imports '{root}'")
            seen.add((root, line))

    for node in ast.walk(source):
        name = None
        if isinstance(node, ast.Name) and node.id in BANNED_IN_SOURCE:
            name = node.id
        elif isinstance(node, ast.Attribute) and node.attr in BANNED_IN_SOURCE:
            name = node.attr
        if name and (name, node.lineno) not in seen:
            seen.add((name, node.lineno))
            findings.append(
                f"{SOURCE_FILE}:{node.lineno}: references '{name}' — the kernel must compute "
                f"with NKI alone"
            )

    # A mention in a comment is not a correctness problem, but it is worth surfacing: it
    # usually means a torch implementation was translated line by line and something was
    # left behind.
    notes: list[str] = []
    for number, raw in enumerate(source_text.splitlines(), start=1):
        stripped = raw.split("#", 1)
        if len(stripped) == 2:
            for banned in ("torch", "numpy", "scipy"):
                if banned in stripped[1]:
                    notes.append(f"{SOURCE_FILE}:{number}: comment mentions '{banned}' (not a failure)")
                    break

    return CheckResult(
        key="c",
        title=CHECK_TITLES["c"],
        passed=not findings,
        summary="kernel is NKI-only" if not findings else f"{len(findings)} reference(s) to torch/numpy/scipy",
        findings=findings + notes[:5],
    )


def check_measurement(inference: ast.Module, repo: Path, run: "RunOutcome") -> CheckResult:
    """(d) Profiles are dumped and a latency is read out of neuron-explorer."""
    findings: list[str] = []

    literals = " ".join(text for text, _ in _string_literals(inference))
    if "neuron-explorer" not in literals:
        findings.append(f"{INFERENCE_FILE} never invokes neuron-explorer")
    if "total_exec_time" not in literals:
        findings.append(f"{INFERENCE_FILE} never reads total_exec_time")

    if not run.ran:
        findings.append(f"{INFERENCE_FILE} did not run, so no profile could be produced")
    else:
        for extension in (".neff", ".ntff"):
            fresh = [
                p for p in repo.rglob(f"*{extension}")
                if p.is_file() and p.stat().st_mtime >= run.started_at - 1
            ]
            if not fresh:
                findings.append(f"the run left no {extension} file behind")

        latency = _marker_value(run.output, LATENCY_MARKER)
        if latency is None:
            findings.append(f"the run printed no ##autohelix[{LATENCY_MARKER}=...] line")
        elif latency <= 0:
            findings.append(f"reported {LATENCY_MARKER}={latency:g}, which cannot be a real latency")

    return CheckResult(
        key="d",
        title=CHECK_TITLES["d"],
        passed=not findings,
        summary="latency measured from a real profile" if not findings else f"{len(findings)} problem(s)",
        findings=findings,
    )


def check_pass_test(
    inference: ast.Module, run: "RunOutcome", tolerance: dict[str, float] | None = None,
) -> CheckResult:
    """(e) It runs, it passes, and it passes at the tolerance it was given."""
    findings: list[str] = []

    constants = _module_level_numbers(inference)
    for name, expected in (tolerance or PINNED_TOLERANCE).items():
        if name not in constants:
            findings.append(
                f"{INFERENCE_FILE} does not declare {name} as a module-level number literal "
                f"(expected {name} = {expected:g})"
            )
            continue
        value, line = constants[name]
        if float(value) != expected:
            # Direction-neutral on purpose. "may not be loosened" was the original wording and
            # it misreads badly when the bar has been *retuned* between runs: the candidate is
            # then told its stricter value is wrong for a reason that says do not loosen.
            findings.append(
                f"{INFERENCE_FILE}:{line}: {name} is {value:g}, but the bar is {expected:g} — "
                f"it must be exactly the value README.md gives, in either direction"
            )

    ceiling = (tolerance or {}).get(CEILING_NAME)

    if not run.ran:
        findings.append(f"{INFERENCE_FILE} could not be run: {run.detail}")
    else:
        if ceiling is not None:
            # The ceiling is checked against the run, not only pinned as a literal: a
            # candidate that declares it and then compares without it has not been held to it.
            worst = _marker_value(run.output, MAX_ABS_ERR_MARKER)
            if worst is None:
                findings.append(
                    f"the run printed no ##autohelix[{MAX_ABS_ERR_MARKER}=...] line, so the "
                    f"{CEILING_NAME} ceiling could not be checked"
                )
            elif worst > ceiling:
                findings.append(
                    f"worst element is off by {worst:.6g}, past the {CEILING_NAME} ceiling of "
                    f"{ceiling:.6g} — no single element may exceed it, whatever the pass fraction"
                )
        if run.return_code != 0:
            tail = "\n".join(run.output.strip().splitlines()[-20:])
            findings.append(f"{INFERENCE_FILE} exited {run.return_code}")
            if tail:
                findings.append(f"last output:\n{tail}")
        passed = _marker_value(run.output, PASSED_MARKER)
        if passed is None:
            findings.append(f"the run printed no ##autohelix[{PASSED_MARKER}=...] line")
        elif passed != 1:
            findings.append(f"the run reported {PASSED_MARKER}={passed:g}: the output does not match the reference")

    return CheckResult(
        key="e",
        title=CHECK_TITLES["e"],
        passed=not findings,
        summary="runs clean and matches at the pinned bar" if not findings else f"{len(findings)} problem(s)",
        findings=findings,
    )


def check_provenance(inference: ast.Module, repo: Path, manifest: dict[str, Any]) -> CheckResult:
    """(f) The tensors fed to the kernel are the recorded ones, unmodified."""
    findings: list[str] = []
    tensors = manifest.get("tensors") or []

    for entry in tensors:
        path = repo / entry["file"]
        if not path.is_file():
            findings.append(f"{entry['file']} is missing from the repo")
            continue
        actual = _sha256(path)
        if actual != entry["sha256"]:
            findings.append(
                f"{entry['file']} has been modified (sha256 {actual[:12]}..., "
                f"recorded {entry['sha256'][:12]}...)"
            )

    referenced = {text for text, _ in _string_literals(inference) if text.endswith(".bin")}
    for entry in tensors:
        if entry.get("required") and entry["file"] not in referenced:
            findings.append(
                f"{INFERENCE_FILE} never reads {entry['file']}, which holds the "
                f"{entry.get('role', 'recorded')} tensor it must be checked against"
            )
    unused = [e["file"] for e in tensors if not e.get("required") and e["file"] not in referenced]

    for node in ast.walk(inference):
        if not isinstance(node, ast.Call):
            continue
        name = None
        if isinstance(node.func, ast.Attribute):
            name = node.func.attr
        elif isinstance(node.func, ast.Name):
            name = node.func.id
        if name in SYNTHETIC_CONSTRUCTORS:
            findings.append(
                f"{INFERENCE_FILE}:{node.lineno}: calls '{name}' — inputs, weights and the "
                f"reference must come from the recorded .bin files, not be generated"
            )
        elif name in SYNTHETIC_INPLACE:
            findings.append(
                f"{INFERENCE_FILE}:{node.lineno}: calls '{name}', which overwrites a tensor "
                f"with generated values"
            )

    notes = (
        [f"{len(unused)} recorded tensor(s) unused: {', '.join(unused[:6])} (not a failure)"]
        if unused else []
    )
    return CheckResult(
        key="f",
        title=CHECK_TITLES["f"],
        passed=not findings,
        summary="fed from the recorded tensors" if not findings else f"{len(findings)} problem(s)",
        findings=findings + notes,
    )


# --------------------------------------------------------------------------------------
# running the candidate
# --------------------------------------------------------------------------------------


@dataclass
class RunOutcome:
    """What happened when inference.py was executed."""

    ran: bool
    return_code: int = -1
    output: str = ""
    detail: str = ""
    started_at: float = 0.0
    duration_s: float = 0.0


def _kill_process_group(proc: subprocess.Popen) -> None:
    """SIGKILL the whole session the run was started in, tolerating an already-dead one."""
    import os
    import signal

    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()


def run_inference(repo: Path, timeout: int) -> RunOutcome:
    """Execute the repo's own inference.py once, and keep everything it said.

    Checks (d) and (e) are both answered from this single run: dumping a profile and
    clearing the numerical bar are two properties of one execution, and running it twice
    would let them disagree.
    """
    started = time.time()
    try:
        # Its own session, so a timeout can kill the whole tree. Tracing a kernel spawns a
        # compiler and a profiler; killing only the direct child leaves those holding the
        # Neuron device, and the next iteration then fails for a reason that has nothing to
        # do with its own work.
        proc = subprocess.Popen(
            [sys.executable, INFERENCE_FILE],
            cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            start_new_session=True,
        )
    except OSError as exc:
        return RunOutcome(ran=False, detail=str(exc), started_at=started)

    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_process_group(proc)
        stdout, stderr = proc.communicate()
        return RunOutcome(
            ran=True, return_code=-1, output=(stdout or "") + (stderr or ""),
            detail=f"timed out after {timeout}s",
            started_at=started, duration_s=time.time() - started,
        )
    return RunOutcome(
        ran=True,
        return_code=proc.returncode,
        output=stdout + stderr,
        started_at=started,
        duration_s=time.time() - started,
    )


def _marker_value(output: str, name: str) -> float | None:
    """The last `##autohelix[name=value]` in the output, as a float.

    The last one wins so a warm-up print cannot shadow the real result.
    """
    import re

    matches = re.findall(rf"##autohelix\[{re.escape(name)}=([^\]]+)\]", output)
    for raw in reversed(matches):
        try:
            return float(raw.strip())
        except ValueError:
            continue
    return None


# --------------------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------------------


def format_report(results: list[CheckResult], run: RunOutcome) -> str:
    """The table a reviewer and a log reader both work from."""
    lines = ["", "NKI bootstrap constraint", ""]
    for result in results:
        mark = "PASS" if result.passed else "FAIL"
        lines.append(f"  [{mark}] ({result.key}) {result.title} — {result.summary}")
        for finding in result.findings:
            for number, part in enumerate(str(finding).splitlines()):
                lines.append(f"         {part}" if number else f"         · {part}")
    failed = [r.key for r in results if not r.passed]
    lines.append("")
    if failed:
        lines.append(f"  {len(failed)} of {len(results)} checks failing: {', '.join(failed)}")
    else:
        lines.append(f"  all {len(results)} checks pass")
    if run.ran:
        lines.append(f"  inference.py ran for {run.duration_s:.1f}s, exit {run.return_code}")
    lines.append("")
    return "\n".join(lines)


#: Where `bootstrap init` writes the manifest, relative to the module repo.
MANIFEST_REL = ".autohelix/bootstrap/manifest.json"


def find_manifest(repo: Path) -> Path:
    """The manifest for this repo, searched for rather than passed in.

    The gate's command is one fixed string shared by every repo, so it cannot name a
    per-repo path — and need not: a worktree lives at ``<repo>/.autohelix/worktrees/iter-N``,
    so walking up reaches the manifest from either.
    """
    for directory in (repo, *repo.parents):
        candidate = directory / MANIFEST_REL
        if candidate.is_file():
            return candidate
    raise CheckerError(
        f"no {MANIFEST_REL} at or above {repo}; this repo was not created by "
        f"`autohelix bootstrap init`"
    )


def load_manifest(path: Path) -> dict[str, Any]:
    """The record `autohelix bootstrap init` wrote of what it put in the repo."""
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise CheckerError(
            f"no manifest at {path}; this repo was not created by `autohelix bootstrap init`"
        ) from exc
    except json.JSONDecodeError as exc:
        raise CheckerError(f"manifest at {path} is not valid JSON: {exc}") from exc


def evaluate(repo: Path, manifest: dict[str, Any], timeout: int) -> tuple[list[CheckResult], RunOutcome]:
    """All six checks, in order, against one repo.

    The static checks run first and the candidate is executed only if the pair at least
    parses — there is nothing to learn from running a file that cannot be imported, and a
    device run is the expensive part.
    """
    source_path = repo / SOURCE_FILE
    inference_path = repo / INFERENCE_FILE
    source_text = source_path.read_text() if source_path.is_file() else ""
    source = _parse(source_path)
    inference = _parse(inference_path)

    allowed_bins = {e["file"] for e in (manifest.get("tensors") or [])}

    static = [
        check_kernel_formalization(source, inference),
        check_self_containment(source, inference, allowed_bins),
        check_nki_only(source, source_text),
    ]
    run = run_inference(repo, timeout)
    return (
        [static[0], static[1], static[2],
         check_measurement(inference, repo, run),
         check_pass_test(inference, run, expected_tolerance(manifest)),
         check_provenance(inference, repo, manifest)],
        run,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Gate a module repo against the six Trainium NKI bootstrap checks.")
    parser.add_argument("--repo", default=".", help="The module repo to check")
    parser.add_argument("--manifest", default=None,
                        help=f"The manifest init wrote (default: {MANIFEST_REL} at or above --repo)")
    parser.add_argument("--json", dest="json_out", default=None,
                        help="Write the machine-readable verdict here")
    parser.add_argument("--timeout", type=int, default=DEFAULT_RUN_TIMEOUT,
                        help=f"Seconds inference.py gets (default {DEFAULT_RUN_TIMEOUT})")
    args = parser.parse_args(argv)

    repo = Path(args.repo).resolve()
    try:
        path = Path(args.manifest) if args.manifest else find_manifest(repo)
        manifest = load_manifest(path)
        results, run = evaluate(repo, manifest, args.timeout)
    except CheckerError as exc:
        # Unusable input is not a verdict. Report every check as failing with the one
        # reason, so the report still names what has to happen next.
        results = [
            CheckResult(key=key, title=title, passed=False, summary="not evaluated",
                        findings=[str(exc)])
            for key, title in CHECK_TITLES.items()
        ]
        run = RunOutcome(ran=False, detail=str(exc))

    report = format_report(results, run)
    print(report)

    if args.json_out:
        payload = {
            "repo": str(repo),
            "passed": all(r.passed for r in results),
            "checks": [r.to_dict() for r in results],
            "run": {
                "ran": run.ran,
                "return_code": run.return_code,
                "duration_s": round(run.duration_s, 3),
                "detail": run.detail,
            },
            "report": report,
        }
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))

    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
