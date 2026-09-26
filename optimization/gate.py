# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the two optimization gates share: verdicts, static analysis, and running a candidate.

`bootstrap/nki_checker.py` already worked out how to read a candidate repo without trusting it —
which string literals are paths, which constructors fabricate tensors, how to kill a hung device
run's whole process group. That work is imported rather than copied: one definition of
"path-like literal" is worth more than an independent one, and a divergence between the two would
show up as a gate that passes something its sibling rejects.

The private names are imported deliberately. They are private to callers *outside* this
distribution; inside it, `optimization/` is a consumer of `bootstrap/` by construction — it starts
from a bootstrapped repo — and `tests/test_optimization_gate.py` fails if any of them disappears,
which turns a refactor over there into a failing test here rather than a silent behaviour change.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bootstrap.nki_checker import (  # noqa: F401 — re-exported for the two checkers
    DYNAMIC_IMPORT_NAMES,
    FILE_IO_NAMES,
    FORBIDDEN_PATH_MARKERS,
    SYNTHETIC_CONSTRUCTORS,
    SYNTHETIC_INPLACE,
    CheckerError,
    CheckResult,
    _decorator_names,
    _import_roots,
    _module_level_numbers,
    _parse,
    _path_like,
    _string_literals,
)

#: The one file the optimizing agent may edit, in both loop stages. `inference.py` is written by
#: a preparation agent, gated once, and then frozen — which is what lets the per-iteration
#: constraint schedule be about the kernel rather than about the validator.
SOURCE_FILE = "source.py"
INFERENCE_FILE = "inference.py"

#: The submodule's declaration of its own cut. Written by the stage-2 agent, read by stage 4.
DECLARATION_FILE = "submodule.json"

#: Markers a validator must print. `latency_ms` is the metric the loop optimizes, so unlike
#: bootstrap — where it was informational — a missing or stale one fails the gate.
LATENCY_MARKER = "latency_ms"
PASSED_MARKER = "passed"
MAX_ABS_ERR_MARKER = "max_abs_err"

#: The five constants a validator pins. Same names as bootstrap's, because the bar means the same
#: thing; the *values* differ per repo and come from the manifest.
TOLERANCE_NAMES = ("RTOL", "ATOL", "MIN_COSINE", "MIN_PASS_FRACTION")
CEILING_NAME = "MAX_ABS_ERR"

#: How long a candidate run gets. The MoE validator took 219s at 384 experts on one core; four
#: ranks under torchrun, compiling on first run, is the case this has to leave room for.
DEFAULT_RUN_TIMEOUT = 2400


@dataclass
class Verdict:
    """What a gate said about one candidate."""

    passed: bool
    results: list[CheckResult] = field(default_factory=list)
    report: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def passing(self) -> list[str]:
        return [r.key for r in self.results if r.passed]

    @property
    def failing(self) -> list[str]:
        return [r.key for r in self.results if not r.passed]

    def summary(self) -> str:
        total = len(self.results)
        if self.passed:
            return f"gate: all {total} checks pass"
        return f"gate: {len(self.passing)}/{total} passing, failing {', '.join(self.failing)}"


@dataclass
class RunOutcome:
    """What happened when a candidate's validator was executed."""

    ran: bool
    return_code: int = -1
    output: str = ""
    detail: str = ""
    duration_s: float = 0.0
    #: Files the run left behind, keyed by kind, for the freshness check.
    artifacts: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ran": self.ran,
            "return_code": self.return_code,
            "duration_s": round(self.duration_s, 3),
            "detail": self.detail,
            "artifacts": self.artifacts,
        }


def _kill_process_group(proc: subprocess.Popen) -> None:
    """SIGKILL the whole session, so a killed run does not leave the device held.

    Tracing spawns a compiler and a profiler, and under `torchrun` there are four more processes
    below that. Killing only the direct child leaves them holding the NeuronCores, and the next
    iteration then fails for a reason that has nothing to do with its own work.
    """
    import signal

    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()


def run_candidate(
    repo: Path,
    command: list[str],
    timeout: int = DEFAULT_RUN_TIMEOUT,
    env_overrides: dict[str, str] | None = None,
) -> RunOutcome:
    """Run the candidate's validator once, and keep everything it said.

    Once, not twice: correctness and latency are two properties of one execution, and measuring
    them separately lets them disagree about which code ran.

    ``env_overrides`` is how `NEURON_RT_NUM_CORES` is pinned — 1 for a submodule, 4 for an
    assembly. Set here rather than left to the validator so the gate, not the candidate, decides
    how much of the device is in play.
    """
    started = time.time()
    env = dict(os.environ)
    env.update(env_overrides or {})
    before = _artifact_mtimes(repo)

    try:
        proc = subprocess.Popen(
            command, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            start_new_session=True, env=env,
        )
    except OSError as exc:
        return RunOutcome(ran=False, detail=str(exc))

    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_process_group(proc)
        stdout, stderr = proc.communicate()
        return RunOutcome(
            ran=True, return_code=-1, output=(stdout or "") + (stderr or ""),
            detail=f"timed out after {timeout}s", duration_s=time.time() - started,
        )

    outcome = RunOutcome(
        ran=True, return_code=proc.returncode, output=stdout + stderr,
        duration_s=time.time() - started,
    )
    outcome.artifacts = _fresh_artifacts(repo, before, started)
    return outcome


#: Profile artifacts a run must leave behind, by extension. A `.ntff` per rank for a collective
#: run, named `*_rank_N.ntff` by `neuron-explorer capture --collectives-profile-id all`.
ARTIFACT_SUFFIXES = (".neff", ".ntff")


def _artifact_mtimes(repo: Path) -> dict[str, float]:
    """Modification times of every profile artifact already in the repo, before the run.

    The reason freshness is checkable at all: a candidate that deletes nothing and produces
    nothing would otherwise pass on the previous iteration's profile.
    """
    found: dict[str, float] = {}
    for suffix in ARTIFACT_SUFFIXES:
        for path in repo.rglob(f"*{suffix}"):
            try:
                found[str(path.relative_to(repo))] = path.stat().st_mtime
            except (OSError, ValueError):
                continue
    return found


def _fresh_artifacts(repo: Path, before: dict[str, float], started: float) -> dict[str, list[str]]:
    """Artifacts this run produced, as opposed to ones it inherited."""
    fresh: dict[str, list[str]] = {s.lstrip("."): [] for s in ARTIFACT_SUFFIXES}
    for suffix in ARTIFACT_SUFFIXES:
        for path in sorted(repo.rglob(f"*{suffix}")):
            try:
                rel, mtime = str(path.relative_to(repo)), path.stat().st_mtime
            except (OSError, ValueError):
                continue
            # Strictly after the run started, and changed if it existed before. `started` alone
            # is not enough on a filesystem with coarse timestamps.
            if mtime >= started - 1.0 and mtime != before.get(rel):
                fresh[suffix.lstrip(".")].append(rel)
    return fresh


_MARKER = r"##autohelix\[{name}=([^\]]+)\]"


def marker_values(output: str, name: str) -> list[float]:
    """Every `##autohelix[name=value]` in the output, in order, as floats."""
    out: list[float] = []
    for raw in re.findall(_MARKER.format(name=re.escape(name)), output):
        try:
            out.append(float(raw.strip()))
        except ValueError:
            continue
    return out


def marker_value(output: str, name: str) -> float | None:
    """The last `##autohelix[name=value]`, so a warm-up print cannot shadow the real result."""
    values = marker_values(output, name)
    return values[-1] if values else None


def format_report(title: str, results: list[CheckResult], run: RunOutcome | None = None) -> str:
    """The table a reviewer and a log reader both work from."""
    lines = ["", title, ""]
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
    if run is not None and run.ran:
        lines.append(f"  the validator ran for {run.duration_s:.1f}s, exit {run.return_code}")
    lines.append("")
    return "\n".join(lines)


def write_verdict(results: list[CheckResult], run: RunOutcome | None, title: str,
                  path: Path | None, extra: dict[str, Any] | None = None) -> Verdict:
    """Assemble a verdict and, when asked, write the machine-readable copy the reviewer reads."""
    payload: dict[str, Any] = {
        "passed": all(r.passed for r in results),
        "checks": [r.to_dict() for r in results],
        "report": format_report(title, results, run),
    }
    if run is not None:
        payload["run"] = run.to_dict()
    payload.update(extra or {})
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))
    return Verdict(
        passed=bool(payload["passed"]), results=results,
        report=str(payload["report"]), payload=payload,
    )


# --------------------------------------------------------------------------------------
# static analysis shared by both gates
# --------------------------------------------------------------------------------------


def traced_entry_points(tree: ast.Module) -> list[str]:
    """Names of top-level functions carrying an `@nki.jit`-family decorator.

    Used to *describe* the kernel, never to require it: from iteration 4 onward a legal
    `source.py` may be pure torch with no `@nki.jit` anywhere, so a gate that demanded one would
    contradict the constraint schedule. What the gate requires is the entry point by name; how it
    is implemented is the schedule's business.
    """
    found: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if any(d.startswith("nki.") or d == "jit" for d in _decorator_names(node)):
                found.append(node.name)
    return found


def top_level_functions(tree: ast.Module) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    return {
        node.name: node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def called_attributes(tree: ast.Module) -> dict[str, int]:
    """Dotted names of every call in the tree, mapped to the line of the first occurrence.

    `nki.collectives.all_reduce(...)` appears as both the full dotted path and its tail, so a
    check can ask either "was a collective used" or "which module was it taken from".
    """
    found: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        parts: list[str] = []
        target: Any = node.func
        while isinstance(target, ast.Attribute):
            parts.append(target.attr)
            target = target.value
        if isinstance(target, ast.Name):
            parts.append(target.id)
        if parts:
            found.setdefault(".".join(reversed(parts)), node.lineno)
    return found


def referenced_names(tree: ast.Module) -> dict[str, int]:
    """Every bare name and attribute root referenced, mapped to its first line."""
    found: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            found.setdefault(node.id, node.lineno)
        elif isinstance(node, ast.Attribute):
            target: Any = node
            while isinstance(target, ast.Attribute):
                target = target.value
            if isinstance(target, ast.Name):
                found.setdefault(target.id, target.lineno)
    return found


def import_findings(tree: ast.Module, allowed: frozenset[str], filename: str) -> list[str]:
    """Imports outside the allowlist, plus any dynamic-import escape hatch."""
    findings: list[str] = []
    for root, line in sorted(_import_roots(tree).items()):
        if root in allowed or _is_stdlib(root):
            continue
        findings.append(f"{filename}:{line} imports '{root}', which is not allowed here")
    for name, line in sorted(referenced_names(tree).items()):
        if name in DYNAMIC_IMPORT_NAMES:
            findings.append(
                f"{filename}:{line} references '{name}', which can reach any module at runtime "
                f"and so makes the import allowlist unenforceable"
            )
    return findings


def _is_stdlib(root: str) -> bool:
    """Whether an import root is part of the standard library.

    `sys.stdlib_module_names` is exact on 3.10+ and needs no import of the module itself, which
    matters: importing a candidate's dependency to classify it would run its top level.
    """
    return root in getattr(sys, "stdlib_module_names", frozenset()) or root == "__future__"


def path_findings(tree: ast.Module, filename: str,
                  markers: tuple[str, ...] = FORBIDDEN_PATH_MARKERS) -> list[str]:
    """Path-shaped literals that reach outside the repository."""
    findings: list[str] = []
    for text, line in _string_literals(tree):
        if not _path_like(text):
            continue
        if text.startswith("/") or ".." in Path(text).parts:
            findings.append(f"{filename}:{line} names '{text}', which is outside this repository")
            continue
        for marker in markers:
            if marker in text:
                findings.append(
                    f"{filename}:{line} names '{text}', which reaches back into the artifact "
                    f"rather than using this repository's own copy"
                )
                break
    return findings


def fabrication_findings(tree: ast.Module, filename: str) -> list[str]:
    """Calls that could invent tensor data where recorded bytes are required."""
    findings: list[str] = []
    calls = called_attributes(tree)
    for dotted, line in sorted(calls.items(), key=lambda kv: kv[1]):
        tail = dotted.rsplit(".", 1)[-1]
        if tail in SYNTHETIC_CONSTRUCTORS:
            findings.append(
                f"{filename}:{line} calls '{dotted}', which fabricates tensor data. Every input, "
                f"weight and reference must come from the recorded bytes"
            )
        elif tail in SYNTHETIC_INPLACE:
            findings.append(f"{filename}:{line} calls '{dotted}', which overwrites a tensor's data")
    return findings


def pinned_constants(tree: ast.Module, expected: dict[str, float],
                     filename: str) -> list[str]:
    """Findings if the five tolerance constants are absent, computed, or altered.

    Both directions are violations. A tightened bar looks virtuous and is still a different
    experiment from the one the manifest describes, and the manifest is what the report cites.
    """
    findings: list[str] = []
    declared = _module_level_numbers(tree)
    for name, want in sorted(expected.items()):
        entry = declared.get(name)
        if entry is None:
            findings.append(
                f"{filename} does not declare {name} as a module-level number literal "
                f"(expected {want:g})"
            )
            continue
        value, line = entry
        try:
            got = float(value)
        except (TypeError, ValueError):
            findings.append(f"{filename}:{line} declares {name} as {value!r}, not a number")
            continue
        if abs(got - want) > 1e-12:
            findings.append(
                f"{filename}:{line} declares {name} = {got:g}, but this repo's bar is {want:g}. "
                f"The bar is not the agent's to change, in either direction"
            )
    return findings
