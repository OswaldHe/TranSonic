# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Running a compiled slot checker, enforcing or advisory.

A slot's constraint is enforced by being AutoHelix's constraint zero: it fails, the iteration is
rejected, its work is discarded, and — because upstream skips the metric commands when a constraint
fails — nothing is measured. That is right for most of a slot's iterations. It is wrong for the last
one.

By a slot's last iteration the agent has had every iteration that slot allows. If it still cannot
satisfy the constraint but has produced something correct and *faster*, throwing that away buys
nothing: the constraint exists to shape the search, and at the end of the slot the search is over.
So on that last iteration the checker runs **advisory** — it writes the same verdict and
always exits 0, letting the real constraints and the measurement proceed — and the loop applies a
stricter acceptance rule instead of the usual 5% slack: correct, and strictly faster than the best
so far, or it is rejected anyway.

Which makes the escape hatch narrow in the right way. Inside a slot, a violation costs the
iteration. At the end of one, a violation costs the 5% of slack every other iteration gets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

#: Exit code when the checker itself could not be run. Distinct from a violation, so a broken
#: checker is not silently read as a compliant candidate.
BROKEN = 2

#: How long a checker gets before it is killed. Generous for what it does — a static read of one
#: source file — because a false timeout rejects an iteration that complied. The compiler is told
#: this number through `constraints.CHECKER_CONTRACT` rather than being given its own copy.
CHECKER_TIMEOUT_SECONDS = 120


#: The files a checker must leave alone. `source.py` is the one that matters — it is the candidate,
#: and a constraint runs after out-of-scope changes are reverted, so an edit here survives into the
#: measurement. `inference.py` is frozen for the whole run and its hash is a gate check.
GUARDED_FILES = ("source.py", "inference.py")


def _repo_fingerprint(repo: Path) -> dict[str, str]:
    """Content hashes of the files a checker may not change."""
    seen: dict[str, str] = {}
    for name in GUARDED_FILES:
        path = repo / name
        if path.is_file():
            seen[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return seen


def _mutations(before: dict[str, str], after: dict[str, str]) -> list[str]:
    """Which guarded files a checker changed, added or removed."""
    return sorted(name for name in set(before) | set(after)
                  if before.get(name) != after.get(name))


def run_checker(checker: Path, repo: Path, report: Path,
                timeout: int = CHECKER_TIMEOUT_SECONDS) -> tuple[bool, str]:
    """Run one compiled checker and return whether it passed, plus its output.

    The checker writes the verdict itself; this only drives it and decides what the exit code
    means. A checker that crashes counts as a violation, with its traceback as the finding — the
    safe direction, since the alternative is a schedule that quietly stops constraining anything.
    """
    report.parent.mkdir(parents=True, exist_ok=True)
    if report.exists():
        report.unlink()  # a stale verdict would be read as this iteration's

    before = _repo_fingerprint(repo)
    try:
        result = subprocess.run(
            [sys.executable, str(checker), "--repo", str(repo), "--json", str(report)],
            cwd=repo, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        _write(report, False, [f"the checker did not finish within {timeout}s"])
        return False, f"timeout after {timeout}s"
    except OSError as exc:
        _write(report, False, [f"the checker could not be run: {exc}"])
        return False, str(exc)

    # The contract says a checker does not modify anything, and the compiler writes the checker, so
    # the claim needs checking rather than trusting. Constraints run *after* out-of-scope changes
    # are reverted, so a checker that edits the candidate edits the code that is then measured and
    # merged: the program judging the constraint would be able to rewrite what it is judging.
    # Compared by content, so any route to the mutation is caught, not just the ones a static read
    # can name.
    touched = _mutations(before, _repo_fingerprint(repo))
    if touched:
        _write(report, False, [
            f"the checker modified {', '.join(touched)}, which its contract forbids. A checker "
            f"reads the candidate and writes only its JSON report.",
        ])
        return False, f"checker modified {', '.join(touched)}"

    output = result.stdout + result.stderr
    if not report.is_file():
        tail = "\n".join(output.strip().splitlines()[-10:]) or "(no output)"
        _write(report, result.returncode == 0,
               [] if result.returncode == 0 else
               [f"the checker wrote no verdict and exited {result.returncode}", tail])
        return result.returncode == 0, output

    try:
        payload = json.loads(report.read_text())
    except json.JSONDecodeError as exc:
        _write(report, False, [f"the checker's verdict is not valid JSON: {exc}"])
        return False, output
    return bool(payload.get("passed")) and result.returncode == 0, output


def _write(report: Path, passed: bool, findings: list[str]) -> None:
    report.write_text(json.dumps({"passed": passed, "findings": findings}, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checker", required=True, help="the compiled checker for this slot")
    parser.add_argument("--repo", default=".", help="the candidate repository")
    parser.add_argument("--json", required=True, dest="report", help="where the verdict goes")
    parser.add_argument("--timeout", type=int, default=CHECKER_TIMEOUT_SECONDS)
    parser.add_argument(
        "--advisory", action="store_true",
        help="always exit 0, so the measurement still runs. Used on the last iteration of a "
             "slot, where the loop applies a stricter acceptance rule instead of rejecting.",
    )
    args = parser.parse_args(argv)

    checker = Path(args.checker)
    if not checker.is_file():
        print(f"no compiled checker at {checker}", file=sys.stderr)
        return 0 if args.advisory else BROKEN

    passed, output = run_checker(
        checker, Path(args.repo).resolve(), Path(args.report), timeout=args.timeout,
    )
    if output.strip():
        print(output.strip())
    if passed:
        return 0
    print(
        "the candidate does not follow this iteration's constraint"
        + (" (advisory: the measurement will still run, but acceptance now requires a strict "
           "improvement)" if args.advisory else ""),
    )
    return 0 if args.advisory else 1


if __name__ == "__main__":
    raise SystemExit(main())
