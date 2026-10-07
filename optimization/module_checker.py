# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The whole-module gate: the only place this pipeline checks semantics.

Everything upstream is advisory by design. The agent chooses how to cut the module; the submodule
gate asks only module-agnostic questions; ten iterations of optimization are measured against a
golden the agent itself derived. That is deliberate — a script that knew how to shard an MoE would
work for MoE and nothing else — and it is safe only because of what happens here: the ranks
are reassembled, and the result has to be the bootstrapped module's recorded output, at the
bootstrapped module's own bar, faster than the bootstrapped module, and within 10% of the
submodule it was built from.

A cut that was wrong cannot pass. A submodule that was fast because it did a quarter of the work
and dropped the rest cannot pass. A collective that quietly reduced over one rank cannot pass. The
looseness upstream is bounded by this gate, which is why it can be loose.

    a  frozen validator  `inference.py` is the one stage 4 wrote, and it drives `source.py`
    b  self-contained    neither file reaches outside its allowlist or this repository
    c  nki collectives   the reduction is `nki.collectives`, not `torch.distributed` or XLA
    d  every rank        `torchrun --nproc_per_node=<the manifest's count>`, every rank exits 0
    e  matches           the post-collective output equals the module's recorded output
    f  measured          a fresh collective profile covering every rank, and the latency comes from it
    g  provenance        the input, weights and golden are the recorded bytes
    h  faster            quicker than the bootstrapped single-core module
    i  overhead bounded  no slower than 1.1x the submodule it is made of

(c) exists because `torch.distributed.all_reduce` would work and would measure the wrong thing:
the point of the exercise is a kernel whose collective is inside the traced graph, where the
compiler can overlap it. Verified feasible before this gate was written — a 4-rank
`nki.collectives.all_reduce` and `all_to_all` both compile and run on this toolchain at LNC=2, and
`neuron-explorer capture --collectives-profile-id all` reports the collective separately as
`cc_op_time`.

(h) and (i) are constraints, not metrics, and they are checked here rather than left to
`acceptance.metric_gates` because they compare against numbers measured *outside* this run. A
metric gate can only compare an iteration against the best iteration of the same loop.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from bootstrap.nki_checker import CheckResult
from optimization import candidate
from optimization.candidate import (
    CEILING_NAME,
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
    "a": "frozen validator",
    "b": "self-containment",
    "c": "nki collectives",
    "d": "every rank",
    "e": "matches the module",
    "f": "measurement",
    "g": "data provenance",
    "h": "faster than the bootstrap",
    "i": "collective overhead bounded",
}

MANIFEST_REL = ".autohelix/optimization/module.json"

#: The rank count when the manifest does not say. The logical NeuronCores of one trn2 device, which
#: is what `PROJECTION_TARGET_UNITS` projects onto — but only a default: the projection deliberately
#: keeps a placement already narrower than the target, and `floorplan.target_units` is configurable,
#: so a valid assembly can be 1 or 2 ranks wide. Fixed at four, the gate demanded four rank markers
#: from a two-rank assembly its own prompt had asked for, which no candidate could satisfy.
DEFAULT_RANKS = 4


def rank_count(manifest: dict[str, Any]) -> int:
    """How many ranks this assembly runs, from what materialization recorded."""
    try:
        explicit = int(manifest.get("ranks"))
    except (TypeError, ValueError):
        explicit = 0
    return explicit if explicit >= 1 else (candidate.projected_units(manifest) or DEFAULT_RANKS)


def launch(ranks: int) -> list[str]:
    """How the validator is launched. `torchrun` for the correctness run — `torch.distributed` only
    bootstraps the process group, and every reduction happens in `nki.collectives` on device."""
    return ["torchrun", "--nproc_per_node", str(ranks), INFERENCE_FILE]

#: Per-rank latency markers the validator must print, on top of the fastest-rank `latency_ms`.
#: Required rather than optional: without them "the fastest rank" is the candidate's unverifiable
#: claim, and load imbalance — the thing that makes a fastest-rank number optimistic — is invisible.
RANK_LATENCY_MARKER = "latency_rank_{rank}_ms"

#: Where the collective must come from. Any other source works and measures a different machine
#: than the one the floorplan is about, so the check resolves each of these operation names through
#: the file's import table and requires the module it came from to be this one — which catches a
#: host-side or XLA collective under any alias, rather than only the spellings someone listed.
COLLECTIVE_MODULE = "nki.collectives"
COLLECTIVE_OPS = ("all_reduce", "all_gather", "all_to_all", "reduce_scatter", "collective_permute")

#: `torch.distributed` calls that are legitimate: they organize processes, they do not reduce data.
ALLOWED_DIST_CALLS = frozenset({
    "init_process_group", "destroy_process_group", "barrier", "get_rank", "get_world_size",
    "is_initialized", "new_group",
})

#: The accuracy statistic every validator in this pipeline prints and the gate already checks.
ACCURACY_MARKER = "max_abs_err"

SOURCE_ALLOWED_IMPORTS = frozenset({
    "nki", "neuronxcc", "torch", "torch_neuronx", "torch_xla", "numpy",
})
INFERENCE_ALLOWED_IMPORTS = SOURCE_ALLOWED_IMPORTS | {"source"}

#: Longer than the submodule's: every rank compiles its own graph on a cold cache, and then a
#: separate `neuron-explorer capture` pass runs over the result.
DEFAULT_RUN_TIMEOUT = 4800

#: How much of the submodule's latency the collective may add. The bound that makes a lazy cut fail.
OVERHEAD_ALLOWANCE = 1.10


def find_manifest(repo: Path) -> Path:
    # `directory`, not `candidate`: this module imports `candidate` and the loop variable shadowed
    # it, so a reader inside this function sees the wrong meaning for the name.
    for directory in [repo, *repo.parents]:
        path = directory / MANIFEST_REL
        if path.is_file():
            return path
    raise CheckerError(
        f"no {MANIFEST_REL} at or above {repo} — was this repo made by `autohelix optimize assemble`?"
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
    """The bootstrapped module's own five constants, unchanged.

    Not re-derived and not loosened. The golden is the same `tensors/reference.bin` the bootstrap
    loop matched, so the bar that admitted that kernel is the bar that admits this one. The 4-rank
    reduction order differs from the reference's single `dist.all_reduce`, so accumulation order
    shifts slightly — and absorbing that is what a bar with a pass fraction and a cosine is for.
    """
    recorded = manifest.get("tolerance") or {}
    missing = [n for n in (*TOLERANCE_NAMES, CEILING_NAME) if n not in recorded]
    if missing:
        raise CheckerError(f"the manifest records no {', '.join(missing)}")
    return {n: float(recorded[n]) for n in (*TOLERANCE_NAMES, CEILING_NAME)}


# --------------------------------------------------------------------------------------
# the checks
# --------------------------------------------------------------------------------------


def check_frozen_validator(repo: Path, manifest: dict[str, Any],
                           ranks: int = DEFAULT_RANKS) -> CheckResult:
    """(a) `inference.py` is what stage 4 froze, it drives `source.py`, and it takes the gate's
    core count rather than choosing its own.

    Scope enforcement already reverts an edit to `inference.py`, but that depends on git noticing.
    This verifies the hash recorded at the freeze — the same belt-and-braces reasoning as the
    floorplan gate's frozen-platform check, and for the same reason: the validator is the only
    thing standing between a fast kernel and a wrong one.
    """
    findings: list[str] = []
    recorded = (manifest.get("frozen") or {}).get(INFERENCE_FILE)
    path = repo / INFERENCE_FILE
    if not path.is_file():
        return CheckResult("a", CHECK_TITLES["a"], False, "missing",
                           [f"{INFERENCE_FILE} is gone"])
    if recorded:
        actual = _sha256(path)
        if actual != recorded:
            findings.append(
                f"{INFERENCE_FILE} has changed since stage 4 froze it "
                f"(recorded {recorded[:12]}, found {actual[:12]}). It is the validator, not the "
                f"candidate — an edit here changes what passing means"
            )
    else:
        findings.append("the manifest records no hash for the frozen validator")

    entry = str(manifest.get("entry_point") or "kernel")
    source = candidate._parse(repo / SOURCE_FILE)
    # Any top-level binding, not only a `def` — the same rule the submodule gate applies, and for
    # the same reason: `kernel = _impl[CORES]` binds the NKI launch grid at module scope, which is
    # how a kernel reaches both physical cores of an LNC=2 pair while the frozen validator still
    # calls `kernel(*args)` with no subscript. A grid cannot be expressed as a `FunctionDef` at
    # all, so requiring one rejected the only shape that works. The two gates disagreeing cost a
    # four-rank assembly that had already passed the other eight checks.
    names = candidate.top_level_names(source)
    if entry not in names:
        findings.append(
            f"{SOURCE_FILE} defines no top-level '{entry}'. "
            f"Found: {', '.join(sorted(names)) or 'nothing'}"
        )
    inference = candidate._parse(path)
    if "source" not in candidate._import_roots(inference):
        findings.append(f"{INFERENCE_FILE} never imports {SOURCE_FILE}")
    elif not candidate.invokes_entry_point(inference, entry):
        # Importing `source` is not the same as running it. Once this file is frozen by hash, a
        # validator that profiles a helper of its own or compares the golden against itself would
        # let every later `source.py` pass without its output ever being checked.
        findings.append(
            f"{INFERENCE_FILE} imports {SOURCE_FILE} but never traces or calls '{entry}' — "
            f"nothing connects the frozen validator to the kernel it is supposed to measure"
        )
    # The gate sets the core count in the child's environment, and this file can overwrite it. The
    # submodule gate has always refused that; the module gate did not, so a two-rank assembly could
    # keep a hard-coded four and benchmark more of the device than its projection allows — once,
    # before being frozen by hash for the rest of the run.
    findings += candidate.core_allocation_findings(inference, INFERENCE_FILE, str(ranks))
    return CheckResult("a", CHECK_TITLES["a"], not findings,
                       f"frozen, drives '{entry}'" if not findings
                       else f"{len(findings)} problem(s)", findings)


def check_self_contained(repo: Path) -> CheckResult:
    """(b) Neither file reaches outside its allowlist or outside this repository."""
    findings: list[str] = []
    for filename, allowed in ((SOURCE_FILE, SOURCE_ALLOWED_IMPORTS),
                              (INFERENCE_FILE, INFERENCE_ALLOWED_IMPORTS)):
        tree = candidate._parse(repo / filename)
        findings += candidate.import_findings(tree, allowed, filename)
        findings += candidate.path_findings(tree, filename)
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


def check_nki_collectives(repo: Path, ranks: int = DEFAULT_RANKS) -> CheckResult:
    """(c) The reduction is `nki.collectives`, and no host-side collective stands in for it.

    Every call is resolved through the file's import table before it is judged. A textual match is
    defeated by a single rename — `import torch.distributed as ncc; ncc.all_reduce(x)` looks like
    the NKI collective and is the host-side one — and this check is the whole reason the measured
    number describes the machine the floorplan is about.
    """
    findings: list[str] = []
    source = candidate._parse(repo / SOURCE_FILE)
    inference = candidate._parse(repo / INFERENCE_FILE)

    # Kept apart by file. Pooled, a `nki.collectives` call in the agent-written `inference.py`
    # satisfied the "calls no collective at all" test below — but the validator runs outside the
    # traced kernel, so an assembly whose ranks never rejoin on device passed the one check that
    # exists to prove they do.
    resolved_from_nki: list[str] = []
    in_source: list[str] = []
    for where, tree in ((SOURCE_FILE, source), (INFERENCE_FILE, inference)):
        aliases = candidate.import_aliases(tree)
        for dotted, line in sorted(candidate.called_attributes(tree).items(), key=lambda kv: kv[1]):
            tail = dotted.rsplit(".", 1)[-1]
            resolved = candidate.resolve_call(dotted, aliases)

            if tail in COLLECTIVE_OPS:
                if resolved.startswith(COLLECTIVE_MODULE):
                    resolved_from_nki.append(f"{resolved} ({where})")
                    if where == SOURCE_FILE:
                        in_source.append(resolved)
                else:
                    findings.append(
                        f"{where}:{line} calls '{dotted}'"
                        + (f", which resolves to '{resolved}'" if resolved != dotted else "")
                        + f". The reduction has to come from {COLLECTIVE_MODULE} and happen inside "
                        f"the traced graph — a host-side or XLA collective works and measures a "
                        f"different machine"
                    )
                continue

            if resolved.startswith("torch.distributed.") and tail not in ALLOWED_DIST_CALLS:
                findings.append(
                    f"{where}:{line} calls '{dotted}'"
                    + (f" ('{resolved}')" if resolved != dotted else "")
                    + f". torch.distributed may organize the {ranks} processes "
                    f"({', '.join(sorted(ALLOWED_DIST_CALLS))}) but may not move tensor data"
                )

    # A one-rank assembly has nothing to reduce. `project()` deliberately preserves a placement the
    # floorplan already fits on one unit, so this is a legal projection rather than a degenerate
    # one — and requiring a collective of it would fail every natural kernel for it. The host-side
    # ban above still applies: one rank must not reach for `torch.distributed` either.
    if ranks > 1 and not in_source and not findings:
        findings.append(
            f"{SOURCE_FILE} calls no collective at all. {ranks} ranks each holding part of the "
            f"module cannot produce the whole module's output without one"
            + (f". {INFERENCE_FILE} calls {', '.join(sorted(resolved_from_nki))}, but the validator "
               f"runs outside the traced kernel, so a collective there does not rejoin the ranks "
               f"on device" if resolved_from_nki else "")
        )
    return CheckResult("c", CHECK_TITLES["c"], not findings,
                       f"reduction via {', '.join(sorted(resolved_from_nki))}" if not findings
                       else f"{len(findings)} problem(s)", findings)


def check_all_ranks(run: RunOutcome, ranks: int = DEFAULT_RANKS) -> CheckResult:
    """(d) Every rank ran and every rank exited 0."""
    findings: list[str] = []
    if not run.ran:
        findings.append(f"the validator did not start: {run.detail}")
    else:
        if run.detail:
            findings.append(run.detail)
        if run.return_code != 0:
            tail = "\n".join(run.output.strip().splitlines()[-15:])
            findings.append(
                f"torchrun exited {run.return_code}. With {ranks} ranks this is usually one rank "
                f"failing and taking the others down\nlast output:\n{tail}"
            )
    seen = [r for r in range(ranks)
            if candidate.marker_value(run.output, RANK_LATENCY_MARKER.format(rank=r)) is not None]
    if len(seen) != ranks:
        absent = sorted(set(range(ranks)) - set(seen))
        findings.append(
            f"no ##autohelix[{RANK_LATENCY_MARKER.format(rank='N')}=...] for rank(s) {absent}. "
            f"Every rank has to report, or 'the fastest rank' is an unverifiable claim"
        )
    return CheckResult("d", CHECK_TITLES["d"], not findings,
                       f"{ranks} ranks, all exit 0" if not findings
                       else f"{len(findings)} problem(s)", findings)


def check_matches(run: RunOutcome, bar: dict[str, float], repo: Path) -> CheckResult:
    """(e) The reassembled output matches the module's recorded output at the pinned bar."""
    findings: list[str] = []
    findings += candidate.pinned_constants(candidate._parse(repo / INFERENCE_FILE), bar, INFERENCE_FILE)

    passed = candidate.marker_value(run.output, PASSED_MARKER)
    if passed is None:
        findings.append(f"the run printed no ##autohelix[{PASSED_MARKER}=...] line")
    elif passed != 1:
        findings.append(
            f"the run reported {PASSED_MARKER}={passed:g}: the reassembled output does not match "
            f"the module's recorded output"
        )
    worst = candidate.marker_value(run.output, MAX_ABS_ERR_MARKER)
    ceiling = bar[CEILING_NAME]
    if worst is None:
        findings.append(
            f"the run printed no ##autohelix[{MAX_ABS_ERR_MARKER}=...] line, so the "
            f"{CEILING_NAME} ceiling could not be checked"
        )
    elif worst > ceiling:
        findings.append(
            f"worst element off by {worst:g}, over the {CEILING_NAME} ceiling of {ceiling:g}"
        )
    return CheckResult("e", CHECK_TITLES["e"], not findings,
                       f"matches at the module's bar (worst {worst:g})" if not findings and worst
                       else ("matches" if not findings else f"{len(findings)} problem(s)"),
                       findings)


def check_measurement(run: RunOutcome, ranks: int = DEFAULT_RANKS) -> CheckResult:
    """(f) A fresh 4-rank collective profile, and the reported latency is the fastest rank of it."""
    findings: list[str] = []
    if not run.artifacts.get("neff"):
        findings.append("the run left no fresh .neff behind")
    ntffs = run.artifacts.get("ntff") or []
    per_rank = [n for n in ntffs if "_rank_" in Path(n).name]
    if len(per_rank) < ranks:
        findings.append(
            f"the run left {len(per_rank)} fresh per-rank .ntff file(s), expected {ranks} from "
            f"`neuron-explorer capture --collectives-worker-count {ranks} "
            f"--collectives-profile-id all`"
        )

    reported = candidate.marker_value(run.output, LATENCY_MARKER)
    if reported is None:
        findings.append(f"the run printed no ##autohelix[{LATENCY_MARKER}=...] line")
        return CheckResult("f", CHECK_TITLES["f"], False, f"{len(findings)} problem(s)", findings)
    if reported <= 0:
        findings.append(f"the reported latency is {reported:g} ms, which is not a measurement")

    rank_latencies = {
        r: candidate.marker_value(run.output, RANK_LATENCY_MARKER.format(rank=r))
        for r in range(ranks)
    }
    known = {r: v for r, v in rank_latencies.items() if v is not None}
    if known:
        fastest = min(known.values())
        if abs(reported - fastest) > max(1e-6, 0.001 * fastest):
            findings.append(
                f"{LATENCY_MARKER} is {reported:g} ms but the fastest rank reported "
                f"{fastest:g} ms (rank {min(known, key=lambda r: known[r])}). The metric is the "
                f"fastest rank, and it has to be one of the numbers the ranks actually printed"
            )
    summary = f"{reported:g} ms (fastest of {len(known)} ranks)"
    if len(known) == ranks:
        spread = max(known.values()) - min(known.values())
        summary += f", spread {spread:g} ms"
    return CheckResult("f", CHECK_TITLES["f"], not findings,
                       summary if not findings else f"{len(findings)} problem(s)", findings)


def check_provenance(repo: Path, manifest: dict[str, Any]) -> CheckResult:
    """(g) The input, weights and golden are the bootstrapped module's recorded bytes."""
    findings: list[str] = []
    inference = candidate._parse(repo / INFERENCE_FILE)
    findings += candidate.fabrication_findings(inference, INFERENCE_FILE)

    recorded = manifest.get("tensors") or {}
    if not recorded:
        findings.append("the manifest lists no tensors, so provenance cannot be established")
    checked = 0
    for name, entry in sorted(recorded.items()):
        rel = str(entry.get("file") or "")
        path = repo / rel
        if not path.is_file():
            findings.append(f"{rel} is missing from the repo")
            continue
        want = entry.get("sha256")
        if want:
            checked += 1
            if _sha256(path) != want:
                findings.append(f"{rel} has been edited since the repo was built")
    return CheckResult("g", CHECK_TITLES["g"], not findings,
                       f"{checked} tensor(s) match the record" if not findings
                       else f"{len(findings)} problem(s)", findings)


def check_faster(run: RunOutcome, manifest: dict[str, Any]) -> CheckResult:
    """(h) Faster than the bootstrapped single-core module."""
    baseline = (manifest.get("baselines") or {}).get("bootstrap_latency_ms")
    reported = candidate.marker_value(run.output, LATENCY_MARKER)
    if baseline is None:
        return CheckResult("h", CHECK_TITLES["h"], False, "no baseline recorded",
                           ["the manifest records no bootstrap_latency_ms to beat"])
    if reported is None:
        return CheckResult("h", CHECK_TITLES["h"], False, "no latency reported",
                           [f"no ##autohelix[{LATENCY_MARKER}=...] line to compare"])
    baseline = float(baseline)
    if reported >= baseline:
        speedup = baseline / reported if reported else 0.0
        return CheckResult(
            "h", CHECK_TITLES["h"], False, f"{reported:g} ms vs {baseline:g} ms",
            [f"{reported:g} ms is not faster than the bootstrapped module's {baseline:g} ms "
             f"({speedup:.2f}x). Four ranks that do not beat one core have spent their "
             f"parallelism on overhead"],
        )
    return CheckResult("h", CHECK_TITLES["h"], True,
                       f"{reported:g} ms vs {baseline:g} ms ({baseline / reported:.2f}x)")


def check_overhead(run: RunOutcome, manifest: dict[str, Any]) -> CheckResult:
    """(i) No slower than 1.1x the submodule it is made of.

    The bound that makes a lazy cut fail. A submodule that was fast because it quietly did a
    quarter of the work leaves the assembly with three-quarters still to do, and no collective is
    cheap enough to hide that inside 10%.
    """
    submodule = (manifest.get("baselines") or {}).get("submodule_latency_ms")
    reported = candidate.marker_value(run.output, LATENCY_MARKER)
    if submodule is None:
        return CheckResult("i", CHECK_TITLES["i"], False, "no submodule latency recorded",
                           ["the manifest records no submodule_latency_ms"])
    if reported is None:
        return CheckResult("i", CHECK_TITLES["i"], False, "no latency reported",
                           [f"no ##autohelix[{LATENCY_MARKER}=...] line to compare"])
    submodule = float(submodule)
    ceiling = submodule * OVERHEAD_ALLOWANCE
    if reported > ceiling:
        excess = (reported / submodule - 1.0) * 100 if submodule else float("inf")
        return CheckResult(
            "i", CHECK_TITLES["i"], False, f"{reported:g} ms vs {ceiling:g} ms allowed",
            [f"{reported:g} ms is {excess:.1f}% over the submodule's {submodule:g} ms, past the "
             f"{(OVERHEAD_ALLOWANCE - 1) * 100:.0f}% the collective is allowed"],
        )
    overhead = (reported / submodule - 1.0) * 100 if submodule else 0.0
    return CheckResult("i", CHECK_TITLES["i"], True,
                       f"{overhead:+.1f}% over the submodule's {submodule:g} ms")


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------------------
# driving
# --------------------------------------------------------------------------------------


#: Checks about whether the *assembly* was worth optimizing rather than whether this candidate is
#: sound. Stage 4 asks them once; `--loop` skips them. See `evaluate`.
ADMISSION_ONLY = ("i",)


def evaluate(repo: Path, manifest: dict[str, Any], timeout: int,
             loop: bool = False) -> tuple[list[CheckResult], RunOutcome]:
    """One `torchrun` of the frozen validator, then the checks, off that single execution.

    `loop` drops `ADMISSION_ONLY`, which is what every stage-5 iteration wants. Check (i) bounds
    the collective's overhead against the submodule's latency, and that is a question about the
    assembly: stage 4 asks it once, to decide whether a loop is worth running at all. Asked again
    every iteration it stops being a bound and becomes a lottery.

    `mtp.0.attention` is the case. Its ceiling was 0.48969 ms, from one draw of the submodule;
    twelve later draws of the accepted module kernel — byte-identical, nothing changed — spanned
    0.4924 to 0.4998 ms. The whole distribution of the code that set the bound sits above the
    bound, so three of five iterations were rejected for measurement noise, having each spent an
    agent-hour and a device slot. One of them had reverted to the accepted commit exactly.

    What still holds every iteration is everything about the candidate: the validator is the frozen
    one, the reduction is a real collective, every rank ran, the output matches the module at its
    bar, the profile is fresh, the tensors are the recorded bytes, and it beats the bootstrap. A
    regression past that is the metric gate's job, and the metric gate is relative to the best
    accepted iteration rather than to a number measured once before the loop began.
    """
    bar = expected_tolerance(manifest)
    ranks = rank_count(manifest)
    run = candidate.run_candidate(
        repo, launch(ranks), timeout=timeout,
        env_overrides={"NEURON_RT_NUM_CORES": str(ranks)},
    )
    results = [
        check_frozen_validator(repo, manifest, ranks),
        check_self_contained(repo),
        check_nki_collectives(repo, ranks),
        check_all_ranks(run, ranks),
        check_matches(run, bar, repo),
        check_measurement(run, ranks),
        check_provenance(repo, manifest),
        check_faster(run, manifest),
        check_overhead(run, manifest),
    ]
    if loop:
        # `.key`, not `.check`: only `CheckResult.to_dict` renames the field to "check", and
        # reading that name off the dataclass is an AttributeError that failed three paid
        # iterations of 42-DSparkMarkovHead before anything measured them.
        results = [r for r in results if r.key not in ADMISSION_ONLY]
    return results, run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default=".", help="the candidate repository")
    parser.add_argument("--json", default=None, help="where to write the machine-readable verdict")
    parser.add_argument("--timeout", type=int, default=DEFAULT_RUN_TIMEOUT)
    parser.add_argument("--loop", action="store_true",
                        help="per-iteration subset: skip the admission-only checks "
                             f"({', '.join(ADMISSION_ONLY)}), which stage 4 already asked")
    args = parser.parse_args(argv)

    repo = Path(args.repo).resolve()
    try:
        manifest = load_manifest(find_manifest(repo))
        results, run = evaluate(repo, manifest, args.timeout, loop=args.loop)
    except CheckerError as exc:
        report = f"\nwhole-module gate\n\n  [FAIL] the repo is unusable — {exc}\n"
        print(report)
        if args.json:
            candidate.write_verdict([], None, "whole-module gate", Path(args.json),
                               extra={"passed": False, "error": str(exc), "report": report})
        return 2

    extra: dict[str, Any] = {}
    latency = candidate.marker_value(run.output, LATENCY_MARKER)
    if latency is not None:
        extra["latency_ms"] = latency
        extra["rank_latency_ms"] = {
            str(r): candidate.marker_value(run.output, RANK_LATENCY_MARKER.format(rank=r))
            for r in range(rank_count(manifest))
        }
    # The accuracy the validator reported, published beside the latency for the same reason: the
    # verdict is the only thing that outlives the worktree, and `optimization.readback` turns it
    # into a recorded metric the loop can gate on.
    worst = candidate.marker_value(run.output, ACCURACY_MARKER)
    if worst is not None:
        extra[ACCURACY_MARKER] = worst
    verdict = candidate.write_verdict(
        results, run, "whole-module gate", Path(args.json) if args.json else None, extra=extra,
    )
    print(verdict.report)
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
