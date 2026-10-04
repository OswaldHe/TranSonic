# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Building the two module repos: everything the preparation agents are given, and nothing more.

The agents write `source.py`, `inference.py` and the declaration. This writes what they read: the
bootstrapped module, the frozen references, the projection, and — for the assembly — the optimized
submodule and the two latencies it has to hold against.

The division matters. A scaffolding step that guessed at the cut would be a script-based submodule
generator, which is exactly what this pipeline is not: how to divide a module is a judgement about
that module, and the whole design rests on the agent making it and the whole-module gate catching it
if the judgement was wrong. So nothing here computes a shard, slices a tensor or names an expert. It
copies, it records, and it states the constraints.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from optimization.projection import Projection
from optimization.strip import strip_tree

#: What a materialized repo carries, by role. `module/` and `reference/` are read-only inputs;
#: `tensors/` is the agent's to fill from them.
MODULE_DIR = "module"
REFERENCE_DIR = "reference"
SUBMODULE_DIR = "submodule"

#: The bootstrapped repo's files that are inputs here. Its `.autohelix/` is deliberately excluded:
#: the bootstrap loop's notes and reviews are about getting a kernel to work at all, and carrying
#: them in would fill the agent's context with a problem that is already solved.
BOOTSTRAP_INPUTS = ("source.py", "inference.py", "README.md", "MODULE.md", "config.json")

#: The frozen references, copied from the bootstrapped repo when it has them and from the artifact
#: otherwise. Present so the stage-2 agent can work out what one rank computes from the semantics
#: rather than inferring it from the recorded numbers.
REFERENCE_INPUTS = (
    "reference_torch.py", "reference_inference.py", "reference_numerics.py",
)
REFERENCE_TREES = ("vendor", "compat")

#: The five constants, as `bootstrap init` publishes them under "The numerical bar".
TOLERANCE_NAMES = ("RTOL", "ATOL", "MIN_COSINE", "MIN_PASS_FRACTION", "MAX_ABS_ERR")

_BAR = re.compile(r"^\s*(RTOL|ATOL|MIN_COSINE|MIN_PASS_FRACTION|MAX_ABS_ERR)\s*=\s*([0-9.eE+-]+)",
                  re.M)

#: The module README's tensor table rows, which is where the golden's dtype and shape are recorded.
_TENSOR_ROW = re.compile(
    r"^\|\s*`?(?P<name>[^`|]+?)`?\s*\|\s*(?P<role>\w+)\s*\|\s*`(?P<file>[^`]+)`\s*\|"
    r"\s*(?P<dtype>[\w]+)\s*\|\s*\((?P<shape>[^)]*)\)\s*\|",
    re.M,
)


class MaterializeError(RuntimeError):
    """The inputs a repo needs are missing or unreadable."""


@dataclass
class TensorRecord:
    """One tensor from the bootstrapped module's README table."""

    name: str
    role: str
    file: str
    dtype: str
    shape: list[int]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "role": self.role, "file": self.file,
                "dtype": self.dtype, "shape": self.shape}


def read_numerical_bar(readme: Path) -> dict[str, float]:
    """The five constants out of a bootstrapped module's README.

    Parsed from the README rather than from its `inference.py`, because the README is what the
    harness published and the validator is what an agent wrote to match it. If they ever disagreed,
    the README is the one the bootstrap gate held the validator to.
    """
    if not readme.is_file():
        raise MaterializeError(f"{readme} does not exist")
    found = {name: float(value) for name, value in _BAR.findall(readme.read_text())}
    missing = [n for n in TOLERANCE_NAMES if n not in found]
    if missing:
        raise MaterializeError(
            f"{readme} does not publish {', '.join(missing)} under 'The numerical bar' — is it a "
            f"repo `autohelix bootstrap init` produced?"
        )
    return {n: found[n] for n in TOLERANCE_NAMES}


def read_tensor_table(readme: Path) -> list[TensorRecord]:
    """Every tensor the bootstrapped module records, from its README table."""
    records: list[TensorRecord] = []
    for match in _TENSOR_ROW.finditer(readme.read_text()):
        raw_shape = match.group("shape").strip()
        try:
            shape = [int(part.strip()) for part in raw_shape.split(",") if part.strip()]
        except ValueError:
            continue
        records.append(TensorRecord(
            name=match.group("name").strip(), role=match.group("role").strip(),
            file=match.group("file").strip(), dtype=match.group("dtype").strip(), shape=shape,
        ))
    if not records:
        raise MaterializeError(f"{readme} has no parseable tensor table")
    return records


def golden_record(records: list[TensorRecord]) -> TensorRecord:
    """The module's recorded output — the tensor everything downstream is judged against."""
    for record in records:
        if record.role.lower() in {"golden", "reference", "output"}:
            return record
    raise MaterializeError(
        "the module's tensor table names no golden/reference output, so there is nothing for the "
        "assembly to be judged against"
    )


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, symlinks=False)
    else:
        shutil.copy2(src, dst)


def _link_or_copy_tensors(src: Path, dst: Path) -> str:
    """Make the bootstrapped module's tensors readable from the new repo, and not writable.

    Hard-linked, because the MoE module's `tensors/` is 6.8 GiB and both repos plus the agent's own
    slices would otherwise be 20 GiB of identical bytes.

    The links are then made **read-only**, and that part is not cosmetic. A hard link is the same
    inode, and an earlier comment here claimed a write would have to unlink first — it does not:
    `open(path, "wb")` truncates the shared inode, so a validator that opened a tensor for writing
    by mistake would destroy the recorded golden in the bootstrap repo, which is irreplaceable. This
    filesystem is ext4 with no reflink support, so copy-on-write is unavailable and the mode bits are
    the guard. Clearing write permission on the inode covers the original too, which is the right
    outcome: recorded tensors are ground truth and nothing here should be writing to them.

    The mode bits are a speed bump rather than a guarantee — the owning user can restore write
    permission — so they are backed by `fingerprint_tensors`, which records what the bytes were
    somewhere no agent can reach.
    """
    dst.mkdir(parents=True, exist_ok=True)
    linked = copied = 0
    for path in sorted(src.iterdir()):
        if not path.is_file():
            continue
        target = dst / path.name
        if not target.exists():
            try:
                target.hardlink_to(path)
                linked += 1
            except (OSError, AttributeError):
                shutil.copy2(path, target)
                copied += 1
        try:
            # 0o444 on the inode, so the shared original becomes read-only as well.
            target.chmod(0o444)
        except OSError:
            pass
    parts = [f"{linked} hard-linked" if linked else "", f"{copied} copied" if copied else ""]
    return ", ".join(p for p in parts if p) + ", read-only"


def fingerprint_tensors(directory: Path) -> dict[str, dict[str, Any]]:
    """Content hash and size of every recorded tensor: what `_link_or_copy_tensors`' mode bits cannot
    guarantee on their own.

    The gate's provenance check cannot stand in for this, because it compares against hashes the
    *agent* recorded — a corrupted tensor re-hashed afterwards passes. So the pipeline keeps this
    record in its own state directory, where no agent writes. 6.8 GiB hashes in about 7 seconds.
    """
    found: dict[str, dict[str, Any]] = {}
    if not directory.is_dir():
        return found
    for path in sorted(directory.glob("*.bin")):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(4 << 20):
                digest.update(chunk)
        found[path.name] = {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}
    return found


def verify_tensors(record: dict[str, dict[str, Any]], directory: Path) -> list[str]:
    """Which recorded tensors no longer match, as findings. Empty when the bytes are untouched."""
    findings: list[str] = []
    current = fingerprint_tensors(directory)
    for name, expected in sorted(record.items()):
        actual = current.get(name)
        if actual is None:
            findings.append(f"{name} is gone from {directory}")
        elif actual["sha256"] != expected.get("sha256"):
            findings.append(
                f"{name} has changed ({expected.get('bytes')} bytes -> {actual['bytes']}); "
                f"recorded {str(expected.get('sha256'))[:12]}, found {actual['sha256'][:12]}"
            )
    return findings


def git_init(repo: Path, message: str) -> None:
    """Make the repo a git repo with one commit, so the loop has something to branch from."""
    if (repo / ".git").is_dir():
        return
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.name=autohelix", "-c", "user.email=autohelix@localhost",
         "commit", "-q", "-m", message],
        cwd=repo, check=True,
    )


def _git_ignored(repo: Path, path: str) -> bool:
    """Whether the repo's own `.gitignore` excludes this path.

    `git add` on an explicitly named ignored path is an error, not a no-op, so an ignored path has
    to be dropped before staging rather than discovered during it.
    """
    return subprocess.run(
        ["git", "check-ignore", "-q", "--", path], cwd=repo, check=False,
    ).returncode == 0


def git_commit_paths(repo: Path, message: str, paths: Sequence[str]) -> bool:
    """Commit named paths, if the repo is a git repo and any of them actually moved.

    For edits the pipeline makes to an already-initialized repo. The loop refuses to start in a
    dirty repository, so a pipeline-side rewrite of a tracked file has to land in a commit of its
    own or it reads as the agent having left work behind.
    """
    if not (repo / ".git").is_dir():
        return False
    present = [p for p in paths if (repo / p).exists() and not _git_ignored(repo, p)]
    if not present:
        return False
    subprocess.run(["git", "add", "--", *present], cwd=repo, check=True)
    staged = subprocess.run(
        ["git", "diff", "--cached", "--quiet", "--", *present], cwd=repo, check=False)
    if staged.returncode == 0:
        return False
    subprocess.run(
        ["git", "-c", "user.name=autohelix", "-c", "user.email=autohelix@localhost",
         "commit", "-q", "-m", message, "--", *present],
        cwd=repo, check=True,
    )
    return True


#: What the Neuron toolchain regenerates beside the repo root rather than under `build/`: the
#: compiler's own log, the metric store it appends to, and the per-kernel compile caches it names by
#: content hash (a `.colz` and a `.done` in a directory whose name is the hash). Kept in one place
#: because `untrack_regenerated` has to recognize what `write_gitignore` excluded.
REGENERATED = ("log-neuron-cc.txt", "global_metric_store.json", "*.colz", "**/.done")

_REGENERATED_NOTE = (
    "# What the Neuron toolchain writes beside the repo root rather than under build/: the\n"
    "# compiler's own log, the metric store it appends to, and the per-kernel compile caches it\n"
    "# names by content hash. A tracked one of these turns any measurement taken in the main repo\n"
    "# — a gate run, the bar's own re-pin — into uncommitted changes, and the loop then refuses to\n"
    "# start for a reason that has nothing to do with the kernel."
)


def write_gitignore(repo: Path) -> None:
    """Keep run state and regenerated artifacts out of the history.

    Profiles especially: an iteration leaves a `.neff` and one `.ntff` per rank, and committing them
    would put tens of megabytes of binary into every iteration's diff.
    """
    (repo / ".gitignore").write_text(
        "\n".join([
            ".autohelix/",
            "__pycache__/",
            "*.pyc",
            "# Profile artifacts: produced fresh every iteration and checked for freshness, so a",
            "# committed one would be a stale measurement waiting to be believed.",
            "*.neff",
            "*.ntff",
            "profile*.json",
            "# A validator's scratch directory: compile caches, per-rank logs, the device output it",
            "# compares. Regenerated every run, and tracking it means every gate run leaves the",
            "# tree dirty — which the loop refuses to start on.",
            "build/",
            _REGENERATED_NOTE,
            *REGENERATED,
            "",
        ])
    )


def untrack_regenerated(repo: Path) -> list[str]:
    """Ignore and un-stage the artifacts the toolchain rewrites. Returns what it un-staged.

    For repos materialized before `REGENERATED` was excluded, where `git_init`'s `git add -A` took
    the compiler log and the compile cache into the first commit. Every later measurement in the
    main repo then rewrites a tracked file, and the loop — which refuses to start in a dirty
    repository — stops on `M log-neuron-cc.txt`, naming a file that has nothing to do with the
    kernel. `02-Attention`'s stage 3 stopped on exactly that, four minutes in.

    Appends rather than rewriting `.gitignore`, and un-stages with `git rm --cached`, so nothing on
    disk is touched and a cache the next compile wants is still there to be found. Idempotent: on a
    repo already clean of them it appends nothing and un-stages nothing.
    """
    if not (repo / ".git").is_dir():
        return []
    ignore = repo / ".gitignore"
    existing = ignore.read_text() if ignore.is_file() else ""
    missing = [p for p in REGENERATED if p not in existing.splitlines()]
    if missing:
        body = existing if existing.endswith("\n") or not existing else existing + "\n"
        ignore.write_text(body + _REGENERATED_NOTE + "\n" + "\n".join(missing) + "\n")

    tracked = subprocess.run(
        ["git", "ls-files", "-z"], cwd=repo, capture_output=True, text=True, check=True,
    ).stdout.split("\0")
    # `--no-index` because by default `check-ignore` says nothing about a tracked path, and every
    # path here is tracked — that is the whole problem being fixed.
    now_ignored = [
        path for path in tracked
        if path and subprocess.run(
            ["git", "check-ignore", "-q", "--no-index", "--", path], cwd=repo, check=False,
        ).returncode == 0
    ]
    if not now_ignored:
        return []
    subprocess.run(["git", "rm", "-r", "--cached", "-q", "--", *now_ignored],
                   cwd=repo, check=True)
    subprocess.run(["git", "add", "--", ".gitignore"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.name=autohelix", "-c", "user.email=autohelix@localhost",
         "commit", "-q", "-m",
         "Stop tracking what the toolchain regenerates\n\n"
         "A tracked compiler log or compile cache makes every measurement taken in the main "
         "repo leave the tree dirty, and the loop refuses to start on a dirty tree."],
        cwd=repo, check=True,
    )
    return now_ignored


# --------------------------------------------------------------------------------------
# stage 2: the submodule repo
# --------------------------------------------------------------------------------------


def materialize_submodule(
    repo: Path,
    bootstrap_repo: Path,
    artifact: Path,
    projection: Projection,
    module_id: str,
    entry_point: str = "kernel",
) -> dict[str, Any]:
    """Scaffold the repo the stage-2 agent fills in. Returns the partial manifest.

    Partial on purpose: the agent completes it with the tensors it slices, the bar it derives and the
    declaration it writes. What is recorded here is what the *pipeline* knows — the module's own
    golden, the module's own bar, and the projection — so none of it is the agent's to choose.
    """
    if not bootstrap_repo.is_dir():
        raise MaterializeError(f"{bootstrap_repo} is not a directory")
    repo.mkdir(parents=True, exist_ok=True)

    module_dst = repo / MODULE_DIR
    for name in BOOTSTRAP_INPUTS:
        src = bootstrap_repo / name
        if src.exists():
            _copy(src, module_dst / name)
    tensors_note = _link_or_copy_tensors(bootstrap_repo / "tensors", module_dst / "tensors")

    reference_dst = repo / REFERENCE_DIR
    for name in REFERENCE_INPUTS:
        for root in (bootstrap_repo, artifact):
            if (root / name).is_file():
                _copy(root / name, reference_dst / name)
                break
    for tree in REFERENCE_TREES:
        for root in (bootstrap_repo, artifact):
            if (root / tree).is_dir():
                _copy(root / tree, reference_dst / tree)
                break

    readme = module_dst / "README.md"
    bar = read_numerical_bar(readme)
    records = read_tensor_table(readme)
    golden = golden_record(records)

    # The bootstrapped kernel and validator arrive stripped of their commentary. See
    # `optimization/strip.py`: those comments are the bootstrap agent's claims about the hardware
    # and the compiler, some of them wrong, and a wrong claim in a comment reads as established
    # fact to the agent that reads it next. `reference/` is untouched — it is the specification.
    stripped = strip_tree(module_dst)

    (repo / "FLOORPLAN.md").write_text(projection.describe())
    (repo / "README.md").write_text(_submodule_readme(
        module_id, projection, bar, golden, tensors_note,
    ))
    write_gitignore(repo)
    (repo / "source.py").write_text(_source_stub(entry_point))
    (repo / "inference.py").write_text(_inference_stub(entry_point))

    manifest = {
        "module": module_id,
        "entry_point": entry_point,
        "stripped": [s.describe() for s in stripped],
        "projection": projection.to_dict(),
        "module_output": {
            "file": f"{MODULE_DIR}/{golden.file}",
            "dtype": golden.dtype,
            "shape": golden.shape,
        },
        "module_tolerance": bar,
        "module_tensors": [r.to_dict() for r in records],
        # Left for the agent: the submodule's own bar, its own tensors, and its declaration.
        "tolerance": {},
        "tensors": {},
    }
    path = repo / ".autohelix" / "optimization" / "submodule.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    return manifest


def _source_stub(entry_point: str) -> str:
    return f'''"""This rank's kernel. The only file the optimization loop may edit.

Replace this stub with the bootstrapped module's kernel narrowed to one rank. Not with an
empty kernel: iteration 0 has to capture a real latency, and a loop whose baseline does not
run has no metric to improve.
"""


def {entry_point}(*args, **kwargs):
    raise NotImplementedError(
        "the submodule kernel has not been written yet — see the stage-2 prompt"
    )
'''


def _inference_stub(entry_point: str) -> str:
    return f'''"""The validator. Written once, then frozen for the whole optimization loop.

It loads the recorded tensors, traces `{entry_point}` from `source.py`, runs it on the device,
measures the latency from a real profile, and compares against this rank's golden at the bar
`SUBMODULE.md` pins. Ten iterations are judged by this file and by nothing else.
"""

raise NotImplementedError("the validator has not been written yet — see the stage-2 prompt")
'''


def _submodule_readme(
    module_id: str, projection: Projection,
    bar: dict[str, float], golden: TensorRecord, tensors_note: str,
) -> str:
    dims = " * ".join(f"{f.dim}x{f.factor}" for f in projection.projected) or "no split"
    return f"""# Submodule repo: one rank of `{module_id}`

The part of `{module_id}` that runs on a **single logical NeuronCore**, so a loop can optimize it in
isolation before the ranks are put back together.

One rank is **{projection.shard_fraction}** of the module: `{dims}` on one device. `FLOORPLAN.md` has
the planned placement this was projected from and what the projection gave up.

## What is here

| | |
|---|---|
| `source.py` | **editable** — this rank's kernel, and the only file the loop may change |
| `inference.py` | written once by the preparation agent, then **frozen** — the validator |
| `submodule.json` | the declaration: how this rank's work reassembles into the whole module |
| `SUBMODULE.md` | what one rank computes, where each golden came from, how the bar was derived |
| `tensors/` | this rank's input, weights and golden, sliced from `module/tensors/` |
| `goldens/` | all {projection.projected_units} ranks' goldens, so the reassembly can be checked |
| `module/` | frozen — the bootstrapped module: its kernel, validator, README and every recorded tensor ({tensors_note}) |
| `reference/` | frozen — the artifact's own references: what the module computes, how it was run, what counts as matching |

## The module this comes from

Its recorded output is `{MODULE_DIR}/{golden.file}` — `{golden.dtype}`, shape
`{tuple(golden.shape)}` — and its numerical bar is:

```python
{chr(10).join(f'{name} = {bar[name]:g}' for name in TOLERANCE_NAMES)}
```

That is the bar the **assembly** is held to, since the assembly reproduces this tensor. This
submodule's own bar is different and is derived from its own output; `SUBMODULE.md` records it and
the derivation.

## Running it

```bash
NEURON_RT_NUM_CORES=1 python inference.py
```

It must exit 0 and print three markers — `##autohelix[latency_ms=...]`,
`##autohelix[passed=1]`, `##autohelix[max_abs_err=...]` — and leave a fresh `.neff` and `.ntff`.
"""


# --------------------------------------------------------------------------------------
# stage 4: the assembled repo
# --------------------------------------------------------------------------------------


#: How much worse than the bootstrapped kernel a reassembly of the same module may be.
#:
#: The derived bar (`read_numerical_bar`) is a property of the *recorded output* — `RTOL` times its
#: largest element plus `ATOL` — not of what the module can actually be computed to. On
#: `layers.2.attention` the derived ceiling is 0.7875 and the bootstrapped single-core kernel
#: reaches 0.0981445, so the pipeline had 8x of headroom and walked into it: stage 4 reassembled at
#: 0.2773438 and stage 5 finished at 0.609375, with 6,571 of 42 M elements outside the elementwise
#: tolerance against the assembly's 24 — all of it passing a gate that was never the binding
#: constraint. The bootstrapped kernel is a working implementation of the same module against the
#: same golden, so what *it* achieves is the honest reference point.
#:
#: Two margins, because the three statistics fail differently. `MAX_ABS_ERR` is one element and
#: moves only when something structural changes, so it is held tight. Cosine and the pass fraction
#: aggregate over 42 M elements and drift a little whenever the arithmetic is reordered — a rank
#: split legitimately costs a few percent there — so they get more room. Checked against all four
#: attention modules measured so far: this admits `00-Attention-B` (whose assembly cost nothing) and
#: `24-Attention` (whose rewrite improved on its bootstrap), and refuses `02-Attention`'s assembly
#: and `00-Attention-C`'s fp8-PV kernel while admitting C's previous iteration.
WORST_MARGIN = 1.10          # MAX_ABS_ERR
SPREAD_MARGIN = 2.00         # MIN_COSINE, MIN_PASS_FRACTION

#: A floor on the allowance, so a bootstrapped kernel that happens to reach `pass_fraction = 1.0`
#: or a cosine of 1.0 does not produce a bar no reassembly can clear. 1e-5 of 42 M elements is
#: about 420 of them.
ACHIEVED_FLOOR = 1e-5


def tighten_bar(derived: dict[str, float], achieved: dict[str, float] | None,
                worst_margin: float = WORST_MARGIN,
                spread_margin: float = SPREAD_MARGIN) -> dict[str, float]:
    """`derived`, tightened to what the bootstrapped kernel achieved plus a margin.

    Only ever tightens: every value returned is at least as strict as the derived one, so the
    documented rule stays the ceiling and this is a floor under it. A statistic missing from
    `achieved` is left at its derived value rather than guessed — an older run whose baseline
    measurement recorded no numerics gets the old behaviour.

    `RTOL` and `ATOL` are never touched. They define the elementwise test that `MIN_PASS_FRACTION`
    counts, so moving them would silently change what the pass fraction means.
    """
    out = dict(derived)
    if not achieved:
        return out
    worst = achieved.get("max_abs_err")
    if worst is not None and "MAX_ABS_ERR" in out:
        out["MAX_ABS_ERR"] = min(out["MAX_ABS_ERR"], worst * worst_margin)
    for name, key in (("MIN_COSINE", "cosine"), ("MIN_PASS_FRACTION", "pass_fraction")):
        got = achieved.get(key)
        if got is None or name not in out:
            continue
        allowance = max(1.0 - got, ACHIEVED_FLOOR) * spread_margin
        out[name] = max(out[name], 1.0 - allowance)
    return out


def materialize_full(
    repo: Path,
    bootstrap_repo: Path,
    artifact: Path,
    submodule_repo: Path,
    projection: Projection,
    module_id: str,
    bootstrap_latency_ms: float,
    submodule_latency_ms: float,
    best_commit: str | None,
    entry_point: str = "kernel",
    achieved: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Scaffold the repo the stage-4 agent fills in. Returns the partial manifest.

    The submodule arrives at its **best accepted commit**, not at the submodule repo's `HEAD`: with
    5% of regression allowed, `HEAD` after ten iterations can be slower than the best iteration, and
    handing it on would give away part of what the loop achieved.
    """
    repo.mkdir(parents=True, exist_ok=True)

    module_dst = repo / MODULE_DIR
    for name in BOOTSTRAP_INPUTS:
        src = bootstrap_repo / name
        if src.exists():
            _copy(src, module_dst / name)
    _link_or_copy_tensors(bootstrap_repo / "tensors", module_dst / "tensors")
    # The assembly is judged against the module's own recorded tensors, so it gets its own copy at
    # the top level rather than reaching into `module/`: `tensors/` is where its validator looks.
    _link_or_copy_tensors(bootstrap_repo / "tensors", repo / "tensors")

    reference_dst = repo / REFERENCE_DIR
    for name in REFERENCE_INPUTS:
        for root in (bootstrap_repo, artifact):
            if (root / name).is_file():
                _copy(root / name, reference_dst / name)
                break
    for tree in REFERENCE_TREES:
        for root in (bootstrap_repo, artifact):
            if (root / tree).is_dir():
                _copy(root / tree, reference_dst / tree)
                break

    submodule_dst = repo / SUBMODULE_DIR
    submodule_dst.mkdir(parents=True, exist_ok=True)
    for name in ("source.py", "inference.py", "submodule.json", "SUBMODULE.md", "README.md"):
        src = submodule_repo / name
        if not src.exists():
            continue
        if name == "source.py" and best_commit:
            show = subprocess.run(
                ["git", "show", f"{best_commit}:source.py"],
                cwd=submodule_repo, capture_output=True, text=True, check=False,
            )
            if show.returncode == 0:
                (submodule_dst / name).write_text(show.stdout)
                continue
        _copy(src, submodule_dst / name)

    readme = module_dst / "README.md"
    derived_bar = read_numerical_bar(readme)
    bar = tighten_bar(derived_bar, achieved)
    records = read_tensor_table(readme)
    golden = golden_record(records)

    # Both carried-in kernels lose their commentary, for the reason in `optimization/strip.py`.
    # `submodule/` matters as much as `module/` here: its comments are stage 2's and stage 3's
    # claims, written by agents that were themselves reading a stripped copy, and the assembly
    # agent has no way to tell a measured one from a guess.
    stripped = strip_tree(module_dst) + strip_tree(submodule_dst)

    tensors: dict[str, Any] = {}
    for record in records:
        path = repo / "tensors" / Path(record.file).name
        if path.is_file():
            tensors[record.name] = {
                "file": f"tensors/{path.name}",
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "dtype": record.dtype,
                "shape": record.shape,
            }

    (repo / "FLOORPLAN.md").write_text(projection.describe())
    (repo / "README.md").write_text(_full_readme(
        module_id, projection, bar, golden,
        bootstrap_latency_ms, submodule_latency_ms,
    ))
    write_gitignore(repo)
    (repo / "source.py").write_text(_source_stub(entry_point))
    (repo / "inference.py").write_text(_inference_stub(entry_point))

    manifest = {
        "module": module_id,
        "entry_point": entry_point,
        "ranks": projection.projected_units,
        "stripped": [s.describe() for s in stripped],
        "projection": projection.to_dict(),
        "tolerance": bar,
        "tensors": tensors,
        "module_output": {
            "file": f"tensors/{Path(golden.file).name}",
            "dtype": golden.dtype,
            "shape": golden.shape,
        },
        "baselines": {
            "bootstrap_latency_ms": bootstrap_latency_ms,
            "submodule_latency_ms": submodule_latency_ms,
            "overhead_ceiling_ms": round(submodule_latency_ms * 1.10, 6),
        },
        "submodule_commit": best_commit,
        # What the bar would have been from the recorded output alone, and what the bootstrapped
        # kernel actually achieved. `tolerance` above is the two reconciled by `tighten_bar`, and is
        # the only one the gate enforces; these two are here so a reader can see why it is stricter.
        "tolerance_derived": derived_bar,
        "tolerance_achieved": dict(achieved) if achieved else {},
        # Filled by the agent when it freezes the validator it wrote.
        "frozen": {},
    }
    path = repo / ".autohelix" / "optimization" / "module.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    return manifest


def _full_readme(
    module_id: str, projection: Projection, bar: dict[str, float],
    golden: TensorRecord, bootstrap_latency_ms: float, submodule_latency_ms: float,
) -> str:
    ranks = projection.projected_units
    return f"""# Assembled repo: all of `{module_id}` on {ranks} ranks

The whole module, distributed across the {ranks} logical NeuronCores of one device, with the ranks
rejoining through `nki.collectives`.

This is where the pipeline's looseness is paid for. Everything before it was judged against goldens
an agent chose; here the target is the bootstrapped module's own recorded output at the bootstrapped
module's own bar.

## What is here

| | |
|---|---|
| `source.py` | **editable** — the {ranks}-rank kernel, and the only file the loop may change |
| `inference.py` | written once by the preparation agent, then **frozen** — the validator |
| `tensors/` | the module's recorded input, weights and golden, byte-for-byte |
| `submodule/` | frozen — the optimized single-rank kernel this is built from, at its best commit |
| `module/` | frozen — the bootstrapped module |
| `reference/` | frozen — the artifact's own references |
| `ASSEMBLY.md` | how the ranks divide the work and which collective rejoins them |

## What it has to hit

The golden is `tensors/{Path(golden.file).name}` — `{golden.dtype}`, shape
`{tuple(golden.shape)}` — compared against **rank 0's post-collective output**, once, at:

```python
{chr(10).join(f'{name} = {bar[name]:g}' for name in TOLERANCE_NAMES)}
```

Unchanged from the bootstrapped module. Two standing bounds, both measured on this host:

| | |
|---|---|
| faster than | **{bootstrap_latency_ms:g} ms** — the bootstrapped module on one core |
| no slower than | **{submodule_latency_ms * 1.10:g} ms** — 1.1x the submodule's {submodule_latency_ms:g} ms |

## Running it

```bash
NEURON_RT_NUM_CORES={ranks} torchrun --nproc_per_node={ranks} inference.py
```

Every rank must exit 0. The validator prints `##autohelix[latency_rank_<N>_ms=...]` for each of the
{ranks} ranks and the fastest of them as `##autohelix[latency_ms=...]`, plus
`##autohelix[passed=1]` and `##autohelix[max_abs_err=...]`.
"""
