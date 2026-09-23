# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish a run's artifacts to a HuggingFace repository.

The artifacts are the deliverable, and they are useful to someone other than the
machine that produced them: per-module implementations with their launchers and
verifiers, the recorded feature maps they are checked against, the plan, and the
reports. This uploads a run as a dataset repo, with a README saying what it is.

Two things it refuses to do. It will not upload a run that did not pass, unless asked,
because an artifact set nobody verified is worse than none. And it will not upload more
than a stated ceiling: the size is knowable before the first byte moves, so a run that
would blow a quota is stopped here rather than halfway.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from model_partition.extract import refresh_calls
from model_partition.hardware import format_bytes
from model_partition.runtime.artifact import CHECKPOINT_DIR

#: Left out by default. Agent transcripts are large and say nothing about the artifacts.
#: ``hf/`` is what ``fetch_weights.py`` downloads on the *reader's* machine —
#: republishing it would upload a copy of the model, which is the one thing an artifact
#: set exists to avoid, and the shards are a pinned revision of a repo anyone can reach.
#: ``.cache/`` is the Hub client's own resume bookkeeping, which it writes *into* the
#: folder it is uploading: 13562 files on the first run here, none of them content.
DEFAULT_EXCLUDE = ("logs/**", "hf/**", ".cache/**")

#: What a run has to contain before uploading it means anything.
REQUIRED = ("run.yaml", "plan/partition_graph.yaml", "modules/index.yaml")

#: Directories a published subset always carries whole. ``vendor`` is the model's own
#: package, which ``source.py`` imports from; ``compat`` is what made its kernels run on
#: the card that produced the reference; ``runtime`` is the builder and the comparison a
#: module's ``inference.py`` reads, so running one needs only ``torch``. Without any of
#: them a module directory is a description of a computation rather than something anyone
#: can run.
ALWAYS = ("vendor", "compat", "runtime", "plan", "reports")

#: A token file an operator may leave beside the checkout, searched when the Hub's own
#: sources (``HF_TOKEN``, a login) have nothing. Read, never written and never logged;
#: it is in ``.gitignore`` so it cannot be committed by accident.
TOKEN_FILE = "hf_token.txt"


class PublishError(RuntimeError):
    """Raised when a run cannot or should not be published."""


@dataclass
class PublishResult:
    """What was uploaded, or would have been."""

    repo_id: str
    files: int = 0
    total_bytes: int = 0
    dry_run: bool = False
    url: str = ""
    largest: list[tuple[str, int]] = field(default_factory=list)

    def summary(self) -> str:
        verb = "would upload" if self.dry_run else "uploaded"
        text = (f"{verb} {self.files} file(s), {format_bytes(self.total_bytes)} "
                f"to {self.repo_id}")
        return f"{text}\n{self.url}" if self.url else text


def materialize_weights(run_dir: str | Path, module_ids: list[str],
                        report: Callable[[str], None] = lambda _m: None) -> int:
    """Write the selected modules' weights into the run; returns the bytes added.

    A run traced with ``cache_weights: false`` keeps only the *index* of which tensors
    each module owns and reads their values from the checkpoint — which is what makes a
    475 GiB model traceable on a 1 TB disk, and what makes the artifacts unusable to
    anybody else: a module directory alone cannot be verified, because the numbers it
    needs are in a repo that is not being published.

    So before publishing a subset, those tensors are read once and dumped beside the
    activations they are checked against. Block scales come with the weights they scale,
    which is what lets the launcher dequantize a weight the class declares wider than the
    checkpoint stores it — no second, bfloat16 copy of the checkpoint is needed for that.
    """
    from model_partition.runtime.module_runner import TraceBundle

    root = Path(run_dir)
    bundle = TraceBundle.load(root / "trace")
    # The bundle's own store, not a second one opened on the same directory: saving the
    # bundle rewrites the manifest from the store it holds, so anything written through
    # another handle is erased by the save that was meant to record it.
    store = bundle.store
    if bundle.checkpoint is None:
        bundle.checkpoint = _weight_source(root)
    added = 0
    for module_id in module_ids:
        wanted = [name for name in (bundle.weight_params.get(module_id) or [])]
        if not wanted:
            continue
        already = {(entry.extra or {}).get("param") for entry in store.entries
                   if entry.module_id == module_id}
        wanted = [name for name in wanted if name not in already]
        if not wanted:
            continue
        # One tensor at a time. A single n-gram table is 94.4 GiB, and reading a module's
        # weights as one batch would ask for all of them at once — which is the thing this
        # whole run is arranged to avoid.
        written = 0
        for name in wanted:
            for key, tensor in bundle.checkpoint.load([name], device="cpu").items():
                meta = store.write(_safe(key), tensor, role="weight", module_id=module_id,
                                   subdir=f"weights/{module_id}", extra={"param": key})
                added += meta.nbytes
                written += 1
                del tensor
        report(f"  {module_id}: {written} tensor(s), {format_bytes(added)} so far")
    store.save_manifest(store.metadata)
    # And say so in the record the module runner reads. Writing the blobs without listing
    # them leaves a directory that has its weights on disk and reports that it has none.
    for entry in store.entries:
        if entry.role == "weight" and entry.module_id in set(module_ids):
            names = bundle.weights.setdefault(entry.module_id, [])
            if entry.name not in names:
                names.append(entry.name)
    bundle.save()
    # And in each module directory's own record, which is what its `inference.py` reads.
    for directory in sorted((root / "modules").glob("*/")):
        refresh_calls(bundle, directory)
    return added


def _weight_source(root: Path):
    """The checkpoint this run read its weights from, resolved from its own manifest.

    A published run is read back from disk, where the source that fed it is not recorded
    as an object. The manifest names the model, so it can be found again — and if it
    cannot, saying so here is better than uploading a module directory whose numbers are
    somewhere else.
    """
    from model_partition import yamlio
    from model_partition.ingest import ingest
    from model_partition.spec import parse_spec
    from model_partition.weights_index import CheckpointWeights

    manifest = yamlio.load_path(root / "run.yaml") or {}
    payload = manifest.get("spec")
    if not payload:
        raise PublishError(f"{root / 'run.yaml'} does not record the model it partitioned")
    # The revision the trace was actually taken from, which is not always the one the
    # spec asked for: a spec may name a moving branch or name nothing at all, and ingest
    # pins whatever that resolved to. Re-ingesting the spec alone would read weights from
    # whatever the branch points at now, and those do not belong beside these activations.
    payload = dict(payload)
    if resolved := str(manifest.get("revision") or ""):
        payload["revision"] = resolved
    try:
        spec = parse_spec(payload)
        result = ingest(spec)
    except Exception as exc:
        raise PublishError(
            f"this run read its weights from the checkpoint rather than dumping them, and "
            f"the checkpoint cannot be reached to copy them in: {exc}"
        ) from exc
    return CheckpointWeights.from_ingest(result, rename=spec.checkpoint.rename)


def _safe(name: str) -> str:
    from model_partition.trace import _safe_name

    return _safe_name(name)


def check_self_contained(run_dir: str | Path, files: list[Path], module_ids: list[str],
                         report: Callable[[str], None] = lambda _m: None) -> list[str]:
    """Verify the published files on their own; returns what failed.

    Not a scan for suspicious paths — an actual run. The published files are copied
    somewhere else with nothing else of this machine in reach, and each selected group is
    run there against its recorded reference. If a module needs a weight that stayed in
    the checkpoint, or a sibling of the model's package that was not carried along, it
    fails here rather than in somebody else's hands.

    Each group is run twice, because the two runs answer different questions.
    ``verify.py`` is the harness's gate and may use this package; ``inference.py`` is what
    somebody who downloads the artifact runs, and it is launched with this package taken
    off ``sys.path`` — so it passes only if the copies under ``runtime/`` and ``vendor/``
    inside the artifact are enough on their own.

    The fetched checkpoint comes too, when the run has one. It is deliberately *not*
    uploaded — republishing a copy of the model is the one thing an artifact set exists to
    avoid — but a reader runs ``fetch_weights.py`` and has it, so a check without it asks
    whether the modules work having skipped a documented step. Without this, a run that
    traced with weight caching off failed here for every group and the only way to publish
    was to turn the check off.
    """
    import shutil
    import subprocess
    import sys
    import tempfile

    root = Path(run_dir)
    # Beside the run, so the published files can be hardlinked rather than copied: the
    # check is about which paths are reachable, not about having a second hundred
    # gigabytes of the same bytes.
    elsewhere = Path(tempfile.mkdtemp(prefix="published-", dir=root.parent))
    checkpoint = root / CHECKPOINT_DIR
    fetched = sorted(checkpoint.glob("*")) if checkpoint.is_dir() else []
    for path in list(files) + fetched:
        target = elsewhere / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            target.hardlink_to(path)
        except OSError:
            shutil.copy2(path, target)

    failures: list[str] = []
    for group, selected in sorted(_published_groups(root, set(module_ids)).items()):
        directory = elsewhere / "modules" / group
        if not (directory / "verify.py").is_file():
            failures.append(f"{group}: no verify.py was published")
            continue
        # The module whose weights were published, not whichever the group lists first:
        # the others are in the artifact for their code and cannot be built from it.
        module = sorted(selected)[0] if selected else None
        for label, command in (
            # The first recorded pass, not every sample: `verify_modules` already
            # checked every module against every sample, and what is being asked here is
            # whether the published files can answer the question at all.
            ("verify.py", [sys.executable, "verify.py"]
                          + (["--module", module] if module else [])),
            ("inference.py", _isolated_command(module)),
        ):
            finished = subprocess.run(command, cwd=directory, capture_output=True,
                                      text=True, timeout=3600)
            tail = (finished.stdout + finished.stderr).strip().splitlines()
            if finished.returncode == 0:
                report(f"  {group}: {label} passed on its own")
            else:
                failures.append(f"{group}/{label}: {tail[-1] if tail else 'failed'}")
                break
    shutil.rmtree(elsewhere, ignore_errors=True)
    return failures


def _isolated_command(module_id: str | None = None) -> list[str]:
    """Run ``inference.py`` with this package out of reach.

    An import that resolves to the installed harness would make the check pass for a
    module directory that cannot be run by anybody who does not have it — which is the
    failure this is here to catch. So the harness's own directory is dropped from
    ``sys.path`` in the child before it runs the file.
    """
    import sys

    harness = str(Path(__file__).resolve().parent.parent)
    argv = ["inference.py", "--repeat", "1", "--warmup", "0"]
    if module_id:
        argv += ["--module", module_id]
    program = "\n".join([
        "import pathlib, runpy, sys",
        f"harness = pathlib.Path({harness!r})",
        "sys.path = [p for p in sys.path if pathlib.Path(p or '.').resolve() != harness]",
        f"sys.argv = {argv!r}",
        "runpy.run_path('inference.py', run_name='__main__')",
    ])
    return [sys.executable, "-c", program]


def representative_modules(run_dir: str | Path) -> list[str]:
    """One module per implementation group, in plan order.

    What makes a useful set to hand somebody: every *kind* of layer the model has, each
    with its own weights and its recorded reference, rather than 40 copies of the same
    attention. The groups are already the model's distinct kinds — that is what
    deduplicating by structural signature means — so one module from each covers them
    all: a dense attention and a sparse one, an MoE, the n-gram memory, the draft
    stack's attention and its heads, the embedding, the norms, the output head.

    Within a group, a module the trace still holds wins over plan order. Retention keeps
    a few representative layers and deletes the rest, so the first module of a group is
    often one whose feature maps are gone — and publishing that one would upload a
    directory whose weights are materialized and whose reference output is not there.
    """
    from model_partition import yamlio
    from model_partition.planner.graph import PartitionGraph

    root = Path(run_dir)
    graph = PartitionGraph.load(root / "plan" / "partition_graph.yaml")
    order = {module.id: index for index, module in enumerate(graph.partitioned_modules)}
    traced = _traced_modules(root)
    chosen: list[str] = []
    for meta in sorted((root / "modules").glob("*/meta.yaml")):
        listed = (yamlio.load_path(meta) or {}).get("module_ids") or []
        known = [m for m in listed if m in order]
        # A group with nothing traced still gets a module: its code is worth publishing,
        # and the missing reference is reported rather than hidden.
        preferred = [m for m in known if m in traced] or known
        if preferred:
            chosen.append(min(preferred, key=lambda m: order[m]))
    return sorted(set(chosen), key=lambda m: order.get(m, 0))


def _traced_modules(root: Path) -> set[str]:
    """Modules the trace on disk still has a recorded call for."""
    from model_partition.runtime.module_runner import TraceBundle

    try:
        return {record.module_id for record in TraceBundle.load(root / "trace").records}
    except Exception:
        return set()


def collect(run_dir: str | Path, exclude: tuple[str, ...] = DEFAULT_EXCLUDE,
            module_ids: list[str] | None = None) -> tuple[list[Path], int]:
    """Files to upload and their total size, in descending size order.

    ``module_ids`` publishes a subset: those module directories, the activations and
    weights recorded for them, and everything a module needs whatever it is — the
    model's own package, the compatibility patches, the plan, the reports. A whole run
    of a 475 GiB model is not a useful thing to hand anybody; a few modules of each kind,
    each verifiable on its own, is.
    """
    root = Path(run_dir)
    if not root.is_dir():
        raise PublishError(f"No such run directory: {root}")
    missing = [name for name in REQUIRED if not (root / name).exists()]
    if missing:
        raise PublishError(
            f"{root} does not look like a finished run: missing {', '.join(missing)}"
        )
    wanted = set(module_ids or ())
    groups = set(_published_groups(root, wanted)) if wanted else set()
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(root).as_posix()
        if any(fnmatch.fnmatch(relative, pattern) for pattern in exclude):
            continue
        if wanted and not _selected(relative, wanted, groups):
            continue
        files.append(path)
    files.sort(key=lambda p: p.stat().st_size, reverse=True)
    return files, sum(p.stat().st_size for p in files)


def _all_modules(root: Path) -> list[str]:
    """Every module the plan partitions, for a whole-run upload."""
    from model_partition.planner.graph import PartitionGraph

    try:
        graph = PartitionGraph.load(root / "plan" / "partition_graph.yaml")
    except Exception:
        return []
    return [module.id for module in graph.partitioned_modules]


def _published_groups(root: Path, module_ids: set[str]) -> dict[str, set[str]]:
    """Group directory name -> which of these modules it serves.

    Read from each group's ``meta.yaml``, which lists the modules it serves: one
    implementation covers 31 attention layers, so publishing it for one publishes it for
    all of them — and the directory is the same bytes either way. Which of them was
    selected still matters, because that is the one whose weights are in the artifact.
    """
    from model_partition import yamlio

    found: dict[str, set[str]] = {}
    for meta in sorted((root / "modules").glob("*/meta.yaml")):
        listed = (yamlio.load_path(meta) or {}).get("module_ids") or []
        shared = set(listed) & module_ids
        if shared:
            found[meta.parent.name] = shared
    return found


def _largest(root: Path, files: list[Path], count: int = 5) -> list[tuple[str, int]]:
    """The biggest published files, named relative to the run, for a size report."""
    return [(path.relative_to(root).as_posix(), path.stat().st_size)
            for path in files[:count]]


def _ceiling_error(total: int, max_bytes: int,
                   largest: list[tuple[str, int]]) -> PublishError:
    """The refusal when a set is larger than the operator allowed."""
    return PublishError(
        f"{format_bytes(total)} exceeds the {format_bytes(max_bytes)} ceiling. Largest: "
        + "; ".join(f"{name} {format_bytes(size)}" for name, size in largest)
        + ". Exclude what you do not need with --exclude, or raise --max-gib."
    )


def _selected(relative: str, module_ids: set[str], groups: set[str]) -> bool:
    """Whether one path belongs to a published subset."""
    head, _, rest = relative.partition("/")
    if "/" not in relative or head in ALWAYS:
        return True
    if head == "modules":
        first = rest.split("/")[0]
        return first == "index.yaml" or first in groups
    if head == "trace":
        kind, _, tail = rest.partition("/")
        if kind not in ("activations", "weights"):
            return True          # records.yaml and manifest.yaml index both
        return tail.split("/")[0] in module_ids
    return True


def publish_run(
    run_dir: str | Path,
    repo_id: str,
    private: bool = False,
    max_bytes: int | None = None,
    exclude: tuple[str, ...] = DEFAULT_EXCLUDE,
    dry_run: bool = False,
    allow_unverified: bool = False,
    module_ids: list[str] | None = None,
    upload_all: bool = False,
    skip_check: bool = False,
    report: Callable[[str], None] = lambda _message: None,
) -> PublishResult:
    """Upload a run directory, or a selection of its modules, as a dataset repo."""
    root = Path(run_dir)
    selection = None if upload_all else module_ids
    # Both refusals before anything is written. Materializing a subset's weights reads
    # tens or hundreds of gigabytes out of the checkpoint, so reaching "this run did not
    # pass" or "this is over the ceiling" afterwards costs hours and a disk for nothing.
    verdict = _verdict(root)
    if not verdict.get("passed") and not allow_unverified:
        raise PublishError(
            f"this run did not pass ({verdict.get('detail', 'no state recorded')}); "
            "an unverified artifact set is worse than none. Pass allow_unverified to "
            "upload it anyway."
        )
    if max_bytes is not None:
        # What is on disk already is a floor for what will be there once the weights are
        # in, so a set over the ceiling now is over it then. The exact total is checked
        # again below, because a subset's final size is not knowable until then.
        present, floor = collect(root, exclude, module_ids=selection)
        if floor > max_bytes:
            raise _ceiling_error(floor, max_bytes, _largest(root, present))
    if module_ids and not dry_run:
        # The weights first: a subset has to carry its own, or nothing in it can be
        # checked away from the checkpoint it was traced against.
        added = materialize_weights(root, module_ids, report=report)
        if added:
            report(f"materialized {format_bytes(added)} of weights for "
                   f"{len(module_ids)} module(s)")
    elif module_ids:
        # A dry run says what would happen and writes nothing. Copying a selection's
        # weights out of a 475 GiB checkpoint is hours of reading and hundreds of
        # gigabytes on disk, which is not something to do on the way to a size estimate.
        report(f"would materialize the weights of {len(module_ids)} module(s) "
               "(skipped: this is a dry run)")
    # ``upload_all`` separates what is verified from what is sent: the weights of a
    # 475 GiB model cannot all be published, but everything the run *does* hold can be —
    # every module's code, plan, reports and recorded feature maps — and a reader is better
    # served by all of it than by a subset. The selection then says which modules also
    # carry their weights, and so which can be verified where they land.
    files, total = collect(root, exclude, module_ids=selection)
    result = PublishResult(repo_id=repo_id, files=len(files), total_bytes=total,
                           dry_run=dry_run, largest=_largest(root, files))
    if max_bytes is not None and total > max_bytes:
        raise _ceiling_error(total, max_bytes, result.largest)
    report(f"{len(files)} file(s), {format_bytes(total)}")
    for name, size in result.largest:
        report(f"  {format_bytes(size):>10}  {name}")
    if dry_run and not skip_check and module_ids:
        # Nothing was materialized, so a check now would fail on weights that publishing
        # for real would have put there. Saying so beats a misleading failure.
        report("not checking: a selection's weights are materialized on the real run")
    elif not skip_check:
        report("checking the published files verify on their own")
        # No selection means the whole run, so every group is checked — passing an empty
        # list would intersect to nothing and report success without running anything.
        checked = module_ids or _all_modules(root)
        failures = check_self_contained(root, files, checked, report=report)
        if failures:
            raise PublishError(
                "the selection does not verify on its own, so it would not verify for "
                "anybody else:\n  " + "\n  ".join(failures)
            )
    if dry_run:
        return result

    api, token = _api()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    readme = root / "README.md"
    wrote_readme = not readme.exists()
    if wrote_readme:
        readme.write_text(_readme(root, verdict))
    try:
        api.upload_large_folder(
            repo_id=repo_id, repo_type="dataset", folder_path=str(root),
            ignore_patterns=list(exclude) + ["**/__pycache__/**"],
            **({"allow_patterns": sorted({p.relative_to(root).as_posix() for p in files})}
               if module_ids else {}),
        )
    finally:
        if wrote_readme:
            readme.unlink(missing_ok=True)
    result.url = f"https://huggingface.co/datasets/{repo_id}"
    del token
    return result


def _api() -> tuple[Any, str]:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - dependency present in practice
        raise PublishError("huggingface_hub is required to publish") from exc
    token = find_token()
    if not token:
        raise PublishError(
            f"no HuggingFace token found. Run `huggingface-cli login`, set HF_TOKEN, or "
            f"leave a token with write access in {TOKEN_FILE} beside the checkout."
        )
    return HfApi(token=token), token


def find_token() -> str | None:
    """A write token, from the Hub's own sources or a file beside the checkout."""
    from huggingface_hub import get_token

    found = get_token()
    if found:
        return found
    # Up to the checkout root and one level beyond it — which is where a secret
    # naturally goes, being outside anything git tracks — and no further.
    here = Path.cwd().resolve()
    passed_root = False
    for directory in (here, *here.parents):
        candidate = directory / TOKEN_FILE
        if candidate.is_file():
            value = candidate.read_text().strip()
            if value:
                return value
        if passed_root:
            break
        passed_root = (directory / ".git").exists()
    return None


def _verdict(root: Path) -> dict[str, Any]:
    """Whether this run passed, from its own state file."""
    import json

    path = root / "state.json"
    if not path.is_file():
        return {"passed": False, "detail": "no state.json"}
    try:
        state = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        return {"passed": False, "detail": f"unreadable state.json: {exc}"}
    stages = state.get("stages") or {}
    failed = [name for name, record in stages.items()
              if isinstance(record, dict) and record.get("status") == "failed"]
    return {
        "passed": bool(state.get("passed")),
        "iteration": state.get("iteration"),
        "detail": f"stage(s) failed: {', '.join(failed)}" if failed
                  else f"passed={state.get('passed')}",
        "stages": {name: record.get("detail", "") for name, record in stages.items()
                   if isinstance(record, dict)},
    }


def _readme(root: Path, verdict: dict[str, Any]) -> str:
    """A README for the uploaded set: what it is and how to use it."""
    from model_partition import yamlio

    manifest = yamlio.load_path(root / "run.yaml") or {}
    spec = manifest.get("spec") or {}
    stages = "\n".join(f"- **{name}** — {detail}"
                       for name, detail in (verdict.get("stages") or {}).items() if detail)
    return f"""---
tags:
- autohelix
- model-partition
---

# Partition artifacts: {spec.get('source', root.name)}

Produced by `autohelix partition` ({spec.get('loader', '?')} loader, revision
`{manifest.get('revision') or '(local)'}`). Every module here was run from these
artifacts alone and compared against the recorded reference.

## What is in here

- `modules/<group>/` — one directory per deduplicated implementation group.
  `source.py` is the implementation, `inference.py` launches it and reports latency and
  error as `##autohelix[...]` metrics, `verify.py` is the gate, `config.json` is the
  config the module was built from, `README.md` states its pre- and post-conditions.
- `trace/` — the recorded per-module inputs, weights and outputs the modules are
  checked against, in a raw little-endian format with a YAML manifest.
- `plan/partition_graph.yaml` — the partition, with `rationale.md` explaining it.
- `reports/` — per-module verification, chained drift, emulated generation, tokens.
- `compat/` — patches that made the model's own code run on the GPU used here, if any
  were needed. The reference was produced with them applied.
- `runtime/` — the harness a module directory builds itself with, so running one needs
  `torch` and nothing installed from the tool that produced this.
- `fetch_weights.py` — present when the run read weights from the checkpoint rather than
  copying them in here. See below.

## Weights

A module's recorded inputs and outputs are all in `trace/`. Its *parameters* may not be:
this set is published without a second copy of the checkpoint, so each module records the
shard and key each weight lives in instead. `fetch_weights.py` downloads only those shards,
pinned to the revision above, into `hf/`, which is where the modules look:

```bash
python fetch_weights.py --list                 # what it would fetch, and how big
python fetch_weights.py --group <group>        # just one group's shards
```

A module short of a weight says so and names the script, rather than reporting numbers it
computed from values it does not have.

## Checking a module

```bash
cd modules/<group>
python verify.py --all-modules --all-samples   # exits non-zero on a mismatch
python inference.py --device cuda --repeat 50   # latency and error as metrics
```

## This run

{stages or '(no stage details recorded)'}
"""
