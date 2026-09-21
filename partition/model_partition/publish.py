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

from model_partition.hardware import format_bytes

#: Left out by default: agent transcripts, which are large and say nothing about the
#: artifacts themselves.
DEFAULT_EXCLUDE = ("logs/**",)

#: What a run has to contain before uploading it means anything.
REQUIRED = ("run.yaml", "plan/partition_graph.yaml", "modules/index.yaml")

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


def collect(run_dir: str | Path, exclude: tuple[str, ...] = DEFAULT_EXCLUDE
            ) -> tuple[list[Path], int]:
    """Files to upload and their total size, in descending size order."""
    root = Path(run_dir)
    if not root.is_dir():
        raise PublishError(f"No such run directory: {root}")
    missing = [name for name in REQUIRED if not (root / name).exists()]
    if missing:
        raise PublishError(
            f"{root} does not look like a finished run: missing {', '.join(missing)}"
        )
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(root).as_posix()
        if any(fnmatch.fnmatch(relative, pattern) for pattern in exclude):
            continue
        files.append(path)
    files.sort(key=lambda p: p.stat().st_size, reverse=True)
    return files, sum(p.stat().st_size for p in files)


def publish_run(
    run_dir: str | Path,
    repo_id: str,
    private: bool = False,
    max_bytes: int | None = None,
    exclude: tuple[str, ...] = DEFAULT_EXCLUDE,
    dry_run: bool = False,
    allow_unverified: bool = False,
    report: Callable[[str], None] = lambda _message: None,
) -> PublishResult:
    """Upload a run directory as a HuggingFace dataset repo."""
    root = Path(run_dir)
    files, total = collect(root, exclude)
    result = PublishResult(repo_id=repo_id, files=len(files), total_bytes=total,
                           dry_run=dry_run,
                           largest=[(p.relative_to(root).as_posix(), p.stat().st_size)
                                    for p in files[:5]])
    if max_bytes is not None and total > max_bytes:
        raise PublishError(
            f"{format_bytes(total)} exceeds the {format_bytes(max_bytes)} ceiling. "
            f"Largest: "
            + "; ".join(f"{name} {format_bytes(size)}" for name, size in result.largest)
            + ". Exclude what you do not need with --exclude, or raise --max-gib."
        )
    verdict = _verdict(root)
    if not verdict.get("passed") and not allow_unverified:
        raise PublishError(
            f"this run did not pass ({verdict.get('detail', 'no state recorded')}); "
            "an unverified artifact set is worse than none. Pass allow_unverified to "
            "upload it anyway."
        )
    report(f"{len(files)} file(s), {format_bytes(total)}")
    for name, size in result.largest:
        report(f"  {format_bytes(size):>10}  {name}")
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

## Checking a module

```bash
cd modules/<group>
python verify.py --all-modules --all-samples   # exits non-zero on a mismatch
python inference.py --device cuda --repeat 50   # latency and error as metrics
```

## This run

{stages or '(no stage details recorded)'}
"""
