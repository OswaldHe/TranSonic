# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve a spec into a local model: config, code, tokenizer, weight inventory.

Metadata first, weights on demand. The inventory comes from the Hub's
safetensors metadata, so a 765 GB checkpoint can be planned and budgeted before
a single shard is fetched — and :func:`ensure_shards` then pulls only the shards
a given module needs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from model_partition.spec import ModelSpec, SpecError, detect_entry
from model_partition.weights_index import WeightIndex

#: Fetched up front: everything except weight data.
METADATA_PATTERNS = [
    "*.json", "*.txt", "*.jinja", "*.model", "*.py", "*.md",
    "inference/*", "encoding/*", "encoding/tests/*",
]

WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf")


class IngestError(RuntimeError):
    """Raised when a model cannot be resolved or its metadata is unusable."""


@dataclass
class IngestResult:
    """A model resolved to local metadata, with weights possibly still remote."""

    spec: ModelSpec
    root: Path
    config: dict[str, Any]
    loader: str
    index: WeightIndex
    revision: str | None = None
    repo_files: list[str] = field(default_factory=list)
    #: Effective vendor-code entry and search paths — from the spec, or detected.
    entry: str | None = None
    code_paths: list[str] = field(default_factory=list)

    @property
    def is_local(self) -> bool:
        return self.spec.is_local

    def config_path(self) -> Path:
        return self.root / self.spec.config_file

    def shard_path(self, shard: str) -> Path:
        return self.root / shard

    def local_shards(self) -> list[str]:
        return sorted(s for s in self.index.shard_bytes if self.shard_path(s).is_file())

    def missing_shards(self) -> list[str]:
        return sorted(s for s in self.index.shard_bytes if not self.shard_path(s).is_file())

    def entry_path(self) -> Path | None:
        return self.root / self.spec.entry if self.spec.entry else None


def _hf_api(token: str | None = None):
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover
        raise IngestError("huggingface_hub is required for remote model specs") from exc
    return HfApi(token=token)


def list_repo_files(spec: ModelSpec, token: str | None = None) -> list[str]:
    """Repo-relative file list, from the Hub or a local directory."""
    if spec.is_local:
        root = Path(spec.source)
        if not root.is_dir():
            raise IngestError(f"Local model directory not found: {root}")
        return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())
    return sorted(_hf_api(token).list_repo_files(spec.repo_id, revision=spec.revision))


def resolve_revision(spec: ModelSpec, token: str | None = None) -> str | None:
    """Pin the spec to a commit sha so a run is reproducible."""
    if spec.is_local:
        return None
    if spec.revision and len(spec.revision) == 40:
        return spec.revision
    info = _hf_api(token).model_info(spec.repo_id, revision=spec.revision)
    return info.sha


def load_config(root: Path, config_file: str = "config.json") -> dict[str, Any]:
    path = root / config_file
    if not path.is_file():
        raise IngestError(f"Model config not found: {path}")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise IngestError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise IngestError(f"{path} must contain a JSON object")
    return data


def ingest(spec: ModelSpec, token: str | None = None, with_index: bool = True) -> IngestResult:
    """Fetch metadata and build the weight inventory, without weight data.

    ``with_index=False`` skips the tensor inventory, which is all that is needed
    to instantiate structure — so a module can be replayed from artifacts on a
    machine where the checkpoint is absent entirely.
    """
    repo_files = list_repo_files(spec, token=token)
    loader = spec.resolve_loader(repo_files)

    if spec.is_local:
        root = Path(spec.source).resolve()
        revision = None
        index = WeightIndex.from_local(root) if with_index else WeightIndex(source=str(root))
    else:
        revision = resolve_revision(spec, token=token)
        root = Path(_snapshot_metadata(spec, revision, token))
        index = (WeightIndex.from_hub(spec.repo_id, revision=revision, token=token)
                 if with_index else WeightIndex(source=f"hf:{spec.repo_id}"))

    config = load_config(root, spec.config_file)
    entry = spec.entry
    code_paths = list(spec.code_paths)
    if loader == "repo_code":
        if not entry:
            entry = detect_entry(repo_files)
        if not entry or entry not in repo_files:
            raise SpecError(
                f"loader 'repo_code' needs an 'entry' present in the repo; "
                f"{entry!r} is not among {len(repo_files)} files"
            )
        if not code_paths:
            parent = str(Path(entry).parent)
            code_paths = [parent] if parent not in (".", "") else []
    return IngestResult(
        spec=spec, root=root, config=config, loader=loader,
        index=index, revision=revision, repo_files=repo_files,
        entry=entry, code_paths=code_paths,
    )


def _snapshot_metadata(spec: ModelSpec, revision: str | None, token: str | None) -> Path:
    """Download everything except weight data; return the snapshot directory."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover
        raise IngestError("huggingface_hub is required for remote model specs") from exc
    patterns = list(METADATA_PATTERNS)
    for code_path in spec.code_paths:
        patterns.append(code_path.rstrip("/") + "/*")
    return Path(snapshot_download(
        spec.repo_id, revision=revision, token=token,
        allow_patterns=sorted(set(patterns)),
        ignore_patterns=[f"*{suffix}" for suffix in WEIGHT_SUFFIXES],
    ))


def ensure_shards(result: IngestResult, shards: list[str], token: str | None = None) -> list[Path]:
    """Make the named shards available locally, downloading only what is absent."""
    if result.is_local:
        paths = [result.shard_path(s) for s in shards]
        missing = [p for p in paths if not p.is_file()]
        if missing:
            raise IngestError(f"Missing local shards: {', '.join(str(p) for p in missing)}")
        return paths

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover
        raise IngestError("huggingface_hub is required to fetch shards") from exc

    paths: list[Path] = []
    for shard in shards:
        local = result.shard_path(shard)
        if local.is_file():
            paths.append(local)
            continue
        downloaded = Path(hf_hub_download(
            result.spec.repo_id, filename=shard, revision=result.revision, token=token,
        ))
        # Pinning the same revision puts the download in this snapshot directory,
        # which is where every later reader looks. Link it into place if a future
        # hub layout ever puts it elsewhere, rather than returning a path the rest
        # of the run cannot see.
        if downloaded.resolve() != local.resolve():
            local.parent.mkdir(parents=True, exist_ok=True)
            local.symlink_to(downloaded)
        paths.append(local)
    return paths


def ensure_weights(result: IngestResult, token: str | None = None) -> list[Path]:
    """Make every shard available locally.

    Ingest is metadata-only so a model can be planned without its weights; this
    is the explicit step that materializes them before a real forward.
    """
    missing = result.missing_shards()
    if not missing:
        return [result.shard_path(s) for s in sorted(result.index.shard_bytes)]
    return ensure_shards(result, missing, token=token)


def shards_for_tensors(result: IngestResult, tensor_names: list[str]) -> list[str]:
    """Shards holding the named tensors."""
    wanted = set(tensor_names)
    return sorted({e.shard for e in result.index.entries if e.name in wanted})


def evict_shards(result: IngestResult, shards: list[str]) -> int:
    """Delete local copies of the named shards; return bytes reclaimed.

    The streaming path for checkpoints too large to hold at once. Never touches a
    local-source model, whose weights are the user's own files.

    The Hub cache stores one blob per digest and symlinks every snapshot entry at
    it, so a blob can be shared with another revision or another run. Only a blob
    that nothing else points at is deleted; otherwise this run's link goes and the
    bytes stay, which is reported as nothing reclaimed rather than pretended away.
    """
    if result.is_local:
        return 0
    freed = 0
    for shard in shards:
        link = result.shard_path(shard)
        if not link.exists() and not link.is_symlink():
            continue
        blob = link.resolve()
        link.unlink(missing_ok=True)
        if blob.is_file() and not _blob_is_shared(blob, result.root):
            freed += blob.stat().st_size
            blob.unlink()
    return freed


def _blob_is_shared(blob: Path, snapshot: Path) -> bool:
    """True when a cache blob is still referenced by some snapshot entry.

    Walks the repo's ``snapshots/`` tree, which is where every reference lives:
    ``<cache>/models--org--name/snapshots/<sha>/<file>`` symlinked at
    ``<cache>/models--org--name/blobs/<digest>``.
    """
    repo_cache = blob.parent.parent
    snapshots = repo_cache / "snapshots"
    if not snapshots.is_dir():
        # Not a hub cache layout: treat the file as exclusively ours only when it
        # sits inside this run's snapshot.
        return snapshot not in blob.parents and blob.parent != snapshot
    for entry in snapshots.rglob("*"):
        if entry.is_symlink() and entry.resolve() == blob:
            return True
    return False
