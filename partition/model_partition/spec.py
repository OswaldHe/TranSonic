# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model specification: what to partition, where it comes from, how to load it.

The spec is the loop's only input. With ``loader: auto`` a repo that ships its own
inference code gets ``repo_code``, otherwise ``transformers`` — vendor code is
preferred because it defines the model's numerics and may cover architectures
transformers does not know.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

#: Repo files that indicate a vendor-provided reference implementation.
REPO_CODE_MARKERS = (
    "inference/model.py",
    "modeling.py",
    "model.py",
)

VALID_LOADERS = ("auto", "transformers", "repo_code")

_SLUG_RE = re.compile(r"[^a-z0-9._-]+")


class SpecError(ValueError):
    """Raised when a spec file is malformed or internally inconsistent."""


def slugify(value: str) -> str:
    """Turn a repo id or path into a filesystem-safe artifact directory name."""
    return _SLUG_RE.sub("-", value.strip().lower()).strip("-")


def detect_entry(available_files: list[str]) -> str | None:
    """First vendor-code marker present in a repo, if any.

    Lets ``loader: auto`` pick ``repo_code`` and know which module to import,
    so a bare repo id or directory needs no hand-written spec.
    """
    present = set(available_files)
    return next((marker for marker in REPO_CODE_MARKERS if marker in present), None)


@dataclass
class InputSpec:
    """Sample input sets. Paths are relative to the spec file, or absolute."""

    short: str | None = None
    long: str | None = None
    long_token_budgets: list[int] = field(default_factory=lambda: [2048, 8192, 16384])
    max_short: int | None = None
    max_long: int | None = None

    def resolve(self, base: Path) -> tuple[Path | None, Path | None]:
        """Resolve the two input files against ``base``."""

        def _one(value: str | None) -> Path | None:
            if not value:
                return None
            p = Path(value)
            return p if p.is_absolute() else (base / p).resolve()

        return _one(self.short), _one(self.long)


@dataclass
class ScopeSpec:
    """Structural parts the loop owns.

    Excluded parts are still recorded in the graph as unpartitioned nodes, just
    not extracted, traced, or emulated.
    """

    vision: bool = False
    mtp: bool = False
    engram: bool = False

    def excluded(self) -> list[str]:
        return [name for name, on in (("vision", self.vision), ("mtp", self.mtp), ("engram", self.engram)) if not on]


@dataclass
class ModelSpec:
    """A resolved, validated model specification."""

    name: str
    source: str
    revision: str | None = None
    loader: str = "auto"
    code_paths: list[str] = field(default_factory=list)
    entry: str | None = None
    config_file: str = "config.json"
    trust_remote_code: bool = False
    dtype: str = "bfloat16"
    scope: ScopeSpec = field(default_factory=ScopeSpec)
    inputs: InputSpec = field(default_factory=InputSpec)
    overrides: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    notes: str = ""
    spec_path: Path | None = None

    # -- source classification -------------------------------------------------

    @property
    def is_local(self) -> bool:
        """True when ``source`` points at a directory on this machine."""
        return self.source.startswith(("/", "./", "../", "file://")) or Path(self.source).is_dir()

    @property
    def repo_id(self) -> str | None:
        """The HuggingFace repo id, or None for a local source."""
        if self.is_local:
            return None
        return self.source.removeprefix("hf:").strip("/")

    @property
    def slug(self) -> str:
        """Stable artifact-directory name for this spec."""
        return slugify(self.name or self.source)

    def base_dir(self) -> Path:
        """Directory that relative paths in this spec resolve against."""
        return self.spec_path.parent if self.spec_path else Path.cwd()

    # -- loader resolution -----------------------------------------------------

    def resolve_loader(self, available_files: list[str]) -> str:
        """Pick a loader from the repo's file list (repo-relative paths).

        An explicit ``loader`` wins; ``auto`` prefers vendor code when present.
        """
        if self.loader != "auto":
            return self.loader
        return "repo_code" if detect_entry(available_files) else "transformers"

    # -- (de)serialization -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "source": self.source,
            "loader": self.loader,
            "config_file": self.config_file,
            "trust_remote_code": self.trust_remote_code,
            "dtype": self.dtype,
            "enabled": self.enabled,
            "scope": {"vision": self.scope.vision, "mtp": self.scope.mtp, "engram": self.scope.engram},
            "inputs": {
                "short": self.inputs.short,
                "long": self.inputs.long,
                "long_token_budgets": list(self.inputs.long_token_budgets),
                "max_short": self.inputs.max_short,
                "max_long": self.inputs.max_long,
            },
        }
        if self.revision:
            data["revision"] = self.revision
        if self.code_paths:
            data["code_paths"] = list(self.code_paths)
        if self.entry:
            data["entry"] = self.entry
        if self.overrides:
            data["overrides"] = dict(self.overrides)
        if self.notes:
            data["notes"] = self.notes
        return data


def parse_spec(data: dict[str, Any], spec_path: Path | None = None) -> ModelSpec:
    """Build a :class:`ModelSpec` from parsed YAML, validating as we go."""
    if not isinstance(data, dict):
        raise SpecError(f"Spec must be a mapping, got {type(data).__name__}")

    unknown = set(data) - {
        "name", "source", "revision", "loader", "code_paths", "entry", "config_file",
        "trust_remote_code", "dtype", "scope", "inputs", "overrides", "enabled", "notes",
    }
    if unknown:
        raise SpecError(f"Unknown spec key(s): {', '.join(sorted(unknown))}")

    source = data.get("source")
    if not source or not isinstance(source, str):
        raise SpecError("Spec requires a non-empty string 'source' (hf repo id or local path)")

    loader = data.get("loader", "auto")
    if loader not in VALID_LOADERS:
        raise SpecError(f"loader must be one of {VALID_LOADERS}, got {loader!r}")

    raw_scope = data.get("scope") or {}
    if not isinstance(raw_scope, dict):
        raise SpecError("scope must be a mapping")
    scope_unknown = set(raw_scope) - {"vision", "mtp", "engram"}
    if scope_unknown:
        raise SpecError(f"Unknown scope key(s): {', '.join(sorted(scope_unknown))}")
    scope = ScopeSpec(
        vision=bool(raw_scope.get("vision", False)),
        mtp=bool(raw_scope.get("mtp", False)),
        engram=bool(raw_scope.get("engram", False)),
    )

    raw_inputs = data.get("inputs") or {}
    if not isinstance(raw_inputs, dict):
        raise SpecError("inputs must be a mapping")
    inputs_unknown = set(raw_inputs) - {"short", "long", "long_token_budgets", "max_short", "max_long"}
    if inputs_unknown:
        raise SpecError(f"Unknown inputs key(s): {', '.join(sorted(inputs_unknown))}")
    budgets = raw_inputs.get("long_token_budgets", [2048, 8192, 16384])
    if not isinstance(budgets, list) or not all(isinstance(b, int) and b > 0 for b in budgets):
        raise SpecError("inputs.long_token_budgets must be a list of positive ints")
    inputs = InputSpec(
        short=raw_inputs.get("short"),
        long=raw_inputs.get("long"),
        long_token_budgets=list(budgets),
        max_short=raw_inputs.get("max_short"),
        max_long=raw_inputs.get("max_long"),
    )

    code_paths = data.get("code_paths") or []
    if not isinstance(code_paths, list) or not all(isinstance(c, str) for c in code_paths):
        raise SpecError("code_paths must be a list of strings")

    if loader == "repo_code" and not (code_paths or data.get("entry")):
        raise SpecError("loader 'repo_code' requires 'entry' (and usually 'code_paths')")

    overrides = data.get("overrides") or {}
    if not isinstance(overrides, dict):
        raise SpecError("overrides must be a mapping")

    name = data.get("name") or slugify(str(source).removeprefix("hf:").replace("/", "-"))

    return ModelSpec(
        name=str(name),
        source=str(source),
        revision=data.get("revision"),
        loader=loader,
        code_paths=list(code_paths),
        entry=data.get("entry"),
        config_file=str(data.get("config_file", "config.json")),
        trust_remote_code=bool(data.get("trust_remote_code", False)),
        dtype=str(data.get("dtype", "bfloat16")),
        scope=scope,
        inputs=inputs,
        overrides=dict(overrides),
        enabled=bool(data.get("enabled", True)),
        notes=str(data.get("notes", "")),
        spec_path=spec_path,
    )


def load_spec(path: str | Path) -> ModelSpec:
    """Load and validate a spec YAML file."""
    spec_path = Path(path).resolve()
    if not spec_path.is_file():
        raise SpecError(f"Spec file not found: {spec_path}")
    try:
        data = yaml.safe_load(spec_path.read_text())
    except yaml.YAMLError as exc:
        raise SpecError(f"Invalid YAML in {spec_path}: {exc}") from exc
    return parse_spec(data or {}, spec_path=spec_path)
