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
class PartitionSpec:
    """How this model should be cut up.

    ``prompt`` is free text handed to the agent that plans and repairs, so a model
    can carry its own instruction rather than needing one on every command line.
    ``split_attention_ffn`` is the deterministic form of the most common such
    instruction: the seed planner acts on it directly, so the partition does not
    depend on an agent call succeeding.
    """

    prompt: str = ""
    split_attention_ffn: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"prompt": self.prompt, "split_attention_ffn": self.split_attention_ffn}


@dataclass
class CheckpointSpec:
    """How the published checkpoint's names relate to the model's own.

    A vendor's inference code and its published weights do not always agree on names —
    ``self_attn`` against ``attn``, a block scale called ``weight_scale_inv`` — and the
    vendor's conversion script reconciles them by rewriting the checkpoint. ``rename``
    is that mapping as data, applied to each checkpoint key in order, so the weights
    can be read where they are instead of being copied into a second layout.
    """

    rename: tuple[tuple[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"rename": [list(rule) for rule in self.rename]}


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
class ExtraPass:
    """One entry point the model's own ``forward`` does not reach.

    A speculative-decoding stack is part of the model and no part of ``forward``: it
    runs after it, on what it produced. Hooks only fire for code that runs, so those
    modules cannot be traced from ``forward`` alone — the entry point that does call
    them has to be driven, and how to call it is knowledge about this model.

    ``entry`` is a dotted path to a callable on the model and ``args`` the values to
    hand it, by name: ``input_ids``, ``start_pos``, ``hidden`` (the decoder stack's
    last output) or any of :attr:`TraceSpec.returns`. ``decode`` runs the pass against
    a single-token step after the prefill rather than the prefill itself, which a
    draft stack needs — at ``start_pos`` 0 it has no drafting to do.
    """

    entry: str
    args: tuple[str, ...] = ()
    decode: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"entry": self.entry, "args": list(self.args), "decode": self.decode}


@dataclass
class TraceSpec:
    """How to drive this model for a trace, beyond ``model(input_ids)``.

    ``returns`` names the model forward's return tuple positionally, so an extra pass
    can ask for one of its elements by name.
    """

    returns: tuple[str, ...] = ()
    extra_passes: tuple[ExtraPass, ...] = ()
    #: Module-level objects of the model's own file whose tensors pass between modules
    #: without being anybody's argument. DeepSeek V4.1's ``shared_attn`` is the case: a
    #: layer that compresses its KV publishes it there and the next several layers attend
    #: over it, so a consumer's recorded call carries no trace of the largest thing it
    #: reads. Naming it here has the trace record it per call, which is the only way such
    #: a module can be checked, or optimized, on its own.
    state: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"returns": list(self.returns), "state": list(self.state),
                "extra_passes": [p.to_dict() for p in self.extra_passes]}


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
    checkpoint: CheckpointSpec = field(default_factory=CheckpointSpec)
    inputs: InputSpec = field(default_factory=InputSpec)
    partition: PartitionSpec = field(default_factory=PartitionSpec)
    trace: TraceSpec = field(default_factory=TraceSpec)
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
            "checkpoint": self.checkpoint.to_dict(),
            "partition": self.partition.to_dict(),
            "trace": self.trace.to_dict(),
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
        "trust_remote_code", "dtype", "scope", "checkpoint", "inputs", "partition",
        "trace", "overrides", "enabled", "notes",
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

    raw_partition = data.get("partition") or {}
    if not isinstance(raw_partition, dict):
        raise SpecError("partition must be a mapping")
    partition_unknown = set(raw_partition) - {"prompt", "split_attention_ffn"}
    if partition_unknown:
        raise SpecError(f"Unknown partition key(s): {', '.join(sorted(partition_unknown))}")
    partition = PartitionSpec(
        prompt=str(raw_partition.get("prompt") or ""),
        split_attention_ffn=bool(raw_partition.get("split_attention_ffn", False)),
    )

    trace = _parse_trace(data.get("trace") or {})

    raw_checkpoint = data.get("checkpoint") or {}
    if not isinstance(raw_checkpoint, dict):
        raise SpecError("checkpoint must be a mapping")
    checkpoint_unknown = set(raw_checkpoint) - {"rename"}
    if checkpoint_unknown:
        raise SpecError(f"Unknown checkpoint key(s): {', '.join(sorted(checkpoint_unknown))}")
    rules = raw_checkpoint.get("rename") or []
    if not isinstance(rules, list) or not all(
            isinstance(r, (list, tuple)) and len(r) == 2 and all(isinstance(p, str) for p in r)
            for r in rules):
        raise SpecError("checkpoint.rename must be a list of [pattern, replacement] pairs")
    checkpoint = CheckpointSpec(rename=tuple((str(a), str(b)) for a, b in rules))

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
        checkpoint=checkpoint,
        inputs=inputs,
        partition=partition,
        trace=trace,
        overrides=dict(overrides),
        enabled=bool(data.get("enabled", True)),
        notes=str(data.get("notes", "")),
        spec_path=spec_path,
    )


def _parse_trace(raw: dict[str, Any]) -> TraceSpec:
    """Validate the ``trace`` block, including every argument name it uses."""
    if not isinstance(raw, dict):
        raise SpecError("trace must be a mapping")
    unknown = set(raw) - {"returns", "extra_passes", "state"}
    if unknown:
        raise SpecError(f"Unknown trace key(s): {', '.join(sorted(unknown))}")
    returns = raw.get("returns") or []
    if not isinstance(returns, list) or not all(isinstance(name, str) for name in returns):
        raise SpecError("trace.returns must be a list of names for the forward's return tuple")
    state = raw.get("state") or []
    if not isinstance(state, list) or not all(isinstance(name, str) for name in state):
        raise SpecError("trace.state must be a list of module-level object names")
    # Names an extra pass may ask for. A typo here would otherwise surface as a
    # failed trace after the model is loaded, which for a large one is an hour away.
    available = set(returns) | {"input_ids", "start_pos", "hidden"}
    passes = []
    for entry in raw.get("extra_passes") or []:
        if not isinstance(entry, dict):
            raise SpecError("each trace.extra_passes entry must be a mapping")
        pass_unknown = set(entry) - {"entry", "args", "decode"}
        if pass_unknown:
            raise SpecError(f"Unknown trace.extra_passes key(s): {', '.join(sorted(pass_unknown))}")
        target = entry.get("entry")
        if not target or not isinstance(target, str):
            raise SpecError("trace.extra_passes entry requires 'entry', a callable on the model")
        args = entry.get("args") or []
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
            raise SpecError(f"trace.extra_passes[{target}].args must be a list of names")
        missing = [a for a in args if a not in available]
        if missing:
            raise SpecError(
                f"trace.extra_passes[{target}] asks for {', '.join(missing)}, which is "
                f"not available; name it in trace.returns or use one of "
                f"{', '.join(sorted(available))}"
            )
        passes.append(ExtraPass(entry=target, args=tuple(args),
                                decode=bool(entry.get("decode", False))))
    return TraceSpec(returns=tuple(returns), state=tuple(state), extra_passes=tuple(passes))


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
