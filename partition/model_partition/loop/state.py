# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resumable loop state with content-hash stage caching.

Stages re-run only when their inputs change. Because tracing a large model is
expensive, a failing verification must not force a re-trace — so invalidation is
explicit and ordered: touching the plan invalidates everything downstream of it
and nothing upstream.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

#: Stage order. Invalidating a stage invalidates every later stage.
#: Extraction follows tracing: a module's implementation documents the parameter
#: names it is handed, and verification runs that implementation.
STAGES = ("ingest", "plan", "trace", "extract", "verify_modules", "emulate", "retain")

PENDING, OK, FAILED, SKIPPED = "pending", "ok", "failed", "skipped"


def content_hash(*parts: Any) -> str:
    """Stable hash of arbitrary JSON-serializable inputs."""
    digest = hashlib.sha256()
    for part in parts:
        digest.update(json.dumps(part, sort_keys=True, default=str).encode())
    return digest.hexdigest()[:16]


@dataclass
class StageRecord:
    """Status of one stage."""

    name: str
    status: str = PENDING
    input_hash: str = ""
    detail: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0
    iteration: int = 0
    finished_at: float = 0.0

    @property
    def succeeded(self) -> bool:
        return self.status in (OK, SKIPPED)


@dataclass
class LoopState:
    """Persistent state of a partition run."""

    slug: str = ""
    iteration: int = 0
    stages: dict[str, StageRecord] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    finished: bool = False
    passed: bool = False

    def record(self, name: str) -> StageRecord:
        return self.stages.setdefault(name, StageRecord(name=name))

    def mark(
        self, name: str, status: str, input_hash: str = "", detail: str = "",
        metrics: dict[str, Any] | None = None, duration_s: float = 0.0,
    ) -> StageRecord:
        entry = self.record(name)
        entry.status = status
        entry.input_hash = input_hash or entry.input_hash
        entry.detail = detail
        entry.metrics = dict(metrics or {})
        entry.duration_s = duration_s
        entry.iteration = self.iteration
        entry.finished_at = time.time()
        return entry

    def is_fresh(self, name: str, input_hash: str) -> bool:
        """True when a stage already succeeded with this exact input."""
        entry = self.stages.get(name)
        return bool(entry and entry.status == OK and entry.input_hash == input_hash)

    def invalidate_from(self, name: str) -> list[str]:
        """Reset ``name`` and every later stage. Returns what was reset."""
        if name not in STAGES:
            raise ValueError(f"Unknown stage {name!r}")
        cleared: list[str] = []
        for stage in STAGES[STAGES.index(name):]:
            entry = self.stages.get(stage)
            if entry and entry.status != PENDING:
                entry.status = PENDING
                entry.input_hash = ""
                cleared.append(stage)
        return cleared

    def first_incomplete(self) -> str | None:
        for stage in STAGES:
            if not self.record(stage).succeeded:
                return stage
        return None

    def all_passed(self) -> bool:
        return all(self.record(stage).succeeded for stage in STAGES)

    def log_iteration(self, summary: dict[str, Any]) -> None:
        self.history.append({"iteration": self.iteration, "at": time.time(), **summary})

    # -- persistence -----------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "iteration": self.iteration,
            "finished": self.finished,
            "passed": self.passed,
            "stages": {name: asdict(entry) for name, entry in self.stages.items()},
            "history": list(self.history),
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return target

    @classmethod
    def load(cls, path: str | Path) -> LoopState:
        source = Path(path)
        if not source.is_file():
            return cls()
        try:
            payload = json.loads(source.read_text())
        except json.JSONDecodeError:
            return cls()
        state = cls(
            slug=payload.get("slug", ""),
            iteration=int(payload.get("iteration", 0)),
            finished=bool(payload.get("finished", False)),
            passed=bool(payload.get("passed", False)),
            history=list(payload.get("history") or []),
        )
        for name, entry in (payload.get("stages") or {}).items():
            state.stages[name] = StageRecord(
                name=name,
                status=entry.get("status", PENDING),
                input_hash=entry.get("input_hash", ""),
                detail=entry.get("detail", ""),
                metrics=entry.get("metrics") or {},
                duration_s=float(entry.get("duration_s") or 0),
                iteration=int(entry.get("iteration") or 0),
                finished_at=float(entry.get("finished_at") or 0),
            )
        return state

    def render(self) -> str:
        symbols = {OK: "ok", FAILED: "FAIL", SKIPPED: "skip", PENDING: "-"}
        rows = []
        for stage in STAGES:
            entry = self.record(stage)
            mark = symbols.get(entry.status, entry.status)
            detail = f"  {entry.detail}" if entry.detail else ""
            rows.append(f"  {stage:<15} {mark:<5}{detail}")
        return "\n".join(rows)
