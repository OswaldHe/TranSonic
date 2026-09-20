# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""On-disk layout of a partition run.

Artifacts live outside the repo: a rejected loop iteration discards code, never
hundreds of gigabytes of traces.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_ARTIFACT_ROOT = Path(
    os.environ.get("MODEL_PARTITION_ARTIFACTS", "~/transonic_artifacts")
).expanduser()

RUN_FILE = "run.yaml"


@dataclass(frozen=True)
class RunLayout:
    """Paths for one model's partition run."""

    root: Path

    @classmethod
    def create(cls, slug: str, artifact_root: str | Path | None = None) -> RunLayout:
        base = Path(artifact_root).expanduser() if artifact_root else DEFAULT_ARTIFACT_ROOT
        return cls(root=(base / slug).resolve())

    def ensure(self) -> RunLayout:
        for path in (self.root, self.plan_dir, self.trace_dir, self.modules_dir, self.reports_dir):
            path.mkdir(parents=True, exist_ok=True)
        return self

    @property
    def run_file(self) -> Path:
        return self.root / RUN_FILE

    @property
    def plan_dir(self) -> Path:
        return self.root / "plan"

    @property
    def graph_path(self) -> Path:
        return self.plan_dir / "partition_graph.yaml"

    @property
    def trace_dir(self) -> Path:
        return self.root / "trace"

    @property
    def modules_dir(self) -> Path:
        return self.root / "modules"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def state_file(self) -> Path:
        return self.root / "state.json"

    @property
    def tokens_file(self) -> Path:
        return self.reports_dir / "tokens.txt"

    @property
    def summary_file(self) -> Path:
        return self.reports_dir / "summary.md"

    def write_run(self, payload: dict[str, Any]) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        self.run_file.write_text(yaml.safe_dump(payload, sort_keys=False))
        return self.run_file

    def read_run(self) -> dict[str, Any]:
        if not self.run_file.is_file():
            raise FileNotFoundError(f"No run manifest at {self.run_file}")
        return yaml.safe_load(self.run_file.read_text()) or {}

    @classmethod
    def at(cls, run_dir: str | Path) -> RunLayout:
        return cls(root=Path(run_dir).expanduser().resolve())

    def disk_usage(self) -> int:
        return sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
