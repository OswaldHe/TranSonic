# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The pipeline's configuration, and the per-stage AutoHelix configs derived from it.

One file describes the whole run — which module, which bootstrapped repo, which floorplan scheme,
and the two stages' goals, budgets, constraint schedules and reviewers. The loops themselves take an
ordinary AutoHelix config, so this module's other job is to *derive* those: the goal and budget come
from the stage, and the constraint command, the metric and the editable scope are fixed by the
pipeline rather than offered to the operator.

That split is why the derived config is written **outside** the module repos. It names the gate, and
the gate is hidden from the agent — as in `bootstrap` and `floorplan`, a constraint command sitting
in the worktree is a constraint the agent can read for what it does not check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from optimization import PROJECTION_TARGET_UNITS
from optimization.constraints import Schedule, ScheduleError
from optimization.memory import MemorySpec, MemoryError_


class ConfigError(ValueError):
    """The pipeline config is missing something or says something impossible."""


#: What an unfilled field looks like in the template. Caught explicitly so a config run straight
#: from the template fails with "you have not filled in module.id" rather than with a path error
#: fifty lines deeper.
PLACEHOLDER = "<FILL IN>"

#: The interpreter the derived constraint and metric commands invoke. Bare, so the config file is
#: not tied to one absolute path, which means it resolves from the iteration's PATH — and so the
#: preflight has to probe *this*, not `sys.executable`. `driver._preflight` does.
GATE_PYTHON = "python"

#: How long a reviewer gets. Reviewing a kernel here is not the skim upstream's default assumes —
#: it means reading a few hundred lines of NKI against a reference and forming an adversarial view
#: of whether the iteration is real — and a reviewer killed mid-read leaves the iteration with no
#: verdict, which is the one thing the loop cannot recover later.
REVIEWER_TIMEOUT_SECONDS = 2000


@dataclass
class StageConfig:
    """One loop stage: what it optimizes, for how long, under what schedule."""

    name: str
    goal: str
    iterations: int
    iteration_time: str | None = None
    max_regression_pct: float = 5.0
    schedule: Schedule = field(default_factory=Schedule)
    memory: MemorySpec = field(default_factory=MemorySpec)
    reviewer_prompt: str = ""
    reviewer_model: str | None = None
    reviewer_timeout_seconds: int = REVIEWER_TIMEOUT_SECONDS

    @property
    def has_reviewer(self) -> bool:
        return bool(self.reviewer_prompt.strip()) and PLACEHOLDER not in self.reviewer_prompt


@dataclass
class PipelineConfig:
    """Everything one `autohelix optimize` run needs."""

    module_id: str
    bootstrap_repo: Path
    artifact: Path
    scheme: Path
    target_units: int
    on_oversized: str
    workspace_root: Path
    venv: Path | None
    submodule: StageConfig
    full: StageConfig
    agent_type: str = "claude"
    preparation_retries: int = 3
    preparation_timeout: str = "4h"
    compiler_model: str | None = None
    compiler_timeout: str = "1h"
    #: The feedback stage reads ~100k words of notes and reviews, checks claims against the device,
    #: and searches for documentation links, so it gets longer than the constraint compiler.
    feedback_model: str | None = None
    feedback_timeout: str = "3h"
    source: Path | None = None

    # -- derived paths -------------------------------------------------------------

    @property
    def slug(self) -> str:
        """A filesystem-safe name for this module, used for both repo directories."""
        return self.module_id.replace("/", "-").replace(".", "-")

    @property
    def submodule_repo(self) -> Path:
        return self.workspace_root / f"{self.slug}-rank0"

    @property
    def full_repo(self) -> Path:
        return self.workspace_root / f"{self.slug}-full"

    def loop_config_path(self, stage: str) -> Path:
        """Where the derived AutoHelix config for a stage is written.

        Beside the repos, never inside one: it names the gate.
        """
        return self.workspace_root / f"{self.slug}-{stage}.autohelix.yaml"

    # -- loading -------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> "PipelineConfig":
        path = Path(path)
        if not path.is_file():
            raise ConfigError(f"no config at {path}")
        try:
            data = yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"{path} is not valid YAML: {exc}") from exc
        # The config's own directory, so `memory.path` may be written relative to the file the
        # operator edits rather than to whatever directory the command was run from.
        config = cls.from_dict(data, base_dir=path.resolve().parent)
        config.source = path.resolve()
        return config

    @classmethod
    def from_dict(cls, data: dict[str, Any], base_dir: Path | None = None) -> "PipelineConfig":
        if not isinstance(data, dict):
            raise ConfigError("the config did not parse as a mapping")
        unknown = set(data) - {
            "module", "floorplan", "workspace", "submodule", "full", "agent", "preparation",
            "feedback",
            "constraint_compiler",
            # Shared defaults both stages' `memory:` blocks inherit, so `path:` is written once.
            "memory",
        }
        if unknown:
            raise ConfigError(
                f"unknown top-level key(s): {', '.join(sorted(unknown))}. Known: module, "
                f"floorplan, workspace, submodule, full, memory, agent, preparation, "
                f"constraint_compiler, feedback"
            )

        module = _section(data, "module")
        module_id = _required_str(module, "id", "module")
        bootstrap_repo = _required_path(module, "bootstrap_repo", "module")
        artifact = _required_path(module, "artifact", "module")

        floorplan = _section(data, "floorplan")
        scheme = _required_path(floorplan, "scheme", "floorplan")
        target_units = int(floorplan.get("target_units", PROJECTION_TARGET_UNITS))
        if target_units < 1:
            raise ConfigError(f"floorplan.target_units must be >= 1, got {target_units}")
        on_oversized = str(floorplan.get("on_oversized", "project")).strip().lower()
        if on_oversized not in {"project", "error"}:
            raise ConfigError(
                f"floorplan.on_oversized must be 'project' or 'error', got '{on_oversized}'"
            )

        workspace = _section(data, "workspace")
        root = workspace.get("root")
        if not root or str(root) == PLACEHOLDER:
            raise ConfigError("workspace.root must name a directory for the two module repos")
        venv = workspace.get("venv")

        agent = data.get("agent") or {}
        preparation = data.get("preparation") or {}
        compiler = data.get("constraint_compiler") or {}
        feedback_section = data.get("feedback") or {}

        return cls(
            module_id=module_id,
            bootstrap_repo=bootstrap_repo,
            artifact=artifact,
            scheme=scheme,
            target_units=target_units,
            on_oversized=on_oversized,
            workspace_root=Path(str(root)).expanduser(),
            venv=Path(str(venv)).expanduser() if venv and str(venv) != PLACEHOLDER else None,
            submodule=_stage(data, "submodule", base_dir, data.get("memory")),
            full=_stage(data, "full", base_dir, data.get("memory")),
            agent_type=str(agent.get("type", "claude")),
            preparation_retries=int(preparation.get("retries", 3)),
            preparation_timeout=str(preparation.get("timeout", "4h")),
            compiler_model=compiler.get("model"),
            compiler_timeout=str(compiler.get("timeout", "1h")),
            feedback_model=feedback_section.get("model"),
            feedback_timeout=str(feedback_section.get("timeout", "3h")),
        )

    # -- validation ----------------------------------------------------------------

    def validate(self) -> list[str]:
        """Everything wrong with this config, as messages. Empty means usable.

        Returned rather than raised so the CLI can print all of them at once: an operator filling
        in a template wants the whole list, not the first problem.
        """
        problems: list[str] = []
        if not self.bootstrap_repo.is_dir():
            problems.append(f"module.bootstrap_repo does not exist: {self.bootstrap_repo}")
        else:
            for needed in ("source.py", "inference.py", "README.md", "tensors"):
                if not (self.bootstrap_repo / needed).exists():
                    problems.append(
                        f"module.bootstrap_repo has no {needed} — is {self.bootstrap_repo} a "
                        f"finished `autohelix bootstrap` project?"
                    )
        if not self.artifact.is_dir():
            problems.append(f"module.artifact does not exist: {self.artifact}")
        if not self.scheme.is_file():
            problems.append(f"floorplan.scheme does not exist: {self.scheme}")
        if self.venv is not None and not self.venv.is_dir():
            problems.append(f"workspace.venv does not exist: {self.venv}")

        for stage in (self.submodule, self.full):
            if not stage.goal.strip() or PLACEHOLDER in stage.goal:
                problems.append(f"{stage.name}.goal is empty or still the placeholder")
            if stage.iterations < 1:
                problems.append(f"{stage.name}.budget.iterations must be >= 1")
            if PLACEHOLDER in stage.reviewer_prompt:
                problems.append(
                    f"{stage.name}.reviewer.prompt is still '{PLACEHOLDER}'. Write the review "
                    f"you want, or delete the `reviewer:` block to run without one"
                )
        return problems

    def warnings(self) -> list[str]:
        """Things worth saying out loud that are not reasons to refuse the run.

        Kept apart from `validate()` because `Schedule.validate` documents an uncovered iteration as
        legal — `describe_for_prompt` and `run_iteration` both implement it as unconstrained
        exploration — and folding its output into the fatal list made a deliberately partial schedule
        unable to pass `optimize check` at all.
        """
        found: list[str] = []
        for stage in (self.submodule, self.full):
            for warning in stage.schedule.validate(stage.iterations):
                found.append(f"{stage.name}: {warning}")
            for warning in stage.memory.validate():
                found.append(f"{stage.name}: {warning}")
        return found

    # -- the derived AutoHelix config ----------------------------------------------

    def derive_loop_config(self, stage: str) -> dict[str, Any]:
        """The AutoHelix config for one loop stage.

        The operator supplies the goal, the budget, the regression allowance, the schedule and the
        reviewer. The pipeline supplies the rest, and those parts are not configurable on purpose:
        the constraint command is the hidden gate, the metric is the thing every bound is stated in,
        and the editable scope is one file because a movable validator makes the measurements
        incomparable.
        """
        from optimization import candidate, module_checker

        if stage == "submodule":
            spec, checker, timeout = (
                self.submodule, "optimization.submodule_checker", candidate.DEFAULT_RUN_TIMEOUT,
            )
        elif stage == "full":
            spec, checker, timeout = (
                self.full, "optimization.module_checker", module_checker.DEFAULT_RUN_TIMEOUT,
            )
        else:
            raise ConfigError(f"unknown stage '{stage}' (expected 'submodule' or 'full')")

        payload: dict[str, Any] = {
            "goal": spec.goal,
            "constraints": [{
                # `python` from PATH, as bootstrap's preset does, so the file is not specific to an
                # interpreter. The preflight proves it can import what the gate needs.
                "command": (
                    f"{GATE_PYTHON} -m {checker} --repo . "
                    f"--json .autohelix/optimization/gate.json --timeout {timeout}"
                ),
                "timeout": timeout + 300,
            }],
            "metrics": [{
                # The gate already ran the validator and wrote the latency into its JSON verdict;
                # re-running it would double every iteration's device time and could disagree with
                # the number the gate judged. This reads the verdict back instead.
                "command": (
                    f"{GATE_PYTHON} -m optimization.readback "
                    f"--json .autohelix/optimization/gate.json"
                ),
                # Only what this stage can actually produce. `Harness._capture_baseline` requires
                # *every* declared metric at iteration 0 and fails the run when one is missing, so
                # declaring the per-rank numbers for the single-rank stage — which has no ranks —
                # aborted the loop the moment it started.
                "values": (
                    {"latency_ms": "lower"} if stage == "submodule" else {
                        "latency_ms": "lower",
                        # Recorded, never gated: the metric is the fastest rank, and the spread is
                        # what keeps that number's optimism visible in the report.
                        "slowest_rank_ms": "lower",
                        "rank_spread_ms": "lower",
                    }
                ),
                "timeout": 120,
            }],
            "scope": {"editable": ["source.py"]},
            "acceptance": {
                "metric_gates": [
                    {"metric": "latency_ms", "max_regression_pct": spec.max_regression_pct},
                ],
            },
            "agent": {"type": self.agent_type},
            "budget": {"iterations": spec.iterations},
        }
        if spec.iteration_time:
            payload["budget"]["iteration_time"] = spec.iteration_time
        if spec.schedule.slots:
            # `enforcement` and `soften_last`, not the legacy `enforce` boolean. The loop re-parses
            # this derived config, and `enforce` collapses hard and soft to the same `True` — which
            # `_ENFORCE_ALIAS` then reads back as `hard`. So a schedule the operator wrote as
            # `soft` arrived at the loop as `hard`, its checker ran without `--advisory`, and a
            # candidate that only owed a strict improvement was rejected and discarded instead.
            # `soften_last: false` was lost the same way, in the other direction.
            payload["iteration_constraints"] = [
                {
                    "iterations": slot.iterations,
                    "text": slot.text,
                    "enforcement": slot.enforcement,
                    "soften_last": slot.soften_last,
                }
                for slot in spec.schedule.slots
            ]
        if spec.memory.enabled:
            # The loop re-reads this derived file, so the memory has to survive the round trip the
            # same way the schedule does. Absolute path: the derived config lives beside the repos,
            # not beside the operator's config, so a relative path would resolve somewhere else.
            payload["memory"] = spec.memory.to_payload()
        if spec.has_reviewer:
            payload["reviewer"] = {
                "prompt": spec.reviewer_prompt,
                "auto_memory": False,
                "timeout_seconds": spec.reviewer_timeout_seconds,
            }
            if spec.reviewer_model:
                payload["reviewer"]["model"] = spec.reviewer_model
        return payload

    def write_loop_config(self, stage: str) -> Path:
        """Write the derived config beside the repos and return its path."""
        path = self.loop_config_path(stage)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = (
            f"# Derived by `autohelix optimize` from {self.source or 'the pipeline config'}.\n"
            f"# Do not edit: it is regenerated each time the stage starts, and it lives outside\n"
            f"# the module repo because its constraint command names the hidden gate.\n\n"
        )
        path.write_text(header + yaml.safe_dump(
            self.derive_loop_config(stage), sort_keys=False, default_flow_style=False,
        ))
        return path


# --------------------------------------------------------------------------------------
# parsing helpers
# --------------------------------------------------------------------------------------


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    section = data.get(name)
    if section is None:
        raise ConfigError(f"the config has no '{name}:' section")
    if not isinstance(section, dict):
        raise ConfigError(f"'{name}:' must be a mapping")
    return section


def _required_str(section: dict[str, Any], key: str, where: str) -> str:
    value = section.get(key)
    if value is None or not str(value).strip():
        raise ConfigError(f"{where}.{key} is required")
    text = str(value).strip()
    if text == PLACEHOLDER:
        raise ConfigError(f"{where}.{key} is still '{PLACEHOLDER}' — fill it in")
    return text


def _required_path(section: dict[str, Any], key: str, where: str) -> Path:
    return Path(_required_str(section, key, where)).expanduser()


def _stage(
    data: dict[str, Any], name: str, base_dir: Path | None = None,
    shared_memory: Any = None,
) -> StageConfig:
    section = data.get(name) or {}
    if not isinstance(section, dict):
        raise ConfigError(f"'{name}:' must be a mapping")
    unknown = set(section) - {"goal", "budget", "acceptance", "iteration_constraints", "reviewer",
                              "memory"}
    if unknown:
        raise ConfigError(f"{name}: unknown key(s) {', '.join(sorted(unknown))}")

    budget = section.get("budget") or {}
    iterations = int(budget.get("iterations", 10 if name == "submodule" else 5))
    acceptance = section.get("acceptance") or {}
    reviewer = section.get("reviewer") or {}

    try:
        schedule = Schedule.from_config(
            section.get("iteration_constraints"), max_iterations=iterations,
        )
    except ScheduleError as exc:
        raise ConfigError(f"{name}.iteration_constraints: {exc}") from exc

    try:
        memory = MemorySpec.from_config(
            MemorySpec.merge(shared_memory, section.get("memory"), where=f"{name}.memory"),
            max_iterations=iterations, base_dir=base_dir, where=f"{name}.memory",
        )
    except MemoryError_ as exc:
        raise ConfigError(str(exc)) from exc

    return StageConfig(
        name=name,
        goal=str(section.get("goal") or ""),
        iterations=iterations,
        iteration_time=(str(budget["iteration_time"]) if budget.get("iteration_time") else None),
        max_regression_pct=float(acceptance.get("max_regression_pct", 5)),
        schedule=schedule,
        memory=memory,
        reviewer_prompt=str(reviewer.get("prompt") or ""),
        reviewer_model=reviewer.get("model"),
        reviewer_timeout_seconds=int(
            reviewer.get("timeout_seconds", REVIEWER_TIMEOUT_SECONDS)
        ),
    )
