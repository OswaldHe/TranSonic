# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The `autohelix integrate` config: an AutoHelix config, plus agent-judged constraints.

Not a separate schema. The file is an ordinary `autohelix.yaml` — `goal`, `constraints`,
`scope`, `agent`, `budget`, `reviewer` — parsed by `autohelix.config.load_config`, so every key
means there what it means to `autohelix run`. This module adds exactly one thing to it and
derives two.

**Added: a constraint may be `kind: agent`.** `autohelix.config` reads only `command` and
`timeout` off a constraint entry and ignores the rest, so the extra keys ride along in the same
list without a schema fork — the same way `optimization/` carries `iteration_constraints`,
`memory` and `feedback_archive` in a config the shared dataclass does not know about.

**Derived: the metric and the acceptance rule.** Both are the pass's, not the operator's. The
metric is `constraints_passing`; acceptance refuses any iteration satisfying fewer than the best
so far. A config that set either could let an unbounded loop wander, so setting them is an
error rather than an override.
"""

from __future__ import annotations

import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autohelix.config import Config, load_config

#: The config filename `init` writes and `run-baseline` looks for, in the project directory.
#: Not `autohelix.yaml`: a repo may have one of those already, for a plain `autohelix run`.
DEFAULT_CONFIG_NAME = "integrate.yaml"

#: What an unfilled field looks like in the template.
PLACEHOLDER = "<FILL IN"

#: The interpreter the derived gate and metric commands invoke: the one running autohelix.
#:
#: The gate imports `integrate`, which ships with autohelix rather than with the project being
#: integrated, so it has to run in autohelix's environment. Bare `python` would work only while
#: the venv is activated — not from cron or a unit file. The derived config is rewritten on every
#: start, so an absolute path cannot go stale.
GATE_PYTHON = sys.executable

#: How a constraint may be expressed.
KINDS = ("script", "agent")

#: Default per-constraint timeouts, in seconds. An agent constraint gets longer: it is a model
#: call, not a subprocess.
DEFAULT_SCRIPT_TIMEOUT = 1800
DEFAULT_AGENT_TIMEOUT = 1800

#: The metric this pass runs on: how many constraints currently hold. Fixed, for the same reason
#: `optimization`'s `latency_ms` is — the acceptance rule, the gate's verdict and the loop's exit
#: condition all name it.
METRIC = "constraints_passing"

#: Where the gate writes its verdict, relative to the project.
GATE_JSON = ".autohelix/integrate/gate.json"

#: The loop runs until the constraints are satisfied, so an iteration count is a backstop rather
#: than a budget. Large enough not to be the thing that stops a real run.
UNBOUNDED_ITERATIONS = 10_000

#: Keys this pass reads off a constraint entry. `command` and `timeout` are also
#: `autohelix.config`'s; the rest are ignored by it and belong to this pass.
CONSTRAINT_KEYS = {"name", "kind", "rationale", "command", "timeout", "prompt", "criteria",
                   "model", "effort"}


class ConfigError(ValueError):
    """The config is unusable, with the reason in the message."""


@dataclass
class Constraint:
    """One condition the integration has to reach.

    Script constraints are a command and an exit code — what a constraint already is in
    `autohelix.yaml`. Agent constraints are a prompt, a criteria, and a model's verdict.
    """

    name: str
    kind: str = "script"
    rationale: str = ""

    # -- kind: script --------------------------------------------------------------------
    command: str = ""

    # -- kind: agent ---------------------------------------------------------------------
    #: What the agent has to check. Free prose; the thing being judged.
    prompt: str = ""
    #: What counts as passing. Separate from `prompt` so the verdict rule is stated once,
    #: explicitly, rather than buried in a description.
    criteria: str = ""
    model: str | None = None
    effort: str | None = None

    timeout: int = 0

    def __post_init__(self) -> None:
        if not self.timeout:
            self.timeout = (
                DEFAULT_AGENT_TIMEOUT if self.kind == "agent" else DEFAULT_SCRIPT_TIMEOUT
            )

    @property
    def slug(self) -> str:
        """A filename-safe form of the name, for per-constraint verdict files."""
        out = "".join(c if c.isalnum() or c in "-_" else "-" for c in self.name).strip("-")
        return out or "constraint"

    def problems(self) -> list[str]:
        """Everything wrong with this constraint, named."""
        out: list[str] = []
        where = f"constraint '{self.name}'"
        if self.kind not in KINDS:
            out.append(f"{where}: kind '{self.kind}' is not one of {', '.join(KINDS)}")
        if self.kind == "script":
            if not self.command.strip() or self.command.startswith(PLACEHOLDER):
                out.append(f"{where}: a script constraint needs a command")
            else:
                try:
                    shlex.split(self.command)
                except ValueError as exc:
                    out.append(f"{where}: the command does not parse as a shell word list: {exc}")
            if self.prompt.strip() or self.criteria.strip():
                out.append(
                    f"{where}: a script constraint takes a `command:`, not a "
                    "`prompt:`/`criteria:`. Set `kind: agent` if the check needs judgement."
                )
        if self.kind == "agent":
            if not self.prompt.strip() or self.prompt.startswith(PLACEHOLDER):
                out.append(f"{where}: `kind: agent` needs a `prompt:` saying what to check")
            if not self.criteria.strip() or self.criteria.startswith(PLACEHOLDER):
                out.append(
                    f"{where}: `kind: agent` needs a `criteria:` saying what counts as passing. "
                    "Without one the verdict is the model's taste rather than your bar."
                )
            if self.command.strip():
                out.append(f"{where}: `kind: agent` takes no `command:`")
        if self.timeout <= 0:
            out.append(f"{where}: `timeout:` must be positive")
        return out

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"name": self.name, "kind": self.kind, "timeout": self.timeout}
        if self.rationale:
            payload["rationale"] = self.rationale
        if self.kind == "script":
            payload["command"] = self.command
        else:
            payload["prompt"] = self.prompt
            payload["criteria"] = self.criteria
            if self.model:
                payload["model"] = self.model
            if self.effort:
                payload["effort"] = self.effort
        return payload

    @classmethod
    def from_entry(cls, raw: Any, index: int) -> "Constraint":
        """One entry of the native `constraints:` list.

        A bare string is a script constraint, exactly as in `autohelix.yaml`. A mapping may be
        either kind.
        """
        if isinstance(raw, str):
            return cls(name=raw.strip(), kind="script", command=raw.strip())
        if not isinstance(raw, dict):
            raise ConfigError(
                f"constraints[{index}] is neither a command string nor a mapping"
            )
        unknown = set(raw) - CONSTRAINT_KEYS
        if unknown:
            raise ConfigError(
                f"constraints[{index}] has unknown key(s): {', '.join(sorted(unknown))}"
            )
        kind = str(raw.get("kind", "script"))
        command = str(raw.get("command", "") or "")
        # Unnamed script constraints are named by their command, which is what `autohelix run`
        # already shows for them. An unnamed agent constraint has nothing to fall back to.
        default_name = command.strip() or f"constraint-{index + 1}"
        return cls(
            name=str(raw.get("name") or default_name),
            kind=kind,
            rationale=str(raw.get("rationale", "") or ""),
            command=command,
            prompt=str(raw.get("prompt", "") or ""),
            criteria=str(raw.get("criteria", "") or ""),
            model=(str(raw["model"]) if raw.get("model") else None),
            effort=(str(raw["effort"]) if raw.get("effort") else None),
            timeout=int(raw.get("timeout") or 0),
        )


@dataclass
class IntegrateConfig:
    """An AutoHelix config read for this pass: the native parts, plus the constraint kinds."""

    #: The repo the loop works in — AutoHelix's project path, from `-p/--path`.
    project_path: Path
    #: The native config, parsed by `autohelix.config`. Everything the Harness needs.
    config: Config
    #: The raw YAML, carried so the derived file keeps keys the dataclass does not model.
    raw: dict[str, Any] = field(default_factory=dict)
    constraints: list[Constraint] = field(default_factory=list)
    source: Path | None = None

    # ------------------------------------------------------------------ native passthroughs

    @property
    def goal(self) -> str:
        return self.config.goal or ""

    @property
    def editable(self) -> list[str]:
        return list(self.config.editable)

    @property
    def agent(self) -> dict[str, Any]:
        """The whole `agent:` block, for `AgentConfig.from_dict`."""
        return dict(self.raw.get("agent") or {"type": "claude"})

    @property
    def agent_type(self) -> str:
        return str(self.agent.get("type", "claude"))

    @property
    def target_id(self) -> str:
        """What to call this run. The project directory's name, as `autohelix run` does."""
        return self.project_path.name

    @property
    def max_iterations(self) -> int | None:
        """The operator's backstop, or None for unbounded.

        Read from the raw config rather than `Config.max_iterations`, which defaults to a
        number — and a default would silently cap a pass whose whole point is to run until done.
        """
        raw = (self.raw.get("budget") or {}).get("iterations")
        return int(raw) if raw else None

    @property
    def gate_json(self) -> Path:
        return self.project_path / GATE_JSON

    @property
    def loop_config_path(self) -> Path:
        return self.project_path / ".autohelix" / "integrate" / "loop.yaml"

    @property
    def has_reviewer(self) -> bool:
        return self.config.reviewer is not None

    # ------------------------------------------------------------------ loading

    @classmethod
    def load(
        cls, project_path: str | Path, config_file: str | Path | None = None
    ) -> "IntegrateConfig":
        """Parse the config with `autohelix.config.load_config`, then read the constraint kinds."""
        root = Path(project_path).resolve()
        path = Path(config_file) if config_file else root / DEFAULT_CONFIG_NAME
        try:
            config, raw = load_config(root, path)
        except FileNotFoundError as exc:
            raise ConfigError(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - surfaced as a config error either way
            raise ConfigError(f"cannot read {path}: {exc}") from exc

        entries = raw.get("constraints") or raw.get("constraint") or []
        if isinstance(entries, str):
            entries = [entries]
        if not isinstance(entries, list):
            raise ConfigError("`constraints:` is not a list")
        constraints = [Constraint.from_entry(e, i) for i, e in enumerate(entries)]

        out = cls(project_path=root, config=config, raw=raw, constraints=constraints)
        out.source = path
        return out

    # ------------------------------------------------------------------ checking

    def validate(self) -> list[str]:
        """Everything that would stop a run, named. Empty means runnable.

        AutoHelix's own `Config.validate` runs first, so an empty or still-default `goal`, a
        typo'd top-level key and everything else it already knows about are reported by the
        code that owns them rather than re-checked here. Only what is specific to this pass is
        added below.
        """
        problems: list[str] = [
            issue.message
            for issue in self.config.validate(self.raw)
            if issue.level == "error"
        ]
        if not (self.project_path / ".git").exists():
            problems.append(
                f"{self.project_path} is not a git repository. The loop works in worktrees, so "
                "it needs one."
            )
        if not self.editable:
            problems.append(
                "`scope.editable:` is empty — with nothing editable an iteration cannot change "
                "anything, so the loop would run forever against a fixed repo"
            )
        if not self.constraints:
            problems.append(
                "no `constraints:` — this pass exists to drive a repo to a passing state, so "
                "with nothing to satisfy there is nothing for the loop to do"
            )

        # The metric and the acceptance rule are the pass's. A config that set either would be
        # overriding the thing that makes an unbounded loop converge, so say so rather than
        # quietly losing it in the derived file.
        if self.raw.get("metrics") or self.raw.get("observables"):
            problems.append(
                "`metrics:`/`observables:` is set, but this pass supplies its own metric "
                f"({METRIC}) and would overwrite it. Express what you wanted as a constraint."
            )
        if (self.raw.get("acceptance") or {}).get("metric_gates"):
            problems.append(
                "`acceptance.metric_gates:` is set, but this pass supplies its own — refusing "
                "any iteration that satisfies fewer constraints than the best so far. Remove it."
            )

        seen: set[str] = set()
        for constraint in self.constraints:
            problems.extend(constraint.problems())
            if constraint.name in seen:
                problems.append(f"two constraints are both named '{constraint.name}'")
            seen.add(constraint.name)
        return problems

    def warnings(self) -> list[str]:
        """Things worth saying that are not failures.

        AutoHelix's own warnings come first — unknown keys, a missing scope, anything it already
        knows to flag — then the ones specific to this pass.
        """
        out: list[str] = [
            issue.message
            for issue in self.config.validate(self.raw)
            if issue.level != "error"
        ]
        agents = [c for c in self.constraints if c.kind == "agent"]
        if agents and len(agents) == len(self.constraints):
            out.append(
                f"all {len(agents)} constraint(s) are `kind: agent`. Every iteration then costs "
                "one model call per constraint and no part of the bar is mechanically checkable "
                "— consider expressing at least the objective ones as scripts."
            )
        budget = self.raw.get("budget") or {}
        if not self.max_iterations and not budget.get("max_cost_usd") and not budget.get("time"):
            out.append(
                "the loop is unbounded and has no `budget.max_cost_usd` or `budget.time`. That "
                "is the intended mode, but nothing will stop a run that cannot converge."
            )
        if not self.has_reviewer:
            out.append("no `reviewer.prompt`, so iterations are accepted on the constraints alone")
        return out

    # ------------------------------------------------------------------ deriving

    def gate_timeout(self) -> int:
        """Long enough for every constraint to run, plus room for the gate's own overhead."""
        return sum(c.timeout for c in self.constraints) + 300

    def derive_loop_config(self) -> dict[str, Any]:
        """The loop config the Harness is driven with: the operator's, with two keys replaced.

        Everything the operator wrote carries through — `goal`, `scope`, `agent`, `budget`,
        `reviewer`. Only `constraints` and `metrics` are the pass's.
        """
        payload: dict[str, Any] = {
            k: v for k, v in self.raw.items()
            if k not in ("constraints", "constraint", "metrics", "observables", "acceptance")
        }

        # One command for the whole set, script and agent alike. It always exits 0 — it is a
        # measurement, not a gate, because here a failing check is the *normal* state and
        # rejecting on it would discard every iteration's progress before the last one. What
        # stops a regression is the metric gate below.
        payload["constraints"] = [{
            "command": (
                f"{GATE_PYTHON} -m integrate.gate --config {self._config_ref()} "
                f"--json {GATE_JSON} --advisory"
            ),
            "timeout": self.gate_timeout(),
        }]
        payload["metrics"] = [{
            # The gate already evaluated everything; re-running it would double each iteration's
            # cost — for agent constraints, paying for every judge twice — and the two runs could
            # disagree, leaving an iteration accepted on one count and ranked on another.
            "command": f"{GATE_PYTHON} -m integrate.readback --json {GATE_JSON}",
            "values": {METRIC: "higher"},
            "timeout": 120,
        }]
        # Zero tolerance: an iteration satisfying fewer constraints than the best so far is
        # rejected and its work discarded. That is what turns "run until it passes" into
        # something that converges rather than oscillates.
        payload["acceptance"] = {
            "metric_gates": [{"metric": METRIC, "max_regression_pct": 0.0}],
        }

        budget = dict(payload.get("budget") or {})
        budget["iterations"] = self.max_iterations or UNBOUNDED_ITERATIONS
        payload["budget"] = budget
        return payload

    def _config_ref(self) -> str:
        """How the derived gate command refers back to this config.

        Absolute: the gate runs with the iteration's worktree as its working directory, so a
        relative path would resolve inside a tree that does not hold the operator's config.
        """
        return str((self.source or Path(DEFAULT_CONFIG_NAME)).resolve())

    def write_loop_config(self) -> Path:
        """Write the derived loop config under the project's `.autohelix/` and return its path."""
        import yaml

        path = self.loop_config_path
        path.parent.mkdir(parents=True, exist_ok=True)
        header = (
            f"# Derived by `autohelix integrate` from {self.source or 'the integrate config'}.\n"
            f"# Do not edit: it is regenerated whenever the loop starts. `constraints` and\n"
            f"# `metrics` are this pass's; everything else is yours, carried through.\n\n"
        )
        path.write_text(
            header
            + yaml.safe_dump(self.derive_loop_config(), sort_keys=False, default_flow_style=False)
        )
        return path
