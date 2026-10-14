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
gated metric is `constraints_passing`, counting the script constraints; acceptance refuses any
iteration holding fewer of them than the best so far. `all_constraints_passing` carries the whole
count — ranked, recorded, and the loop's exit condition, but not gated, because an agent judge
can flip over an unchanged tree. A config that set either could let an unbounded loop wander, so
setting them is an error rather than an override.
"""

from __future__ import annotations

import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autohelix.config import Config

#: The config filename `init` writes and `run-baseline` looks for, in the project directory.
#: Not `autohelix.yaml`: a repo may have one of those already, for a plain `autohelix run`.
DEFAULT_CONFIG_NAME = "integrate.yaml"

#: What `run` looks for. A separate file from the baseline stage's, because the two stages ask
#: for different things — the baseline has no metrics and runs unbounded, `run` has metrics and a
#: budget — and because keeping them apart is what lets a repo hold both at once: the
#: constraints that were reached, and the measurements now being improved.
RUN_CONFIG_NAME = "integrate-run.yaml"

#: The two stages. `baseline` drives a failing repo to a passing one; `run` keeps it passing and
#: makes its measurements better.
STAGES = ("baseline", "run")

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

#: The metric acceptance gates on: how many *script* constraints currently hold. Fixed, for the
#: same reason `optimization`'s `latency_ms` is — the acceptance rule, the gate's verdict and the
#: readback all name it.
#:
#: Script constraints only, because this is the count an iteration is *rejected* for. A script
#: constraint is deterministic: the same tree gives the same verdict, so a drop is a regression
#: the iteration caused. An agent constraint is a model call, and two calls over the same tree can
#: disagree — in the vllm-neuron run `heavy-ops-use-nki-kernels` went PASS at iteration 3 and FAIL
#: at iteration 4 over lines that were byte-identical between them, and the zero-tolerance rule
#: below discarded the first iteration that had ever produced a token. Agent verdicts still decide
#: `satisfied`, so the loop does not finish until every constraint holds, and they still reach the
#: next iteration as findings. They just cannot cost an iteration its work.
METRIC = "constraints_passing"

#: Every constraint, script and agent alike. What the run is actually trying to reach, so it is
#: what the loop's exit condition and the summary are measured on — but never gated, per `METRIC`.
METRIC_ALL = "all_constraints_passing"

#: Where the gate writes its verdict, relative to the project.
GATE_JSON = ".autohelix/integrate/gate.json"

#: How much a metric may regress, in percent, before `run` rejects the iteration that did it.
#: The operator's `acceptance.max_regression_pct` overrides it.
#:
#: Not zero. A measurement on real hardware is noisy — this pass's first user measured the same
#: prefill at 21 s and 33 s — so a zero-tolerance gate would reject work for the noise floor and
#: a loop would spend its budget re-measuring. Not large either: the point of the gate is that an
#: iteration cannot trade away what earlier ones won.
DEFAULT_MAX_REGRESSION_PCT = 5.0

#: Iterations `run` takes when the config does not say. A measured iteration here is expensive —
#: the gate runs the constraints, one of which is usually the thing being measured — so the
#: budget is small and deliberate rather than a backstop.
DEFAULT_RUN_ITERATIONS = 10

#: How long the readback gets. It parses a JSON file; a minute is already generous.
READBACK_TIMEOUT = 120

#: The loop runs until the constraints are satisfied, so an iteration count is a backstop rather
#: than a budget. Large enough not to be the thing that stops a real run.
UNBOUNDED_ITERATIONS = 10_000

#: Keys this pass reads off a constraint entry. `command` and `timeout` are also
#: `autohelix.config`'s; the rest are ignored by it and belong to this pass.
CONSTRAINT_KEYS = {"name", "kind", "rationale", "command", "timeout", "prompt", "criteria",
                   "model", "effort"}

#: Keys a `metrics:` entry may carry in `run`. `command`, `values`, `timeout`, `capture` and
#: `capture_stdout` are `autohelix.config`'s; `gate` is this pass's.
METRIC_KEYS = {"command", "values", "timeout", "capture", "capture_stdout", "gate"}


class ConfigError(ValueError):
    """The config is unusable, with the reason in the message."""


def _read_yaml(path: Path) -> dict[str, Any]:
    """The config file as a dict, with the same not-found error `load_config` raises.

    Read here rather than through `autohelix.config.load_config` because this pass has to
    normalize the YAML *before* a `Config` is built from it — see `IntegrateConfig.load`.
    `load_config` is two lines over this, so nothing is lost but the indirection.
    """
    import yaml

    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} is not a YAML mapping")
    return data


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
class Measurement:
    """One `metrics:` entry of a `run` config.

    The shape is `autohelix.yaml`'s, with one addition and one relaxation.

    **`command:` may be omitted.** The measurements this pass is built for are the ones a
    *constraint* already produces: the thing that decides whether the port still works is also
    the thing that prints its latencies, and running it a second time to get a number would
    double the most expensive part of an iteration and let the two runs disagree. So an entry
    with no command is read back out of the gate's verdict, where every
    `##autohelix[name=value]` a constraint printed was recorded.

    **`gate:` decides whether acceptance acts on it.** Default true. Some numbers are worth
    recording and wrong to gate — a ratio that moves with what the model happened to generate,
    a figure measured on one sample — and declaring them anyway is how they stay visible.
    """

    values: dict[str, str] = field(default_factory=dict)
    command: str = ""
    timeout: int = 0
    gate: bool = True
    capture: list[str] = field(default_factory=list)
    capture_stdout: bool = True

    @property
    def from_constraints(self) -> bool:
        """Whether this entry is read back from the gate rather than run."""
        return not self.command.strip()

    @property
    def names(self) -> list[str]:
        return list(self.values)

    def problems(self, index: int) -> list[str]:
        out: list[str] = []
        where = f"metrics[{index}]"
        if not self.values:
            out.append(
                f"{where}: no `values:`. A metric entry exists to name the numbers it produces "
                "and which direction is better, so one with none measures nothing."
            )
        for name, direction in self.values.items():
            if direction not in ("higher", "lower"):
                out.append(
                    f"{where}: metric '{name}' has direction '{direction}' — it has to be "
                    "'higher' or 'lower', which is what says whether a change is an improvement"
                )
        if self.timeout < 0:
            out.append(f"{where}: `timeout:` cannot be negative")
        return out

    @classmethod
    def from_entry(cls, raw: Any, index: int) -> "Measurement":
        if not isinstance(raw, dict):
            raise ConfigError(f"metrics[{index}] is not a mapping")
        unknown = set(raw) - METRIC_KEYS
        if unknown:
            raise ConfigError(
                f"metrics[{index}] has unknown key(s): {', '.join(sorted(unknown))}. "
                f"Known: {', '.join(sorted(METRIC_KEYS))}"
            )
        values = raw.get("values") or {}
        if not isinstance(values, dict):
            raise ConfigError(f"metrics[{index}].values must be a mapping of name to direction")
        capture = raw.get("capture") or []
        return cls(
            values={str(k): str(v) for k, v in values.items()},
            command=str(raw.get("command", "") or ""),
            timeout=int(raw.get("timeout") or 0),
            gate=bool(raw.get("gate", True)),
            capture=[str(c) for c in capture] if isinstance(capture, list) else [],
            capture_stdout=bool(raw.get("capture_stdout", True)),
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
    #: Which stage this config is for: one of `STAGES`. It decides whether `metrics:` is the
    #: operator's (`run`) or the pass's (`baseline`), and what acceptance gates on.
    stage: str = "baseline"
    #: The `metrics:` entries, in `run`. Empty in `baseline`, where the pass supplies them.
    measurements: list[Measurement] = field(default_factory=list)
    #: The raw YAML with any command-less metric entry filled in, which is what the native
    #: `Config` was built from — `Config.validate` rejects an empty observable command, and here
    #: an empty one means "read it back from the gate".
    normalized: dict[str, Any] = field(default_factory=dict)

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
        Unbounded is a `baseline` notion only; `run` has a budget, so see `iterations`.
        """
        raw = (self.raw.get("budget") or {}).get("iterations")
        return int(raw) if raw else None

    @property
    def iterations(self) -> int:
        """How many iterations this stage runs.

        `run` is bounded: its iterations each pay for a full constraint evaluation, and the thing
        being measured is usually one of the constraints, so "until it stops improving" is not a
        budget anyone should discover by accident.
        """
        return self.max_iterations or DEFAULT_RUN_ITERATIONS

    @property
    def max_regression_pct(self) -> float:
        """How far a gated metric may regress before the iteration is rejected."""
        raw = (self.raw.get("acceptance") or {}).get("max_regression_pct")
        try:
            return DEFAULT_MAX_REGRESSION_PCT if raw is None else float(raw)
        except (TypeError, ValueError):
            return DEFAULT_MAX_REGRESSION_PCT

    @property
    def metric_names(self) -> list[str]:
        """Every metric the operator declared, in declaration order."""
        return [name for m in self.measurements for name in m.names]

    @property
    def gated_metric_names(self) -> list[str]:
        """The operator's metrics acceptance acts on."""
        return [name for m in self.measurements if m.gate for name in m.names]

    @property
    def iteration_constraints(self) -> Any:
        """The raw `iteration_constraints:` block, parsed by `integrate.schedule`."""
        return self.raw.get("iteration_constraints")

    @property
    def gate_json(self) -> Path:
        return self.project_path / GATE_JSON

    @property
    def loop_config_path(self) -> Path:
        """Where the derived loop config goes — one per stage.

        Separate files because the Harness records which config a run was started with and
        prompts to archive when it changes. Two stages sharing one path would look like the same
        run continuing, and `run` would inherit the baseline's iteration count and believe its
        budget was already spent.
        """
        name = "loop.yaml" if self.stage == "baseline" else f"{self.stage}-loop.yaml"
        return self.project_path / ".autohelix" / "integrate" / name

    @property
    def has_reviewer(self) -> bool:
        return self.config.reviewer is not None

    # ------------------------------------------------------------------ loading

    @classmethod
    def load(
        cls,
        project_path: str | Path,
        config_file: str | Path | None = None,
        stage: str = "baseline",
    ) -> "IntegrateConfig":
        """Parse the config, then read the constraint kinds and (in `run`) the metrics.

        The native `Config` is built from a *normalized* copy of the YAML rather than the file
        itself: in `run` a metric entry may omit its command, meaning "read it back from the
        gate", and `Config.validate` rejects an empty observable command. Filling it in here
        keeps the operator's file declarative and the shared validator honest — rather than
        teaching this pass to ignore one of the validator's errors.
        """
        if stage not in STAGES:
            raise ConfigError(f"unknown stage '{stage}' (expected one of {', '.join(STAGES)})")
        root = Path(project_path).resolve()
        default = DEFAULT_CONFIG_NAME if stage == "baseline" else RUN_CONFIG_NAME
        path = Path(config_file) if config_file else root / default
        try:
            raw = _read_yaml(path)
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

        measurements: list[Measurement] = []
        if stage == "run":
            raw_metrics = raw.get("metrics")
            if raw_metrics is None:
                raw_metrics = raw.get("observables") or []
            if isinstance(raw_metrics, dict):
                raw_metrics = [raw_metrics]
            if not isinstance(raw_metrics, list):
                raise ConfigError("`metrics:` is not a list")
            measurements = [Measurement.from_entry(e, i) for i, e in enumerate(raw_metrics)]

        out = cls(
            project_path=root,
            config=Config.from_dict(raw),
            raw=raw,
            constraints=constraints,
            stage=stage,
            measurements=measurements,
        )
        out.source = path
        # Built once the paths are known, since a filled-in readback command names the gate's
        # verdict file. The native config is then re-made from it, so what `Config.validate`
        # sees is what the Harness will run.
        out.normalized = out._normalize_raw()
        out.config = Config.from_dict(out.normalized)
        return out

    def _normalize_raw(self) -> dict[str, Any]:
        """The operator's YAML with command-less metric entries filled in."""
        if self.stage != "run" or not self.measurements:
            return dict(self.raw)
        payload = dict(self.raw)
        payload.pop("observables", None)
        payload["metrics"] = [self._metric_entry(m) for m in self.measurements]
        return payload

    def _metric_entry(self, measurement: Measurement) -> dict[str, Any]:
        """One metrics entry as the Harness should see it: always with a command."""
        entry: dict[str, Any] = {
            "command": measurement.command.strip() or self.readback_command(measurement.names),
            "values": dict(measurement.values),
            "timeout": measurement.timeout or (
                DEFAULT_SCRIPT_TIMEOUT if measurement.command.strip() else READBACK_TIMEOUT
            ),
        }
        if measurement.capture:
            entry["capture"] = list(measurement.capture)
        if not measurement.capture_stdout:
            entry["capture_stdout"] = False
        return entry

    def readback_command(self, names: list[str] | None = None) -> str:
        """The command that reports the gate's counts, and optionally named measurements.

        Relative `GATE_JSON`, like the gate's own `--json`: both run with the iteration's
        worktree as the working directory, so this reads the verdict that iteration produced
        rather than whatever was last written to the project.
        """
        extra = f" --metrics {','.join(names)}" if names else ""
        return f"{GATE_PYTHON} -m integrate.readback --json {GATE_JSON}{extra}"

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
            for issue in self.config.validate(self.normalized or self.raw)
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

        problems.extend(
            self._baseline_problems() if self.stage == "baseline" else self._run_problems()
        )

        seen: set[str] = set()
        for constraint in self.constraints:
            problems.extend(constraint.problems())
            if constraint.name in seen:
                problems.append(f"two constraints are both named '{constraint.name}'")
            seen.add(constraint.name)
        return problems

    def _baseline_problems(self) -> list[str]:
        """What is wrong with this config *as a baseline config*.

        The metric and the acceptance rule are the pass's there. A config that set either would
        be overriding the thing that makes an unbounded loop converge, so say so rather than
        quietly losing it in the derived file.
        """
        out: list[str] = []
        if self.raw.get("metrics") or self.raw.get("observables"):
            out.append(
                "`metrics:`/`observables:` is set, but the baseline stage supplies its own "
                f"metrics ({METRIC}, {METRIC_ALL}) and would overwrite them. Express what you "
                "wanted as a constraint, or write a `run` config — that stage is the one that "
                "takes measurements."
            )
        if (self.raw.get("acceptance") or {}).get("metric_gates"):
            out.append(
                "`acceptance.metric_gates:` is set, but the baseline stage supplies its own — "
                "refusing any iteration that satisfies fewer constraints than the best so far. "
                "Remove it."
            )
        if self.raw.get("iteration_constraints"):
            out.append(
                "`iteration_constraints:` is set, but the baseline stage has no schedule: it "
                "runs until the constraints hold, so there is no iteration count for slots to "
                "be indexed against. It belongs in a `run` config."
            )
        return out

    def _run_problems(self) -> list[str]:
        """What is wrong with this config *as a run config*."""
        out: list[str] = []
        if not self.measurements:
            out.append(
                "no `metrics:` — this stage keeps the constraints holding while making a "
                "measurement better, so with nothing measured it would accept every iteration "
                "that did not break anything. Declare at least one metric and its direction."
            )
        for index, measurement in enumerate(self.measurements):
            out.extend(measurement.problems(index))

        names = self.metric_names
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            out.append(
                f"metric name(s) declared twice: {', '.join(duplicates)}. A name is the loop's "
                "handle on a number, so two entries claiming one leaves the gate reading "
                "whichever ran last."
            )
        reserved = sorted(set(names) & {METRIC, METRIC_ALL})
        if reserved:
            out.append(
                f"metric name(s) {', '.join(reserved)} are this pass's own constraint counts. "
                "Rename yours."
            )
        if names and not self.gated_metric_names:
            out.append(
                "every metric is `gate: false`, so nothing can reject an iteration for "
                "regressing. At least one measurement has to count."
            )

        gates = (self.raw.get("acceptance") or {}).get("metric_gates") or []
        for gate in gates:
            if isinstance(gate, dict) and gate.get("metric") not in names:
                out.append(
                    f"`acceptance.metric_gates` names '{gate.get('metric')}', which no "
                    "`metrics:` entry produces"
                )
        pct = (self.raw.get("acceptance") or {}).get("max_regression_pct")
        if pct is not None:
            try:
                if float(pct) < 0:
                    out.append("`acceptance.max_regression_pct:` cannot be negative")
            except (TypeError, ValueError):
                out.append(f"`acceptance.max_regression_pct:` '{pct}' is not a number")

        from integrate.schedule import Schedule, ScheduleError

        try:
            Schedule.from_config(self.iteration_constraints, max_iterations=self.iterations)
        except ScheduleError as exc:
            out.append(f"`iteration_constraints:` {exc}")
        return out

    def warnings(self) -> list[str]:
        """Things worth saying that are not failures.

        AutoHelix's own warnings come first — unknown keys, a missing scope, anything it already
        knows to flag — then the ones specific to this pass.
        """
        out: list[str] = [
            issue.message
            for issue in self.config.validate(self.normalized or self.raw)
            if issue.level != "error"
        ]
        if self.stage == "run":
            from integrate.schedule import Schedule, ScheduleError

            try:
                schedule = Schedule.from_config(
                    self.iteration_constraints, max_iterations=self.iterations
                )
            except ScheduleError:
                schedule = Schedule()  # reported as an error by `validate`
            out.extend(schedule.validate(self.iterations))
            ungated = [n for m in self.measurements if not m.gate for n in m.names]
            if ungated:
                out.append(
                    f"recorded but never gated: {', '.join(ungated)} — a regression in one of "
                    "those cannot reject an iteration"
                )
            if not self.iteration_constraints:
                out.append(
                    "no `iteration_constraints:`, so every iteration gets the same prompt. That "
                    "is the mode where a run tends to repeat its first idea."
                )
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
        """The loop config the Harness is driven with: the operator's, with keys replaced.

        Everything the operator wrote carries through — `goal`, `scope`, `agent`, `budget`,
        `reviewer`, and in `run` the `iteration_constraints` schedule. What the pass replaces is
        `constraints` (one command, the gate), `metrics` and `acceptance`.
        """
        payload: dict[str, Any] = {
            k: v for k, v in self.raw.items()
            if k not in ("constraints", "constraint", "metrics", "observables", "acceptance")
        }

        # One command for the whole set, script and agent alike. It always exits 0 — it is a
        # measurement, not a gate, because here a failing check is the *normal* state and
        # rejecting on it would discard every iteration's progress before the last one. What
        # stops a regression is the metric gate below.
        #
        # `run` keeps it advisory for a different reason: its constraints *do* hold at the
        # baseline, but one of them is usually an agent judge, and a judge that flips over an
        # unchanged tree would otherwise throw away an iteration's work. The zero-tolerance gate
        # on the script count is what makes a real constraint loss cost the iteration.
        payload["constraints"] = [{
            "command": (
                f"{GATE_PYTHON} -m integrate.gate --config {self._config_ref()} "
                f"--json {GATE_JSON} --advisory"
            ),
            "timeout": self.gate_timeout(),
        }]
        payload["metrics"] = self._derive_metrics()
        payload["acceptance"] = {"metric_gates": self._derive_gates()}

        budget = dict(payload.get("budget") or {})
        budget["iterations"] = (
            (self.max_iterations or UNBOUNDED_ITERATIONS) if self.stage == "baseline"
            else self.iterations
        )
        payload["budget"] = budget
        return payload

    def _derive_metrics(self) -> list[dict[str, Any]]:
        """The metric commands the Harness runs after the gate.

        The gate already evaluated everything; re-running it would double each iteration's cost —
        for agent constraints, paying for every judge twice — and the two runs could disagree,
        leaving an iteration accepted on one count and ranked on another. So the first entry
        *reads back* the gate's verdict, and in `run` it reads the operator's measurements out of
        it too: a constraint that prints `##autohelix[...]` has already produced them.
        """
        from_constraints = [
            name for m in self.measurements if m.from_constraints for name in m.names
        ]
        counts: dict[str, Any] = {
            "command": self.readback_command(from_constraints),
            "values": {METRIC: "higher", METRIC_ALL: "higher"},
            "timeout": READBACK_TIMEOUT,
        }
        for measurement in self.measurements:
            if measurement.from_constraints:
                counts["values"].update(measurement.values)
        out = [counts]
        # Anything the operator gave its own command runs as an ordinary observable, which is
        # what `autohelix run` would have done with it.
        out.extend(
            self._metric_entry(m) for m in self.measurements if not m.from_constraints
        )
        return out

    def _derive_gates(self) -> list[dict[str, Any]]:
        """The acceptance gates: the constraint count, and in `run` each gated measurement.

        Zero tolerance on the script count, in both stages: an iteration holding fewer script
        constraints than the best so far is rejected and its work discarded. In `baseline` that
        is what turns "run until it passes" into something that converges rather than
        oscillates; in `run` it is what stops a faster iteration from being a broken one.
        `METRIC_ALL` is deliberately never gated — it moves with judge verdicts, which can
        differ between two runs over the same tree, and a flip there is not a regression to
        discard an iteration for. See `METRIC`.
        """
        gates: list[dict[str, Any]] = [{"metric": METRIC, "max_regression_pct": 0.0}]
        if self.stage != "run":
            return gates
        explicit = {
            str(g.get("metric")): float(g.get("max_regression_pct", self.max_regression_pct))
            for g in ((self.raw.get("acceptance") or {}).get("metric_gates") or [])
            if isinstance(g, dict) and g.get("metric")
        }
        for name in self.gated_metric_names:
            gates.append({
                "metric": name,
                "max_regression_pct": explicit.get(name, self.max_regression_pct),
            })
        return gates

    def _config_ref(self) -> str:
        """How the derived gate command refers back to this config.

        Absolute: the gate runs with the iteration's worktree as its working directory, so a
        relative path would resolve inside a tree that does not hold the operator's config.
        """
        default = DEFAULT_CONFIG_NAME if self.stage == "baseline" else RUN_CONFIG_NAME
        return str((self.source or Path(default)).resolve())

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
