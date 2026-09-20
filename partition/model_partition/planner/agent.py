# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Claude-driven planning and repair.

Reuses AutoHelix's own Claude backend (``ClaudeCodeAgent``), so the transport,
flags and Bedrock routing are identical to the rest of the project — the agent
inherits ``CLAUDE_CODE_USE_BEDROCK`` from the environment.

The agent's editable surface is the partition graph. Artifacts are never at risk:
the previous plan is snapshotted, and a proposal that fails validation is rolled
back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from model_partition.planner.graph import GraphError, PartitionGraph

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


class AgentPlannerError(RuntimeError):
    """Raised when the agent backend cannot be run."""


@dataclass
class AgentOutcome:
    """Result of one agent invocation."""

    ok: bool
    text: str = ""
    error: str = ""
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def cost_usd(self) -> float:
        value = self.usage.get("cost_usd") or self.usage.get("total_cost_usd") or 0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0


def render_prompt(name: str, **context: Any) -> str:
    """Render a prompt template from ``prompts/``."""
    from jinja2 import Environment, FileSystemLoader, StrictUndefined

    env = Environment(loader=FileSystemLoader(str(PROMPTS_DIR)), undefined=StrictUndefined,
                      keep_trailing_newline=True)
    return env.get_template(name).render(**context)


@dataclass
class AgentPlanner:
    """Runs Claude in a working directory to revise the partition plan."""

    model: str | None = None
    timeout_seconds: int = 1800
    reasoning_effort: str | None = None
    command: str | None = None

    def _agent(self):
        try:
            from autohelix.agents import AgentConfig
            from autohelix.agents.claudecode import ClaudeCodeAgent
        except ImportError as exc:  # pragma: no cover
            raise AgentPlannerError("autohelix is required for the agent planner") from exc
        return ClaudeCodeAgent(AgentConfig(
            type="claude", command=self.command, model=self.model,
            reasoning_effort=self.reasoning_effort,
            timeout_seconds=self.timeout_seconds, auto_memory=False,
        ))

    def invoke(self, workdir: Path, prompt: str, iteration: int, log_path: Path) -> AgentOutcome:
        """Run the agent with ``workdir`` as its working directory."""
        agent = self._agent()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        collected: list[str] = []

        def on_event(event) -> None:
            text = getattr(event, "text", "") or ""
            if text and not (getattr(event, "metadata", {}) or {}).get("stream"):
                collected.append(text)

        try:
            result = agent.run(
                worktree_path=workdir, prompt=prompt, iteration=iteration,
                log_path=log_path, event_callback=on_event, project_path=None,
            )
        except Exception as exc:
            return AgentOutcome(ok=False, error=f"agent run failed: {exc}")

        error = getattr(result, "error", None)
        return AgentOutcome(
            ok=not error,
            text="\n".join(collected)[-20000:],
            error=str(error or ""),
            usage=dict(getattr(result, "usage", {}) or {}),
        )

    # -- high-level operations -------------------------------------------------

    def refine_plan(
        self,
        layout,
        graph: PartitionGraph,
        context: dict[str, Any],
        iteration: int,
    ) -> tuple[AgentOutcome, PartitionGraph]:
        """Ask the agent to improve the plan for kernel-development convenience."""
        prompt = render_prompt("planner.md", graph=graph, **context)
        return self._edit_plan(layout, graph, prompt, iteration, "plan")

    def repair_plan(
        self,
        layout,
        graph: PartitionGraph,
        context: dict[str, Any],
        iteration: int,
    ) -> tuple[AgentOutcome, PartitionGraph]:
        """Ask the agent to fix a plan whose verification failed."""
        prompt = render_prompt("repair.md", graph=graph, **context)
        return self._edit_plan(layout, graph, prompt, iteration, "repair")

    def _edit_plan(
        self, layout, graph: PartitionGraph, prompt: str, iteration: int, tag: str,
    ) -> tuple[AgentOutcome, PartitionGraph]:
        snapshot = snapshot_plan(layout, iteration)
        outcome = self.invoke(
            workdir=layout.root, prompt=prompt, iteration=iteration,
            log_path=layout.root / "logs" / f"agent-{tag}-{iteration}.log",
        )
        try:
            revised = PartitionGraph.load(layout.graph_path)
            revised.validate(graph.budget_bytes)
        except (GraphError, OSError) as exc:
            restore_plan(layout, snapshot)
            return AgentOutcome(
                ok=False, text=outcome.text, usage=outcome.usage,
                error=f"agent produced an invalid plan, rolled back: {exc}",
            ), graph
        return outcome, revised


def snapshot_plan(layout, iteration: int) -> Path | None:
    """Copy the current plan aside so a bad edit can be undone."""
    if not layout.graph_path.is_file():
        return None
    history = layout.plan_dir / "history"
    history.mkdir(parents=True, exist_ok=True)
    target = history / f"iter-{iteration}.yaml"
    target.write_text(layout.graph_path.read_text())
    return target


def restore_plan(layout, snapshot: Path | None) -> bool:
    if snapshot is None or not snapshot.is_file():
        return False
    layout.graph_path.write_text(snapshot.read_text())
    return True
