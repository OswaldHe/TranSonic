# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Claude-driven review, planning and repair.

Reuses AutoHelix's own Claude backend (``ClaudeCodeAgent``), so the transport,
flags and Bedrock routing are identical to the rest of the project — the agent
inherits ``CLAUDE_CODE_USE_BEDROCK`` from the environment.

Two roles, as in AutoHelix's own loop. A **reviewer** runs only when something
failed: it diagnoses from the artifacts and writes ``reports/review.md``, editing
nothing. The **planner** then makes the change, with that review as its brief.
Splitting them keeps diagnosis honest — the agent that has to produce a fix is
the wrong one to decide what went wrong.

The editable surface is the partition graph and the module implementations.
Everything else under the run directory is snapshotted around each call and put
back, so an agent cannot weaken the check it has to pass.
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
    #: What the guard had to put back, if anything.
    guard: Any = None

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
        """Run the agent with ``workdir`` as its working directory.

        Harness-owned files under ``workdir`` are snapshotted first and restored
        afterwards. ``reports/review.md`` is deliberately not among them, so the
        reviewer can write its diagnosis under the same guard as everyone else.
        """
        from model_partition.loop.guard import HarnessGuard

        agent = self._agent()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        collected: list[str] = []

        def on_event(event) -> None:
            text = getattr(event, "text", "") or ""
            if text and not (getattr(event, "metadata", {}) or {}).get("stream"):
                collected.append(text)

        snapshot = HarnessGuard.capture(workdir)
        try:
            result = agent.run(
                worktree_path=workdir, prompt=prompt, iteration=iteration,
                log_path=log_path, event_callback=on_event, project_path=None,
            )
        except Exception as exc:
            return AgentOutcome(ok=False, error=f"agent run failed: {exc}",
                                guard=snapshot.restore())

        report = snapshot.restore()
        error = getattr(result, "error", None)
        if report.tampered:
            error = (f"agent modified {len(report.tampered)} reference tensor(s), which "
                     f"cannot be restored: {', '.join(report.tampered[:4])}")
        return AgentOutcome(
            ok=not error,
            text="\n".join(collected)[-20000:],
            error=str(error or ""),
            usage=dict(getattr(result, "usage", {}) or {}),
            guard=report,
        )

    # -- high-level operations -------------------------------------------------

    def review(self, layout, context: dict[str, Any], iteration: int) -> AgentOutcome:
        """Diagnose a failure and write ``reports/review.md``. Changes nothing else.

        The review is the brief the next agent works from, so it is kept even when
        the run later stops: ``reports/reviews/iter-N.md`` holds the trail.
        """
        layout.review_file.parent.mkdir(parents=True, exist_ok=True)
        layout.review_file.unlink(missing_ok=True)
        prompt = render_prompt("reviewer.md", **context)
        outcome = self.invoke(
            workdir=layout.root, prompt=prompt, iteration=iteration,
            log_path=layout.root / "logs" / f"agent-review-{iteration}.log",
        )
        if not layout.review_file.is_file():
            return AgentOutcome(
                ok=False, text=outcome.text, usage=outcome.usage,
                error=outcome.error or f"reviewer wrote no {layout.review_file.name}",
            )
        layout.archive_review(iteration)
        return outcome

    def edit_plan(
        self,
        layout,
        graph: PartitionGraph,
        context: dict[str, Any],
        iteration: int,
        tag: str = "plan",
    ) -> tuple[AgentOutcome, PartitionGraph]:
        """Ask the agent to edit the plan.

        One prompt for both jobs. Refining a sound plan and fixing a broken one are
        the same edit to the same file against the same constraints; what differs is
        the brief, which arrives as the failure context and the reviewer's report.
        """
        prompt = render_prompt("planner.md", graph=graph, **context)
        return self._edit_plan(layout, graph, prompt, iteration, tag)

    def repair_modules(
        self,
        layout,
        context: dict[str, Any],
        iteration: int,
    ) -> AgentOutcome:
        """Ask the agent to fix a module's inference code.

        The editable surface here is ``modules/*/inference.py`` — not the plan and
        not the verifier. Implementations are snapshotted so a proposal that makes
        things worse can be rolled back, and :meth:`invoke` puts back anything the
        agent touched outside that surface.
        """
        prompt = render_prompt("repair_module.md", **context)
        snapshot = snapshot_impls(layout, iteration)
        outcome = self.invoke(
            workdir=layout.root, prompt=prompt, iteration=iteration,
            log_path=layout.root / "logs" / f"agent-module-{iteration}.log",
        )
        if not outcome.ok:
            restore_impls(layout, snapshot)
        return outcome

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


def snapshot_impls(layout, iteration: int) -> Path | None:
    """Copy every extracted implementation aside before the agent edits them."""
    import shutil

    sources = sorted(layout.modules_dir.glob("*/inference.py"))
    if not sources:
        return None
    target = layout.modules_dir / "history" / f"iter-{iteration}"
    target.mkdir(parents=True, exist_ok=True)
    for source in sources:
        shutil.copy2(source, target / f"{source.parent.name}.py")
    return target


def restore_impls(layout, snapshot: Path | None) -> int:
    """Put snapshotted implementations back; returns how many were restored."""
    import shutil

    if snapshot is None or not snapshot.is_dir():
        return 0
    restored = 0
    for saved in snapshot.glob("*.py"):
        destination = layout.modules_dir / saved.stem / "inference.py"
        if destination.parent.is_dir():
            shutil.copy2(saved, destination)
            restored += 1
    return restored


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
