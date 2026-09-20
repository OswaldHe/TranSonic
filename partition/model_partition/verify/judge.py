# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM judge for emulated-inference output.

Uses the same transport as AutoHelix's Claude backend — the ``claude`` CLI with
``--print --output-format json`` — so Bedrock routing comes from the inherited
environment (``CLAUDE_CODE_USE_BEDROCK``) with no extra dependency or credential
handling. :class:`StubJudge` keeps tests offline.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Protocol

DEFAULT_TIMEOUT = 240
DEFAULT_MIN_SCORE = 4

JUDGE_PROMPT = """\
You are judging whether a language model's continuation is coherent.

The continuation was produced by a model reassembled from partitioned weights, so
the question is whether it reads like fluent, on-topic text — not whether it is
factually correct or complete. Truncation at the end is expected and fine.

PROMPT:
{prompt}

CONTINUATION:
{continuation}

Reply with ONLY a JSON object, no other text:
{{"fluent": true|false, "score": 1-5, "reason": "<one sentence>"}}

score 5 = fluent and clearly on-topic; 3 = readable but odd or repetitive;
1 = gibberish, degenerate repetition, or unrelated tokens.
"""


class JudgeError(RuntimeError):
    """Raised when a judge cannot produce a verdict."""


@dataclass
class Verdict:
    """A judge's assessment of one continuation."""

    fluent: bool
    score: int
    reason: str = ""
    sample_id: str = ""
    raw: str = ""
    error: str = ""

    def passed(self, min_score: int = DEFAULT_MIN_SCORE) -> bool:
        return self.fluent and self.score >= min_score and not self.error

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id, "fluent": self.fluent, "score": self.score,
            "reason": self.reason, "error": self.error,
        }


class Judge(Protocol):
    def judge(self, prompt: str, continuation: str, sample_id: str = "") -> Verdict: ...


def parse_verdict(text: str, sample_id: str = "") -> Verdict:
    """Extract a verdict from a model reply, tolerating surrounding prose."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return Verdict(fluent=False, score=0, sample_id=sample_id, raw=text,
                       error="no JSON object in judge reply")
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return Verdict(fluent=False, score=0, sample_id=sample_id, raw=text,
                       error=f"invalid JSON in judge reply: {exc}")
    if not isinstance(payload, dict):
        return Verdict(fluent=False, score=0, sample_id=sample_id, raw=text,
                       error="judge reply JSON is not an object")
    try:
        score = int(payload.get("score", 0))
    except (TypeError, ValueError):
        score = 0
    return Verdict(
        fluent=bool(payload.get("fluent", False)),
        score=max(0, min(score, 5)),
        reason=str(payload.get("reason", ""))[:500],
        sample_id=sample_id,
        raw=text,
    )


@dataclass
class ClaudeJudge:
    """Judge backed by the ``claude`` CLI."""

    model: str | None = None
    timeout: int = DEFAULT_TIMEOUT
    command: str | None = None
    extra_args: list[str] = field(default_factory=list)

    def _command_path(self) -> str:
        return self.command or os.environ.get("AUTOHELIX_CLAUDE_CMD") or shutil.which("claude") or "claude"

    def _model(self) -> str:
        return (self.model
                or os.environ.get("MODEL_PARTITION_JUDGE_MODEL")
                or os.environ.get("ANTHROPIC_DEFAULT_SONNET_MODEL")
                or "sonnet")

    def available(self) -> bool:
        return shutil.which(self._command_path()) is not None or os.path.exists(self._command_path())

    def judge(self, prompt: str, continuation: str, sample_id: str = "") -> Verdict:
        if not self.available():
            return Verdict(fluent=False, score=0, sample_id=sample_id,
                           error="claude CLI not found on PATH")
        body = JUDGE_PROMPT.format(prompt=prompt[-4000:], continuation=continuation[:4000])
        command = [
            self._command_path(), "--print", "--output-format", "json",
            "--model", self._model(), *self.extra_args, body,
        ]
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=self.timeout,
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            return Verdict(fluent=False, score=0, sample_id=sample_id,
                           error=f"judge timed out after {self.timeout}s")
        except OSError as exc:
            return Verdict(fluent=False, score=0, sample_id=sample_id, error=f"judge failed: {exc}")

        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()[:400]
            return Verdict(fluent=False, score=0, sample_id=sample_id,
                           error=f"claude exited {completed.returncode}: {detail}")
        return parse_verdict(_extract_text(completed.stdout), sample_id=sample_id)


def _extract_text(stdout: str) -> str:
    """Pull the assistant text out of ``--output-format json`` output."""
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return stdout
    if isinstance(payload, dict):
        for key in ("result", "text", "content", "message"):
            value = payload.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, list):
                parts = [p.get("text", "") for p in value if isinstance(p, dict)]
                if parts:
                    return "".join(parts)
    return stdout


@dataclass
class StubJudge:
    """Offline judge: flags degenerate output without a network call.

    Used by tests and by ``--no-judge`` runs. The heuristics are deliberately
    crude — they catch the failure mode that matters (a broken partition emitting
    repeated or empty tokens) without pretending to assess fluency.
    """

    min_unique_ratio: float = 0.3
    min_length: int = 2

    def judge(self, prompt: str, continuation: str, sample_id: str = "") -> Verdict:
        tokens = continuation.split()
        if len(tokens) < self.min_length:
            return Verdict(fluent=False, score=1, reason="continuation too short",
                           sample_id=sample_id)
        unique_ratio = len(set(tokens)) / len(tokens)
        if unique_ratio < self.min_unique_ratio:
            return Verdict(fluent=False, score=1, sample_id=sample_id,
                           reason=f"degenerate repetition (unique ratio {unique_ratio:.2f})")
        return Verdict(fluent=True, score=4, sample_id=sample_id,
                       reason=f"stub judge: {len(tokens)} tokens, unique ratio {unique_ratio:.2f}")


def build_judge(kind: str = "claude", **kwargs: Any) -> Judge:
    if kind == "claude":
        return ClaudeJudge(**kwargs)
    if kind == "stub":
        return StubJudge()
    raise JudgeError(f"Unknown judge kind {kind!r}")
