# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sample inputs: load JSONL prompts and tokenize them to fixed budgets."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from model_partition.storage import LONG_THRESHOLD_TOKENS


class InputError(RuntimeError):
    """Raised when an input set is missing or malformed."""


@dataclass
class SampleInput:
    """One tokenized prompt."""

    id: str
    prompt: str
    token_ids: list[int] = field(default_factory=list)
    role: str = "raw"
    source: str = ""
    target_tokens: int | None = None

    @property
    def n_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def is_long(self) -> bool:
        return self.n_tokens >= LONG_THRESHOLD_TOKENS

    def tensor(self, device: str = "cpu"):
        import torch

        return torch.tensor([self.token_ids], dtype=torch.long, device=device)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "n_tokens": self.n_tokens, "role": self.role,
            "source": self.source, "target_tokens": self.target_tokens,
            "prompt_preview": self.prompt[:200],
        }


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file, reporting the offending line on a parse error."""
    file_path = Path(path)
    if not file_path.is_file():
        raise InputError(f"Input file not found: {file_path}")
    records: list[dict[str, Any]] = []
    for lineno, line in enumerate(file_path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise InputError(f"{file_path}:{lineno} is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise InputError(f"{file_path}:{lineno} must be a JSON object")
        records.append(payload)
    return records


def tokenize_samples(
    records: list[dict[str, Any]],
    tokenizer: Any,
    limit: int | None = None,
    pad_to_target: bool = True,
) -> list[SampleInput]:
    """Tokenize records, trimming or repeating text to hit ``target_tokens``.

    A record may declare ``target_tokens`` to pin a context length; text is
    truncated to it, or repeated up to it when ``pad_to_target`` is set, so long
    samples land on the intended budget instead of whatever the source happened
    to be.
    """
    samples: list[SampleInput] = []
    for index, record in enumerate(records):
        if limit is not None and len(samples) >= limit:
            break
        prompt = record.get("prompt") or record.get("text") or ""
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        role = str(record.get("role", "raw"))
        target = record.get("target_tokens")
        target = int(target) if isinstance(target, (int, float)) else None

        ids = _encode(tokenizer, prompt, role)
        if target:
            if len(ids) > target:
                ids = ids[:target]
            elif pad_to_target and len(ids) < target and len(ids) > 0:
                repeats = (target // len(ids)) + 1
                ids = _encode(tokenizer, (prompt + "\n") * repeats, role)[:target]
        samples.append(SampleInput(
            id=str(record.get("id") or f"sample-{index:03d}"),
            prompt=prompt, token_ids=list(ids), role=role,
            source=str(record.get("source", "")), target_tokens=target,
        ))
    return samples


def _encode(tokenizer: Any, text: str, role: str) -> list[int]:
    try:
        return list(tokenizer.encode(text, role=role))
    except TypeError:
        return list(tokenizer.encode(text))


def load_input_set(
    short_path: str | Path | None,
    long_path: str | Path | None,
    tokenizer: Any,
    max_short: int | None = None,
    max_long: int | None = None,
) -> list[SampleInput]:
    """Load and tokenize the short and long input sets."""
    samples: list[SampleInput] = []
    if short_path:
        samples.extend(tokenize_samples(read_jsonl(short_path), tokenizer, limit=max_short))
    if long_path and Path(long_path).is_file():
        samples.extend(tokenize_samples(read_jsonl(long_path), tokenizer, limit=max_long))
    if not samples:
        raise InputError("Input set is empty; nothing to trace")
    return samples


def summarize(samples: list[SampleInput]) -> str:
    short = [s for s in samples if not s.is_long]
    long = [s for s in samples if s.is_long]
    parts = [f"{len(samples)} sample(s)"]
    if short:
        parts.append(f"short: {len(short)} ({min(s.n_tokens for s in short)}"
                     f"-{max(s.n_tokens for s in short)} tokens)")
    if long:
        parts.append(f"long: {len(long)} ({min(s.n_tokens for s in long)}"
                     f"-{max(s.n_tokens for s in long)} tokens)")
    return ", ".join(parts)
