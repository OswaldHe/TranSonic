# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tokenizer loading with fallbacks.

Tries transformers' AutoTokenizer, then a bare ``tokenizer.json``, then a
vendor-supplied encoder module (DeepSeek ships ``encoding/encoding.py``). Falls
back to a byte tokenizer so a model without one can still be traced.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class TokenizerError(RuntimeError):
    """Raised when text cannot be tokenized."""


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...


@dataclass
class HFTokenizer:
    """Wraps a transformers tokenizer, applying a chat template when present."""

    impl: Any
    use_chat_template: bool = True
    kind: str = "transformers"

    def encode(self, text: str, role: str = "raw") -> list[int]:
        if self.use_chat_template and role == "user" and getattr(self.impl, "chat_template", None):
            return list(self.impl.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True, tokenize=True,
            ))
        return list(self.impl.encode(text, add_special_tokens=True))

    def decode(self, ids: list[int]) -> str:
        return self.impl.decode(ids, skip_special_tokens=True)

    @property
    def eos_token_id(self) -> int | None:
        return getattr(self.impl, "eos_token_id", None)


@dataclass
class RawTokenizer:
    """Wraps a ``tokenizers.Tokenizer`` loaded straight from tokenizer.json."""

    impl: Any
    kind: str = "tokenizers"
    eos_token_id: int | None = None

    def encode(self, text: str, role: str = "raw") -> list[int]:
        return list(self.impl.encode(text).ids)

    def decode(self, ids: list[int]) -> str:
        return self.impl.decode(ids)


@dataclass
class ByteTokenizer:
    """UTF-8 byte fallback, so tracing never blocks on a missing tokenizer."""

    vocab_size: int = 256
    kind: str = "bytes"
    eos_token_id: int | None = None

    def encode(self, text: str, role: str = "raw") -> list[int]:
        return [b % self.vocab_size for b in text.encode("utf-8")]

    def decode(self, ids: list[int]) -> str:
        return bytes(i % 256 for i in ids).decode("utf-8", errors="replace")


def load_tokenizer(root: str | Path, trust_remote_code: bool = False,
                   vocab_size: int | None = None) -> Tokenizer:
    """Load the best available tokenizer for a model directory."""
    directory = Path(root)
    try:
        from transformers import AutoTokenizer

        return HFTokenizer(AutoTokenizer.from_pretrained(
            str(directory), trust_remote_code=trust_remote_code,
        ))
    except Exception:
        pass

    tokenizer_json = directory / "tokenizer.json"
    if tokenizer_json.is_file():
        try:
            from tokenizers import Tokenizer as RawImpl

            return RawTokenizer(RawImpl.from_file(str(tokenizer_json)))
        except Exception:
            pass

    return ByteTokenizer(vocab_size=vocab_size or 256)
