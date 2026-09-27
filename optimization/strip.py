# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Removing an earlier agent's prose from the code a later agent reads.

`source.py` and `inference.py` in a bootstrapped repo were written by an agent, and their comments
and docstrings are that agent's *claims* — about the hardware, the compiler, why a tile is the size
it is. Some are hard-won and right. Some are wrong, and the wrong ones are indistinguishable from the
right ones at a glance, which is the problem: a claim in a comment reads as established fact to the
next agent, and it will design around it without testing it.

This repository has already produced one worked example of exactly that failure. `floorplan/README.md`
states that NKI 0.6.0 exposes no collective primitive, and the floorplan's intra-device bandwidth is
*derived* rather than measured on the strength of it. The claim is false — the collectives are in
`nki.collectives`, absent from `nl`/`nisa` — and it went unchallenged because it was written down
confidently. A comment inside a kernel is the same hazard with less visibility.

So the copies carried into an optimization repo are stripped. The code is the fact; the commentary
about the code is not. Anything genuinely load-bearing survives as behaviour that the validator
checks, and anything that was only ever an opinion is gone.

What is *not* stripped: `reference_torch.py`, `reference_numerics.py`, `reference_inference.py`,
`vendor/` and `compat/`. Those are the vendor's and the harness's own code, carried in verbatim as the
specification, and their comments are authoritative rather than inferred. Stripping them would delete
the thing the agent is supposed to read.
"""

from __future__ import annotations

import ast
import io
import tokenize
from dataclasses import dataclass
from pathlib import Path

#: The files this applies to, by name. Agent-authored, in every repo this pipeline builds.
#: Deliberately a name list rather than a directory walk: `reference_*.py` and `vendor/` must be
#: left alone, and matching on the two names an agent writes is the narrowest rule that does that.
STRIPPED_FILES = ("source.py", "inference.py")

#: Prepended to a stripped file. Without it a 448-line kernel with no commentary looks accidental,
#: and an agent may spend effort restoring what it thinks was lost. Says what happened and nothing
#: about the hardware, so it cannot itself become the next wrong fact.
BANNER = (
    "# Comments and docstrings were removed from this copy on purpose: they were written by an\n"
    "# earlier agent and are its claims, not established fact. The original repository keeps them.\n"
)


class StripError(RuntimeError):
    """Stripping produced something that does not parse, so the original is kept."""


@dataclass
class StripResult:
    """What stripping one file removed."""

    path: Path
    comments: int
    docstrings: int
    lines_before: int
    lines_after: int
    skipped: str = ""

    @property
    def ok(self) -> bool:
        return not self.skipped

    def describe(self) -> str:
        if self.skipped:
            return f"{self.path.name}: left as-is ({self.skipped})"
        return (
            f"{self.path.name}: removed {self.comments} comment(s) and "
            f"{self.docstrings} docstring(s), {self.lines_before} -> {self.lines_after} lines"
        )


def _docstring_line_ranges(tree: ast.Module) -> list[tuple[int, int]]:
    """1-based inclusive line ranges of every docstring statement.

    Collected as statements rather than as string constants so the whole `Expr` goes, leaving no
    orphaned expression behind.
    """
    ranges: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if not (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            continue
        end = getattr(first, "end_lineno", None) or first.lineno
        # A function or class whose *only* statement is its docstring needs a body left behind.
        # `pass` is substituted at the docstring's own indentation rather than the line being
        # deleted, so the result still parses.
        needs_pass = len(body) == 1 and not isinstance(node, ast.Module)
        ranges.append((first.lineno, end, needs_pass))  # type: ignore[arg-type]
    return ranges  # type: ignore[return-value]


def strip_source(text: str) -> tuple[str, int, int]:
    """Return ``text`` with comments and docstrings removed, plus how many of each there were.

    Two passes, because the two live at different levels. Comments are lexical, so they come out
    through `tokenize`, which is the only way to tell a `#` in code from a `#` inside a string.
    Docstrings are syntactic, so they come out through `ast` line ranges.

    Raises :class:`StripError` if the result does not parse — the caller then keeps the original,
    because shipping code that an agent cannot run is worse than shipping code it might misread.
    """
    original = ast.parse(text)  # raises SyntaxError if the input was already broken

    # -- docstrings, by line, before any renumbering happens ---------------------------
    drop: dict[int, str | None] = {}
    docstrings = 0
    for start, end, needs_pass in _docstring_line_ranges(original):  # type: ignore[misc]
        docstrings += 1
        for line in range(start, end + 1):
            drop[line] = None
        if needs_pass:
            drop[start] = "pass"

    lines = text.splitlines(keepends=True)
    kept: list[str] = []
    for number, line in enumerate(lines, start=1):
        if number not in drop:
            kept.append(line)
            continue
        replacement = drop[number]
        if replacement is not None:
            indent = line[: len(line) - len(line.lstrip())]
            kept.append(f"{indent}{replacement}\n")
    without_docstrings = "".join(kept)

    # -- comments, by position ----------------------------------------------------------
    # Truncating each line at the comment's start column rather than rebuilding the file from
    # tokens. `tokenize.untokenize` does not preserve spacing — it renders `import nki.language`
    # as `import nki .language` and `DIM = 5120` as `DIM =5120` — and handing an agent code that
    # looks mechanically mangled invites it to "fix" the formatting instead of the kernel.
    #
    # A Python comment runs to end of line, so there is at most one per line and the column is
    # all that is needed. `tokenize` is still what finds them, because it is the only thing that
    # can tell a `#` in code from a `#` inside a string.
    tokens = list(tokenize.generate_tokens(io.StringIO(without_docstrings).readline))
    starts: dict[int, int] = {}
    for token in tokens:
        if token.type == tokenize.COMMENT:
            row, column = token.start
            starts.setdefault(row, column)
    comments = len(starts)

    rebuilt: list[str] = []
    for number, line in enumerate(without_docstrings.splitlines(), start=1):
        if number not in starts:
            rebuilt.append(line)
            continue
        head = line[: starts[number]].rstrip()
        if head:
            rebuilt.append(head)  # a trailing comment on a line of code
        # otherwise the line was only a comment, and it goes entirely

    # A removed comment block leaves a hole. Collapse runs of three or more blank lines to two,
    # which is what the code looked like before the block was written.
    collapsed: list[str] = []
    blanks = 0
    for line in rebuilt:
        if line.strip():
            blanks = 0
            collapsed.append(line)
        else:
            blanks += 1
            if blanks <= 2:
                collapsed.append("")
    result = "\n".join(collapsed).rstrip() + "\n"

    try:
        ast.parse(result)
    except SyntaxError as exc:
        raise StripError(f"stripping produced code that does not parse: {exc}") from exc
    return result, comments, docstrings


def strip_file(path: Path, banner: bool = True) -> StripResult:
    """Strip one file in place. A file that cannot be stripped safely is left untouched."""
    if not path.is_file():
        return StripResult(path, 0, 0, 0, 0, skipped="not present")
    text = path.read_text()
    before = len(text.splitlines())
    try:
        stripped, comments, docstrings = strip_source(text)
    except (SyntaxError, StripError, tokenize.TokenError) as exc:
        # Loud in the caller's log, but not fatal: the point of stripping is to remove a hazard,
        # and failing the whole stage over it would trade a small hazard for a blocked run.
        return StripResult(path, 0, 0, before, before, skipped=str(exc))
    if banner:
        stripped = BANNER + stripped
    path.write_text(stripped)
    return StripResult(path, comments, docstrings, before, len(stripped.splitlines()))


def strip_tree(root: Path, names: tuple[str, ...] = STRIPPED_FILES) -> list[StripResult]:
    """Strip every `source.py` and `inference.py` at or under ``root``.

    Used on the `module/` and `submodule/` directories a repo carries. `reference/` holds no file
    with either name, so it is excluded by the name rule rather than by a path exception — which
    means adding a directory to a repo cannot accidentally expose it.
    """
    results: list[StripResult] = []
    for name in names:
        for path in sorted(root.rglob(name)):
            results.append(strip_file(path))
    return results
