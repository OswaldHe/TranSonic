#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A PreToolUse hook that refuses a Bash command which would signal the agent itself.

The agent is started with its whole prompt as the last argv element of the `claude`
process (`agents/claudecode.py`), and the prompt quotes every constraint's command line. So
`pgrep -f <anything in the prompt>` matches the agent's own process, and piping that into
`kill` ends the iteration — three hours of work discarded, with the harness recording
`agent complete — 0s, $0.0000` and a bare `exited with status -9`. That is not hypothetical:
iteration 5 of the vllm-neuron integrate run died exactly this way at 08:04:53Z on
2026-10-11, cleaning up a hung `run_verifier`, five seconds after `pgrep -f "run_verifier
--model" | wc -l` had told it the pattern matched two processes.

An agent cleaning up a hung device process has no way to see this from inside. So the check
lives here, where the process tree is visible: walk our own ancestors — the hook runs as a
child of `claude`, which is a child of the harness — and refuse the command if it names one
of them, by pid or by a `-f` pattern that matches its command line.

Fails open. A guard that blocks real work because it crashed on an unusual command line is
worse than the thing it guards against, so every error path exits 0.
"""

from __future__ import annotations

import json
import os
import re
import sys

#: `pkill`/`pgrep` with a `-f` pattern: the whole command line is matched, which is what makes
#: the prompt in our own argv reachable.
PATTERN_RE = re.compile(
    r"\b(?:pkill|pgrep)\b(?:\s+-[^\s]+)*\s+(?:\"([^\"]+)\"|'([^']+)'|([^\s;|&]+))"
)
#: A signal sent to literal pids: `kill -9 1234`, `kill 1234 5678`.
PID_RE = re.compile(r"\bkill(?:all)?\b((?:\s+-[^\s]+)*(?:\s+\d+)+)")
#: `pkill`/`killall`/`pgrep` matching on the executable name rather than the command line.
#: `pgrep` is in here because `pgrep -x python | xargs kill` is the same mistake by another
#: route; a bare `pgrep` with no `kill` anywhere in the command never reaches this.
NAME_RE = re.compile(r"\b(?:pkill|killall|pgrep)\b((?:\s+-[^\s]+)*)\s+([^\s;|&]+)")


def ancestors() -> list[tuple[int, str, str]]:
    """`(pid, comm, cmdline)` for every process between this hook and pid 1."""
    out: list[tuple[int, str, str]] = []
    pid = os.getppid()
    seen: set[int] = set()
    while pid and pid > 1 and pid not in seen:
        seen.add(pid)
        try:
            with open(f"/proc/{pid}/stat", "rb") as fh:
                stat = fh.read().decode("utf-8", "replace")
            # comm is parenthesised and may itself contain spaces or parens.
            close = stat.rindex(")")
            comm = stat[stat.index("(") + 1 : close]
            ppid = int(stat[close + 2 :].split()[1])
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                cmdline = fh.read().decode("utf-8", "replace").replace("\0", " ")
        except (OSError, ValueError, IndexError):
            break
        out.append((pid, comm, cmdline))
        pid = ppid
    return out


def offence(command: str, tree: list[tuple[int, str, str]]) -> str | None:
    """Why this command would signal one of our own processes, or `None` if it would not."""
    kills = bool(re.search(r"\bkill(?:all)?\b|\bpkill\b", command))
    if not kills:
        return None

    for match in PID_RE.finditer(command):
        for token in match.group(1).split():
            if not token.isdigit():
                continue
            for pid, comm, _ in tree:
                if int(token) == pid:
                    return (
                        f"`kill {token}` targets pid {pid} ({comm}), which is this agent's own "
                        f"process tree."
                    )

    for match in PATTERN_RE.finditer(command):
        flags_and_pattern = match.group(0)
        if " -f" not in flags_and_pattern and not re.search(r"-\w*f", flags_and_pattern):
            continue
        pattern = match.group(1) or match.group(2) or match.group(3)
        if not pattern:
            continue
        try:
            matcher = re.compile(pattern)
            hit = lambda text: matcher.search(text) is not None  # noqa: E731
        except re.error:
            hit = lambda text: pattern in text  # noqa: E731
        for pid, comm, cmdline in tree:
            if hit(cmdline):
                return (
                    f"the pattern {pattern!r} matches pid {pid} ({comm}), which is this agent's "
                    f"own process tree — the whole prompt is in its argv, so a `-f` pattern "
                    f"drawn from the prompt matches it."
                )

    for match in NAME_RE.finditer(command):
        flags, name = match.group(1) or "", match.group(2)
        if re.search(r"-\w*f", flags):
            continue  # handled above, against the full command line
        for pid, comm, cmdline in tree:
            argv0 = os.path.basename(cmdline.split()[0]) if cmdline.strip() else ""
            # `python` vs `python3` vs `python3.12`, `node` vs `node22`: either direction of
            # prefix counts, because `pkill python` does kill a `python3.12` parent.
            for own in filter(None, {comm, argv0}):
                if name == own or name.startswith(own) or own.startswith(name):
                    return (
                        f"`{name}` names pid {pid} ({comm}), which is this agent's own process "
                        f"tree."
                    )
    return None


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0
    if payload.get("tool_name") != "Bash":
        return 0
    command = (payload.get("tool_input") or {}).get("command") or ""
    if not isinstance(command, str):
        return 0

    reason = offence(command, ancestors())
    if reason is None:
        return 0

    print(
        "Refused: this command would kill the agent running it — " + reason + "\n"
        "Kill the pid you launched (`$!`, or the pid in `neuron-ls`'s last column), or filter "
        "by executable and skip your own tree, e.g.\n"
        "  for p in $(pgrep -x python3 python); do\n"
        "    grep -qa <pattern> /proc/$p/cmdline || continue\n"
        "    [ \"$p\" = \"$$\" ] || kill -9 \"$p\"\n"
        "  done\n"
        "A previous iteration ended itself with `pgrep -f ... | xargs kill -9` and lost three "
        "hours of work; this is that guard.",
        file=sys.stderr,
    )
    return 2  # PreToolUse: non-zero blocks the call and shows stderr to the agent


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001 — fail open, never block real work on a guard bug
        raise SystemExit(0)
