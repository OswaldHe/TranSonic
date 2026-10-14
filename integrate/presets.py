# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reading the package's own templates.

Same reasoning as `optimization/presets.py`: the prompt that states the specification lives in
the package, not in the repo a run edits. A file copied into the repo is a file an iteration can
edit, and the prompt constraining an agent is the last thing that should be editable by it.

The operator's escape hatch is `--config` and `.autohelix/integrate/prompt.md`, both explicit.
"""

from __future__ import annotations

from importlib.resources import files as pkg_files
from pathlib import Path

_PKG = "integrate.templates"

#: The config `autohelix integrate init` copies into a project for the operator to fill in.
CONFIG_TEMPLATE = "integrate.yaml"

#: The config `autohelix integrate init --stage run` writes: the same constraints, plus the
#: measurements the loop then improves.
RUN_CONFIG_TEMPLATE = "integrate-run.yaml"


def read(name: str) -> str:
    """One packaged template, by filename."""
    return pkg_files(_PKG).joinpath(name).read_text()


def baseline_prompt() -> str:
    """The per-iteration prompt for the baseline loop."""
    return read("baseline_prompt.md")


def run_prompt() -> str:
    """The per-iteration prompt for the run loop."""
    return read("run_prompt.md")


def config_template() -> str:
    return read(CONFIG_TEMPLATE)


def run_config_template() -> str:
    return read(RUN_CONFIG_TEMPLATE)


def project_prompt(repo: Path) -> str:
    """The loop's prompt template, letting a repo override it.

    `.autohelix/integrate/prompt.md` wins when it exists, which is how an operator tries a
    different prompt without editing the installed package. `init` does not seed it, so the
    default is the packaged one and an override is always deliberate.
    """
    override = repo / ".autohelix" / "integrate" / "prompt.md"
    if override.is_file():
        return override.read_text()
    return baseline_prompt()


def project_run_prompt(repo: Path) -> str:
    """The run loop's prompt template, with the same override rule under its own filename.

    A separate file from the baseline stage's, because the two prompts say different things —
    one asks for a constraint to be reached, the other for a number to come down while they all
    keep holding — and overriding one should not silently change the other.
    """
    override = repo / ".autohelix" / "integrate" / "run-prompt.md"
    if override.is_file():
        return override.read_text()
    return run_prompt()
