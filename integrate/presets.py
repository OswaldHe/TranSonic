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


def read(name: str) -> str:
    """One packaged template, by filename."""
    return pkg_files(_PKG).joinpath(name).read_text()


def baseline_prompt() -> str:
    """The per-iteration prompt for the baseline loop."""
    return read("baseline_prompt.md")


def config_template() -> str:
    return read(CONFIG_TEMPLATE)


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
