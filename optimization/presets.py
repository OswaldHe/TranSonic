# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reading the package's own templates.

The prompts and the config template live in the package, not in the project a run creates. Same
reasoning as `bootstrap/preset.py`: a file that is copied into the project is a file an iteration
can edit, and the prompt that states the specification is the last thing that should be editable by
the agent it constrains. The operator's escape hatch is `--config` and `.autohelix/prompt.md`, both
explicit.
"""

from __future__ import annotations

from importlib.resources import files as pkg_files
from pathlib import Path

_PKG = "optimization.templates"  # the data package beside this module

#: The config `autohelix optimize init` copies into a project for the operator to fill in.
CONFIG_TEMPLATE = "optimization.yaml"


def read(name: str) -> str:
    """One packaged template, by filename."""
    return pkg_files(_PKG).joinpath(name).read_text()


def loop_prompt() -> str:
    """The per-iteration prompt for the optimization loop."""
    return read("loop_prompt.md")


def submodule_prompt() -> str:
    """The stage-2 prompt: cut the module down to one rank."""
    return read("submodule_prompt.md")


def assemble_prompt() -> str:
    """The stage-4 prompt: put the ranks back together with a collective."""
    return read("assemble_prompt.md")


def compiler_prompt() -> str:
    """The constraint compiler's prompt: turn each slot's prose into a checker."""
    return read("compiler_prompt.md")


def config_template() -> str:
    return read(CONFIG_TEMPLATE)


def project_prompt(project_path: Path) -> str:
    """The loop's prompt template, letting a project override it.

    `.autohelix/prompt.md` wins when it exists, which is how an operator tries a different prompt
    without editing the installed package. It is not seeded by `init`, so the default is the
    packaged one and an override is always a deliberate act.
    """
    override = project_path / ".autohelix" / "prompt.md"
    if override.is_file():
        return override.read_text()
    return loop_prompt()
