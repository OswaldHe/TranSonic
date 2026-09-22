# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Locate the bootstrap preset. It is data, not something this module builds.

`bootstrap/preset.yaml` is the config the loop runs with and `bootstrap/prompt.md` is the
agent's prompt template. Both are fixed files, read straight from the package: nothing
generates them, nothing substitutes into them, and no copy is written into a module repo.
Editing `preset.yaml` changes the next run, with no re-init.

Keeping the preset as one reviewable file matters more here than in an ordinary loop. The
agent never sees `nki_checker.py`; the `goal` in `preset.yaml` is the entire specification
it works from. So the preset and the checker are a pair, and the preset has to be readable
in a diff for that pairing to be checkable at all. `tests/test_bootstrap_driver.py` fails if
they drift on the tolerances, the markers or the banned constructs, but only a human reading
the goal can tell whether a requirement is stated *clearly*.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

#: The package directory, which is where both fixed files live.
HERE = Path(__file__).resolve().parent

#: The config the loop is pointed at, and the prompt template beside it.
PRESET_PATH = HERE / "preset.yaml"
PROMPT_PATH = HERE / "prompt.md"


class PresetError(RuntimeError):
    """The preset is missing or does not say what the loop needs it to say."""


def load_preset() -> dict[str, Any]:
    """`preset.yaml`, parsed.

    The loop itself does not go through this — it hands the path to AutoHelix's own
    `load_config`, so there is exactly one parser for it and no chance of this module and
    the harness disagreeing about what the file says. This exists for the tests and for
    anything that wants to read one field out of it.
    """
    try:
        data = yaml.safe_load(PRESET_PATH.read_text())
    except FileNotFoundError as exc:
        raise PresetError(f"the bootstrap preset is missing: {PRESET_PATH}") from exc
    if not isinstance(data, dict):
        raise PresetError(f"{PRESET_PATH.name} did not parse as a mapping")
    for key in ("goal", "constraints", "scope", "reviewer"):
        if key not in data:
            raise PresetError(f"{PRESET_PATH.name} has no '{key}'")
    return data


def load_prompt_template() -> str:
    """`prompt.md`, the Jinja template the agent's prompt is rendered from."""
    try:
        return PROMPT_PATH.read_text()
    except FileNotFoundError as exc:
        raise PresetError(f"the bootstrap prompt template is missing: {PROMPT_PATH}") from exc


def render_goal() -> str:
    """Just the goal, for callers that want the specification without the config."""
    return str(load_preset()["goal"])


#: Read once at import: the tests and the reviewer plumbing refer to these by name.
GOAL = render_goal()
REVIEWER_PROMPT = str(load_preset()["reviewer"]["prompt"])
PROMPT_TEMPLATE = load_prompt_template()


def dump_manifest(path: Path, payload: dict[str, Any]) -> None:
    """Write the tensor manifest the gate reads.

    The one thing about a run that cannot be a fixed file: it records which tensors this
    particular repo holds and what they hash to.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
