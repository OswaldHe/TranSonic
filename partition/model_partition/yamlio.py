# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""YAML reading and writing, through libyaml when it is available.

PyYAML's default loader is pure Python and parses at a few hundred KB/s. A run's
tensor manifest is comfortably a megabyte for a small model and far more for a
large one, and it is read once per module — so the default loader alone accounted
for most of a verification pass. libyaml is an order of magnitude faster and is
what PyYAML ships against on every platform we run on; the pure-Python loader
stays as a fallback so an environment without it still works.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_Loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
_Dumper = getattr(yaml, "CSafeDumper", yaml.SafeDumper)

#: True when the compiled parser is in use. Reported by the loop so a slow run on
#: a machine without libyaml is explainable rather than mysterious.
FAST = _Loader is not yaml.SafeLoader


def loads(text: str) -> Any:
    """Parse YAML text. Raises :class:`yaml.YAMLError` on malformed input."""
    return yaml.load(text, Loader=_Loader)


def load_path(path: str | Path) -> Any:
    return loads(Path(path).read_text())


def dumps(payload: Any, sort_keys: bool = False, **options: Any) -> str:
    return yaml.dump(payload, Dumper=_Dumper, sort_keys=sort_keys,
                     default_flow_style=False, **options)


def dump_path(path: str | Path, payload: Any, **options: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(dumps(payload, **options))
    return target
