# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime pieces a published module directory builds itself with."""

from __future__ import annotations

import hashlib
from pathlib import Path


def private_module_name(prefix: str, path: str | Path) -> str:
    """A private ``sys.modules`` name for a file, stable across processes.

    Derived from the path so re-importing an edited file *replaces* its predecessor
    rather than accumulating beside it, and from a digest rather than ``hash()`` so two
    runs agree. ``hash()`` of a string is salted per process, so the name changed on
    every run — which is recorded in each group's ``meta.yaml`` and so rewrote 32 files
    of an artifact for nothing, adding a version of each to the published history every
    time. Nothing reads the recorded value for a flat name like these, which is the only
    reason the churn was invisible.
    """
    digest = hashlib.blake2s(str(path).encode()).hexdigest()[:16]
    return f"{prefix}{digest}"
