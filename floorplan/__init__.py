# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`autohelix floorplan` — decide what runs where on a Trainium instance.

See `floorplan/README.md`. The short version: `parser.py` turns a system YAML into a
hardware model deterministically, an agent writes the per-module cost models on top of it,
and then a loop edits one file — the floorplan — trying to make the simulated latencies
smaller without breaking any of them.
"""

__all__ = ["schema", "parser", "baseline", "checker", "invariants", "rank", "sim"]
