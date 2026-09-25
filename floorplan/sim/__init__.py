# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The simulation platform.

Framework, written once and frozen: `engine.py` (the timeline), `collectives.py` (what
rejoining a split costs), `memory.py` (where the bytes are), `api.py` (the contract),
`runner.py` (the executable the loop measures).

Agent-written, per project: `sim/modules/*.py` (one cost model per module archetype) and
`sim/constraints.py` (the platform's prose constraints, made executable).
"""
