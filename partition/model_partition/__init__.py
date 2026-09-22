# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model partitioning, tracing, and verification for TranSonic.

Takes a model specification (config, inference code, weights) plus sample inputs
and runs a specialized loop: partition into locally-runnable modules, trace real
IO per module, verify each module replays, then emulate end-to-end inference and
judge the sampled tokens.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
