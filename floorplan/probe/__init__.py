# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Micro-benchmarks that turn a datasheet into a cost model.

Primitives only — a matmul at a shape, a DMA of a size, a collective over four cores. Never
a module, because a module's measured latency on this workspace's bootstrapped kernels
reflects how optimized those kernels happen to be, and anchoring the simulator to that would
bake their inefficiency in as if it were a property of the silicon.
"""
