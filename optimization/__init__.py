# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The optimization loop: make one bootstrapped module fast on a single Trainium device.

`autohelix bootstrap` produces a correct-but-slow NKI kernel per module. `autohelix floorplan`
decides where each module runs. This takes both and closes the gap: it projects the floorplan's
placement onto the one device in front of us, has an agent cut the module down to what a single
NeuronCore runs, optimizes that under a per-iteration constraint schedule, then reassembles the
whole module across four ranks with `nki.collectives` and optimizes that.

See `optimization/README.md` for the design and why the stages are separated the way they are.
"""

__all__ = ["PROJECTION_TARGET_UNITS"]

#: How many logical NeuronCores one device of the dev host has, and therefore the width every
#: placement is projected onto. Lives here because it is the single number the whole pipeline
#: is organized around: the submodule is 1/this of the module, the assembly is this many ranks,
#: and `NEURON_RT_NUM_CORES` is 1 in stage 3 and this in stage 5.
PROJECTION_TARGET_UNITS = 4
