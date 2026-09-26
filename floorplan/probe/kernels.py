# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""NKI kernels that exist only to be timed.

Each isolates one engine, with its operands where the cost model assumes they are. That
separation mirrors `sim/api.py`: `matmul_seconds` charges the tensor engine and
`dma_seconds` charges the movement, so a probe that conflated them would have every module
paying for its data twice.

Two measurement traps found the hard way here, both worth stating because a probe that falls
into either produces a confidently wrong number rather than an error:

**Operands must already be on the device.** A `torch_neuronx.trace` input is copied
host-to-device on every call, so a kernel that reads its input measures PCIe — 12.9 GB/s on
this host — and not HBM. The HBM probes therefore fill a `private_hbm` scratch buffer inside
the kernel and read *that*, which measures 205 GB/s. A 16x error, silently.

**Operands must be in SBUF to measure arithmetic.** A textbook tiled matmul that loads its
tiles inside the innermost loop measures 6.4 TFLOPS — 3.8% of per-core peak — because it is
bound by 1.3 GB of redundant HBM traffic per call. Hoisted into SBUF, the same silicon
measures 63 TFLOPS. Feeding the first number to the simulator would model a machine 10x
slower at arithmetic than the one we have, making every scheme compute-bound with
communication free, which is exactly backwards.

These are not good kernels and are not trying to be. They are the simplest construction that
keeps one engine busy, which is what makes the number they produce easy to argue about.
"""

from __future__ import annotations

import nki
import nki.isa as nisa
import nki.language as nl

#: nc_matmul calls per dispatch. Sized so device time dominates the ~120 us host round trip.
MATMUL_REPS = 2048

#: Stationary tiles cycled through, so the compiler cannot fold the repetitions into one.
STATIONARY_TILES = 8

#: Elementwise/gather repetitions, chosen the same way.
VECTOR_REPS = 512
GATHER_REPS = 64

#: HBM scratch geometry for the bandwidth probes: 256 tiles x 2048 columns of bf16 = 128 MiB,
#: read ``HBM_PASSES`` times. Large enough that the fill is a small share of the measurement.
HBM_TILES = 256
HBM_COLS = 2048
HBM_PASSES = 8

#: Columns read per tile in the strided probe. Far below a tile's width, so transfers are
#: short and scattered — the Engram row-lookup regime.
STRIDE_COLS = 8


@nki.jit
def dispatch_overhead(x):
    """The cheapest kernel that still moves a byte. Every measurement subtracts this."""
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    nl.store(out[0:128, 0:1], value=nl.load(x[0:128, 0:1]))
    return out


@nki.jit
def matmul_pe(stationary, moving):
    """Tensor engine only: both operands SBUF-resident, accumulating in one PSUM bank.

    ``stationary`` is ``(K, 128 * STATIONARY_TILES)`` and ``moving`` is ``(K, N)``; the
    contraction runs along the partition axis, so ``K`` is the partition count. K = 128 with
    N = 512 is the streaming peak; a small K is the regime MoE and LoRA projections live in,
    where the systolic array cannot be filled.
    """
    out = nl.ndarray((128, moving.shape[1]), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    stat = nl.load(stationary)
    mov = nl.load(moving)
    acc = nl.zeros((128, moving.shape[1]), nl.float32, buffer=nl.psum)
    for rep in nl.affine_range(MATMUL_REPS):
        offset = (rep % STATIONARY_TILES) * 128
        nisa.nc_matmul(
            dst=acc, stationary=stat[:, offset:offset + 128], moving=mov, accumulate=True,
        )
    nl.store(out, value=nl.copy(acc, dtype=nl.bfloat16))
    return out


@nki.jit
def vector_elementwise(a, b):
    """Vector engine: repeated tensor-tensor adds over SBUF-resident tiles.

    Each iteration reads ``acc`` as well as writing it, so the loop is a genuine dependency
    chain. That is load-bearing rather than stylistic: ``tensor_tensor`` *overwrites* its
    destination, so a loop of ``dst=acc, data1=x, data2=y`` is 511 redundant iterations the
    compiler may or may not eliminate — and the coefficient this probe produced swung 8x
    between runs (0.30 to 2.35) depending on whether it did. The gather probe, which chained
    through ``acc`` from the start, was stable to within 5% across the same runs.
    """
    out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
    x = nl.load(a)
    y = nl.load(b)
    acc = nl.copy(x, dtype=nl.float32)
    for _ in nl.affine_range(VECTOR_REPS):
        nisa.tensor_tensor(dst=acc, data1=acc, data2=y, op=nl.add, engine=nisa.engine.vector)
    nl.store(out, value=nl.copy(acc, dtype=a.dtype))
    return out


@nki.jit
def scalar_activation(a):
    """Scalar engine: repeated transcendental passes over an SBUF-resident tile.

    Chained through ``acc`` for the same reason as ``vector_elementwise``.
    """
    out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
    x = nl.load(a)
    acc = nl.copy(x, dtype=nl.float32)
    for _ in nl.affine_range(VECTOR_REPS):
        nisa.activation(dst=acc, op=nl.exp, data=acc)
    nl.store(out, value=nl.copy(acc, dtype=a.dtype))
    return out


@nki.jit
def gpsimd_gather(table, indices):
    """GPSIMD: gather along the free axis, per partition.

    The Engram and MoE primitive. ``indices`` must be uint32 and the same shape as the
    destination — it names one element per output position, not one row. The cast happens
    here because torch hands the trace an int32 tensor.
    """
    width = indices.shape[1]
    out = nl.ndarray((128, width), dtype=nl.float32, buffer=nl.shared_hbm)
    tab = nl.load(table)
    idx = nl.copy(nl.load(indices), dtype=nl.uint32)
    gathered = nl.ndarray((128, width), dtype=table.dtype, buffer=nl.sbuf)
    acc = nl.zeros((128, width), dtype=nl.float32, buffer=nl.sbuf)
    for _ in nl.affine_range(GATHER_REPS):
        nisa.nc_n_gather(dst=gathered, data=tab, indices=idx)
        nisa.tensor_tensor(
            dst=acc, data1=acc, data2=gathered, op=nl.add, engine=nisa.engine.vector,
        )
    nl.store(out, value=acc)
    return out


@nki.jit
def pcie_consume(src):
    """Read every tile of a *traced input*, which is the host-to-device path.

    The mirror image of the HBM probes: they avoid the input copy, this one measures it. A
    kernel that merely *receives* a large input is not enough — the runtime is free not to
    materialize bytes nothing reads, and a probe built that way reported 3.0 GB/s where
    consuming the whole input reports 12.9. So every tile is read and reduced.
    """
    rows, cols = src.shape
    out = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    acc = nl.zeros((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    for index in nl.affine_range(rows // 128):
        chunk = nl.load(src[index * 128:(index + 1) * 128, :])
        nisa.tensor_reduce(dst=acc, op=nl.add, data=chunk, axis=(1,))
    nl.store(out, value=acc)
    return out


@nki.jit
def hbm_fill(seed):
    """Fill the HBM scratch and read one tile. The baseline the read probes subtract.

    Separating this out is what makes the read measurement a read measurement: the fill
    writes 128 MiB, which is not free, and attributing it to the read would halve the
    reported bandwidth.
    """
    scratch = nl.ndarray((HBM_TILES * 128, HBM_COLS), dtype=nl.bfloat16, buffer=nl.private_hbm)
    out = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    tile = nl.load(seed)
    for index in nl.affine_range(HBM_TILES):
        nl.store(scratch[index * 128:(index + 1) * 128, :], value=tile)
    acc = nl.zeros((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=acc, op=nl.add, data=nl.load(scratch[0:128, :]), axis=(1,))
    nl.store(out, value=acc)
    return out


@nki.jit
def hbm_read_contiguous(seed):
    """Stream the whole HBM scratch into SBUF, full tiles at a time."""
    scratch = nl.ndarray((HBM_TILES * 128, HBM_COLS), dtype=nl.bfloat16, buffer=nl.private_hbm)
    out = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    tile = nl.load(seed)
    for index in nl.affine_range(HBM_TILES):
        nl.store(scratch[index * 128:(index + 1) * 128, :], value=tile)
    acc = nl.zeros((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    for _ in nl.affine_range(HBM_PASSES):
        for index in nl.affine_range(HBM_TILES):
            chunk = nl.load(scratch[index * 128:(index + 1) * 128, :])
            nisa.tensor_reduce(dst=acc, op=nl.add, data=chunk, axis=(1,))
    nl.store(out, value=acc)
    return out


@nki.jit
def hbm_read_strided(seed):
    """Read the same *number of transfers* as the contiguous probe, but tiny slices each.

    The ratio between the two is what makes a scattered row lookup expensive in the
    simulator, and it is a ratio rather than an absolute so it survives being wrong about
    the absolute bandwidth.
    """
    scratch = nl.ndarray((HBM_TILES * 128, HBM_COLS), dtype=nl.bfloat16, buffer=nl.private_hbm)
    out = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    tile = nl.load(seed)
    for index in nl.affine_range(HBM_TILES):
        nl.store(scratch[index * 128:(index + 1) * 128, :], value=tile)
    acc = nl.zeros((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    for pass_index in nl.affine_range(HBM_PASSES):
        for index in nl.affine_range(HBM_TILES):
            # A stride that is not the tile pitch, so successive reads do not coalesce.
            offset = ((index * 37 + pass_index * 11) % (HBM_COLS - STRIDE_COLS))
            chunk = nl.load(scratch[index * 128:(index + 1) * 128,
                                    offset:offset + STRIDE_COLS])
            nisa.tensor_reduce(dst=acc, op=nl.add, data=chunk, axis=(1,))
    nl.store(out, value=acc)
    return out
