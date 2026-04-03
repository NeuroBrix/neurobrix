"""Dot product — pure @triton.jit kernels.

Ported from FlagGems dot (Apache-2.0 license).
Two-phase reduction for large N: partial sums then final reduce.
Single-phase for small N.
"""

import triton
import triton.language as tl


@triton.jit
def dot_kernel_small(
    x_ptr, y_ptr, out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """Single-block dot product for small vectors (N < 4096).

    Accumulates in fp32 for numerical stability.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    result = tl.sum(x * y)
    tl.store(out_ptr, result)


@triton.jit
def dot_kernel_partial(
    x_ptr, y_ptr, mid_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """Phase 1: compute partial dot products per block.

    Each block computes sum(x[block] * y[block]) and stores to mid_ptr[pid].
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    partial_sum = tl.sum(x * y)
    tl.store(mid_ptr + pid, partial_sum)


@triton.jit
def dot_kernel_reduce(
    mid_ptr, out_ptr,
    M,
    BLOCK_MID: tl.constexpr,
):
    """Phase 2: reduce partial sums into final scalar result."""
    offsets = tl.arange(0, BLOCK_MID)
    mask = offsets < M
    mid_val = tl.load(mid_ptr + offsets, mask=mask, other=0.0)
    result = tl.sum(mid_val)
    tl.store(out_ptr, result)
