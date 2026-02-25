"""
Prod (Reduce Product) - Pure Triton
Source: FlagGems
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _reduce_mul(a, b):
    """Reduce helper: multiply."""
    return a * b

@triton.jit
def _prod_kernel_1d(
    x_ptr, mid_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    """First pass: reduce blocks to intermediate values."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=1.0)
    x = x.to(tl.float32)
    prod_val = tl.reduce(x, axis=0, combine_fn=_reduce_mul)
    tl.store(mid_ptr + pid, prod_val)

@triton.jit
def _prod_kernel_final(
    mid_ptr, out_ptr, mid_size,
    BLOCK_MID: tl.constexpr
):
    """Second pass: reduce intermediate values to final result."""
    offsets = tl.arange(0, BLOCK_MID)
    mask = offsets < mid_size

    mid_val = tl.load(mid_ptr + offsets, mask=mask, other=1.0)
    mid_val = mid_val.to(tl.float32)
    prod_val = tl.reduce(mid_val, axis=0, combine_fn=_reduce_mul)
    tl.store(out_ptr, prod_val)

@triton.jit
def _prod_dim_kernel(
    x_ptr, out_ptr, M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    """Reduce along one dimension."""
    pid_m = tl.program_id(0)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    acc = tl.full((BLOCK_M, BLOCK_N), value=1.0, dtype=tl.float32)
    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N + n_offset[None, :]
        mask = (m_offset[:, None] < M) & (n_offset[None, :] < N)
        x = tl.load(x_ptr + offset, mask=mask, other=1.0)
        x = x.to(tl.float32)
        acc *= x

    result = tl.reduce(acc, axis=1, combine_fn=_reduce_mul)
    out_mask = m_offset < M
    tl.store(out_ptr + m_offset, result, mask=out_mask)
