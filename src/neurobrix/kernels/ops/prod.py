"""Product reduction — pure @triton.jit kernels.

Global reduction via two-pass (mid → result) and dim-specific reduction.
Extracted from FlagGems (Apache-2.0).
"""

import triton
import triton.language as tl


@triton.jit
def reduce_mul(a, b):
    return a * b


# ---------------------------------------------------------------------------
# Global reduction: two-pass partial products
# ---------------------------------------------------------------------------

@triton.jit
def prod_kernel_mid(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    """Pass 1: each program computes a partial product over its BLOCK_SIZE chunk."""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    inp_val = tl.load(inp_ptrs, mask=mask, other=1.0).to(tl.float32)
    mid_value = tl.reduce(inp_val, axis=0, combine_fn=reduce_mul)
    mid_ptr = mid + pid
    tl.store(mid_ptr, mid_value.to(inp_val.dtype))


@triton.jit
def prod_kernel_result(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    """Pass 2: reduce all partial products into a single scalar."""
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    mid_val = tl.load(mid_ptrs, mask=mask, other=1.0).to(tl.float32)
    prod_val = tl.reduce(mid_val, axis=0, combine_fn=reduce_mul)
    tl.store(out, prod_val)


# ---------------------------------------------------------------------------
# Dim-specific reduction: product along one axis
# ---------------------------------------------------------------------------

@triton.jit
def prod_kernel(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Reduce product along the compressed inner dimension N.

    inp: [M, N] (dim-compressed) → out: [M]
    """
    pid_m = tl.program_id(0)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    acc = tl.full((BLOCK_M, BLOCK_N), value=1.0, dtype=tl.float32)
    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N + n_offset[None, :]
        mask = m_offset[:, None] < M and n_offset[None, :] < N
        inp_ptrs = inp + offset
        inp_vals = tl.load(inp_ptrs, mask=mask, other=1.0).to(tl.float32)
        acc *= inp_vals
    result = tl.reduce(acc, axis=1, combine_fn=reduce_mul)

    out_ptrs = out + m_offset
    mask1 = m_offset < M
    tl.store(out_ptrs, result, mask=mask1)
