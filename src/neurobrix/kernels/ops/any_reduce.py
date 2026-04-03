"""Any reduction (logical OR) — pure @triton.jit kernels.

Tests if ANY element evaluates to True (non-zero).
Global two-pass reduction and dim-specific kernel.
Extracted from FlagGems (Apache-2.0).
"""

import triton
import triton.language as tl


@triton.jit
def reduce_any(a, b):
    return a or b


# ---------------------------------------------------------------------------
# Global reduction: two-pass
# ---------------------------------------------------------------------------

@triton.jit
def any_kernel_mid(
    inp,
    mid,
    n_elements,
    mid_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Pass 1: check if any element in this block is non-zero."""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < n_elements
    inp_val = tl.load(inp_ptrs, mask=mask, other=0.0)
    any_val = tl.reduce(inp_val != 0, axis=0, combine_fn=reduce_any)
    mid_ptr = mid + pid
    tl.store(mid_ptr, any_val)


@triton.jit
def any_kernel_result(mid, out, MID_SIZE, BLOCK_MID: tl.constexpr):
    """Pass 2: combine partial results into scalar bool."""
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < MID_SIZE
    mid_val = tl.load(mid_ptrs, mask=mask, other=0).to(tl.int1)
    any_val = tl.reduce(mid_val, axis=0, combine_fn=reduce_any)
    tl.store(out, any_val)


# ---------------------------------------------------------------------------
# Dim-specific: any along one axis
# ---------------------------------------------------------------------------

@triton.jit
def any_kernel_dim(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Check any non-zero along compressed inner dim N.

    inp: [M, N] (dim-compressed) → out: [M] (bool)
    """
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows * N
    out = out + rows
    row_mask = rows < M

    _any = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int1)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(inp + cols, mask, other=0.0)
        _any = _any or (a != 0)
    result = tl.reduce(_any, axis=1, combine_fn=reduce_any)
    tl.store(out, result[:, None], row_mask)
