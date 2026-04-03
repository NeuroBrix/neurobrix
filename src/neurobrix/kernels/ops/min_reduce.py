"""Min reduction — pure @triton.jit kernels.

Global two-pass min and dim-specific min (returns value + index).
Extracted from FlagGems (Apache-2.0).

Note: FlagGems uses `get_dtype_max` from utils.limits. We inline the
sentinel via float('inf') which Triton handles for all float dtypes.
For the dim kernel, bf16 accumulation is promoted to fp32.
"""

import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Global reduction: two-pass min
# ---------------------------------------------------------------------------

@triton.jit
def min_kernel_mid(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    """Pass 1: compute block-local min."""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    inp_val = tl.load(inp_ptrs, mask=mask, other=float('inf'))
    min_val = tl.min(inp_val)
    mid_ptr = mid + pid
    tl.store(mid_ptr, min_val)


@triton.jit
def min_kernel_result(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    """Pass 2: reduce partial mins into scalar."""
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    mid_val = tl.load(mid_ptrs, mask=mask, other=float('inf'))
    min_val = tl.min(mid_val)
    tl.store(out, min_val)


# ---------------------------------------------------------------------------
# Dim-specific: min along one axis (returns value + argmin index)
# ---------------------------------------------------------------------------

@triton.jit
def min_kernel(
    inp,
    out_value,
    out_index,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Min reduction along compressed inner dim N.

    inp: [M, N] (dim-compressed) → out_value: [M], out_index: [M] (int64)
    Tracks both minimum value and its index (argmin).
    bf16 inputs are accumulated in fp32 to avoid precision issues.
    """
    pid_m = tl.program_id(0)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    dtype = inp.type.element_ty
    acc_type = tl.float32 if dtype is tl.bfloat16 else dtype
    min_values = tl.full([BLOCK_M], dtype=acc_type, value=float('inf'))
    argmin_values = tl.full([BLOCK_M], dtype=tl.int64, value=0)
    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N + n_offset[None, :]
        mask = m_offset[:, None] < M and n_offset[None, :] < N
        inp_ptrs = inp + offset
        inp_vals = tl.load(inp_ptrs, mask=mask, other=float('inf'))
        local_min, local_argmin = tl.min(inp_vals, 1, return_indices=True)
        update = local_min < min_values
        min_values = tl.where(update, local_min, min_values)
        argmin_values = tl.where(update, start_n + local_argmin, argmin_values)

    offset_index = m_offset
    out_value_ptrs = out_value + offset_index
    out_index_ptrs = out_index + offset_index
    mask1 = m_offset < M
    tl.store(out_value_ptrs, min_values, mask=mask1)
    tl.store(out_index_ptrs, argmin_values, mask=mask1)
