"""Max reduction — pure @triton.jit kernels.

Global two-pass max and dim-specific max (returns value + index).
Extracted from FlagGems (Apache-2.0).

Note: FlagGems uses `get_dtype_min` from utils.limits. We inline the
sentinel via float('-inf') which Triton handles for all float dtypes.
For the dim kernel, bf16 accumulation is promoted to fp32.
"""

import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Global reduction: two-pass max
# ---------------------------------------------------------------------------

@triton.jit
def max_kernel_mid(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    """Pass 1: compute block-local max."""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    inp_val = tl.load(inp_ptrs, mask=mask, other=float('-inf'))
    max_val = tl.max(inp_val)
    mid_ptr = mid + pid
    tl.store(mid_ptr, max_val)


@triton.jit
def max_kernel_result(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    """Pass 2: reduce partial maxes into scalar."""
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    mid_val = tl.load(mid_ptrs, mask=mask, other=float('-inf'))
    max_val = tl.max(mid_val)
    tl.store(out, max_val)


# ---------------------------------------------------------------------------
# Dim-specific: max along one axis (returns value + argmax index)
# ---------------------------------------------------------------------------

@triton.jit
def max_kernel(
    inp,
    out_value,
    out_index,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Max reduction along compressed inner dim N.

    inp: [M, N] (dim-compressed) → out_value: [M], out_index: [M] (int64)
    Tracks both maximum value and its index (argmax).
    bf16 inputs are accumulated in fp32 to avoid precision issues.
    """
    pid_m = tl.program_id(0)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    dtype = inp.type.element_ty
    acc_type = tl.float32 if dtype is tl.bfloat16 else dtype
    result_value = tl.full([BLOCK_M], value=float('-inf'), dtype=acc_type)
    result_index = tl.zeros([BLOCK_M], dtype=tl.int64)
    for i in range(0, N, BLOCK_N):
        n_offset = i + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N + n_offset[None, :]
        mask = m_offset[:, None] < M and n_offset[None, :] < N
        inp_ptrs = inp + offset
        inp_vals = tl.load(inp_ptrs, mask=mask, other=float('-inf'))
        max_value, max_index = tl.max(inp_vals, axis=1, return_indices=True)
        update_mask = max_value > result_value
        result_value = tl.where(update_mask, max_value, result_value)
        result_index = tl.where(update_mask, i + max_index, result_index)

    mask1 = m_offset < M
    offset_index = m_offset
    out_value_ptrs = out_value + offset_index
    out_index_ptrs = out_index + offset_index
    tl.store(out_value_ptrs, result_value, mask=mask1)
    tl.store(out_index_ptrs, result_index, mask=mask1)
