"""
Index Select - Pure Triton
Source: FlagGems
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _index_select_kernel(
    inp_ptr, out_ptr, index_ptr,
    M, N, index_len,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    """
    Select elements along a dimension using index tensor.

    inp: (M, N) input tensor (after dim compression)
    index: (index_len,) indices to select along N dimension
    out: (M, index_len) output tensor
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Row offsets
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    m_mask = m_offset < M

    # Column offsets (index positions)
    n_offset = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offset < index_len

    # Load indices
    indices = tl.load(index_ptr + n_offset, mask=n_mask, other=0)

    # Bounds check on indices
    valid_idx = (indices >= 0) & (indices < N)

    # Calculate input and output offsets
    inp_off = m_offset * N + indices[None, :]
    out_off = m_offset * index_len + n_offset[None, :]

    # Combined mask
    final_mask = m_mask & n_mask[None, :] & valid_idx[None, :]

    # Load and store
    selected = tl.load(inp_ptr + inp_off, mask=final_mask, other=0.0)
    tl.store(out_ptr + out_off, selected, mask=m_mask & n_mask[None, :])
