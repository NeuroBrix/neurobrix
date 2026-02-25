# Gather - Pure Triton
# NeuroBrix - NVIDIA Common

import torch
import triton
import triton.language as tl

@triton.jit
def _gather_nd_kernel(
    x_ptr, indices_ptr, out_ptr,
    n_indices, index_dim,
    stride_x, stride_out,
    BLOCK_SIZE: tl.constexpr
):
    """General gather along specified dimension."""
    pid = tl.program_id(0)
    if pid >= n_indices: return
    idx = tl.load(indices_ptr + pid)
    for d_start in range(0, index_dim, BLOCK_SIZE):
        d_offs = d_start + tl.arange(0, BLOCK_SIZE)
        mask = d_offs < index_dim
        val = tl.load(x_ptr + idx * stride_x + d_offs, mask=mask, other=0.0)
        tl.store(out_ptr + pid * stride_out + d_offs, val, mask=mask)

@triton.jit
def _gather_kernel(
    out_ptr, x_ptr, indices_ptr,
    stride_out_row, stride_x_row, stride_idx_row,
    n_cols, BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    idx_val = tl.load(indices_ptr + row_idx * stride_idx_row + cols, mask=mask, other=0)
    val = tl.load(x_ptr + row_idx * stride_x_row + idx_val, mask=mask, other=0.0)
    tl.store(out_ptr + row_idx * stride_out_row + cols, val, mask=mask)
