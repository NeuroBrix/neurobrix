"""
Scatter - Pure Triton
Source: FlagGems (simplified)
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _scatter_kernel(
    src_ptr, index_ptr, out_ptr,
    n_elements, dim_stride, dim_size,
    BLOCK_SIZE: tl.constexpr
):
    """
    Scatter src values into out at positions given by index.

    This is a simplified 1D scatter kernel.
    For multi-dimensional scatter, the adapter should handle reshaping.

    out[index[i]] = src[i]
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load source values and indices
    src = tl.load(src_ptr + offsets, mask=mask)
    idx = tl.load(index_ptr + offsets, mask=mask, other=0)

    # Bounds check
    valid = (idx >= 0) & (idx < dim_size)
    final_mask = mask & valid

    # Calculate output positions
    out_offsets = idx * dim_stride

    # Scatter store
    tl.store(out_ptr + out_offsets, src, mask=final_mask)

@triton.jit
def _scatter_add_kernel(
    src_ptr, index_ptr, out_ptr,
    n_elements, dim_stride, dim_size,
    BLOCK_SIZE: tl.constexpr
):
    """Scatter with atomic add reduction."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    src = tl.load(src_ptr + offsets, mask=mask)
    idx = tl.load(index_ptr + offsets, mask=mask, other=0)

    valid = (idx >= 0) & (idx < dim_size)
    final_mask = mask & valid

    out_offsets = idx * dim_stride
    tl.atomic_add(out_ptr + out_offsets, src, mask=final_mask)
