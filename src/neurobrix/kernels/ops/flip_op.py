"""Flip — pure @triton.jit kernel.

Reverses elements along one dimension. The wrapper computes the
flipped strides and offset; this kernel just copies the data.

Ported from FlagGems flip (Apache-2.0 license).
"""

import triton
import triton.language as tl

@triton.jit
def flip_1d_kernel(
    src_ptr, dst_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Reverse a 1D contiguous tensor: dst[i] = src[n_elements - 1 - i]."""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    src_idx = n_elements - 1 - offset
    data = tl.load(src_ptr + src_idx, mask=mask)
    tl.store(dst_ptr + offset, data, mask=mask)

@triton.jit
def flip_strided_kernel(
    src_ptr, dst_ptr,
    n_elements,
    # Dimension being flipped
    dim_size,
    stride_dim,
    stride_post,
    BLOCK_SIZE: tl.constexpr,
):
    """Flip along a single dimension of an N-D contiguous tensor.

    For each element, decomposes the flat index into (pre, dim, post)
    coordinates and reverses the dim coordinate.

    src and dst are both contiguous with the same layout; only the
    dim-axis index is mirrored.
    """
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    # Decompose flat index
    post_size = stride_dim  # product of all dims after the flip dim
    pre_idx = offset // (dim_size * post_size)
    dim_idx = (offset // post_size) % dim_size
    post_idx = offset % post_size

    # Flip the dim coordinate
    flipped_dim_idx = dim_size - 1 - dim_idx

    src_flat = pre_idx * (dim_size * post_size) + flipped_dim_idx * post_size + post_idx
    data = tl.load(src_ptr + src_flat, mask=mask)
    tl.store(dst_ptr + offset, data, mask=mask)
