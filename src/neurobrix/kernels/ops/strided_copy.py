"""Strided-to-contiguous copy — pure @triton.jit kernel.

Copies data from a non-contiguous (strided) source tensor into a
contiguous destination tensor. Handles arbitrary strides including
stride=0 (from expand/broadcast views).

Why this exists: NBXTensor.expand() creates views with stride=0.
A simple cudaMemcpy copies raw bytes from the base pointer, which
is wrong for stride-0 layouts — it reads past the actual data.
This kernel computes the correct source offset per element using
the source strides and shape decomposition.

Supports tensors up to 5D (padded with shape=1, stride=0 for unused dims).
"""

import triton
import triton.language as tl


@triton.jit
def strided_copy_kernel(
    src_ptr, dst_ptr,
    n_elements,
    s0, s1, s2, s3, s4,
    d0, d1, d2, d3, d4,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy strided src to contiguous dst.

    Each thread computes the multi-dimensional index from its flat output
    position, then uses the source strides to find the correct read offset.

    Args:
        src_ptr: Source tensor pointer (may be non-contiguous).
        dst_ptr: Destination tensor pointer (contiguous).
        n_elements: Total number of elements to copy.
        s0..s4: Source strides per dimension (0-padded for unused dims).
        d0..d4: Shape per dimension (1-padded for unused dims).
        BLOCK_SIZE: Elements per thread block.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Flat dst index → multi-dim indices → strided src offset
    remaining = offsets
    i0 = remaining // (d1 * d2 * d3 * d4)
    remaining = remaining % (d1 * d2 * d3 * d4)
    i1 = remaining // (d2 * d3 * d4)
    remaining = remaining % (d2 * d3 * d4)
    i2 = remaining // (d3 * d4)
    remaining = remaining % (d3 * d4)
    i3 = remaining // d4
    i4 = remaining % d4

    src_offsets = i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3 + i4 * s4

    vals = tl.load(src_ptr + src_offsets, mask=mask)
    tl.store(dst_ptr + offsets, vals, mask=mask)


@triton.jit
def strided_scatter_kernel(
    src_ptr, dst_ptr,
    n_elements,
    s0, s1, s2, s3, s4,
    d0, d1, d2, d3, d4,
    BLOCK_SIZE: tl.constexpr,
):
    """Scatter contiguous src into strided dst.

    Inverse of strided_copy_kernel: reads src linearly and writes
    to dst using the dst's strides. Used for KV cache indexed writes
    where the destination is a narrow view (non-contiguous).

    Args:
        src_ptr: Source tensor pointer (contiguous).
        dst_ptr: Destination tensor pointer (may be non-contiguous).
        n_elements: Total number of elements.
        s0..s4: Destination strides per dimension.
        d0..d4: Shape per dimension.
        BLOCK_SIZE: Elements per thread block.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Flat src index → multi-dim indices → strided dst offset
    remaining = offsets
    i0 = remaining // (d1 * d2 * d3 * d4)
    remaining = remaining % (d1 * d2 * d3 * d4)
    i1 = remaining // (d2 * d3 * d4)
    remaining = remaining % (d2 * d3 * d4)
    i2 = remaining // (d3 * d4)
    remaining = remaining % (d3 * d4)
    i3 = remaining // d4
    i4 = remaining % d4

    dst_offsets = i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3 + i4 * s4

    vals = tl.load(src_ptr + offsets, mask=mask)
    tl.store(dst_ptr + dst_offsets, vals, mask=mask)
