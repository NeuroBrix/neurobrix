"""Repeat (tile) — pure @triton.jit kernel.

Copies input data into output with tiling via modular indexing.
For each output element, the source index is computed as:
    src[i] = (output_idx[i] % input_shape[i]) for each dimension.

This is a rank-generic kernel that handles up to 6 dimensions.
Grid: (ceil(num_output_elements / BLOCK_SIZE),)
"""

import triton
import triton.language as tl


@triton.jit
def repeat_1d_kernel(
    in_ptr, out_ptr,
    in_stride0,
    out_stride0,
    out_shape0,
    in_shape0,
    num_tasks,
    BLOCK_SIZE: tl.constexpr,
):
    """Repeat kernel for 1D tensors."""
    pid = tl.program_id(0)
    tid = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = tid < num_tasks

    i0 = tid

    # Compute source index via modular indexing
    in_offset = (i0 % in_shape0) * in_stride0
    out_offset = i0 * out_stride0

    val = tl.load(in_ptr + in_offset, mask=mask)
    tl.store(out_ptr + out_offset, val, mask=mask)


@triton.jit
def repeat_2d_kernel(
    in_ptr, out_ptr,
    in_stride0, in_stride1,
    out_stride0, out_stride1,
    out_shape0, out_shape1,
    in_shape0, in_shape1,
    num_tasks,
    BLOCK_SIZE: tl.constexpr,
):
    """Repeat kernel for 2D tensors."""
    pid = tl.program_id(0)
    tid = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = tid < num_tasks

    # Decompose flat index into multi-index
    i1 = tid % out_shape1
    i0 = tid // out_shape1

    # Source via modular indexing
    in_offset = (i0 % in_shape0) * in_stride0 + (i1 % in_shape1) * in_stride1
    out_offset = i0 * out_stride0 + i1 * out_stride1

    val = tl.load(in_ptr + in_offset, mask=mask)
    tl.store(out_ptr + out_offset, val, mask=mask)


@triton.jit
def repeat_3d_kernel(
    in_ptr, out_ptr,
    in_stride0, in_stride1, in_stride2,
    out_stride0, out_stride1, out_stride2,
    out_shape0, out_shape1, out_shape2,
    in_shape0, in_shape1, in_shape2,
    num_tasks,
    BLOCK_SIZE: tl.constexpr,
):
    """Repeat kernel for 3D tensors."""
    pid = tl.program_id(0)
    tid = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = tid < num_tasks

    i2 = tid % out_shape2
    tmp = tid // out_shape2
    i1 = tmp % out_shape1
    i0 = tmp // out_shape1

    in_offset = (
        (i0 % in_shape0) * in_stride0
        + (i1 % in_shape1) * in_stride1
        + (i2 % in_shape2) * in_stride2
    )
    out_offset = i0 * out_stride0 + i1 * out_stride1 + i2 * out_stride2

    val = tl.load(in_ptr + in_offset, mask=mask)
    tl.store(out_ptr + out_offset, val, mask=mask)


@triton.jit
def repeat_4d_kernel(
    in_ptr, out_ptr,
    in_stride0, in_stride1, in_stride2, in_stride3,
    out_stride0, out_stride1, out_stride2, out_stride3,
    out_shape0, out_shape1, out_shape2, out_shape3,
    in_shape0, in_shape1, in_shape2, in_shape3,
    num_tasks,
    BLOCK_SIZE: tl.constexpr,
):
    """Repeat kernel for 4D tensors."""
    pid = tl.program_id(0)
    tid = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = tid < num_tasks

    i3 = tid % out_shape3
    tmp = tid // out_shape3
    i2 = tmp % out_shape2
    tmp = tmp // out_shape2
    i1 = tmp % out_shape1
    i0 = tmp // out_shape1

    in_offset = (
        (i0 % in_shape0) * in_stride0
        + (i1 % in_shape1) * in_stride1
        + (i2 % in_shape2) * in_stride2
        + (i3 % in_shape3) * in_stride3
    )
    out_offset = (
        i0 * out_stride0 + i1 * out_stride1
        + i2 * out_stride2 + i3 * out_stride3
    )

    val = tl.load(in_ptr + in_offset, mask=mask)
    tl.store(out_ptr + out_offset, val, mask=mask)
