"""Concatenation — pure @triton.jit kernel.

Ported from FlagGems cat_copy_func_kernel_4 (Apache-2.0 license).
Copies up to 4 input tensors per launch into the correct position
in the output tensor along the cat dimension.
"""

import triton
import triton.language as tl


@triton.jit
def cat_copy_kernel_4(
    out_ptr,
    in_ptr_a,
    in_ptr_b,
    in_ptr_c,
    in_ptr_d,
    dim_size_in_a,
    dim_size_in_b,
    dim_size_in_c,
    dim_size_in_d,
    dim_size_out,
    dim_prod_post,
    dim_offset_a,
    dim_offset_b,
    dim_offset_c,
    dim_offset_d,
    total_elements_a,
    total_elements_b,
    total_elements_c,
    total_elements_d,
    BLOCK_X: tl.constexpr,
):
    """Copy up to 4 input tensors into their correct positions in the output.

    pid_x indexes blocks within a single input tensor.
    pid_y selects which of the 4 input tensors this program handles.

    The index remapping converts flat input indices to flat output indices
    by splitting into (pre_dim, cat_dim, post_dim) components and applying
    the cat dimension offset.
    """
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)

    # Select the correct input tensor and its metadata
    if pid_y == 0:
        in_ptr = in_ptr_a
        dim_size_in = dim_size_in_a
        dim_offset = dim_offset_a
        total_elements = total_elements_a
    elif pid_y == 1:
        in_ptr = in_ptr_b
        dim_size_in = dim_size_in_b
        dim_offset = dim_offset_b
        total_elements = total_elements_b
    elif pid_y == 2:
        in_ptr = in_ptr_c
        dim_size_in = dim_size_in_c
        dim_offset = dim_offset_c
        total_elements = total_elements_c
    else:
        in_ptr = in_ptr_d
        dim_size_in = dim_size_in_d
        dim_offset = dim_offset_d
        total_elements = total_elements_d

    block_start = pid_x * BLOCK_X
    offsets = tl.arange(0, BLOCK_X)
    mask = block_start + offsets < total_elements

    idx = block_start + offsets

    # Decompose flat index into (pre, dim, post) coordinates
    pre_idx = idx // (dim_size_in * dim_prod_post)
    dim_idx = (idx // dim_prod_post) % dim_size_in
    post_idx = idx % dim_prod_post

    # Compute output index with the cat dimension offset applied
    out_idx = (
        pre_idx * dim_size_out * dim_prod_post
        + (dim_idx + dim_offset) * dim_prod_post
        + post_idx
    )

    data = tl.load(in_ptr + idx, mask=mask)
    tl.store(out_ptr + out_idx, data, mask=mask)
