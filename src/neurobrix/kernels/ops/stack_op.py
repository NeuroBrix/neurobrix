"""Stack — pure @triton.jit kernel.

Stack = unsqueeze + cat along new dim. Copies up to 4 input tensors
per kernel launch into the output tensor at their respective offsets.
Extracted from FlagGems (Apache 2.0).

Grid: (ceil(max_elements / BLOCK_X), num_tensors_in_batch)
"""

import triton
import triton.language as tl


@triton.jit
def stack_copy_kernel(
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
    """Copies up to 4 tensors into stacked output at correct offsets.

    pid_y selects which of the 4 tensors this program handles.
    pid_x tiles over the elements of that tensor.

    The output index is computed from the input flat index by decomposing
    into (pre, in_dim, post) and inserting the stack dimension offset.
    """
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)

    if pid_y == 0:
        in_ptr = in_ptr_a
        dim_offset = dim_offset_a
        total_elements = total_elements_a
    elif pid_y == 1:
        in_ptr = in_ptr_b
        dim_offset = dim_offset_b
        total_elements = total_elements_b
    elif pid_y == 2:
        in_ptr = in_ptr_c
        dim_offset = dim_offset_c
        total_elements = total_elements_c
    else:
        in_ptr = in_ptr_d
        dim_offset = dim_offset_d
        total_elements = total_elements_d

    block_start = pid_x * BLOCK_X
    offsets = tl.arange(0, BLOCK_X)
    idx = block_start + offsets
    mask = idx < total_elements

    # Decompose flat index into (pre_idx, post_idx) around the stack dim
    pre_idx = idx // dim_prod_post
    post_idx = idx % dim_prod_post

    # Compute output index with stack dimension inserted
    out_idx = (
        pre_idx * dim_size_out * dim_prod_post
        + dim_offset * dim_prod_post
        + post_idx
    )

    data = tl.load(in_ptr + idx, mask=mask)
    tl.store(out_ptr + out_idx, data, mask=mask)
