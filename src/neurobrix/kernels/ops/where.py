"""Where — pure @triton.jit kernel."""

import triton
import triton.language as tl

@triton.jit
def where_forward_kernel(
    cond_ptr, x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """out = where(cond, x, y)"""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    cond = tl.load(cond_ptr + offset, mask=mask)
    x = tl.load(x_ptr + offset, mask=mask)
    y = tl.load(y_ptr + offset, mask=mask)
    tl.store(output_ptr + offset, tl.where(cond, x, y), mask=mask)
