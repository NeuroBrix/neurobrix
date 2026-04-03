"""Full (fill) — pure @triton.jit kernel.

Creates a tensor filled with a scalar value.
Simple element-wise write: tl.store(out + offset, value, mask=mask).

Grid: (ceil(n_elements / BLOCK_SIZE),)
"""

import triton
import triton.language as tl

@triton.jit
def full_kernel(
    output_ptr,
    fill_value,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fill output tensor with a constant scalar value."""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    tl.store(output_ptr + offset, fill_value, mask=mask)
