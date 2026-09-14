"""Fill — pure @triton.jit kernel.

Fills a tensor with a scalar value.
"""

import triton
import triton.language as tl

@triton.jit
def fill_kernel(
    output_ptr,
    value,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fill output with a scalar value."""
    pid = tl.program_id(0)
    offset = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 64-bit from the program id: the product itself wraps past 2^31 elements (register 58)
    mask = offset < n_elements
    tl.store(output_ptr + offset, value, mask=mask)
