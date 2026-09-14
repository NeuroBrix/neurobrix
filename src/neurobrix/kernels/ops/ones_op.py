"""Ones — pure @triton.jit kernel.

Fills a tensor with ones. Specialization of fill.
"""

import triton
import triton.language as tl

@triton.jit
def ones_kernel(
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fill output with ones."""
    pid = tl.program_id(0)
    offset = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 64-bit from the program id: the product itself wraps past 2^31 elements (register 58)
    mask = offset < n_elements
    one = tl.full([BLOCK_SIZE], 1.0, dtype=output_ptr.type.element_ty)
    tl.store(output_ptr + offset, one, mask=mask)
