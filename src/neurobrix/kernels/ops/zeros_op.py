"""Zeros — pure @triton.jit kernel.

Fills a tensor with zeros. Specialization of fill.
"""

import triton
import triton.language as tl

@triton.jit
def zeros_kernel(
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fill output with zeros."""
    pid = tl.program_id(0)
    offset = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 64-bit from the program id: the product itself wraps past 2^31 elements (register 58)
    mask = offset < n_elements
    zero = tl.zeros([BLOCK_SIZE], dtype=output_ptr.type.element_ty)
    tl.store(output_ptr + offset, zero, mask=mask)
