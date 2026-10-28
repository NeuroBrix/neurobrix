"""Masked fill — pure @triton.jit kernel."""

import triton
import triton.language as tl

@triton.jit
def masked_fill_forward_kernel(
    input_ptr, mask_ptr, output_ptr,
    n_elements,
    fill_value,
    BLOCK_SIZE: tl.constexpr,
):
    """out = fill_value where mask is True, else input."""
    pid = tl.program_id(0).to(tl.int64)
    offset = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 64-bit from the program id: the product itself wraps past 2^31 elements (register 58)
    mask = offset < n_elements

    x = tl.load(input_ptr + offset, mask=mask)
    m = tl.load(mask_ptr + offset, mask=mask)
    out = tl.where(m, fill_value, x)
    tl.store(output_ptr + offset, out, mask=mask)
