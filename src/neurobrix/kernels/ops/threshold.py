"""Threshold activation — pure @triton.jit kernel. Logic from FlagGems."""

import triton
import triton.language as tl

@triton.jit
def threshold_forward_kernel(
    input_ptr, output_ptr,
    n_elements,
    threshold_val, value,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    x = tl.load(input_ptr + offset, mask=mask)
    out = tl.where(x > threshold_val, x, value)
    tl.store(output_ptr + offset, out, mask=mask)
