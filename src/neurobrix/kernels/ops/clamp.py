"""Clamp — pure @triton.jit kernel."""

import triton
import triton.language as tl

@triton.jit
def clamp_forward_kernel(
    input_ptr, output_ptr,
    n_elements,
    min_val, max_val,
    has_min: tl.constexpr,
    has_max: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    x = tl.load(input_ptr + offset, mask=mask)
    if has_min:
        x = tl.maximum(min_val, x)
    if has_max:
        x = tl.minimum(max_val, x)
    tl.store(output_ptr + offset, x, mask=mask)
