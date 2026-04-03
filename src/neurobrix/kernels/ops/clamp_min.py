"""Clamp min — pure @triton.jit kernel.

aten::clamp_min(x, min_val) = max(x, min_val). Separate file per ATen op convention.
"""

import triton
import triton.language as tl

@triton.jit
def clamp_min_forward_kernel(
    input_ptr, output_ptr,
    n_elements,
    min_val,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    x = tl.load(input_ptr + offset, mask=mask)
    result = tl.maximum(min_val, x)
    tl.store(output_ptr + offset, result, mask=mask)
