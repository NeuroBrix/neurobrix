"""Hard Sigmoid — pure @triton.jit kernel."""

import triton
import triton.language as tl

@triton.jit
def hardsigmoid_forward_kernel(
    input_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    x = tl.load(input_ptr + offset, mask=mask)
    out = tl.maximum(0, tl.minimum(1, x / 6.0 + 0.5))
    tl.store(output_ptr + offset, out, mask=mask)
