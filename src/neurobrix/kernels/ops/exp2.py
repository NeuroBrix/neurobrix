"""Exp2 (base-2 exponential) — pure @triton.jit kernel. Logic from FlagGems."""

import triton
import triton.language as tl

@triton.jit
def exp2_forward_kernel(
    input_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    x = tl.load(input_ptr + offset, mask=mask)
    out = tl.exp2(x.to(tl.float32))
    tl.store(output_ptr + offset, out, mask=mask)
