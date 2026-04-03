"""CELU activation — pure @triton.jit kernel. Logic from FlagGems."""

import triton
import triton.language as tl

@triton.jit
def celu_forward_kernel(
    input_ptr, output_ptr,
    n_elements,
    alpha,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    x = tl.load(input_ptr + offset, mask=mask)
    x_fp32 = x.to(tl.float32)
    out = tl.where(x_fp32 > 0, x_fp32, alpha * (tl.exp(x_fp32 / alpha) - 1.0))
    tl.store(output_ptr + offset, out, mask=mask)
