"""SELU — pure @triton.jit kernel."""

import triton
import triton.language as tl

@triton.jit
def selu_forward_kernel(
    input_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    scale = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717

    x = tl.load(input_ptr + offset, mask=mask)
    x_fp32 = x.to(tl.float32)
    out = scale * (tl.maximum(0, x_fp32) + tl.minimum(0, alpha * (tl.exp(x_fp32) - 1.0)))
    tl.store(output_ptr + offset, out, mask=mask)
