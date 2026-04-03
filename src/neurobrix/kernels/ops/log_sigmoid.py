"""Log-sigmoid — pure @triton.jit kernel. Logic from FlagGems."""

import triton
import triton.language as tl

@triton.jit
def log_sigmoid_forward_kernel(
    input_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    x = tl.load(input_ptr + offset, mask=mask)
    x_fp32 = x.to(tl.float32)
    out = tl.where(
        x_fp32 >= 0,
        -tl.log(1.0 + tl.exp(-x_fp32)),
        x_fp32 - tl.log(1.0 + tl.exp(x_fp32)),
    )
    tl.store(output_ptr + offset, out, mask=mask)
