"""Power — pure @triton.jit kernel."""

import triton
import triton.language as tl

@triton.jit
def pow_forward_kernel(
    input_ptr, output_ptr,
    n_elements,
    exponent,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    x = tl.load(input_ptr + offset, mask=mask)
    x_fp32 = x.to(tl.float32)
    # Use libm pow for correctness with negative inputs and arbitrary exponents.
    # tl.exp(e * tl.log(x)) fails for negative x (log of negative = NaN).
    # tl.math.pow is the safe path — handles negative bases, integer/float exponents.
    out = tl.extra.cuda.libdevice.pow(x_fp32, exponent)
    tl.store(output_ptr + offset, out, mask=mask)
