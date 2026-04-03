"""Softplus activation — pure @triton.jit kernel.

Logic from FlagGems softplus.py: softplus(x) = log(1 + exp(beta * x)) / beta.
For large beta*x, returns x directly (numerical stability).
"""

import triton
import triton.language as tl

@triton.jit
def softplus_forward_kernel(
    x_ptr, output_ptr,
    n_elements,
    beta,
    threshold,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    x = tl.load(x_ptr + offset, mask=mask)
    x_fp = x.to(tl.float32)
    z = x_fp * beta
    soft_z = tl.where(z > threshold, z, tl.log(1.0 + tl.exp(z)))
    result = soft_z / beta
    tl.store(output_ptr + offset, result, mask=mask)
