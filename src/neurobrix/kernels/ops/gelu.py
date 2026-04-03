"""GELU — pure @triton.jit kernel (exact and approximate)."""

import triton
import triton.language as tl

from ._common import tanh_fn

@triton.jit
def gelu_forward_kernel(
    input_ptr, output_ptr,
    n_elements,
    approximate: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """GELU forward. approximate=True for tanh approximation, False for exact."""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    x = tl.load(input_ptr + offset, mask=mask)
    x_fp32 = x.to(tl.float32)

    if approximate:
        # Tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        cdf = 0.5 * (1.0 + tanh_fn(0.7978845608 * x_fp32 * (1.0 + 0.044715 * x_fp32 * x_fp32)))
    else:
        # Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
        cdf = 0.5 * (1.0 + tl.math.erf(0.707106781 * x_fp32))

    out = cdf * x_fp32
    tl.store(output_ptr + offset, out, mask=mask)
