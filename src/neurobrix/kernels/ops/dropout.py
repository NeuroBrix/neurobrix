"""Dropout — pure @triton.jit kernel.

At inference time (train=False), dropout is identity (passthrough).
This kernel implements the identity passthrough for inference mode.
Training dropout with Philox RNG would be a separate kernel.
"""

import triton
import triton.language as tl

@triton.jit
def dropout_inference_kernel(
    x_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Dropout at inference = identity copy."""
    pid = tl.program_id(0)
    offset = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 64-bit from the program id: the product itself wraps past 2^31 elements (register 58)
    mask = offset < n_elements

    x = tl.load(x_ptr + offset, mask=mask)
    tl.store(output_ptr + offset, x, mask=mask)
