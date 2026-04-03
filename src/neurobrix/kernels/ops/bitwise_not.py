"""Bitwise NOT kernel — extracted from FlagGems.

Pure Triton. ZERO PyTorch imports.
Logic: result = ~x
"""

import triton
import triton.language as tl

@triton.jit
def bitwise_not_forward_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    x = tl.load(x_ptr + offset, mask=mask)

    result = ~x
    tl.store(output_ptr + offset, result, mask=mask)
