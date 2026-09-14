"""Logical OR kernel — extracted from FlagGems.

Pure Triton. ZERO PyTorch imports.
Logic: result = (x != 0) | (y != 0)
"""

import triton
import triton.language as tl

@triton.jit
def logical_or_forward_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 64-bit from the program id: the product itself wraps past 2^31 elements (register 58)
    mask = offset < n_elements

    x = tl.load(x_ptr + offset, mask=mask)
    y = tl.load(y_ptr + offset, mask=mask)

    result = (x != 0) | (y != 0)
    tl.store(output_ptr + offset, result, mask=mask)
