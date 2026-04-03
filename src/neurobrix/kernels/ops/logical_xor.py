"""Logical XOR — pure @triton.jit kernel.

Computes: output = (x != 0) XOR (y != 0)

Result is boolean (stored as int8: 0 or 1).

Extracted from FlagGems reference (logical_xor.py).
"""

import triton
import triton.language as tl

@triton.jit
def logical_xor_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Logical XOR: out = (x != 0) ^ (y != 0).

    Output is int8 (0 or 1).
    """
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    x = tl.load(x_ptr + offset, mask=mask, other=0.0)
    y = tl.load(y_ptr + offset, mask=mask, other=0.0)

    # Convert to bool then XOR
    x_bool = (x != 0).to(tl.int1)
    y_bool = (y != 0).to(tl.int1)
    result = x_bool ^ y_bool

    tl.store(output_ptr + offset, result.to(tl.int8), mask=mask)
