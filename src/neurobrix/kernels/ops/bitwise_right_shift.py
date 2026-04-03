"""Bitwise right shift — pure @triton.jit kernel.

Computes: output = x >> y (element-wise, arithmetic shift).

Both inputs must be integer tensors.

Extracted from FlagGems reference (bitwise_right_shift.py).
"""

import triton
import triton.language as tl

@triton.jit
def bitwise_right_shift_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Bitwise right shift: out = x >> y."""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    x = tl.load(x_ptr + offset, mask=mask)
    y = tl.load(y_ptr + offset, mask=mask)

    result = x >> y
    tl.store(output_ptr + offset, result, mask=mask)

@triton.jit
def bitwise_right_shift_scalar_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    shift_amount,
    BLOCK_SIZE: tl.constexpr,
):
    """Bitwise right shift by scalar: out = x >> shift_amount."""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    x = tl.load(x_ptr + offset, mask=mask)
    result = x >> shift_amount
    tl.store(output_ptr + offset, result, mask=mask)
