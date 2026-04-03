"""Remainder (modulo) — pure @triton.jit kernel."""

import triton
import triton.language as tl

@triton.jit
def remainder_forward_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    x = tl.load(x_ptr + offset, mask=mask)
    y = tl.load(y_ptr + offset, mask=mask)
    result = x % y
    tl.store(output_ptr + offset, result, mask=mask)

@triton.jit
def remainder_scalar_kernel(
    x_ptr, output_ptr,
    n_elements,
    divisor,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    x = tl.load(x_ptr + offset, mask=mask)
    result = x % divisor
    tl.store(output_ptr + offset, result, mask=mask)
