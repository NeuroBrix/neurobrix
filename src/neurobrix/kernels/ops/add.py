"""Add — pure @triton.jit kernel (tensor + tensor, tensor + scalar)."""

import triton
import triton.language as tl

@triton.jit
def add_forward_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    alpha,
    BLOCK_SIZE: tl.constexpr,
):
    """out = x + alpha * y (tensor + tensor)"""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    x = tl.load(x_ptr + offset, mask=mask)
    y = tl.load(y_ptr + offset, mask=mask)
    tl.store(output_ptr + offset, x + alpha * y, mask=mask)

@triton.jit
def add_scalar_kernel(
    x_ptr, output_ptr,
    n_elements,
    scalar,
    BLOCK_SIZE: tl.constexpr,
):
    """out = x + scalar"""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    x = tl.load(x_ptr + offset, mask=mask)
    tl.store(output_ptr + offset, x + scalar, mask=mask)
