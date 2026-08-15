"""Division — pure @triton.jit kernel."""

import triton
import triton.language as tl

@triton.jit
def div_forward_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """out = x / y (tensor / tensor)"""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    x = tl.load(x_ptr + offset, mask=mask)
    y = tl.load(y_ptr + offset, mask=mask)
    tl.store(output_ptr + offset, x / y, mask=mask)

@triton.jit
def div_scalar_kernel(
    x_ptr, output_ptr,
    n_elements,
    scalar,
    BLOCK_SIZE: tl.constexpr,
):
    """out = x / scalar"""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    x = tl.load(x_ptr + offset, mask=mask)
    tl.store(output_ptr + offset, x / scalar, mask=mask)


@triton.jit
def div_scalar_dev_kernel(
    x_ptr, s_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """out = x / s where s is a 0-d DEVICE tensor. Bit-exact mirror
    of div_scalar_kernel (host float(s) -> f32 arg == load -> f64 ->
    f32). Device-scalar increment 2026-08-15."""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    s = tl.load(s_ptr).to(tl.float64).to(tl.float32)
    x = tl.load(x_ptr + offset, mask=mask)
    tl.store(output_ptr + offset, x / s, mask=mask)
