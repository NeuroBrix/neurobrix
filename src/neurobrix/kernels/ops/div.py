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
    pid = tl.program_id(0).to(tl.int64)
    offset = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 64-bit from the program id: the product itself wraps past 2^31 elements (register 58)
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
    pid = tl.program_id(0).to(tl.int64)
    offset = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 64-bit from the program id: the product itself wraps past 2^31 elements (register 58)
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
    pid = tl.program_id(0).to(tl.int64)
    offset = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 64-bit from the program id: the product itself wraps past 2^31 elements (register 58)
    mask = offset < n_elements
    s = tl.load(s_ptr).to(tl.float64).to(tl.float32)
    x = tl.load(x_ptr + offset, mask=mask)
    tl.store(output_ptr + offset, x / s, mask=mask)


@triton.jit
def int_div_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    FLOOR: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """out = x // y on integers, computed in the integers: truncated toward zero (C), or toward
    -inf when FLOOR (ATen's floor_divide). Exact at every magnitude of the dtype, where a float
    round-trip is not: Metal's float32 divide is not correctly rounded (-3403868 / 13 gave
    -261836.02, floor -261837; 4435 of 10000 quotients differ from IEEE, 2026-10-08)."""
    pid = tl.program_id(0).to(tl.int64)
    offset = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 64-bit from the program id: the product itself wraps past 2^31 elements (register 58)
    mask = offset < n_elements

    x = tl.load(x_ptr + offset, mask=mask, other=0)
    y = tl.load(y_ptr + offset, mask=mask, other=1)
    q = x // y
    if FLOOR:
        r = x - q * y
        q = q - ((r != 0) & ((r < 0) != (y < 0))).to(q.dtype)
    tl.store(output_ptr + offset, q, mask=mask)
