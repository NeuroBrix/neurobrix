"""Less than — pure @triton.jit kernel."""

import triton
import triton.language as tl

@triton.jit
def lt_forward_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 64-bit from the program id: the product itself wraps past 2^31 elements (register 58)
    mask = offset < n_elements

    x = tl.load(x_ptr + offset, mask=mask)
    y = tl.load(y_ptr + offset, mask=mask)
    tl.store(output_ptr + offset, (x < y).to(tl.int1), mask=mask)

@triton.jit
def lt_scalar_kernel(
    x_ptr, output_ptr,
    n_elements,
    scalar,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 64-bit from the program id: the product itself wraps past 2^31 elements (register 58)
    mask = offset < n_elements

    x = tl.load(x_ptr + offset, mask=mask)
    tl.store(output_ptr + offset, (x < scalar).to(tl.int1), mask=mask)
