"""Where — pure @triton.jit kernel."""

import triton
import triton.language as tl

@triton.jit
def where_forward_kernel(
    cond_ptr, x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """out = where(cond, x, y)"""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    cond = tl.load(cond_ptr + offset, mask=mask)
    x = tl.load(x_ptr + offset, mask=mask)
    y = tl.load(y_ptr + offset, mask=mask)
    # `!= 0` and not `cond` alone: a boolean crosses the NBX boundary in a
    # uint8 container (that is how a bool tensor is stored), so the loaded
    # value is an INTEGER. Triton accepts a non-boolean condition today with a
    # deprecation warning and will raise on it in a future version — one of the
    # two scheduled breakages a user's A40 report surfaced, four warnings per
    # run. Comparing here is exact for any integer width and costs nothing.
    tl.store(output_ptr + offset, tl.where(cond != 0, x, y), mask=mask)
