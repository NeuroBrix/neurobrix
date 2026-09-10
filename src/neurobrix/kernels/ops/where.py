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
    # `.to(tl.int1)` and not `cond` alone: a boolean crosses the NBX boundary in
    # a uint8 container (that is how a bool tensor is stored), so the loaded
    # value is an INTEGER. `tl.where` accepts that today with a deprecation
    # warning and will raise on it in a future version — one of the two
    # scheduled breakages a user's A40 report surfaced, four warnings per run.
    #
    # This is a forward-compatibility edit and it changes NO instruction: the
    # cast `tl.where` already performed internally on a non-boolean condition is
    # `not_equal(x, null_of(x.dtype))` (semantic.py, the is_bool() branch of
    # cast), which is exactly what `.to(tl.int1)` emits. Proven, not asserted —
    # the TTIR is identical to the pre-edit form
    # (`tools/kernel_boolean_ir_equality.py`).
    #
    # Why NOT `cond != 0`, which was the first form written here: the Python
    # literal `0` promotes to i32, so the comparison gains an `arith.extsi`
    # i8 -> i32 on every element before it. Same boolean for every input —
    # extsi maps 0 to 0 and is injective — but one extra elementwise op, on the
    # kernel that 39 of the 56 local containers reach. "Costs nothing" was a
    # claim nobody had checked; the instrument checked it.
    tl.store(output_ptr + offset, tl.where(cond.to(tl.int1), x, y), mask=mask)
