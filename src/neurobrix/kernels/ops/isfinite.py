"""IsFinite — pure @triton.jit kernel. Logic from FlagGems."""

import triton
import triton.language as tl

@triton.jit
def isfinite_forward_kernel(
    input_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 64-bit from the program id: the product itself wraps past 2^31 elements (register 58)
    mask = offset < n_elements

    x = tl.load(input_ptr + offset, mask=mask)
    pos_inf = float('inf')
    neg_inf = float('-inf')
    out = (x == x) & (x != pos_inf) & (x != neg_inf)
    tl.store(output_ptr + offset, out, mask=mask)
