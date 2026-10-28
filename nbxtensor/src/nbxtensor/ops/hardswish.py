"""Hard Swish — pure @triton.jit kernel."""

import triton
import triton.language as tl

@triton.jit
def hardswish_forward_kernel(
    input_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offset = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 64-bit from the program id: the product itself wraps past 2^31 elements (register 58)
    mask = offset < n_elements

    x = tl.load(input_ptr + offset, mask=mask)
    # hardswish = x * relu6(x + 3) / 6
    out = x * tl.minimum(tl.maximum(0, x + 3.0), 6.0) / 6.0
    tl.store(output_ptr + offset, out, mask=mask)
