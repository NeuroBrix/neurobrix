"""atan2(y, x) — pure @triton.jit kernel (element-wise, fp32 via libdevice)."""

import triton
import triton.language as tl


@triton.jit
def atan2_kernel(y_ptr, x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    y = tl.load(y_ptr + offs, mask=mask).to(tl.float32)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    tl.store(out_ptr + offs, tl.extra.cuda.libdevice.atan2(y, x), mask=mask)
