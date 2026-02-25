"""
Universal Mul - Pure Triton
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _mul_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    # Use fp32 for multiplication to avoid precision loss, then clamp
    output = (x.to(tl.float32) * y.to(tl.float32))
    output = tl.minimum(tl.maximum(output, -65000.0), 65000.0)
    tl.store(out_ptr + offsets, output.to(x.dtype), mask=mask)
