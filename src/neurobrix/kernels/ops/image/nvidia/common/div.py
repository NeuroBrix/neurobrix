"""
Universal Div - Pure Triton (Robust)
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _div_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Avoid div by zero - use a safer threshold
    y_safe = tl.where(tl.abs(y) < 1e-7, tl.where(y >= 0, 1e-7, -1e-7), y)

    output = x / y_safe
    # Clamp output to prevent fp16 overflow
    output = tl.minimum(tl.maximum(output, -65000.0), 65000.0)
    tl.store(out_ptr + offsets, output, mask=mask)
