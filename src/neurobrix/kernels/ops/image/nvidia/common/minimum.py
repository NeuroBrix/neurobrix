"""
Universal Minimum - Pure Triton
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _minimum_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = tl.minimum(x, y)
    tl.store(out_ptr + offsets, output, mask=mask)
