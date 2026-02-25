"""
Universal Rsqrt - Pure Triton (Robust)
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _rsqrt_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    # Robust clamping to avoid div by zero or huge values
    x = tl.maximum(x, 1e-10)
    output = tl.rsqrt(x)
    tl.store(out_ptr + offsets, output, mask=mask)
