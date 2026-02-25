"""
Softsign - Pure Triton
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _softsign_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Softsign: x / (1 + |x|)"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    output = x / (1.0 + tl.abs(x))
    tl.store(out_ptr + offsets, output, mask=mask)

@triton.jit
def _softsign_grad_kernel(grad_out_ptr, x_ptr, grad_in_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Backward: grad_in = grad_out / (1 + |x|)^2"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    grad_out = tl.load(grad_out_ptr + offsets, mask=mask)
    x = tl.load(x_ptr + offsets, mask=mask)
    denom = 1.0 + tl.abs(x)
    grad_in = grad_out / (denom * denom)
    tl.store(grad_in_ptr + offsets, grad_in, mask=mask)
