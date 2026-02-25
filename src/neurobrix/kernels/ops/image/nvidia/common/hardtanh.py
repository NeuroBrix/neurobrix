"""
Hard Tanh - Pure Triton
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _hardtanh_kernel(
    x_ptr, out_ptr, n_elements,
    min_val, max_val,
    BLOCK_SIZE: tl.constexpr
):
    """Hard Tanh: clamp(x, min_val, max_val)"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.maximum(min_val, tl.minimum(max_val, x))
    tl.store(out_ptr + offsets, output, mask=mask)

@triton.jit
def _hardtanh_grad_kernel(
    grad_out_ptr, x_ptr, grad_in_ptr, n_elements,
    min_val, max_val,
    BLOCK_SIZE: tl.constexpr
):
    """Backward: grad_in = grad_out * (1 if min < x < max else 0)"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    grad_out = tl.load(grad_out_ptr + offsets, mask=mask)
    x = tl.load(x_ptr + offsets, mask=mask)
    grad_in = tl.where((min_val < x) & (x < max_val), 1.0, 0.0) * grad_out
    tl.store(grad_in_ptr + offsets, grad_in, mask=mask)
