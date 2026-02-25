"""
Universal Sub/Rsub - Pure Triton
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _sub_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    # Use fp32 for subtraction to avoid precision loss, then clamp
    output = (x.to(tl.float32) - y.to(tl.float32))
    output = tl.minimum(tl.maximum(output, -65000.0), 65000.0)
    tl.store(out_ptr + offsets, output.to(x.dtype), mask=mask)

@triton.jit
def _rsub_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """rsub(a, b) = b - a. Adapter swaps inputs, so we receive (b, a) and compute x - y = b - a."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    # Adapter swaps inputs for rsub, so x - y gives correct result
    output = (x.to(tl.float32) - y.to(tl.float32))
    output = tl.minimum(tl.maximum(output, -65000.0), 65000.0)
    tl.store(out_ptr + offsets, output.to(x.dtype), mask=mask)
