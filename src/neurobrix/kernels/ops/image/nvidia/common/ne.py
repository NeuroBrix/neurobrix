"""
Not Equal (!=) - Pure Triton
Source: FlagGems
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _ne_kernel(
    x_ptr, y_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    """Element-wise not-equal: out = (x != y)"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    x = x.to(tl.float32)
    y = y.to(tl.float32)
    output = x != y
    tl.store(out_ptr + offsets, output, mask=mask)

@triton.jit
def _ne_scalar_kernel(
    x_ptr, out_ptr, n_elements, scalar,
    BLOCK_SIZE: tl.constexpr
):
    """Element-wise not-equal with scalar: out = (x != scalar)"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    x = x.to(tl.float32)
    output = x != scalar
    tl.store(out_ptr + offsets, output, mask=mask)
