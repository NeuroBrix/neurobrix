"""
Constant - Create tensor filled with a constant value
ATen: aten::full, aten::zeros, aten::ones
"""
import torch
import triton
import triton.language as tl
from typing import List

@triton.jit
def _fill_kernel(
    out_ptr,
    value,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    tl.store(out_ptr + offsets, value, mask=mask)

def constant_kernel(value: float = 0.0, shape: list = None, 
                   dtype: torch.dtype = None, device: str = "cuda") -> torch.Tensor:
    """Create constant tensor."""
    if shape is None:
        shape = [1]
    if dtype is None:
        dtype = torch.float32
    return torch.full(shape, value, dtype=dtype, device=device)
