"""
Constant - Create tensor filled with a constant value
Type: Triton Kernel (Creation)

NeuroBrix - NVIDIA Common (All architectures)
"""
import torch
import triton
import triton.language as tl
from typing import List, Union

from neurobrix.kernels.registry import register_kernel


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
    
    # Fill with constant value
    tl.store(out_ptr + offsets, value, mask=mask)


def _constant_impl(shape: List[int], value: float, dtype: torch.dtype, device: str) -> torch.Tensor:
    out = torch.empty(shape, dtype=dtype, device=device)
    n_elements = out.numel()
    
    if n_elements == 0:
        return out
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    _fill_kernel[grid](out, value, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
    return out


@register_kernel(family="llm", vendor="nvidia", tier="common", op_name="constant")
def constant_kernel(
    shape: Union[List[int], torch.Size], 
    value: float = 0.0,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Create a tensor filled with a constant value.
    
    Args:
        shape: Output tensor shape
        value: Fill value (default: 0.0)
        dtype: Output dtype (default: float32)
        device: Device (default: cuda)
    """
    return _constant_impl(list(shape), value, dtype, device)