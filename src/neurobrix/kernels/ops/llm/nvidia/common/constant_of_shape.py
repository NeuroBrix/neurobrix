"""
ConstantOfShape - Create tensor of given shape filled with value
ATen: aten::full, aten::zeros, aten::ones
"""
import torch
import triton
import triton.language as tl
from typing import Union

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
    
    tl.store(out_ptr + offsets, value, mask=mask)

def _constant_of_shape_impl(
    shape: list,
    value: float = 0.0,
    dtype: torch.dtype = None,
    device: str = "cuda",
) -> torch.Tensor:
    """ConstantOfShape implementation."""
    if dtype is None:
        dtype = torch.float32
    return torch.full(shape, value, dtype=dtype, device=device)


@register_kernel(family="llm", vendor="nvidia", tier="common", op_name="constant_of_shape")
def constant_of_shape_kernel(
    shape: list,
    value: float = 0.0,
    dtype: torch.dtype = None,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Create constant tensor of given shape.
    """
    return _constant_of_shape_impl(shape, value, dtype, device)

