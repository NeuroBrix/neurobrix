"""
MINIMUM Kernel - NVIDIA (All architectures)
Element-wise minimum: out = minimum(x, y)

Source: FlagGems (adapted)
Tier: common
"""
import torch
import triton
import triton.language as tl

from neurobrix.kernels.registry import register_kernel


@triton.jit
def _minimum_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    out = tl.minimum(x, y)
    tl.store(out_ptr + offsets, out, mask=mask)


def _minimum_impl(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda, "Tensors must be on CUDA"
    # Assuming same shape for simplicity as per batch instruction
    x = x.contiguous()
    y = y.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    _minimum_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
    return out


@register_kernel(family="audio", vendor="nvidia", tier="common", op_name="minimum")
def minimum_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return _minimum_impl(x, y)
