"""
SIGMOID Kernel - NVIDIA (All architectures)
Element-wise sigmoid: out = sigmoid(x)

Source: FlagGems (adapted)
Tier: common
"""
import torch
import triton
import triton.language as tl

from neurobrix.kernels.registry import register_kernel


@triton.jit
def _sigmoid_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    out = tl.sigmoid(x)
    tl.store(out_ptr + offsets, out, mask=mask)


def _sigmoid_impl(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Tensor must be on CUDA"
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    _sigmoid_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
    return out


@register_kernel(family="audio", vendor="nvidia", tier="common", op_name="sigmoid")
def sigmoid_kernel(x: torch.Tensor) -> torch.Tensor:
    return _sigmoid_impl(x)
