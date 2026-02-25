"""
EXP Kernel - NVIDIA (All architectures)
Element-wise exp: out = exp(x)

Source: FlagGems (adapted)
Tier: common
"""
import torch
import triton
import triton.language as tl

from neurobrix.kernels.registry import register_kernel


@triton.jit
def _exp_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    out = tl.exp(x)
    tl.store(out_ptr + offsets, out, mask=mask)


def _exp_impl(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Tensor must be on CUDA"
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    _exp_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
    return out


@register_kernel(family="video", vendor="nvidia", tier="common", op_name="exp")
def exp_kernel(x: torch.Tensor) -> torch.Tensor:
    return _exp_impl(x)
