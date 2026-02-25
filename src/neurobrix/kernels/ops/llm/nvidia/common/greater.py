"""
Greater - Element-wise comparison: out = (x > y)
Type: Triton Kernel

NeuroBrix - NVIDIA Common (All architectures)
"""
import torch
import triton
import triton.language as tl

from neurobrix.kernels.registry import register_kernel


@triton.jit
def _greater_kernel(
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
    out = (x > y)
    tl.store(out_ptr + offsets, out, mask=mask)


def _greater_impl(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda, "Tensors must be on CUDA"
    
    if x.shape != y.shape:
        x, y = torch.broadcast_tensors(x, y)
    
    x = x.contiguous()
    y = y.contiguous()
    out = torch.empty(x.shape, dtype=torch.bool, device=x.device)
    n_elements = x.numel()
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    _greater_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
    return out


@register_kernel(family="llm", vendor="nvidia", tier="common", op_name="greater")
def greater_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Element-wise greater: out = (x > y)"""
    return _greater_impl(x, y)