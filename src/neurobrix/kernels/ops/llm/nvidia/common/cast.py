"""
Cast - Convert tensor dtype
Type: Triton Kernel (Pure Triton copy with conversion)

NeuroBrix - NVIDIA Common (All architectures)
"""
import torch
import triton
import triton.language as tl

from neurobrix.kernels.registry import register_kernel


@triton.jit
def _cast_fp32_to_fp16_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(in_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x.to(tl.float16), mask=mask)


@triton.jit
def _cast_fp16_to_fp32_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(in_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x.to(tl.float32), mask=mask)


@triton.jit
def _cast_fp32_to_bf16_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(in_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x.to(tl.bfloat16), mask=mask)


@triton.jit
def _cast_bf16_to_fp32_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(in_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x.to(tl.float32), mask=mask)


@triton.jit
def _cast_fp16_to_bf16_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(in_ptr + offsets, mask=mask)
    # fp16 -> fp32 -> bf16
    tl.store(out_ptr + offsets, x.to(tl.float32).to(tl.bfloat16), mask=mask)


@triton.jit
def _cast_bf16_to_fp16_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(in_ptr + offsets, mask=mask)
    # bf16 -> fp32 -> fp16
    tl.store(out_ptr + offsets, x.to(tl.float32).to(tl.float16), mask=mask)


# Kernel dispatch table
_CAST_KERNELS = {
    (torch.float32, torch.float16): _cast_fp32_to_fp16_kernel,
    (torch.float16, torch.float32): _cast_fp16_to_fp32_kernel,
    (torch.float32, torch.bfloat16): _cast_fp32_to_bf16_kernel,
    (torch.bfloat16, torch.float32): _cast_bf16_to_fp32_kernel,
    (torch.float16, torch.bfloat16): _cast_fp16_to_bf16_kernel,
    (torch.bfloat16, torch.float16): _cast_bf16_to_fp16_kernel,
}


def _cast_impl(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    assert x.is_cuda, "Tensor must be on CUDA"
    
    # Same dtype - return as-is
    if x.dtype == dtype:
        return x
    
    x = x.contiguous()
    out = torch.empty(x.shape, dtype=dtype, device=x.device)
    n_elements = x.numel()
    
    if n_elements == 0:
        return out
    
    # Get the appropriate kernel
    key = (x.dtype, dtype)
    if key not in _CAST_KERNELS:
        raise NotImplementedError(f"Cast from {x.dtype} to {dtype} not implemented in Triton. "
                                  f"Supported: {list(_CAST_KERNELS.keys())}")
    
    kernel = _CAST_KERNELS[key]
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
    return out


@register_kernel(family="llm", vendor="nvidia", tier="common", op_name="cast")
def cast_kernel(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """
    Cast tensor to new dtype using pure Triton.
    
    Supported conversions:
    - float32 <-> float16
    - float32 <-> bfloat16
    - float16 <-> bfloat16
    
    Args:
        x: Input tensor
        dtype: Target dtype
    """
    return _cast_impl(x, dtype)
