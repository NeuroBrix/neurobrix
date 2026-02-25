"""
Universal Neg - Triton Optimized
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _neg_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    output = -x
    tl.store(out_ptr + offsets, output, mask=mask)

def _unary_impl(x: torch.Tensor) -> torch.Tensor:
    """Unary op implementation using pure Triton."""
    assert x.is_cuda, "Tensor must be on CUDA"
    x = x.contiguous()
    
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    if n_elements == 0:
        return output
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    _neg_kernel[grid](
        x, output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def neg_kernel(x: torch.Tensor) -> torch.Tensor:
    """
    Neg activation using pure Triton.
    
    Args:
        x: Input tensor
    
    Returns:
        Result tensor
    """
    return _unary_impl(x)
