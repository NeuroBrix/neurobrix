"""
Universal Erf Kernel - ATen erf
"""
import torch
from typing import List, Dict, Any

def _erf_impl(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    x = x.contiguous()
    output = torch.empty_like(x)
    n = output.numel()
    if n == 0:
        return output
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    _erf_kernel[grid](x, output, n, BLOCK_SIZE=BLOCK)
    return output

def erf_kernel(x: torch.Tensor) -> torch.Tensor:
    """Error function."""
    return _erf_impl(x)
