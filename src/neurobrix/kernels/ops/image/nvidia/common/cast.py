"""
Universal Cast Kernel - ATen to() operation
"""
import torch
from typing import List, Dict, Any

def cast_kernel(x: torch.Tensor, dtype: torch.dtype = None) -> torch.Tensor:
    """Cast tensor to dtype."""
    if dtype is None:
        return x
    return x.to(dtype)
