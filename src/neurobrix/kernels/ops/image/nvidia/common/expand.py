"""
Expand - Broadcast tensor to new shape
ATen: aten::expand
"""
import torch
from typing import List

def expand_kernel(x: torch.Tensor, size: list = None) -> torch.Tensor:
    """Expand tensor to new size."""
    if size is None:
        return x
    return x.expand(*size)
