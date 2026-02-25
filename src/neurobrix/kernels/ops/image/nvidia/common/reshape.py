"""
Reshape - Reshape tensor to new shape
ATen: aten::reshape, aten::view
"""
import torch
from typing import List

def reshape_kernel(x: torch.Tensor, shape: list = None, size: list = None) -> torch.Tensor:
    """Reshape tensor."""
    target_shape = shape if shape is not None else size
    if target_shape is None:
        return x
    return x.reshape(*target_shape)
