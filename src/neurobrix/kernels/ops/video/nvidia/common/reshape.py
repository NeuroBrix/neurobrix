"""
Reshape - Reshape tensor to new shape
Type: Metadata Op (ZERO compute if contiguous, copy otherwise)

NeuroBrix - NVIDIA Common (All architectures)
"""
import torch
from typing import List, Union

from neurobrix.kernels.registry import register_kernel


@register_kernel(family="video", vendor="nvidia", tier="common", op_name="reshape")
def reshape_kernel(x: torch.Tensor, shape: Union[List[int], torch.Size]) -> torch.Tensor:
    """
    Reshape - Reshape tensor to new shape.
    
    Uses torch.reshape():
    - If tensor is contiguous and new shape is compatible: returns view (ZERO compute)
    - Otherwise: returns contiguous copy
    
    Supports -1 for automatic dimension inference.
    """
    return x.reshape(*shape)