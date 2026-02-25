"""
Flatten - Flatten tensor dimensions
Type: Metadata Op (ZERO compute if contiguous, copy otherwise)

NeuroBrix - NVIDIA Common (All architectures)
"""
import torch

from neurobrix.kernels.registry import register_kernel


@register_kernel(family="video", vendor="nvidia", tier="common", op_name="flatten")
def flatten_kernel(x: torch.Tensor, start_dim: int = 0, end_dim: int = -1) -> torch.Tensor:
    """
    Flatten - Flatten dimensions from start_dim to end_dim.
    
    Uses torch.flatten():
    - If tensor is contiguous: returns view (ZERO compute)
    - If tensor is non-contiguous: returns contiguous copy
    
    This is the optimal behavior - PyTorch handles it automatically.
    """
    return x.flatten(start_dim, end_dim)