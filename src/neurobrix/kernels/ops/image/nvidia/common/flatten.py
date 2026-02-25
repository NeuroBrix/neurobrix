"""
Flatten - Flatten tensor dimensions
ATen: aten::flatten
"""
import torch

def flatten_kernel(x: torch.Tensor, start_dim: int = 0, end_dim: int = -1) -> torch.Tensor:
    """Flatten tensor."""
    return x.flatten(start_dim, end_dim)
