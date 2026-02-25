"""
Unsqueeze - Add dimension of size 1
ATen: aten::unsqueeze
"""
import torch

def unsqueeze_kernel(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Unsqueeze tensor."""
    return x.unsqueeze(dim)
