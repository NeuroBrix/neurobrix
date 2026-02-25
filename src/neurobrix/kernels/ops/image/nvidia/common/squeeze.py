"""
Squeeze - Remove dimensions of size 1
ATen: aten::squeeze
"""
import torch

def squeeze_kernel(x: torch.Tensor, dim: int = None) -> torch.Tensor:
    """Squeeze tensor."""
    if dim is None:
        return x.squeeze()
    return x.squeeze(dim)
