"""
Shape - Return tensor shape as tensor
ATen: aten::size
"""
import torch

def shape_kernel(x: torch.Tensor) -> torch.Tensor:
    """Return tensor shape as tensor."""
    return torch.tensor(list(x.shape), dtype=torch.int64, device=x.device)
