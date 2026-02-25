"""
Transpose - Permute tensor dimensions
ATen: aten::permute, aten::transpose
"""
import torch

def transpose_kernel(x: torch.Tensor, dim0: int = 0, dim1: int = 1) -> torch.Tensor:
    """Transpose tensor."""
    return x.transpose(dim0, dim1)
