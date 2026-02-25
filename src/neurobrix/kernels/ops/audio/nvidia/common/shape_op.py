"""
Shape - Return tensor shape as tensor
ATen: aten::size
"""
import torch

from neurobrix.kernels.registry import register_kernel

@register_kernel(family="audio", vendor="nvidia", tier="common", op_name="shape")
def shape_kernel(x: torch.Tensor) -> torch.Tensor:
    """Return tensor shape as tensor."""
    return torch.tensor(list(x.shape), dtype=torch.int64, device=x.device)

