"""
Squeeze - Remove dimensions of size 1
Type: Metadata Op (ZERO compute, uses torch view)

NeuroBrix - NVIDIA Common (All architectures)
"""
import torch

from neurobrix.kernels.registry import register_kernel


@register_kernel(family="video", vendor="nvidia", tier="common", op_name="squeeze")
def squeeze_kernel(x: torch.Tensor, dim: int = None) -> torch.Tensor:
    """
    Squeeze - Remove dimensions of size 1.
    
    Metadata operation - ZERO GPU compute.
    Uses torch.squeeze() which is just a view (no data copy).
    """
    if dim is None:
        return x.squeeze()
    else:
        return x.squeeze(dim)