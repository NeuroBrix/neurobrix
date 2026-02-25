"""
Unsqueeze - Add dimension of size 1
Type: Metadata Op (ZERO compute, uses torch view)

NeuroBrix - NVIDIA Common (All architectures)
"""
import torch

from neurobrix.kernels.registry import register_kernel


@register_kernel(family="llm", vendor="nvidia", tier="common", op_name="unsqueeze")
def unsqueeze_kernel(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Unsqueeze - Add dimension of size 1 at position dim.
    
    Metadata operation - ZERO GPU compute.
    Uses torch.unsqueeze() which is just a view (no data copy).
    """
    return x.unsqueeze(dim)