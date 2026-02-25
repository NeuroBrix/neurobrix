"""
Expand - Broadcast tensor to new shape
Type: Metadata Op (ZERO compute, uses torch view)

NeuroBrix - NVIDIA Common (All architectures)
"""
import torch
from typing import List, Union

from neurobrix.kernels.registry import register_kernel


@register_kernel(family="audio", vendor="nvidia", tier="common", op_name="expand")
def expand_kernel(x: torch.Tensor, size: Union[List[int], torch.Size]) -> torch.Tensor:
    """
    Expand - Broadcast tensor to new shape.
    
    Metadata operation - ZERO GPU compute.
    Uses torch.expand() which is just a view with stride=0 for broadcast dims.
    No data is copied.
    """
    # handle size if it is a single list/tuple argument or varargs (though signature enforces one arg)
    # torch.expand takes varargs usually, but here we expect 'size' as a list/tuple
    return x.expand(*size)