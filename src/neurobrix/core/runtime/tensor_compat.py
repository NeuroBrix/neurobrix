"""Dual-tensor type compatibility helpers.

The triton runtime substitutes NBXTensor for torch.Tensor at component
boundaries. isinstance(x, torch.Tensor) returns False for NBXTensor, so
every site that gates logic on isinstance(*, torch.Tensor) silently
short-circuits in --triton mode.

This module centralizes the dual-type recognition. NBXTensor is imported
lazily so core handlers and runtime modules don't carry a hard dependency
on the triton subsystem.
"""
from __future__ import annotations

from typing import Any

import torch


def is_tensor(x: Any) -> bool:
    """True for both torch.Tensor and NBXTensor."""
    if isinstance(x, torch.Tensor):
        return True
    try:
        from neurobrix.kernels.nbx_tensor import NBXTensor
        return isinstance(x, NBXTensor)
    except ImportError:
        return False
