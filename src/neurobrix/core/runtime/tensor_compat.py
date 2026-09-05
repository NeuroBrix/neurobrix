"""Dual-tensor type compatibility helpers.

The triton runtime substitutes NBXTensor for torch.Tensor at component
boundaries. isinstance(x, torch.Tensor) returns False for NBXTensor, so
every site that gates logic on isinstance(*, torch.Tensor) silently
short-circuits in --triton mode.

This module centralizes the dual-type recognition — and it never imports
torch (R33). An object can only be a torch.Tensor if the ATen library is
already loaded in the process, so the check reads ``sys.modules``: exact
on the compiled branch, and free of any import on the Triton branch, where
torch is absent by construction.
"""
from __future__ import annotations

import sys
from typing import Any, Optional


def aten() -> Optional[Any]:
    """The ATen library (torch) if the process has loaded it, else None.

    Shared orchestration code that must act differently when the compiled
    branch owns the device (its caching allocator, its RNG) asks here. It
    never imports the library: on the Triton branch the answer is None
    for the whole life of the process (R33).
    """
    return sys.modules.get("torch")


def is_torch_tensor(x: Any) -> bool:
    """True for a torch.Tensor — without importing torch."""
    t = sys.modules.get("torch")
    return t is not None and isinstance(x, t.Tensor)


def is_nbx_tensor(x: Any) -> bool:
    """True for an NBXTensor (the kernel layer is imported lazily)."""
    try:
        from neurobrix.kernels.nbx_tensor import NBXTensor
    except ImportError:
        return False
    return isinstance(x, NBXTensor)


def is_tensor(x: Any) -> bool:
    """True for both torch.Tensor and NBXTensor."""
    return is_torch_tensor(x) or is_nbx_tensor(x)
