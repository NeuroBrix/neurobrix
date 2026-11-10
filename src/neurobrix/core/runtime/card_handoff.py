"""Moves a torch tensor from one card to another without the framework's device-to-device copy.

Why it exists. A run or a daemon holds a context only on the cards its plan names (the profile is the machine's
identity card; a card nobody planned is somebody else's). Since PyTorch 2.10 the first GPU-to-GPU `tensor.to(card)` of
a process builds an event pool that creates events — hence a context, 306 MiB on a V100 — on EVERY visible card
(aten/src/ATen/native/cuda/Copy.cu, c10/cuda/CUDAEvent.h; upstream PR #165647; no switch). Measured 2026-11-10 with
the same five lines: PyTorch 2.14.0 puts the process on all four cards of the rack, 2.5.1 on the two it uses, a copy
through the host on the two.

Why through the host. The other way round the pool — the library's raw peer copy on the tensors' pointers
(`DeviceAllocator.peer_copy`) — is refused by the driver (cudaMemcpy rc 1) as soon as PyTorch's allocator runs with
expandable segments, which the engine turns on for the plans that need them (core/runtime/torch_allocator.py): that
memory is mapped for its own card only and only PyTorch's allocator can grant it to a peer, inside the very copy
this module avoids (measured 2026-11-10: plain allocator copied, expandable segments refused). One path that always
holds: the source card to host memory, host memory to the target card. What crosses cards in today's plans is
hidden states and routing tensors (a few KB a decoded token, a few MB a prefill chunk); a site that moved a weight
card to card would pay the two hops for it.

Contract.
- In: a torch tensor anywhere; the target device (a `torch.device` or its string); `non_blocking`, passed to torch.
- Out: the tensor on the target device — same dtype, shape and values. A tensor already there is returned as it is.
  From or to the host, and between two devices that are not both cards, it is torch's own single `.to()`; card to
  card it is two: to the host, then to the target.
- Refuses nothing of its own.

torch is imported inside the function: the module is importable where the Triton engine's proof forbids torch.
"""
from __future__ import annotations

from typing import Any


def to_card(tensor: Any, device: Any, non_blocking: bool = False) -> Any:
    """`tensor` on `device` — the module's contract."""
    import torch
    target, source = torch.device(device), tensor.device
    if source.type == "cuda" and target.type == "cuda" and source != target:
        tensor = tensor.to("cpu")
    return tensor.to(target, non_blocking=non_blocking)
