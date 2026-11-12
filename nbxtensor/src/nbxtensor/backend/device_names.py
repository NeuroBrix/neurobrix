"""The names a device is known by.

- The arch string: the key its device file and its certified kernel settings are found under. A device reports a model
  name ("Apple M4 Pro", "Tesla V100-SXM2-16GB"); the arch string is that name lowercased with spaces hyphenated —
  "apple-m4-pro". One rule, owned by the library (it was a service the application provided, `nbxtensor.host`), read
  by the Metal backend and by any application keying its own files the same way.
- A placement's device name ("cuda:1", "hip:0", "mps:0", "cpu"): which accelerator it names (`accelerator_index`), a
  tensor moved there (`to_named_device`), that device waited for (`synchronize_named`) — read once for every backend.
"""
from __future__ import annotations

import functools
from typing import Optional

__all__ = ["arch_string", "ACCELERATOR_PREFIXES", "accelerator_index", "to_named_device", "synchronize_named"]


@functools.lru_cache(maxsize=16)
def arch_string(model: Optional[str]) -> str:
    """The arch string of a device's model name: lowercased, spaces hyphenated ("Apple M4 Pro" -> "apple-m4-pro")."""
    return str(model or "").strip().lower().replace(" ", "-")


#: the prefixes of a device name that name an accelerator, as a placement writes them: cuda (NVIDIA), hip (AMD), xpu
#: (Intel), mps (Apple), tt (Tenstorrent), musa, npu. "cpu" names the host.
ACCELERATOR_PREFIXES = frozenset({"cuda", "hip", "xpu", "mps", "tt", "musa", "npu"})


def accelerator_index(device) -> Optional[int]:
    """The ordinal of the accelerator a device name names ("cuda:1" -> 1, "hip:0" -> 0, "mps" -> 0: no ordinal is
    device 0); None for the host ("cpu") or a name that is no accelerator's (a compound placement, "fgp:cuda:0,...")."""
    prefix, _, idx = str(device or "").partition(":")
    if prefix not in ACCELERATOR_PREFIXES:
        return None
    return int(idx) if idx.isdigit() else 0


def to_named_device(tensor, device):
    """`tensor` (an NBXTensor or a QuantizedTensor) on the device `device` names: "cpu" -> `to_cpu()`, an accelerator
    -> `to_cuda(ordinal)` (the move every backend dispatches: an NBXTensor on Metal reports "cuda" too). A name that is
    neither moves nothing."""
    if str(device).partition(":")[0] == "cpu":
        return tensor.to_cpu()
    idx = accelerator_index(device)
    return tensor if idx is None else tensor.to_cuda(idx)


def synchronize_named(device=None) -> None:
    """Wait for the device `device` names to finish its work (the current device when it names no accelerator). The
    current device is selected for the wait and restored after it, even when the wait raises."""
    from nbxtensor.memory.allocator import DeviceAllocator
    idx = accelerator_index(device)
    previous = None
    if idx is not None:
        previous = DeviceAllocator.get_device()
        DeviceAllocator.set_device(idx)
    try:
        DeviceAllocator.sync_device()
    finally:
        if previous is not None:
            DeviceAllocator.set_device(previous)
