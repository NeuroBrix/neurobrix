"""Zero3 (CPU offload + GPU compute) — triton path (NBXTensor, zero torch).

Zero3's block-wise ratchet pipelining is ALREADY mode-polymorphic: it drives
H2D/evict through DeviceAllocator stream + event ctypes APIs, pins host
weights via NBXTensor.pin_host(), and rebinds arena slots through the
TritonSequence handle — all torch-free on the NBXTensor path. So the triton
zero3 reuses that validated ratchet (mixin) rather than duplicating ~660
lines; it only swaps the transfer/sync surface for the NBXTensor one
(TritonStrategy) and drops the torch.cuda device-detection fallback.

NOTE: the shared module still carries a `torch` import for the compiled
path's polymorphic branches; a fully torch-free shared ratchet is a
follow-up for the two-version split. The triton entry point itself adds no
torch.
"""

from ..zero3 import Zero3Strategy as _SharedZero3
from .base import TritonStrategy


class Zero3Strategy(TritonStrategy, _SharedZero3):
    """Zero3 CPU-offload-with-GPU-compute on the triton path.

    MRO: TritonStrategy (NBXTensor transfer + DeviceAllocator sync) →
    _SharedZero3 (block-pipelining ratchet, NBXTensor-capable) →
    ExecutionStrategy. The shared `__init__` runs and calls
    `self._get_exec_device()`, which resolves to the torch-free override
    below.
    """

    def _get_exec_device(self) -> str:
        """Pick the GPU execution device from Prism allocations (torch-free).

        Zero3 pins all weights to "cpu"; the compute device is encoded as
        "zero3:cuda:N". Iterate for the first accelerator entry; no
        torch.cuda fallback (the NBX path is GPU-only by construction —
        a GPU-less host has no NBX compute backend anyway)."""
        for alloc_info in self.context.allocations.values():
            device_str = ""
            if isinstance(alloc_info, dict):
                device_str = alloc_info.get('device', '') or ''
            elif isinstance(alloc_info, tuple) and alloc_info:
                device_str = alloc_info[0]
            if device_str.startswith("zero3:"):
                device_str = device_str.split(":", 1)[1]
            if device_str and device_str.startswith(("cuda", "hip", "xpu", "mps")):
                return device_str
        return "cuda:0"
