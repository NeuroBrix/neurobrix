"""Apple Metal's memory primitives (`DeviceMemory`) — the Metal device behind the one account.

The account (`memory.allocator.DeviceAllocator`) counts every byte this process holds, refuses past the device's
budget and reports. This object only does what the account asks of the device, over the Metal runtime
(`backend.metal_device.MetalRuntime`): shared-storage buffers whose one address the CPU and the GPU both use, their
release once the GPU is done with them, the in-place binding of host pages, copies and fills in the one address
space. It keeps NO account of its own: until 2026-10-29 the runtime kept a second one (`_live_bytes`) beside the
account's, two figures of one fact.

- `budget()`: `MTLDevice.recommendedMaxWorkingSetSize`, Apple's own ceiling for the process on unified memory. Metal
  hands out buffers past it, backed by memory the whole machine shares, until the system pages: the ACCOUNT refuses
  past it (`supports("budget_refusal")`), with this backend's allocation-refused status (`ALLOC_REFUSED`).
- `mem_get_info()`: (budget less what the account holds, budget) — `MTLDevice.currentAllocatedSize` is cumulative and
  never comes down (measured 2026-09-05), so the headroom is read from the account, never from the device.
- `free(ptr)`: the buffer is released once the command buffers in flight are done with it; the account counts it
  freed at once.

Torch-free (R33).
"""
from __future__ import annotations

import ctypes
from typing import Tuple

from .device_memory import DeviceMemory

#: this backend's status for an allocation refused (the runtime's `_ERR_ALLOC`): the account returns it when the budget
#: refuses, so a refusal reads the same whichever of the two said no
ALLOC_REFUSED = 1

_PRIMITIVES = frozenset({"malloc", "free", "mem_get_info", "memset", "memcpy", "host_alloc", "host_free",
                         "set_device", "get_device", "device_count", "budget", "map_host", "unmap_host",
                         "budget_refusal"})


class MetalMemory(DeviceMemory):
    """The memory primitives of the one Metal device of this process, over its runtime."""

    __slots__ = ("_rt",)
    ALLOC_REFUSED = ALLOC_REFUSED

    def __init__(self, runtime):
        self._rt = runtime

    def supports(self, primitive):
        return primitive in _PRIMITIVES

    def malloc(self, nbytes):
        p = ctypes.c_void_p()
        rc = self._rt.malloc(ctypes.byref(p), ctypes.c_size_t(int(nbytes)))
        return rc, (p.value or 0)

    def free(self, ptr):
        return self._rt.free(ctypes.c_void_p(int(ptr)))

    def budget(self) -> Tuple[int, int]:
        return 0, int(self._rt.budget_bytes())

    def mem_get_info(self):
        from .allocator import DeviceAllocator as DA
        total = int(self._rt.budget_bytes())
        held = DA._cuda_live_bytes.get(0, 0) + DA._host_pinned_live_bytes
        return 0, max(0, total - held), total

    def memset(self, ptr, value, nbytes):
        return self._rt.memset(int(ptr), int(value), int(nbytes))

    def memcpy(self, dst, src, nbytes, kind):
        return self._rt.memcpy(int(dst), int(src), int(nbytes), int(kind))

    def host_alloc(self, nbytes):
        p = ctypes.c_void_p()
        rc = self._rt.malloc_host(ctypes.byref(p), ctypes.c_size_t(int(nbytes)))
        return rc, (p.value or 0)

    def host_free(self, ptr):
        return self._rt.free_host(ctypes.c_void_p(int(ptr)))

    def set_device(self, index):
        return self._rt.set_device(int(index))

    def get_device(self):
        d = ctypes.c_int()
        rc = self._rt.get_device(ctypes.byref(d))
        return rc, d.value

    def device_count(self):
        n = ctypes.c_int()
        rc = self._rt.device_count(ctypes.byref(n))
        return rc, n.value

    def map_host(self, ptr, nbytes):
        return self._rt.bind_mapped(int(ptr), int(nbytes))

    def unmap_host(self, start):
        self._rt.unbind_mapped(start)
