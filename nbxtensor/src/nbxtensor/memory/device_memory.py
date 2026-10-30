"""The memory primitives of one device backend — what the allocator's account (`memory.allocator.DeviceAllocator`) asks
a backend for, and nothing else: no byte is counted here (the account counts), no policy is decided here (the account
decides: the pool, the rung, the OOM report).

A primitive returns the backend's own status code beside its value (0 = success), as the runtime gives it: the account
checks it and names the failure, exactly as it did when it called the runtime's symbols itself. One implementation
serves every backend that exposes the C ABI of a CUDA-style runtime (`AbiMemory`: CUDA, HIP, and Metal's runtime
emulation today); a backend with a native API implements `DeviceMemory` directly (Metal next, the Mac's M2).

Built per call from the runtime in force (`backend.runtime.device_memory()`), never cached: a test that substitutes the
runtime or its table is read at once. Torch-free, numpy-free.
"""
from __future__ import annotations

import ctypes
from typing import Tuple


class DeviceMemory:
    """The contract. Every method returns the backend's status code first (0 = success)."""

    def supports(self, primitive: str) -> bool:                  # e.g. "host_alloc": a backend may not offer it
        raise NotImplementedError

    def malloc(self, nbytes: int) -> Tuple[int, int]:            # (rc, device pointer)
        raise NotImplementedError

    def free(self, ptr: int) -> int:                             # rc; a backend may release when its queue is done
        raise NotImplementedError

    def mem_get_info(self) -> Tuple[int, int, int]:              # (rc, free bytes, total bytes) of the current device
        raise NotImplementedError

    def memset(self, ptr: int, value: int, nbytes: int) -> int:
        raise NotImplementedError

    def memcpy(self, dst: int, src: int, nbytes: int, kind: int) -> int:   # kind: 0 H2H, 1 H2D, 2 D2H, 3 D2D
        raise NotImplementedError

    def host_alloc(self, nbytes: int) -> Tuple[int, int]:        # (rc, page-locked / shared host pointer)
        raise NotImplementedError

    def host_free(self, ptr: int) -> int:
        raise NotImplementedError

    def host_register(self, ptr: int, nbytes: int) -> int:       # rc; page-locks existing host bytes (mapped, portable)
        raise NotImplementedError

    def host_unregister(self, ptr: int) -> int:
        raise NotImplementedError

    def set_device(self, index: int) -> int:
        raise NotImplementedError

    def get_device(self) -> Tuple[int, int]:                     # (rc, current device index)
        raise NotImplementedError

    def device_count(self) -> Tuple[int, int]:                   # (rc, number of devices)
        raise NotImplementedError

    def budget(self) -> Tuple[int, int]:                         # (rc, bytes the device grants this process at most)
        raise NotImplementedError

    def map_host(self, ptr: int, nbytes: int) -> int:            # unified memory: a device-visible address of host bytes
        raise NotImplementedError

    def unmap_host(self, start: int) -> None:
        raise NotImplementedError


class AbiMemory(DeviceMemory):
    """A runtime exposing the C ABI of a CUDA-style runtime, through its table of symbol names
    (`backend.runtime._GPU_BACKENDS[...]`, or Metal's emulation table): the calls the account made itself before."""

    __slots__ = ("_rt", "_table")

    def __init__(self, rt, table: dict):
        self._rt = rt
        self._table = table

    #: the contract's names that are table keys of another spelling
    _KEYS = {"host_alloc": "malloc_host", "host_free": "free_host"}

    def supports(self, primitive):
        return self._KEYS.get(primitive, primitive) in self._table

    def _fn(self, name: str):
        return getattr(self._rt, self._table[name])

    def malloc(self, nbytes):
        p = ctypes.c_void_p()
        rc = self._fn("malloc")(ctypes.byref(p), ctypes.c_size_t(int(nbytes)))
        return rc, (p.value or 0)

    def free(self, ptr):
        return self._fn("free")(ctypes.c_void_p(int(ptr)))

    def mem_get_info(self):
        free_b, total_b = ctypes.c_size_t(), ctypes.c_size_t()
        rc = getattr(self._rt, self._table.get("mem_get_info", "cudaMemGetInfo"))(ctypes.byref(free_b),
                                                                                    ctypes.byref(total_b))
        return rc, free_b.value, total_b.value

    def memset(self, ptr, value, nbytes):
        return self._fn("memset")(ctypes.c_void_p(int(ptr)), ctypes.c_int(int(value)), ctypes.c_size_t(int(nbytes)))

    def memcpy(self, dst, src, nbytes, kind):
        return self._fn("memcpy")(ctypes.c_void_p(int(dst)), ctypes.c_void_p(int(src)), ctypes.c_size_t(int(nbytes)),
                                  ctypes.c_int(int(kind)))

    def host_alloc(self, nbytes):
        p = ctypes.c_void_p()
        rc = self._fn("malloc_host")(ctypes.byref(p), ctypes.c_size_t(int(nbytes)))
        return rc, (p.value or 0)

    def host_free(self, ptr):
        return self._fn("free_host")(ctypes.c_void_p(int(ptr)))

    def host_register(self, ptr, nbytes):
        # flags 0 (the Default flag): under unified addressing the bytes are mapped and portable, as the
        # runtime's own page-locked allocation is
        return self._fn("host_register")(ctypes.c_void_p(int(ptr)), ctypes.c_size_t(int(nbytes)), ctypes.c_uint(0))

    def host_unregister(self, ptr):
        return self._fn("host_unregister")(ctypes.c_void_p(int(ptr)))

    def set_device(self, index):
        return self._fn("set_device")(ctypes.c_int(int(index)))

    def get_device(self):
        d = ctypes.c_int()
        rc = self._fn("get_device")(ctypes.byref(d))
        return rc, d.value

    def device_count(self):
        n = ctypes.c_int()
        rc = self._fn("device_count")(ctypes.byref(n))
        return rc, n.value

    def budget(self):
        """The device's total, as the runtime reads it (a C-ABI runtime grants the whole device)."""
        rc, _free, total = self.mem_get_info()
        return rc, total

    def map_host(self, ptr, nbytes):
        """Metal's runtime emulation maps host bytes for the device (`bind_mapped`); a C-ABI runtime has no such call."""
        return self._rt.bind_mapped(int(ptr), int(nbytes))

    def unmap_host(self, start):
        self._rt.unbind_mapped(start)
