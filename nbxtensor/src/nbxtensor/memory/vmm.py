"""The device's virtual memory management primitives (CUDA's driver API: cuMemAddressReserve, cuMemCreate, cuMemMap,
cuMemSetAccess, cuMemUnmap, cuMemRelease) — what an expandable range (`memory.expandable`) builds on: ONE virtual
address range per device, physical chunks mapped into it as it grows and unmapped where a free block covers them whole
(R16, the supervisor's GO 2026-11-02 01:30; the method of torch's `expandable_segments`, PR #96995).

Reached through ctypes on the driver library the backend table names (`driver_libs`); a backend without the API (HIP,
Metal) or a device without the attribute (`CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED`) answers
`supported() == False` and the pool keeps its driver segments. Every call returns the driver's status code; the caller
names a failure. Measured on V100 (registres/benchmarks/vmm_v100.md, 2026-11-02): the granularity is 2 MiB; a map of a
chunk (create + map + set access) costs what an ordinary allocation of the same size costs (~0.36 ms), an unmap ~0.1 ms.

Torch-free, numpy-free.
"""
from __future__ import annotations

import ctypes
from typing import Optional, Tuple

#: cuMemAllocationProp.type: CU_MEM_ALLOCATION_TYPE_PINNED; .location.type: CU_MEM_LOCATION_TYPE_DEVICE
_ALLOC_PINNED, _LOC_DEVICE = 1, 1
#: CU_MEM_ACCESS_FLAGS_PROT_READWRITE
_PROT_RW = 3
#: CU_MEM_ALLOC_GRANULARITY_MINIMUM
_GRAN_MINIMUM = 0
#: CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED
_ATTR_VMM = 102


class _Loc(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _AllocFlags(ctypes.Structure):
    _fields_ = [("compressionType", ctypes.c_ubyte), ("gpuDirectRDMACapable", ctypes.c_ubyte),
                ("usage", ctypes.c_ushort), ("reserved", ctypes.c_ubyte * 4)]


class _Prop(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("requestedHandleTypes", ctypes.c_int), ("location", _Loc),
                ("win32HandleMetaData", ctypes.c_void_p), ("allocFlags", _AllocFlags)]


class _Access(ctypes.Structure):
    _fields_ = [("location", _Loc), ("flags", ctypes.c_int)]


class Vmm:
    """One device's VMM calls. `lib`: the driver library (ctypes CDLL); `device`: the device ordinal the process sees."""

    def __init__(self, lib, device: int) -> None:
        self.lib = lib
        self.device = int(device)
        self._prop = _Prop()
        self._prop.type = _ALLOC_PINNED
        self._prop.location = _Loc(_LOC_DEVICE, self.device)
        self._access = _Access(_Loc(_LOC_DEVICE, self.device), _PROT_RW)

    @staticmethod
    def load(libs) -> Optional[object]:
        """The first driver library of `libs` that loads, initialised; None when none does."""
        for name in libs or ():
            try:
                lib = ctypes.CDLL(name)
            except OSError:
                continue
            if lib.cuInit(0) == 0:
                return lib
        return None

    def supported(self) -> bool:
        """The device answers the VMM attribute with 1."""
        dev, val = ctypes.c_int(), ctypes.c_int()
        if self.lib.cuDeviceGet(ctypes.byref(dev), self.device) != 0:
            return False
        if self.lib.cuDeviceGetAttribute(ctypes.byref(val), _ATTR_VMM, dev) != 0:
            return False
        return val.value == 1

    def granularity(self) -> Tuple[int, int]:
        """(rc, the minimum allocation granularity in bytes)."""
        g = ctypes.c_size_t()
        rc = self.lib.cuMemGetAllocationGranularity(ctypes.byref(g), ctypes.byref(self._prop), _GRAN_MINIMUM)
        return rc, int(g.value)

    def reserve(self, nbytes: int) -> Tuple[int, int]:
        """(rc, the base of a reserved virtual range of `nbytes`)."""
        va = ctypes.c_uint64()
        rc = self.lib.cuMemAddressReserve(ctypes.byref(va), ctypes.c_size_t(nbytes), ctypes.c_size_t(0),
                                          ctypes.c_uint64(0), ctypes.c_ulonglong(0))
        return rc, int(va.value)

    def unreserve(self, va: int, nbytes: int) -> int:
        return self.lib.cuMemAddressFree(ctypes.c_uint64(va), ctypes.c_size_t(nbytes))

    def create(self, nbytes: int) -> Tuple[int, int]:
        """(rc, a physical allocation handle of `nbytes` on the device)."""
        h = ctypes.c_ulonglong()
        rc = self.lib.cuMemCreate(ctypes.byref(h), ctypes.c_size_t(nbytes), ctypes.byref(self._prop),
                                  ctypes.c_ulonglong(0))
        return rc, int(h.value)

    def map(self, va: int, nbytes: int, handle: int) -> int:
        """Map `handle` at `va` and open it to the device; on a failed access grant the mapping is undone."""
        rc = self.lib.cuMemMap(ctypes.c_uint64(va), ctypes.c_size_t(nbytes), ctypes.c_size_t(0),
                               ctypes.c_ulonglong(handle), ctypes.c_ulonglong(0))
        if rc != 0:
            return rc
        rc = self.lib.cuMemSetAccess(ctypes.c_uint64(va), ctypes.c_size_t(nbytes), ctypes.byref(self._access),
                                     ctypes.c_size_t(1))
        if rc != 0:
            self.lib.cuMemUnmap(ctypes.c_uint64(va), ctypes.c_size_t(nbytes))
        return rc

    def unmap(self, va: int, nbytes: int) -> int:
        return self.lib.cuMemUnmap(ctypes.c_uint64(va), ctypes.c_size_t(nbytes))

    def release(self, handle: int) -> int:
        return self.lib.cuMemRelease(ctypes.c_ulonglong(handle))
