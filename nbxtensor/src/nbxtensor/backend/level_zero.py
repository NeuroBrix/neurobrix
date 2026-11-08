"""Intel GPUs through Level Zero (oneAPI's driver API), torch-free and SYCL-free: the runtime the allocator speaks to
(`LevelZeroRuntime`, the C ABI of a CUDA-style runtime emulated over Level Zero, as Metal's runtime emulates it) and the
launch driver for the kernels Intel's Triton backend compiles (`LevelZeroDriver`). Written 2026-11-06 at the owner's
order (AMD and Intel in 0.0.2, to the NVIDIA and Apple drivers' level); no Intel card has run it.

Sources, read 2026-11-06:
- The Level Zero API: the loader's own ctypes bindings, include/ze.py and include/ze_api.h of oneapi-src/level-zero
  (v1.20-r1.20.41): every structure layout, `ZE_STRUCTURE_TYPE_*` value and signature used here.
- Intel's Triton backend v3.8.0 (intel/intel-xpu-backend-for-triton, third_party/intel/backend/): the binary is SPIR-V
  (`asm["spv"]`, `ZE_MODULE_FORMAT_IL_SPIRV`) built with `metadata.build_flags`, rebuilt once with
  `-cl-intel-256-GRF-per-thread` when the build fails or the kernel spills past 1000 bytes with no GRF mode chosen
  (driver.c, `compileLevelZeroObjects`); the kernel created with ZE_KERNEL_FLAG_FORCE_RESIDENCY; a work-group of
  `num_warps * threads_per_warp`, refused past the device's `maxTotalGroupSize`; the arguments set by index — the
  kernel's own, then the global- and profile-scratch pointers, then, when the kernel uses shared local memory, one
  argument of that size and no value (driver.c `_launch`: `sycl::local_accessor`; compute-runtime `setArgSlm` reads
  the size only). Intel launches through a SYCL queue; the Level Zero launch below is that ABI on an in-order
  immediate command list (zeKernelSetGroupSize, zeCommandListAppendLaunchKernel with the grid as group count).
- The device: PCI ids from Linux's include/drm/intel/pciids.h select its device file (`devices/intel/*.yml`), which
  states the facts Level Zero does not answer (the architecture name Intel's compiler keys on, fp64, 16-bit atomics,
  2D block I/O); what Level Zero answers is asked of it.
"""
from __future__ import annotations

import ctypes
import functools
import threading
from typing import Dict, Optional, Tuple

#: the Level Zero loader (`libze1` on Ubuntu), tried in order
from nbxtensor.backend.libnames import LEVEL_ZERO as LIBS   # noqa: E402

# ZE_STRUCTURE_TYPE_* (ze.py)
_ST_DRIVER_PROPERTIES, _ST_DEVICE_COMPUTE, _ST_QUEUE_GROUP, _ST_DEVICE_MEMORY = 0x1, 0x4, 0x6, 0x7
_ST_CONTEXT_DESC, _ST_QUEUE_DESC, _ST_EVENT_POOL_DESC, _ST_EVENT_DESC = 0xD, 0xE, 0x10, 0x11
_ST_DEVICE_MEM_ALLOC, _ST_HOST_MEM_ALLOC, _ST_MODULE_DESC, _ST_KERNEL_DESC, _ST_KERNEL_PROPS = 0x15, 0x16, 0x1B, 0x1D, 0x1E
_ST_DEVICE_PROPERTIES_1_2, _ST_USABLE_MEM = 0x00020006, 0x00020041
_DEVICE_TYPE_GPU, _QUEUE_GROUP_COMPUTE, _QUEUE_FLAG_IN_ORDER = 1, 1, 2
_MODULE_FORMAT_IL_SPIRV, _KERNEL_FLAG_FORCE_RESIDENCY = 0, 1
_EVENT_POOL_HOST_VISIBLE, _EVENT_SCOPE_HOST = 1, 4
_UINT64_MAX = 0xFFFFFFFFFFFFFFFF
_GRF_RETRY_FLAG, _SPILL_RETRY_BYTES = "-cl-intel-256-GRF-per-thread", 1000      # Intel's driver.c
_ERR_UNSUPPORTED_FEATURE, _ERR_INVALID_ARGUMENT, _ERR_DEVICE_LOST = 0x78000003, 0x78000004, 0x70000001

#: ze_result_t names (ze_api.h), for a refusal that says what Level Zero said
RESULT_NAMES = {
    0x1: "ZE_RESULT_NOT_READY", 0x70000001: "ZE_RESULT_ERROR_DEVICE_LOST",
    0x70000002: "ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY", 0x70000003: "ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY",
    0x70000004: "ZE_RESULT_ERROR_MODULE_BUILD_FAILURE", 0x70000005: "ZE_RESULT_ERROR_MODULE_LINK_FAILURE",
    0x78000001: "ZE_RESULT_ERROR_UNINITIALIZED", 0x78000002: "ZE_RESULT_ERROR_UNSUPPORTED_VERSION",
    0x78000003: "ZE_RESULT_ERROR_UNSUPPORTED_FEATURE", 0x78000004: "ZE_RESULT_ERROR_INVALID_ARGUMENT",
    0x78000005: "ZE_RESULT_ERROR_INVALID_NULL_HANDLE", 0x78000007: "ZE_RESULT_ERROR_INVALID_NULL_POINTER",
    0x78000008: "ZE_RESULT_ERROR_INVALID_SIZE", 0x7800000F: "ZE_RESULT_ERROR_INVALID_NATIVE_BINARY",
    0x78000011: "ZE_RESULT_ERROR_INVALID_KERNEL_NAME", 0x78000013: "ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION",
    0x78000015: "ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX",
    0x78000016: "ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE", 0x7FFFFFFE: "ZE_RESULT_ERROR_UNKNOWN",
}


class _Desc(ctypes.Structure):
    _fields_ = [("stype", ctypes.c_uint32), ("pNext", ctypes.c_void_p), ("flags", ctypes.c_uint32)]


class _QueueDesc(ctypes.Structure):
    _fields_ = [("stype", ctypes.c_uint32), ("pNext", ctypes.c_void_p), ("ordinal", ctypes.c_uint32),
                ("index", ctypes.c_uint32), ("flags", ctypes.c_uint32), ("mode", ctypes.c_uint32),
                ("priority", ctypes.c_uint32)]


class _QueueGroupProps(ctypes.Structure):
    _fields_ = [("stype", ctypes.c_uint32), ("pNext", ctypes.c_void_p), ("flags", ctypes.c_uint32),
                ("maxMemoryFillPatternSize", ctypes.c_size_t), ("numQueues", ctypes.c_uint32)]


class _DeviceMemAllocDesc(ctypes.Structure):
    _fields_ = [("stype", ctypes.c_uint32), ("pNext", ctypes.c_void_p), ("flags", ctypes.c_uint32),
                ("ordinal", ctypes.c_uint32)]


class _UsableMem(ctypes.Structure):
    _fields_ = [("stype", ctypes.c_uint32), ("pNext", ctypes.c_void_p), ("currUsableMemSize", ctypes.c_uint64)]


class _DeviceProps(ctypes.Structure):
    _fields_ = [("stype", ctypes.c_uint32), ("pNext", ctypes.c_void_p), ("type", ctypes.c_uint32),
                ("vendorId", ctypes.c_uint32), ("deviceId", ctypes.c_uint32), ("flags", ctypes.c_uint32),
                ("subdeviceId", ctypes.c_uint32), ("coreClockRate", ctypes.c_uint32),
                ("maxMemAllocSize", ctypes.c_uint64), ("maxHardwareContexts", ctypes.c_uint32),
                ("maxCommandQueuePriority", ctypes.c_uint32), ("numThreadsPerEU", ctypes.c_uint32),
                ("physicalEUSimdWidth", ctypes.c_uint32), ("numEUsPerSubslice", ctypes.c_uint32),
                ("numSubslicesPerSlice", ctypes.c_uint32), ("numSlices", ctypes.c_uint32),
                ("timerResolution", ctypes.c_uint64), ("timestampValidBits", ctypes.c_uint32),
                ("kernelTimestampValidBits", ctypes.c_uint32), ("uuid", ctypes.c_uint8 * 16),
                ("name", ctypes.c_char * 256)]


class _ComputeProps(ctypes.Structure):
    _fields_ = [("stype", ctypes.c_uint32), ("pNext", ctypes.c_void_p), ("maxTotalGroupSize", ctypes.c_uint32),
                ("maxGroupSizeX", ctypes.c_uint32), ("maxGroupSizeY", ctypes.c_uint32),
                ("maxGroupSizeZ", ctypes.c_uint32), ("maxGroupCountX", ctypes.c_uint32),
                ("maxGroupCountY", ctypes.c_uint32), ("maxGroupCountZ", ctypes.c_uint32),
                ("maxSharedLocalMemory", ctypes.c_uint32), ("numSubGroupSizes", ctypes.c_uint32),
                ("subGroupSizes", ctypes.c_uint32 * 8)]


class _MemoryProps(ctypes.Structure):
    _fields_ = [("stype", ctypes.c_uint32), ("pNext", ctypes.c_void_p), ("flags", ctypes.c_uint32),
                ("maxClockRate", ctypes.c_uint32), ("maxBusWidth", ctypes.c_uint32), ("totalSize", ctypes.c_uint64),
                ("name", ctypes.c_char * 256)]


class _DriverProps(ctypes.Structure):
    _fields_ = [("stype", ctypes.c_uint32), ("pNext", ctypes.c_void_p), ("uuid", ctypes.c_uint8 * 16),
                ("driverVersion", ctypes.c_uint32)]


class _ModuleDesc(ctypes.Structure):
    _fields_ = [("stype", ctypes.c_uint32), ("pNext", ctypes.c_void_p), ("format", ctypes.c_uint32),
                ("inputSize", ctypes.c_size_t), ("pInputModule", ctypes.c_void_p), ("pBuildFlags", ctypes.c_char_p),
                ("pConstants", ctypes.c_void_p)]


class _KernelDesc(ctypes.Structure):
    _fields_ = [("stype", ctypes.c_uint32), ("pNext", ctypes.c_void_p), ("flags", ctypes.c_uint32),
                ("pKernelName", ctypes.c_char_p)]


class _KernelProps(ctypes.Structure):
    _fields_ = [("stype", ctypes.c_uint32), ("pNext", ctypes.c_void_p), ("numKernelArgs", ctypes.c_uint32),
                ("requiredGroupSizeX", ctypes.c_uint32), ("requiredGroupSizeY", ctypes.c_uint32),
                ("requiredGroupSizeZ", ctypes.c_uint32), ("requiredNumSubGroups", ctypes.c_uint32),
                ("requiredSubgroupSize", ctypes.c_uint32), ("maxSubgroupSize", ctypes.c_uint32),
                ("maxNumSubgroups", ctypes.c_uint32), ("localMemSize", ctypes.c_uint32),
                ("privateMemSize", ctypes.c_uint32), ("spillMemSize", ctypes.c_uint32),
                ("kid", ctypes.c_uint8 * 16), ("mid", ctypes.c_uint8 * 16)]


class _GroupCount(ctypes.Structure):
    _fields_ = [("x", ctypes.c_uint32), ("y", ctypes.c_uint32), ("z", ctypes.c_uint32)]


class _EventPoolDesc(ctypes.Structure):
    _fields_ = [("stype", ctypes.c_uint32), ("pNext", ctypes.c_void_p), ("flags", ctypes.c_uint32),
                ("count", ctypes.c_uint32)]


class _EventDesc(ctypes.Structure):
    _fields_ = [("stype", ctypes.c_uint32), ("pNext", ctypes.c_void_p), ("index", ctypes.c_uint32),
                ("signal", ctypes.c_uint32), ("wait", ctypes.c_uint32)]


def result_name(rc: int) -> str:
    return RESULT_NAMES.get(int(rc) & 0xFFFFFFFF, f"ze_result_t {int(rc):#x}")


def _as_int(v) -> int:
    v = getattr(v, "value", v)
    return int(v or 0)


def _deref(ref, ctype=ctypes.c_uint64):
    """The cell a `ctypes.byref(...)` argument points to (the allocator passes byref cells, as to a C runtime)."""
    obj = getattr(ref, "_obj", None)
    if obj is not None:
        return obj
    return ctypes.cast(ref, ctypes.POINTER(ctype)).contents


def load_library():
    """The Level Zero loader, or None when the system has none (`LIBS`)."""
    for name in LIBS:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def _gpu_devices(lib) -> list:
    """[(driver, device)] of every GPU the loader reports that a device file describes, after zeInit(GPU only); [] when
    none or no driver."""
    if lib.zeInit(ctypes.c_uint32(1)) != 0:            # ZE_INIT_FLAG_GPU_ONLY
        return []
    n = ctypes.c_uint32(0)
    if lib.zeDriverGet(ctypes.byref(n), None) != 0 or n.value == 0:
        return []
    drivers = (ctypes.c_void_p * n.value)()
    if lib.zeDriverGet(ctypes.byref(n), drivers) != 0:
        return []
    out = []
    for drv in drivers:
        m = ctypes.c_uint32(0)
        if lib.zeDeviceGet(ctypes.c_void_p(drv), ctypes.byref(m), None) != 0 or m.value == 0:
            continue
        devs = (ctypes.c_void_p * m.value)()
        if lib.zeDeviceGet(ctypes.c_void_p(drv), ctypes.byref(m), devs) != 0:
            continue
        for dev in devs:
            props = _DeviceProps(stype=_ST_DEVICE_PROPERTIES_1_2)
            if lib.zeDeviceGetProperties(ctypes.c_void_p(dev), ctypes.byref(props)) == 0 \
                    and props.type == _DEVICE_TYPE_GPU and _described(props.deviceId):
                out.append((drv, dev))
    return out


def _described(device_id: int) -> bool:
    """Whether a device file lists the PCI id: a GPU none describes has no compile target (a Tiger Lake iGPU: Intel's
    Triton names none), so it is no device of this runtime — never the backend's device 0, never its probe's answer."""
    from nbxtensor.backend import device_files
    return device_files.for_pci_id("intel", int(device_id)) is not None


@functools.lru_cache(maxsize=1)
def level_zero_device_available() -> bool:
    """Whether the system's Level Zero loader opens and reports at least one GPU — the backend probe
    (`backend.runtime.gpu_backend`): a loader with no Intel GPU behind it does not name the backend."""
    lib = load_library()
    return lib is not None and bool(_gpu_devices(lib))


#: ze_device_property_flag_t ZE_DEVICE_PROPERTY_FLAG_INTEGRATED = ZE_BIT(0): "Device is integrated with the Host"
#: (ze_api.h v1.20-r1.20.41)
_DEVICE_FLAG_INTEGRATED = 1


def gpu_facts(lib=None) -> Optional[list]:
    """What each GPU the Level Zero loader reports and a device file describes states of itself, in the runtime's device
    order (its index is the runtime's): `pci_device_id`, `name`, `integrated` (ZE_DEVICE_PROPERTY_FLAG_INTEGRATED: the device draws on the
    host's memory), `memory_bytes` (its memory heaps, zeDeviceGetMemoryProperties: on an integrated GPU, the share of
    the host's memory the driver lets it hold) and `max_allocation_bytes` (maxMemAllocSize: the largest single
    allocation, which this runtime asks without the relaxed-limits extension). None when no loader is installed."""
    lib = lib if lib is not None else load_library()
    if lib is None:
        return None
    out = []
    for _drv, dev in _gpu_devices(lib):
        props = _DeviceProps(stype=_ST_DEVICE_PROPERTIES_1_2)
        n = ctypes.c_uint32(0)
        if lib.zeDeviceGetProperties(ctypes.c_void_p(dev), ctypes.byref(props)) != 0 \
                or lib.zeDeviceGetMemoryProperties(ctypes.c_void_p(dev), ctypes.byref(n), None) != 0:
            raise RuntimeError("nbxtensor Level Zero: a GPU the loader reports does not answer its properties")
        heaps = (_MemoryProps * n.value)(*[_MemoryProps(stype=_ST_DEVICE_MEMORY) for _ in range(n.value)])
        if lib.zeDeviceGetMemoryProperties(ctypes.c_void_p(dev), ctypes.byref(n), heaps) != 0:
            raise RuntimeError("nbxtensor Level Zero: a GPU the loader reports does not answer its memory")
        out.append({"pci_device_id": int(props.deviceId), "name": props.name.decode(errors="replace"),
                    "integrated": bool(props.flags & _DEVICE_FLAG_INTEGRATED),
                    "memory_bytes": sum(int(h.totalSize) for h in heaps),
                    "max_allocation_bytes": int(props.maxMemAllocSize)})
    return out


#: the runtime table of the `xpu` backend: the contract's names, served by `LevelZeroRuntime`'s methods of the same
#: names (the allocator calls `getattr(runtime, table[key])`, as on Metal). No host registration (Level Zero imports a
#: host pointer only through an extension), no stream-ordered allocator, no peer access across cards yet.
XPU_BACKEND = {
    "rt_libs": list(LIBS),
    "malloc": "malloc", "free": "free", "memcpy": "memcpy", "memset": "memset",
    "set_device": "set_device", "get_device": "get_device", "device_count": "device_count",
    "mem_get_info": "mem_get_info", "sync": "sync",
    "malloc_host": "malloc_host", "free_host": "free_host", "malloc_host_flags": None,
    "stream_create": "stream_create", "stream_destroy": "stream_destroy", "stream_sync": "stream_sync",
    "event_create": "event_create", "event_create_flags": "event_create_flags", "event_destroy": "event_destroy",
    "event_record": "event_record", "event_sync": "event_sync", "event_elapsed": "event_elapsed",
    "stream_wait_event": "stream_wait_event", "memcpy_async": "memcpy_async",
}


class LevelZeroRuntime:
    """The C ABI of a CUDA-style runtime (`cudaMalloc(&p, n)`, `cudaMemcpy(dst, src, n, kind)`, streams, events),
    served over Level Zero: one context per driver, an in-order immediate command list per device as its default
    stream, device memory from zeMemAllocDevice, page-locked host memory from zeMemAllocHost. Every method returns the
    ze_result_t (0 = success), which the allocator checks and names."""

    def __init__(self, lib=None) -> None:
        self.lib = lib if lib is not None else load_library()
        if self.lib is None:
            raise RuntimeError(f"nbxtensor: the Level Zero loader is not installed (tried {LIBS})")
        pairs = _gpu_devices(self.lib)
        if not pairs:
            raise RuntimeError("nbxtensor: the Level Zero loader reports no GPU")
        self._lock = threading.RLock()
        self._devices = [dev for _drv, dev in pairs]
        self._drivers = [drv for drv, _dev in pairs]
        self._contexts: Dict[int, ctypes.c_void_p] = {}     # driver handle -> its context (one per driver)
        self._thread = threading.local()                    # the current device is per thread, as cudaSetDevice's
        self._default_lists: Dict[int, int] = {}        # device index -> its default in-order immediate list
        self._lists: Dict[int, int] = {}                # every list this runtime made -> its device index
        self._sizes: Dict[int, int] = {}                # live allocation -> bytes
        self._events: Dict[int, dict] = {}              # event handle -> its pool and timestamp slot
        self._timer_hz: Dict[int, Tuple[int, int]] = {}

    @property
    def _current(self) -> int:
        return getattr(self._thread, "index", 0)

    # -- handles ---------------------------------------------------------------
    def _check(self, rc: int, what: str) -> None:
        if rc != 0:
            raise RuntimeError(f"nbxtensor Level Zero: {what} failed ({result_name(rc)})")

    def context(self, index: Optional[int] = None) -> ctypes.c_void_p:
        """The context of the driver the device belongs to: one per driver, shared by its devices."""
        drv = self._drivers[self._current if index is None else index]
        with self._lock:
            ctx = self._contexts.get(drv)
            if ctx is None:
                ctx = ctypes.c_void_p()
                desc = _Desc(stype=_ST_CONTEXT_DESC)
                self._check(self.lib.zeContextCreate(ctypes.c_void_p(drv), ctypes.byref(desc), ctypes.byref(ctx)),
                            "zeContextCreate")
                self._contexts[drv] = ctx
        return ctx

    def device(self, index: Optional[int] = None) -> ctypes.c_void_p:
        return ctypes.c_void_p(self._devices[self._current if index is None else index])

    def _compute_ordinal(self, index: int) -> int:
        n = ctypes.c_uint32(0)
        self._check(self.lib.zeDeviceGetCommandQueueGroupProperties(self.device(index), ctypes.byref(n), None),
                    "zeDeviceGetCommandQueueGroupProperties")
        groups = (_QueueGroupProps * n.value)(*[_QueueGroupProps(stype=_ST_QUEUE_GROUP) for _ in range(n.value)])
        self._check(self.lib.zeDeviceGetCommandQueueGroupProperties(self.device(index), ctypes.byref(n), groups),
                    "zeDeviceGetCommandQueueGroupProperties")
        for ordinal, g in enumerate(groups):
            if g.flags & _QUEUE_GROUP_COMPUTE:
                return ordinal
        raise RuntimeError("nbxtensor Level Zero: the device has no compute command-queue group")

    def _try_new_list(self, index: int) -> Tuple[int, int]:
        """(handle, ze_result) of a new in-order immediate command list on the device's compute group."""
        lst = ctypes.c_void_p()
        desc = _QueueDesc(stype=_ST_QUEUE_DESC, ordinal=self._compute_ordinal(index), index=0,
                          flags=_QUEUE_FLAG_IN_ORDER, mode=0, priority=0)
        rc = self.lib.zeCommandListCreateImmediate(self.context(index), self.device(index), ctypes.byref(desc),
                                                   ctypes.byref(lst))
        if rc == 0:
            self._lists[int(lst.value)] = index
        return int(lst.value or 0), rc

    def _new_list(self, index: int) -> int:
        lst, rc = self._try_new_list(index)
        self._check(rc, "zeCommandListCreateImmediate")
        return lst

    def command_list(self, stream: int = 0) -> ctypes.c_void_p:
        """The in-order immediate command list of `stream` (0: the current device's default one)."""
        s = _as_int(stream)
        if s:
            return ctypes.c_void_p(s)
        with self._lock:
            lst = self._default_lists.get(self._current)
            if lst is None:
                lst = self._default_lists[self._current] = self._new_list(self._current)
        return ctypes.c_void_p(lst)

    # -- devices ---------------------------------------------------------------
    def device_count(self, out_count) -> int:
        _deref(out_count, ctypes.c_int).value = len(self._devices)
        return 0

    def set_device(self, index) -> int:
        i = _as_int(index)
        if not 0 <= i < len(self._devices):
            return _ERR_INVALID_ARGUMENT
        self._thread.index = i
        return 0

    def get_device(self, out_index) -> int:
        _deref(out_index, ctypes.c_int).value = self._current
        return 0

    def properties(self, index: Optional[int] = None) -> _DeviceProps:
        props = _DeviceProps(stype=_ST_DEVICE_PROPERTIES_1_2)
        self._check(self.lib.zeDeviceGetProperties(self.device(index), ctypes.byref(props)), "zeDeviceGetProperties")
        return props

    def compute_properties(self, index: Optional[int] = None) -> _ComputeProps:
        props = _ComputeProps(stype=_ST_DEVICE_COMPUTE)
        self._check(self.lib.zeDeviceGetComputeProperties(self.device(index), ctypes.byref(props)),
                    "zeDeviceGetComputeProperties")
        return props

    def driver_version(self, index: Optional[int] = None) -> str:
        """The driver version as SYCL's Level Zero adapter states it (intel/llvm, unified-runtime level_zero
        common/platform.cpp): Intel's version string (zeIntelGetDriverVersionString) when the driver exports it, else
        driverVersion read as | 31-24 major | 23-16 minor | 15-0 build | — what Intel's compiler parses (is_lts)."""
        drv = ctypes.c_void_p(self._drivers[self._current if index is None else index])
        fn = ctypes.c_void_p()
        if self.lib.zeDriverGetExtensionFunctionAddress(drv, b"zeIntelGetDriverVersionString", ctypes.byref(fn)) == 0 \
                and fn.value:
            call = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_char_p,
                                    ctypes.POINTER(ctypes.c_size_t))(fn.value)
            n = ctypes.c_size_t(0)
            call(drv, None, ctypes.byref(n))
            buf = ctypes.create_string_buffer(n.value or 1)
            self._check(call(drv, buf, ctypes.byref(n)), "zeIntelGetDriverVersionString")
            return buf.value.decode(errors="replace")
        props = _DriverProps(stype=_ST_DRIVER_PROPERTIES)
        self._check(self.lib.zeDriverGetProperties(drv, ctypes.byref(props)), "zeDriverGetProperties")
        v = int(props.driverVersion)
        return f"{(v >> 24) & 0xFF}.{(v >> 16) & 0xFF}.{v & 0xFFFF}"

    def mem_get_info(self, out_free, out_total) -> int:
        """Total: the device's memory heaps (zeDeviceGetMemoryProperties). Free: the memory the device can still
        allocate (`ze_device_usablemem_size_ext_properties_t`, chained to its properties) — refused with
        ZE_RESULT_ERROR_UNSUPPORTED_FEATURE when the driver does not answer it, never a guess."""
        n = ctypes.c_uint32(0)
        rc = self.lib.zeDeviceGetMemoryProperties(self.device(), ctypes.byref(n), None)
        if rc != 0:
            return rc
        heaps = (_MemoryProps * n.value)(*[_MemoryProps(stype=_ST_DEVICE_MEMORY) for _ in range(n.value)])
        rc = self.lib.zeDeviceGetMemoryProperties(self.device(), ctypes.byref(n), heaps)
        if rc != 0:
            return rc
        usable = _UsableMem(stype=_ST_USABLE_MEM)
        props = _DeviceProps(stype=_ST_DEVICE_PROPERTIES_1_2, pNext=ctypes.cast(ctypes.pointer(usable),
                                                                                  ctypes.c_void_p))
        rc = self.lib.zeDeviceGetProperties(self.device(), ctypes.byref(props))
        if rc != 0:
            return rc
        if not usable.currUsableMemSize:
            return _ERR_UNSUPPORTED_FEATURE
        _deref(out_total, ctypes.c_size_t).value = sum(int(h.totalSize) for h in heaps)
        _deref(out_free, ctypes.c_size_t).value = int(usable.currUsableMemSize)
        return 0

    # -- memory ----------------------------------------------------------------
    def malloc(self, out_ptr, size) -> int:
        n, slot, p = _as_int(size), _deref(out_ptr, ctypes.c_void_p), ctypes.c_void_p()
        desc = _DeviceMemAllocDesc(stype=_ST_DEVICE_MEM_ALLOC, flags=0, ordinal=0)
        rc = self.lib.zeMemAllocDevice(self.context(), ctypes.byref(desc), ctypes.c_size_t(n), ctypes.c_size_t(0),
                                       self.device(), ctypes.byref(p))
        slot.value = p.value if rc == 0 else 0
        if rc == 0 and p.value:
            self._sizes[int(p.value)] = (n, self.context())
        return rc

    def malloc_host(self, out_ptr, size) -> int:
        n, slot, p = _as_int(size), _deref(out_ptr, ctypes.c_void_p), ctypes.c_void_p()
        desc = _Desc(stype=_ST_HOST_MEM_ALLOC, flags=0)
        rc = self.lib.zeMemAllocHost(self.context(), ctypes.byref(desc), ctypes.c_size_t(n), ctypes.c_size_t(0),
                                     ctypes.byref(p))
        slot.value = p.value if rc == 0 else 0
        if rc == 0 and p.value:
            self._sizes[int(p.value)] = (n, self.context())
        return rc

    def free(self, ptr) -> int:
        p = _as_int(ptr)
        if not p:
            return 0
        held = self._sizes.pop(p, None)
        if held is None:
            return _ERR_INVALID_ARGUMENT                     # not an allocation of this runtime: refused, never freed
        return self.lib.zeMemFree(held[1], ctypes.c_void_p(p))   # in the context that allocated it

    free_host = free

    def memcpy_async(self, dst, src, size, kind=None, stream=None) -> int:
        return self.lib.zeCommandListAppendMemoryCopy(self.command_list(stream or 0), ctypes.c_void_p(_as_int(dst)),
                                                      ctypes.c_void_p(_as_int(src)), ctypes.c_size_t(_as_int(size)),
                                                      None, ctypes.c_uint32(0), None)

    def memcpy(self, dst, src, size, kind=None) -> int:
        """A copy in any direction (Level Zero's unified addresses carry it), complete on return as cudaMemcpy's."""
        rc = self.memcpy_async(dst, src, size, kind, 0)
        return rc if rc != 0 else self.sync()

    def memset(self, ptr, value, size) -> int:
        pattern = ctypes.c_uint8(_as_int(value) & 0xFF)
        rc = self.lib.zeCommandListAppendMemoryFill(self.command_list(0), ctypes.c_void_p(_as_int(ptr)),
                                                    ctypes.byref(pattern), ctypes.c_size_t(1),
                                                    ctypes.c_size_t(_as_int(size)), None, ctypes.c_uint32(0), None)
        return rc if rc != 0 else self.sync()

    # -- ordering --------------------------------------------------------------
    def sync(self) -> int:
        for lst, index in list(self._lists.items()):
            if index == self._current:
                rc = self.lib.zeCommandListHostSynchronize(ctypes.c_void_p(lst), ctypes.c_uint64(_UINT64_MAX))
                if rc != 0:
                    return rc
        return 0

    def stream_create(self, out_handle) -> int:
        lst, rc = self._try_new_list(self._current)
        if rc == 0:
            _deref(out_handle, ctypes.c_void_p).value = lst
        return rc

    def stream_destroy(self, handle) -> int:
        h = _as_int(handle)
        self._lists.pop(h, None)
        return self.lib.zeCommandListDestroy(ctypes.c_void_p(h))

    def stream_sync(self, handle) -> int:
        return self.lib.zeCommandListHostSynchronize(self.command_list(handle), ctypes.c_uint64(_UINT64_MAX))

    def event_create(self, out_handle) -> int:
        return self._event_create(out_handle, timing=True)

    def event_create_flags(self, out_handle, flags) -> int:
        return self._event_create(out_handle, timing=not (_as_int(flags) & 0x02))     # cudaEventDisableTiming

    def _event_create(self, out_handle, timing: bool) -> int:
        """One host-visible event in a pool of its own, and for a timing event a host slot the device writes its
        global timestamp into when the event is recorded (zeCommandListAppendWriteGlobalTimestamp)."""
        pool, event = ctypes.c_void_p(), ctypes.c_void_p()
        pdesc = _EventPoolDesc(stype=_ST_EVENT_POOL_DESC, flags=_EVENT_POOL_HOST_VISIBLE, count=1)
        dev = self.device()
        rc = self.lib.zeEventPoolCreate(self.context(), ctypes.byref(pdesc), ctypes.c_uint32(1), ctypes.byref(dev),
                                        ctypes.byref(pool))
        if rc != 0:
            return rc
        edesc = _EventDesc(stype=_ST_EVENT_DESC, index=0, signal=_EVENT_SCOPE_HOST, wait=_EVENT_SCOPE_HOST)
        rc = self.lib.zeEventCreate(pool, ctypes.byref(edesc), ctypes.byref(event))
        if rc != 0:
            return rc
        slot = 0
        if timing:
            rc, slot = self._host_slot()
            if rc != 0:
                return rc
        self._events[int(event.value)] = {"pool": pool, "slot": slot, "device": self._current}
        _deref(out_handle, ctypes.c_void_p).value = event.value
        return 0

    def _host_slot(self) -> Tuple[int, int]:
        p = ctypes.c_void_p()
        rc = self.malloc_host(ctypes.byref(p), 8)
        return rc, (p.value or 0)

    def event_destroy(self, handle) -> int:
        e = self._events.pop(_as_int(handle), None)
        if e is None:
            return _ERR_INVALID_ARGUMENT
        rc = self.free(e["slot"]) if e["slot"] else 0
        rc = rc or self.lib.zeEventDestroy(ctypes.c_void_p(_as_int(handle)))
        return rc or self.lib.zeEventPoolDestroy(e["pool"])

    def event_record(self, handle, stream=None) -> int:
        h = _as_int(handle)
        e = self._events.get(h)
        if e is None:
            return _ERR_INVALID_ARGUMENT
        lst = self.command_list(stream or 0)
        rc = self.lib.zeEventHostReset(ctypes.c_void_p(h))
        if rc != 0:
            return rc
        if e["slot"]:
            return self.lib.zeCommandListAppendWriteGlobalTimestamp(lst, ctypes.c_void_p(e["slot"]), ctypes.c_void_p(h),
                                                                    ctypes.c_uint32(0), None)
        return self.lib.zeCommandListAppendSignalEvent(lst, ctypes.c_void_p(h))

    def event_sync(self, handle) -> int:
        return self.lib.zeEventHostSynchronize(ctypes.c_void_p(_as_int(handle)), ctypes.c_uint64(_UINT64_MAX))

    def stream_wait_event(self, stream, handle, flags=None) -> int:
        ev = ctypes.c_void_p(_as_int(handle))
        return self.lib.zeCommandListAppendWaitOnEvents(self.command_list(stream), ctypes.c_uint32(1),
                                                        ctypes.byref(ev))

    def event_elapsed(self, out_ms, start, end) -> int:
        """Milliseconds between two recorded timing events: the device's global timestamps, in its timer's cycles
        per second (`timerResolution` of DEVICE_PROPERTIES_1_2), masked to its valid bits."""
        a, b = self._events.get(_as_int(start)), self._events.get(_as_int(end))
        if a is None or b is None or not a["slot"] or not b["slot"]:
            return _ERR_INVALID_ARGUMENT
        for h in (start, end):
            rc = self.event_sync(h)
            if rc != 0:
                return rc
        hz, bits = self._timer(a["device"])
        mask = (1 << bits) - 1 if bits else _UINT64_MAX
        t0 = ctypes.c_uint64.from_address(a["slot"]).value & mask
        t1 = ctypes.c_uint64.from_address(b["slot"]).value & mask
        _deref(out_ms, ctypes.c_float).value = ((t1 - t0) & mask) * 1000.0 / hz
        return 0

    def _timer(self, index: int) -> Tuple[int, int]:
        if index not in self._timer_hz:
            p = self.properties(index)
            self._timer_hz[index] = (int(p.timerResolution), int(p.timestampValidBits))
        return self._timer_hz[index]


_RUNTIME: Optional[LevelZeroRuntime] = None
_RUNTIME_LOCK = threading.Lock()


def runtime() -> LevelZeroRuntime:
    """The process-wide Level Zero runtime."""
    global _RUNTIME
    with _RUNTIME_LOCK:
        if _RUNTIME is None:
            _RUNTIME = LevelZeroRuntime()
        return _RUNTIME


#: the OpenCL ICD loader (`ocl-icd-libopencl1`; Intel's runtime registers itself in it with intel-opencl-icd)
from nbxtensor.backend.libnames import OPENCL as OPENCL_LIBS   # noqa: E402
_CL_DEVICE_TYPE_GPU, _CL_DEVICE_VENDOR_ID, _CL_DEVICE_EXTENSIONS, _CL_DEVICE_ID_INTEL = 4, 0x1001, 0x1030, 0x4251

#: target facts Intel's compiler reads (intel/backend/compiler.py `parse_target`) -> the OpenCL extension that states
#: each, as Intel's own query does (intel/backend/extension_utils.py `query_device_extensions`; cl_khr_fp64 for fp64)
EXTENSION_FACTS = {
    "has_subgroup_matrix_multiply_accumulate": "cl_intel_subgroup_matrix_multiply_accumulate",
    "has_subgroup_matrix_multiply_accumulate_tensor_float32":
        "cl_intel_subgroup_matrix_multiply_accumulate_tensor_float32",
    "has_2d_block_io": "cl_intel_subgroup_2d_block_io",
    "has_bfloat16_conversion": "cl_intel_bfloat16_conversions",
    "has_fp64": "cl_khr_fp64",
}


def device_extensions(device_id: int, lib=None) -> list:
    """The OpenCL extensions of the Intel GPU whose PCI device id is `device_id`, asked of the OpenCL ICD
    (clGetDeviceInfo CL_DEVICE_EXTENSIONS, the device matched by CL_DEVICE_ID_INTEL of cl_intel_device_attribute_query)
    — what Intel's Triton asks through SYCL. Refused by name when the ICD or the device is not there."""
    if lib is None:
        for name in OPENCL_LIBS:
            try:
                lib = ctypes.CDLL(name)
                break
            except OSError:
                continue
    if lib is None:
        raise RuntimeError(f"nbxtensor: the OpenCL ICD loader is not installed (tried {OPENCL_LIBS}); Intel's "
                           f"compiler needs the device's extensions — install ocl-icd-libopencl1 and intel-opencl-icd")
    n = ctypes.c_uint32(0)
    rc = lib.clGetPlatformIDs(ctypes.c_uint32(0), None, ctypes.byref(n))
    platforms = (ctypes.c_void_p * n.value)()
    if rc != 0 or lib.clGetPlatformIDs(n, platforms, None) != 0:
        raise RuntimeError(f"nbxtensor: the OpenCL ICD lists no platform (clGetPlatformIDs {rc}) — install "
                           f"intel-opencl-icd (the device's extensions are what Intel's compiler is told)")
    for plat in platforms:
        m = ctypes.c_uint32(0)
        if lib.clGetDeviceIDs(ctypes.c_void_p(plat), ctypes.c_uint64(_CL_DEVICE_TYPE_GPU), ctypes.c_uint32(0), None,
                              ctypes.byref(m)) != 0 or not m.value:
            continue
        devs = (ctypes.c_void_p * m.value)()
        if lib.clGetDeviceIDs(ctypes.c_void_p(plat), ctypes.c_uint64(_CL_DEVICE_TYPE_GPU), m, devs, None) != 0:
            continue
        for dev in devs:
            vendor, did = ctypes.c_uint32(0), ctypes.c_uint32(0)
            lib.clGetDeviceInfo(ctypes.c_void_p(dev), ctypes.c_uint32(_CL_DEVICE_VENDOR_ID), ctypes.c_size_t(4),
                                ctypes.byref(vendor), None)
            if vendor.value != 0x8086 or lib.clGetDeviceInfo(ctypes.c_void_p(dev), ctypes.c_uint32(_CL_DEVICE_ID_INTEL),
                                                             ctypes.c_size_t(4), ctypes.byref(did), None) != 0:
                continue
            if did.value != int(device_id):
                continue
            size = ctypes.c_size_t(0)
            lib.clGetDeviceInfo(ctypes.c_void_p(dev), ctypes.c_uint32(_CL_DEVICE_EXTENSIONS), ctypes.c_size_t(0), None,
                                ctypes.byref(size))
            buf = ctypes.create_string_buffer(size.value or 1)
            lib.clGetDeviceInfo(ctypes.c_void_p(dev), ctypes.c_uint32(_CL_DEVICE_EXTENSIONS), size, buf, None)
            return buf.value.decode(errors="replace").split()
    raise RuntimeError(f"nbxtensor: the OpenCL ICD reports no Intel GPU {int(device_id):#06x} — install "
                       f"intel-opencl-icd (the device's extensions are what Intel's compiler is told)")


def device_file_for(device_id: int):
    """(name, description) of the library's Intel device file whose `pci_device_ids` hold `device_id`; refused by
    name when none does — the facts Level Zero does not answer are never guessed."""
    from nbxtensor.backend import device_files
    found = device_files.for_pci_id("intel", device_id)
    if found is not None:
        return found
    raise RuntimeError(f"nbxtensor: no Intel device file lists PCI device {int(device_id):#06x} "
                       f"(backend/devices/intel/*.yml, pci_device_ids) — its target is not guessed")
