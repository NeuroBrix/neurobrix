"""NBXTensor — Tensor type for --triton mode.

Wraps raw CUDA device pointers with shape/strides/dtype metadata.
All metadata ops are pure Python CPU computation.
Triton kernels receive NBXTensor via .data_ptr() and .dtype.

Dependencies: ctypes (stdlib), math (stdlib), triton (lazy, for dtype mapping).

Weight loading: safetensors → numpy → cudaMemcpy H2D → NBXTensor
Allocation: raw cudaMalloc via ctypes
Metadata ops: pure Python stride/shape computation
"""

from __future__ import annotations

import ctypes
import functools
import math
import struct
from enum import IntEnum
from typing import Dict, Optional, Tuple


# ============================================================================
# DTYPE SYSTEM
# ============================================================================

class NBXDtype(IntEnum):
    float16 = 0
    bfloat16 = 1
    float32 = 2
    float64 = 3
    int8 = 4
    int16 = 5
    int32 = 6
    int64 = 7
    uint8 = 8
    bool_ = 9
    complex64 = 10
    complex128 = 11


_DTYPE_SIZES = {
    NBXDtype.float16: 2, NBXDtype.bfloat16: 2,
    NBXDtype.float32: 4, NBXDtype.float64: 8,
    NBXDtype.int8: 1, NBXDtype.int16: 2, NBXDtype.int32: 4, NBXDtype.int64: 8,
    NBXDtype.uint8: 1, NBXDtype.bool_: 1,
    NBXDtype.complex64: 8, NBXDtype.complex128: 16,
}

_DTYPE_FROM_STR = {
    'float16': NBXDtype.float16, 'fp16': NBXDtype.float16, 'half': NBXDtype.float16,
    'bfloat16': NBXDtype.bfloat16, 'bf16': NBXDtype.bfloat16,
    'float32': NBXDtype.float32, 'fp32': NBXDtype.float32, 'float': NBXDtype.float32,
    'float64': NBXDtype.float64, 'fp64': NBXDtype.float64, 'double': NBXDtype.float64,
    'int8': NBXDtype.int8, 'int16': NBXDtype.int16,
    'int32': NBXDtype.int32, 'int': NBXDtype.int32,
    'int64': NBXDtype.int64, 'long': NBXDtype.int64,
    'uint8': NBXDtype.uint8, 'byte': NBXDtype.uint8,
    'bool': NBXDtype.bool_,
    'complex64': NBXDtype.complex64, 'complex128': NBXDtype.complex128,
}

# numpy dtype string → NBXDtype
_NUMPY_DTYPE_MAP = {
    'float16': NBXDtype.float16, 'float32': NBXDtype.float32,
    'float64': NBXDtype.float64, 'int8': NBXDtype.int8,
    'int16': NBXDtype.int16, 'int32': NBXDtype.int32,
    'int64': NBXDtype.int64, 'uint8': NBXDtype.uint8,
    'bool': NBXDtype.bool_,
}

# __cuda_array_interface__ typestr
_DTYPE_TYPESTR = {
    NBXDtype.float16: '<f2', NBXDtype.bfloat16: '<V2',
    NBXDtype.float32: '<f4', NBXDtype.float64: '<f8',
    NBXDtype.int8: '<i1', NBXDtype.int16: '<i2',
    NBXDtype.int32: '<i4', NBXDtype.int64: '<i8',
    NBXDtype.uint8: '<u1', NBXDtype.bool_: '|b1',
    NBXDtype.complex64: '<c8', NBXDtype.complex128: '<c16',
}

_FLOATING_DTYPES = frozenset({
    NBXDtype.float16, NBXDtype.bfloat16, NBXDtype.float32, NBXDtype.float64,
})

_COMPLEX_DTYPES = frozenset({NBXDtype.complex64, NBXDtype.complex128})


def dtype_size(dtype: NBXDtype) -> int:
    return _DTYPE_SIZES[dtype]


def parse_dtype(d) -> NBXDtype:
    """Parse dtype from string or NBXDtype."""
    if isinstance(d, NBXDtype):
        return d
    if isinstance(d, str):
        s = d.replace('torch.', '')
        return _DTYPE_FROM_STR[s]
    # Unknown type — try string conversion
    s = str(d).replace('torch.', '')
    return _DTYPE_FROM_STR[s]


# ============================================================================
# TORCH DTYPE BOUNDARY — converts NBXDtype ↔ torch.dtype at system edges
# ============================================================================

def nbx_dtype_to_torch(d: NBXDtype):
    """Convert NBXDtype to torch.dtype. Used at NBXTensor↔torch boundaries."""
    import torch
    _MAP = {
        NBXDtype.float16: torch.float16, NBXDtype.bfloat16: torch.bfloat16,
        NBXDtype.float32: torch.float32, NBXDtype.float64: torch.float64,
        NBXDtype.int8: torch.int8, NBXDtype.int16: torch.int16,
        NBXDtype.int32: torch.int32, NBXDtype.int64: torch.int64,
        NBXDtype.uint8: torch.uint8, NBXDtype.bool_: torch.bool,
        NBXDtype.complex64: torch.complex64, NBXDtype.complex128: torch.complex128,
    }
    return _MAP[d]


def nbx_to_torch(tensor: 'NBXTensor'):
    """Convert NBXTensor to torch.Tensor via D2D copy.

    Used at the boundary between triton execution and torch-based pipeline.
    """
    import torch
    t = torch.empty(tensor.shape, dtype=nbx_dtype_to_torch(tensor._dtype),
                    device=f"cuda:{tensor._device_idx}")
    if tensor._nbytes > 0:
        DeviceAllocator.memcpy(t.data_ptr(), tensor.data_ptr(), tensor._nbytes)
    return t


# ============================================================================
# TRITON DTYPE MAPPING (lazy import)
# ============================================================================

_TL_DTYPE_MAP = None


def _get_tl_dtype(nbx_dtype: NBXDtype):
    """Map NBXDtype → triton.language dtype. Lazy import of triton."""
    global _TL_DTYPE_MAP
    if _TL_DTYPE_MAP is None:
        import triton.language as tl
        _TL_DTYPE_MAP = {
            NBXDtype.float16: tl.float16,
            NBXDtype.bfloat16: tl.bfloat16,
            NBXDtype.float32: tl.float32,
            NBXDtype.float64: tl.float64,
            NBXDtype.int8: tl.int8,
            NBXDtype.int16: tl.int16,
            NBXDtype.int32: tl.int32,
            NBXDtype.int64: tl.int64,
            NBXDtype.uint8: tl.uint8,
            NBXDtype.bool_: tl.uint8,  # bool stored as uint8 (same as PyTorch)
        }
    result = _TL_DTYPE_MAP.get(nbx_dtype)
    if result is None:
        import triton.language as tl
        return tl.float32
    return result


# ============================================================================
# GPU RUNTIME — hardware-agnostic via ctypes (CUDA/ROCm)
# ============================================================================

# Runtime API mapping per backend
_GPU_BACKENDS = {
    "cuda": {
        "rt_libs": ["libcudart.so", "libcudart.so.12", "libcudart.so.11.0"],
        "malloc": "cudaMalloc", "free": "cudaFree",
        "memcpy": "cudaMemcpy", "memset": "cudaMemset",
        "set_device": "cudaSetDevice", "get_device": "cudaGetDevice",
        "sync": "cudaDeviceSynchronize",
        "malloc_host": "cudaMallocHost", "free_host": "cudaFreeHost",
        # Stream-ordered memory allocator (CUDA 11.2+). The driver
        # handles kernel-lifetime correctness per-stream, so freeing a
        # tensor while an async kernel is still reading from it is
        # safe — the free is ordered against the stream's pending work.
        # Using stream=0 (default stream) keeps us aligned with the
        # rest of NeuroBrix's single-default-stream execution model;
        # when pipelining introduces a transfer stream in a future
        # session we'll route those allocs to it.
        "malloc_async": "cudaMallocAsync", "free_async": "cudaFreeAsync",
        # Stream + event API — used by zero3 pipelining for async H2D
        # overlap of block N+1 with compute on block N.
        "stream_create": "cudaStreamCreate",
        "stream_destroy": "cudaStreamDestroy",
        "stream_sync": "cudaStreamSynchronize",
        "event_create": "cudaEventCreate",
        "event_destroy": "cudaEventDestroy",
        "event_record": "cudaEventRecord",
        "stream_wait_event": "cudaStreamWaitEvent",
        "memcpy_async": "cudaMemcpyAsync",
    },
    "hip": {
        "rt_libs": ["libamdhip64.so", "libamdhip64.so.5"],
        "malloc": "hipMalloc", "free": "hipFree",
        "memcpy": "hipMemcpy", "memset": "hipMemset",
        "set_device": "hipSetDevice", "get_device": "hipGetDevice",
        "sync": "hipDeviceSynchronize",
        "malloc_host": "hipHostMalloc", "free_host": "hipHostFree",
        # ROCm 4.3+ exposes the same stream-ordered API under hip*
        # names. Not tested in this pass (investigation was V100-only),
        # but the symbol mapping is kept for symmetry.
        "malloc_async": "hipMallocAsync", "free_async": "hipFreeAsync",
        "stream_create": "hipStreamCreate",
        "stream_destroy": "hipStreamDestroy",
        "stream_sync": "hipStreamSynchronize",
        "event_create": "hipEventCreate",
        "event_destroy": "hipEventDestroy",
        "event_record": "hipEventRecord",
        "stream_wait_event": "hipStreamWaitEvent",
        "memcpy_async": "hipMemcpyAsync",
    },
}


class DeviceAllocator:
    """GPU + pinned-host memory allocator via raw runtime API.

    Tracks running byte counters per device and for pinned host memory
    so diagnostics (especially for zero3 CPU offload) don't need to
    round-trip through cudaMemGetInfo or torch.cuda.memory_allocated —
    both of which are unavailable or incorrect for NBXTensor.
    """

    # Running counters — populated by every malloc_cuda/free_cuda and
    # malloc_host_pinned/free_host_pinned. Class-level so the accounting
    # is process-wide. Not thread-safe; NeuroBrix runtime is
    # single-threaded per component by convention.
    _cuda_live_bytes: Dict[int, int] = {}            # device_idx -> live
    _cuda_peak_bytes: Dict[int, int] = {}            # device_idx -> peak
    _cuda_ptr_size: Dict[int, int] = {}              # ptr -> nbytes (for free accounting)
    _cuda_ptr_device: Dict[int, int] = {}            # ptr -> device_idx (REQUIRED for cudaFreeAsync correctness)
    _host_pinned_live_bytes: int = 0
    _host_pinned_peak_bytes: int = 0
    _host_pinned_ptr_size: Dict[int, int] = {}       # ptr -> nbytes

    @staticmethod
    def set_device(device_id: int):
        """Set current GPU device for allocations (runtime API only).

        Uses only cudaSetDevice/hipSetDevice — does NOT touch the driver
        context. The driver context is managed solely by ensure_triton_device
        which is called just before Triton kernel launches.
        """
        rt = _gpu_runtime()
        getattr(rt, _active_backend()["set_device"])(ctypes.c_int(device_id))

    @staticmethod
    def get_device() -> int:
        """Get current GPU device."""
        dev = ctypes.c_int()
        rt = _gpu_runtime()
        getattr(rt, _active_backend()["get_device"])(ctypes.byref(dev))
        return dev.value

    @staticmethod
    def malloc_cuda(nbytes: int) -> int:
        """Allocate GPU memory via the synchronous allocator.

        Uses cudaMalloc. The stream-ordered allocator (cudaMallocAsync)
        was prototyped but disabled because its per-device memory pools
        don't grant cross-device peer access by default — freeing a
        cuda:N pointer from a Python-GC site while cuda:M is current
        produced illegal memory accesses on the multi-GPU
        pipeline_parallel path (Qwen3-30B --triton without --hardware
        override). The synchronous path combined with TritonSequence's
        _deferred batch-release is stable across single-GPU, zero3,
        and pipeline_parallel. See tests/scratch/zero3_triton_impl/
        LEAK_PINPOINT_REPORT.md for history; the actual block-sized
        leak was fixed by TritonSequence._find_cuda_arg, not by the
        allocator swap.
        """
        if nbytes == 0:
            return 0
        ptr = ctypes.c_void_p()
        rt = _gpu_runtime()
        backend = _active_backend()
        ret = getattr(rt, backend["malloc"])(
            ctypes.byref(ptr), ctypes.c_size_t(nbytes))
        if ret != 0:
            raise RuntimeError(
                f"GPU malloc failed (error {ret}) for {nbytes} bytes")
        p = ptr.value or 0
        dev = DeviceAllocator.get_device()
        DeviceAllocator._cuda_ptr_size[p] = nbytes
        DeviceAllocator._cuda_ptr_device[p] = dev
        live = DeviceAllocator._cuda_live_bytes.get(dev, 0) + nbytes
        DeviceAllocator._cuda_live_bytes[dev] = live
        peak = DeviceAllocator._cuda_peak_bytes.get(dev, 0)
        if live > peak:
            DeviceAllocator._cuda_peak_bytes[dev] = live
        return p

    @staticmethod
    def free_cuda(ptr: int):
        """Free GPU memory via the synchronous allocator.

        cudaFree blocks the calling thread until the device is idle
        for the relevant stream, so TritonSequence batches freeable
        tensors into _deferred and releases them after a sync at the
        end of each forward pass. This avoids serialising on every
        kernel while still being UAF-safe (sync before free).
        """
        if ptr:
            rt = _gpu_runtime()
            backend = _active_backend()
            alloc_dev = DeviceAllocator._cuda_ptr_device.pop(ptr, None)
            getattr(rt, backend["free"])(ctypes.c_void_p(ptr))
            nbytes = DeviceAllocator._cuda_ptr_size.pop(ptr, None)
            if nbytes is not None:
                if alloc_dev is not None:
                    # Accurate decrement: always use the allocation's
                    # own device, independent of current context.
                    live = DeviceAllocator._cuda_live_bytes.get(alloc_dev, 0) - nbytes
                    DeviceAllocator._cuda_live_bytes[alloc_dev] = live
                else:
                    dev = DeviceAllocator.get_device()
                    live = DeviceAllocator._cuda_live_bytes.get(dev, 0) - nbytes
                    if live < 0:
                        for d in DeviceAllocator._cuda_live_bytes:
                            if DeviceAllocator._cuda_live_bytes[d] >= nbytes:
                                DeviceAllocator._cuda_live_bytes[d] -= nbytes
                                break
                    else:
                        DeviceAllocator._cuda_live_bytes[dev] = live

    @staticmethod
    def malloc_host_pinned(nbytes: int) -> int:
        """Allocate page-locked host memory. Enables non_blocking H2D DMA.

        Uses cudaMallocHost (or hipHostMalloc on ROCm). The returned
        pointer is a HOST address — numpy can wrap it via
        np.frombuffer(ctypes.string_at(ptr, nbytes), ...) or ctypes
        casting, and cudaMemcpy kind=1 (H2D) uses it as src.
        """
        if nbytes == 0:
            return 0
        ptr = ctypes.c_void_p()
        rt = _gpu_runtime()
        backend = _active_backend()
        fn_name = backend.get("malloc_host")
        if fn_name is None:
            raise RuntimeError(
                f"Pinned host allocation unsupported on backend "
                f"{_detect_gpu_backend()!r}")
        ret = getattr(rt, fn_name)(
            ctypes.byref(ptr), ctypes.c_size_t(nbytes))
        if ret != 0:
            raise RuntimeError(
                f"Host pinned malloc failed (error {ret}) for {nbytes} bytes")
        p = ptr.value or 0
        DeviceAllocator._host_pinned_ptr_size[p] = nbytes
        DeviceAllocator._host_pinned_live_bytes += nbytes
        if DeviceAllocator._host_pinned_live_bytes > DeviceAllocator._host_pinned_peak_bytes:
            DeviceAllocator._host_pinned_peak_bytes = DeviceAllocator._host_pinned_live_bytes
        return p

    @staticmethod
    def free_host_pinned(ptr: int):
        if ptr:
            rt = _gpu_runtime()
            backend = _active_backend()
            fn_name = backend.get("free_host")
            if fn_name is None:
                return
            getattr(rt, fn_name)(ctypes.c_void_p(ptr))
            nbytes = DeviceAllocator._host_pinned_ptr_size.pop(ptr, None)
            if nbytes is not None:
                DeviceAllocator._host_pinned_live_bytes -= nbytes

    # ------------------------------------------------------------------
    # Accounting API — use instead of cudaMemGetInfo for per-NeuroBrix
    # observability (cudaMemGetInfo returns device-wide numbers
    # including other processes, which is useless for debugging a
    # specific zero3 leak).
    # ------------------------------------------------------------------

    @staticmethod
    def memory_allocated(device_idx: Optional[int] = None) -> int:
        """Live bytes allocated via malloc_cuda on the given device.
        With device_idx=None, returns the total across all devices."""
        if device_idx is None:
            return sum(DeviceAllocator._cuda_live_bytes.values())
        return DeviceAllocator._cuda_live_bytes.get(device_idx, 0)

    @staticmethod
    def peak_memory_allocated(device_idx: Optional[int] = None) -> int:
        """Highest live-bytes watermark observed for the device.
        With device_idx=None, returns the total peak across all devices."""
        if device_idx is None:
            return sum(DeviceAllocator._cuda_peak_bytes.values())
        return DeviceAllocator._cuda_peak_bytes.get(device_idx, 0)

    @staticmethod
    def reset_peak_memory(device_idx: Optional[int] = None) -> None:
        """Reset the peak watermark to the current live value.
        With device_idx=None, resets all devices."""
        if device_idx is None:
            for d in list(DeviceAllocator._cuda_live_bytes.keys()):
                DeviceAllocator._cuda_peak_bytes[d] = (
                    DeviceAllocator._cuda_live_bytes[d])
        else:
            DeviceAllocator._cuda_peak_bytes[device_idx] = (
                DeviceAllocator._cuda_live_bytes.get(device_idx, 0))

    @staticmethod
    def host_pinned_allocated() -> int:
        """Live bytes allocated via malloc_host_pinned."""
        return DeviceAllocator._host_pinned_live_bytes

    @staticmethod
    def host_pinned_peak() -> int:
        """Peak pinned host bytes observed."""
        return DeviceAllocator._host_pinned_peak_bytes

    @staticmethod
    def memset_cuda(ptr: int, value: int, nbytes: int):
        if nbytes > 0 and ptr:
            rt = _gpu_runtime()
            getattr(rt, _active_backend()["memset"])(
                ctypes.c_void_p(ptr), ctypes.c_int(value),
                ctypes.c_size_t(nbytes))

    @staticmethod
    def memcpy(dst: int, src: int, nbytes: int, kind: int = 3):
        """Copy memory. kind: 0=H2H, 1=H2D, 2=D2H, 3=D2D."""
        if nbytes > 0:
            rt = _gpu_runtime()
            getattr(rt, _active_backend()["memcpy"])(
                ctypes.c_void_p(dst), ctypes.c_void_p(src),
                ctypes.c_size_t(nbytes), ctypes.c_int(kind))

    @staticmethod
    def sync_device():
        """Synchronize current GPU device. Waits for all pending GPU ops."""
        rt = _gpu_runtime()
        backend = _active_backend()
        sync_name = backend.get("sync")
        if sync_name:
            getattr(rt, sync_name)()

    # ------------------------------------------------------------------
    # Async stream + event primitives (zero torch).
    # Zero3 pipelining opens a dedicated transfer stream so that the H2D
    # copy of block N+1 overlaps with the compute on block N which runs
    # on the default stream. Events synchronize the two streams so we
    # never read a weight that's still being copied.
    #
    # Handles are opaque ctypes.c_void_p values surfaced as Python ints
    # (0 is the default/NULL stream). Keep them in Python state; do not
    # re-wrap them on each use — ctypes creates a fresh opaque object
    # each call.
    # ------------------------------------------------------------------

    @staticmethod
    def create_stream() -> int:
        """Create a new non-blocking GPU stream. Returns opaque handle."""
        rt = _gpu_runtime()
        backend = _active_backend()
        fn_name = backend.get("stream_create")
        if fn_name is None:
            raise RuntimeError(
                f"Stream create unsupported on backend {_detect_gpu_backend()!r}")
        h = ctypes.c_void_p()
        ret = getattr(rt, fn_name)(ctypes.byref(h))
        if ret != 0:
            raise RuntimeError(f"stream create failed (error {ret})")
        return h.value or 0

    @staticmethod
    def destroy_stream(stream: int):
        """Destroy a stream created by create_stream. No-op for default (0)."""
        if not stream:
            return
        rt = _gpu_runtime()
        backend = _active_backend()
        fn_name = backend.get("stream_destroy")
        if fn_name is None:
            return
        getattr(rt, fn_name)(ctypes.c_void_p(stream))

    @staticmethod
    def stream_synchronize(stream: int):
        """Block the caller until the given stream drains."""
        rt = _gpu_runtime()
        backend = _active_backend()
        fn_name = backend.get("stream_sync")
        if fn_name is None:
            return
        getattr(rt, fn_name)(ctypes.c_void_p(stream))

    @staticmethod
    def create_event() -> int:
        """Create a new event. Returns opaque handle."""
        rt = _gpu_runtime()
        backend = _active_backend()
        fn_name = backend.get("event_create")
        if fn_name is None:
            raise RuntimeError(
                f"Event create unsupported on backend {_detect_gpu_backend()!r}")
        h = ctypes.c_void_p()
        ret = getattr(rt, fn_name)(ctypes.byref(h))
        if ret != 0:
            raise RuntimeError(f"event create failed (error {ret})")
        return h.value or 0

    @staticmethod
    def destroy_event(event: int):
        """Destroy an event created by create_event."""
        if not event:
            return
        rt = _gpu_runtime()
        backend = _active_backend()
        fn_name = backend.get("event_destroy")
        if fn_name is None:
            return
        getattr(rt, fn_name)(ctypes.c_void_p(event))

    @staticmethod
    def record_event(event: int, stream: int = 0):
        """Capture the current state of `stream` into `event`."""
        rt = _gpu_runtime()
        backend = _active_backend()
        fn_name = backend.get("event_record")
        if fn_name is None:
            return
        getattr(rt, fn_name)(
            ctypes.c_void_p(event), ctypes.c_void_p(stream))

    @staticmethod
    def stream_wait_event(stream: int, event: int, flags: int = 0):
        """Make `stream` wait for `event` without blocking the host."""
        rt = _gpu_runtime()
        backend = _active_backend()
        fn_name = backend.get("stream_wait_event")
        if fn_name is None:
            return
        getattr(rt, fn_name)(
            ctypes.c_void_p(stream), ctypes.c_void_p(event),
            ctypes.c_uint(flags))

    @staticmethod
    def memcpy_async(dst: int, src: int, nbytes: int,
                     kind: int = 1, stream: int = 0):
        """Asynchronous memcpy on a specific stream.

        kind: 0=H2H, 1=H2D, 2=D2H, 3=D2D (matches cudaMemcpyKind).
        For H2D to actually overlap with compute, `src` must be pinned.
        """
        if nbytes <= 0:
            return
        rt = _gpu_runtime()
        backend = _active_backend()
        fn_name = backend.get("memcpy_async")
        if fn_name is None:
            # Fallback to synchronous memcpy if async is unsupported.
            DeviceAllocator.memcpy(dst, src, nbytes, kind)
            return
        getattr(rt, fn_name)(
            ctypes.c_void_p(dst), ctypes.c_void_p(src),
            ctypes.c_size_t(nbytes), ctypes.c_int(kind),
            ctypes.c_void_p(stream))

    @staticmethod
    def ensure_triton_device(device_idx: int):
        """Set GPU device for Triton JIT context. Zero torch. Universal.

        Uses cudaSetDevice (runtime) + Triton's own driver API.
        Does NOT call cuCtxSetCurrent — that conflicts with the runtime
        context on multi-GPU systems.
        """
        # 1. Set runtime device (cudaSetDevice / hipSetDevice)
        backend = _active_backend()
        rt = _gpu_runtime()
        getattr(rt, backend["set_device"])(ctypes.c_int(device_idx))
        # 2. Tell Triton which device to target
        import triton.runtime.driver
        triton.runtime.driver.active.set_current_device(device_idx)


def _set_device(t):
    """Sync Triton/CUDA driver context with the tensor's physical device.

    Called before every Triton kernel launch. Triton's internal driver
    state must match `t._device_idx`, or the kernel will target the
    wrong device and raise "Pointer argument cannot be accessed from
    Triton" when the pointer lives on a different GPU. Single source of
    truth — imported by wrappers.py, dispatch.py, shape_utils.py,
    triton/weight_loader.py so every kernel-launching module honours
    the contract.
    """
    if hasattr(t, '_device_idx'):
        DeviceAllocator.ensure_triton_device(t._device_idx)


@functools.lru_cache(maxsize=1)
def _detect_gpu_backend() -> str:
    """Detect GPU backend: 'cuda' or 'hip'."""
    # Try Triton runtime first (most reliable)
    try:
        import triton.runtime.driver
        backend = triton.runtime.driver.active.get_current_target().backend
        if backend in ("cuda", "hip"):
            return backend
    except Exception:
        pass
    # Fallback: try loading CUDA runtime
    for name in ("libcudart.so", "libcudart.so.12"):
        try:
            ctypes.cdll.LoadLibrary(name)
            return "cuda"
        except OSError:
            continue
    # Fallback: try HIP runtime
    for name in ("libamdhip64.so", "libamdhip64.so.5"):
        try:
            ctypes.cdll.LoadLibrary(name)
            return "hip"
        except OSError:
            continue
    raise RuntimeError("No GPU runtime found (tried CUDA and ROCm/HIP)")


def _active_backend() -> dict:
    return _GPU_BACKENDS[_detect_gpu_backend()]


@functools.lru_cache(maxsize=1)
def _gpu_runtime():
    backend = _active_backend()
    for name in backend["rt_libs"]:
        try:
            return ctypes.cdll.LoadLibrary(name)
        except OSError:
            continue
    raise RuntimeError(f"GPU runtime not found for {_detect_gpu_backend()}")


# ============================================================================
# STRIDED COPY
# ============================================================================

def _strided_copy(src: 'NBXTensor', dst: 'NBXTensor'):
    """Copy non-contiguous src into contiguous dst via Triton kernel.

    Uses strided_copy_kernel from kernels/ops/strided_copy.py.
    Pads shape/strides to 5D for the kernel interface.
    """
    import triton
    from neurobrix.kernels.ops.strided_copy import strided_copy_kernel

    n = src._numel
    shape5 = list(src._shape) + [1] * (5 - src.ndim)
    stride5 = list(src._strides) + [0] * (5 - src.ndim)

    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    _set_device(src)
    strided_copy_kernel[grid](
        src, dst, n,
        stride5[0], stride5[1], stride5[2], stride5[3], stride5[4],
        shape5[0], shape5[1], shape5[2], shape5[3], shape5[4],
        BLOCK_SIZE=BLOCK,
    )


def _strided_scatter(src: 'NBXTensor', dst: 'NBXTensor'):
    """Scatter contiguous src into non-contiguous dst via Triton kernel.

    Inverse of _strided_copy. Used by __setitem__ for KV cache writes
    where the destination is a narrow view with non-contiguous strides.
    """
    import triton
    from neurobrix.kernels.ops.strided_copy import strided_scatter_kernel

    n = src._numel
    shape5 = list(dst._shape) + [1] * (5 - dst.ndim)
    stride5 = list(dst._strides) + [0] * (5 - dst.ndim)

    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    _set_device(src)
    strided_scatter_kernel[grid](
        src, dst, n,
        stride5[0], stride5[1], stride5[2], stride5[3], stride5[4],
        shape5[0], shape5[1], shape5[2], shape5[3], shape5[4],
        BLOCK_SIZE=BLOCK,
    )


# ============================================================================
# STRIDE COMPUTATION
# ============================================================================

def _contiguous_strides(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    if len(shape) == 0:
        return ()
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)


def _broadcast_shapes(*shapes: Tuple[int, ...]) -> Tuple[int, ...]:
    """Compute broadcast output shape for 2+ tensors. Pure Python."""
    if len(shapes) == 1:
        return shapes[0]
    result = shapes[0]
    for s in shapes[1:]:
        ndim = max(len(result), len(s))
        a = (1,) * (ndim - len(result)) + result
        b = (1,) * (ndim - len(s)) + s
        new = []
        for sa, sb in zip(a, b):
            if sa == sb:
                new.append(sa)
            elif sa == 1:
                new.append(sb)
            elif sb == 1:
                new.append(sa)
            else:
                raise ValueError(f"Cannot broadcast {result} and {s}")
        result = tuple(new)
    return result


# ============================================================================
# NBXTensor
# ============================================================================

class NBXTensor:
    """Tensor descriptor for --triton mode.

    Properties for Triton kernel compatibility:
      .data_ptr() → raw CUDA pointer (int) — Triton reads this
      .dtype      → triton.language dtype (tl.float16 etc.) — Triton JIT uses this
      .nbx_dtype  → NBXDtype enum — internal, zero dependency
      .shape      → tuple of ints
      .stride()   → tuple of ints (element strides)

    All metadata ops (view, reshape, permute...) are pure Python.
    Allocation via raw cudaMalloc.
    """

    __slots__ = ('_data_ptr', '_shape', '_strides', '_dtype', '_device',
                 '_nbytes', '_numel', '_offset', '_owns_data', '_device_idx',
                 '_base',     # reference to parent tensor (prevents GC of underlying memory);
                              # also holds the numpy buffer for unpinned CPU tensors
                 '_pinned',   # True for pinned host memory (cudaMallocHost); used by __del__
                              # to dispatch to free_host_pinned instead of free_cuda
                 )

    def __init__(self, data_ptr: int, shape: Tuple[int, ...],
                 strides: Tuple[int, ...], dtype: NBXDtype,
                 device: str = 'cuda', offset: int = 0, owns_data: bool = False,
                 device_idx: int = 0, base=None, pinned: bool = False):
        self._data_ptr = data_ptr
        self._shape = tuple(shape)
        self._strides = tuple(strides)
        self._dtype = dtype
        self._device = device
        self._offset = offset
        self._owns_data = owns_data
        self._device_idx = device_idx
        self._base = base  # keep parent alive to prevent use-after-free on views
        self._pinned = pinned
        self._numel = math.prod(shape) if shape else 1
        self._nbytes = self._numel * dtype_size(dtype)

    def __del__(self):
        if self._owns_data and self._data_ptr:
            if self._device == 'cuda':
                DeviceAllocator.free_cuda(self._data_ptr)
            elif self._device == 'cpu' and self._pinned:
                # Pinned host memory allocated via cudaMallocHost.
                DeviceAllocator.free_host_pinned(self._data_ptr)
            # Unpinned CPU: backed by self._base (numpy array) — Python
            # GC drops it automatically when self goes out of scope.

    # ========================================================================
    # PROPERTIES
    # ========================================================================

    def data_ptr(self) -> int:
        """Raw CUDA pointer. Triton kernels read this."""
        return self._data_ptr + self._offset * dtype_size(self._dtype)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def dtype(self):
        """Returns triton.language dtype for Triton JIT compatibility."""
        return _get_tl_dtype(self._dtype)

    @property
    def nbx_dtype(self) -> NBXDtype:
        """Internal dtype — zero dependency."""
        return self._dtype

    @property
    def device(self):
        """Device descriptor. Returns self (NBXTensor is its own device context)."""
        return self

    @property
    def type(self) -> str:
        """Device type string for device.type compatibility."""
        return self._device

    @property
    def index(self) -> int:
        """Device index for device.index compatibility."""
        return 0  # default GPU 0

    @property
    def raw_device(self) -> str:
        return self._device

    @property
    def ndim(self) -> int:
        return len(self._shape)

    def numel(self) -> int:
        """Element count. Method (not property) for ATen API compat."""
        return self._numel

    def nbytes(self) -> int:
        return self._nbytes

    def element_size(self) -> int:
        return dtype_size(self._dtype)

    def is_contiguous(self) -> bool:
        """Method for ATen API compat."""
        return self._strides == _contiguous_strides(self._shape)

    def is_expanded(self) -> bool:
        """True when the view replays elements along a broadcast axis.

        Expand views have one or more (shape > 1, stride == 0) axes, which
        means `numel * element_size > actual_backing_bytes`. Any caller that
        intends to `memcpy(nbytes)` across a device boundary MUST first
        materialise via `contiguous()` — otherwise the H2D / D2D read runs
        past the real allocation and copies garbage.
        """
        return any(st == 0 and sh > 1
                   for sh, st in zip(self._shape, self._strides))

    @property
    def is_cuda(self) -> bool:
        return self._device == 'cuda'

    @property
    def is_cpu(self) -> bool:
        """True for NBXTensor allocated on host memory (zero3 offload,
        pinned or unpinned). Every NBXTensor is either CUDA or CPU
        today; there is no third state."""
        return self._device == 'cpu'

    @property
    def is_pinned(self) -> bool:
        """True for CPU-backed tensors allocated via cudaMallocHost
        (fast non_blocking H2D DMA). False otherwise, including all
        CUDA tensors."""
        return self._pinned

    # ========================================================================
    # TYPE CHECKS
    # ========================================================================

    def is_floating_point(self) -> bool:
        return self._dtype in _FLOATING_DTYPES

    def is_complex(self) -> bool:
        return self._dtype in _COMPLEX_DTYPES

    # ========================================================================
    # SHAPE QUERY METHODS
    # ========================================================================

    def size(self, dim=None):
        if dim is not None:
            return self._shape[dim]
        return self._shape

    def dim(self) -> int:
        return len(self._shape)

    def stride(self, dim=None):
        if dim is not None:
            return self._strides[dim]
        return self._strides

    # ========================================================================
    # __cuda_array_interface__ — standard CUDA interop
    # ========================================================================

    @property
    def __cuda_array_interface__(self):
        return {
            'shape': self._shape,
            'typestr': _DTYPE_TYPESTR[self._dtype],
            'data': (self.data_ptr(), False),
            'version': 3,
            'strides': tuple(s * dtype_size(self._dtype) for s in self._strides),
        }

    # ========================================================================
    # FACTORY METHODS — raw cudaMalloc
    # ========================================================================

    @staticmethod
    def empty(shape, dtype=None, device='cuda') -> 'NBXTensor':
        if isinstance(shape, int):
            shape = (shape,)
        if isinstance(shape, list):
            shape = tuple(shape)
        nbx_dt = dtype if isinstance(dtype, NBXDtype) else parse_dtype(str(dtype)) if dtype else NBXDtype.float32
        dev_str = 'cuda'
        dev_idx = DeviceAllocator.get_device()  # default: current CUDA device
        if isinstance(device, str):
            if ':' in device:
                dev_idx = int(device.split(':')[1])
        elif hasattr(device, '_device_idx'):
            dev_idx = device._device_idx
        elif hasattr(device, 'index') and device.index is not None:
            dev_idx = device.index

        numel = math.prod(shape) if shape else 1
        nbytes = numel * dtype_size(nbx_dt)
        if nbytes == 0:
            return NBXTensor(0, shape, _contiguous_strides(shape), nbx_dt, dev_str, device_idx=dev_idx)

        # Multi-GPU: allocate on correct device
        cur = DeviceAllocator.get_device()
        if cur != dev_idx:
            DeviceAllocator.set_device(dev_idx)
        ptr = DeviceAllocator.malloc_cuda(nbytes)
        if cur != dev_idx:
            DeviceAllocator.set_device(cur)

        return NBXTensor(ptr, shape, _contiguous_strides(shape), nbx_dt, dev_str,
                         owns_data=True, device_idx=dev_idx)

    @staticmethod
    def zeros(shape, dtype=None, device='cuda') -> 'NBXTensor':
        t = NBXTensor.empty(shape, dtype, device)
        if t._nbytes > 0:
            DeviceAllocator.memset_cuda(t._data_ptr, 0, t._nbytes)
        return t

    @staticmethod
    def empty_like(other: 'NBXTensor', dtype=None, device=None) -> 'NBXTensor':
        dev = device if device else f"cuda:{other._device_idx}"
        return NBXTensor.empty(other._shape,
                               dtype if dtype else other._dtype,
                               dev)

    @staticmethod
    def zeros_like(other: 'NBXTensor', dtype=None, device=None) -> 'NBXTensor':
        return NBXTensor.zeros(other._shape,
                               dtype if dtype else other._dtype,
                               device if device else other._device)

    @staticmethod
    def from_numpy(arr) -> 'NBXTensor':
        """Load numpy array to GPU. For weight loading from safetensors."""
        import numpy as np
        dtype_name = str(arr.dtype)
        if dtype_name in _NUMPY_DTYPE_MAP:
            nbx_dt = _NUMPY_DTYPE_MAP[dtype_name]
        else:
            nbx_dt = NBXDtype.float32
        arr_c = np.ascontiguousarray(arr)
        nbytes = arr_c.nbytes
        shape = arr_c.shape
        dev_idx = DeviceAllocator.get_device()
        ptr = DeviceAllocator.malloc_cuda(nbytes)
        # H2D copy
        DeviceAllocator.memcpy(ptr, arr_c.ctypes.data, nbytes, kind=1)
        return NBXTensor(ptr, shape, _contiguous_strides(shape), nbx_dt, 'cuda',
                         owns_data=True, device_idx=dev_idx)

    @staticmethod
    def from_raw(ptr: int, shape: Tuple[int, ...], dtype: NBXDtype,
                 device: str = 'cuda', owns_data: bool = False,
                 device_idx: int = 0, base=None) -> 'NBXTensor':
        """Wrap an existing raw pointer. For interop at boundaries.

        Args:
            base: Reference to the object owning the memory (prevents GC).
                  For torch.Tensor → NBXTensor conversion, pass the torch.Tensor
                  so it stays alive as long as the NBXTensor exists.
        """
        return NBXTensor(ptr, shape, _contiguous_strides(shape), dtype, device,
                         owns_data=owns_data, device_idx=device_idx, base=base)

    # ------------------------------------------------------------------
    # CPU form — zero3 weight offload and similar scenarios where
    # weights live on host memory between promotions to GPU. Uses numpy
    # for unpinned backing (default) or cudaMallocHost for pinned
    # backing (faster H2D DMA). Zero torch dependency.
    # ------------------------------------------------------------------

    @staticmethod
    def empty_cpu(shape, dtype: 'Optional[NBXDtype]' = None,
                  pinned: bool = False) -> 'NBXTensor':
        """Allocate a CPU-backed NBXTensor.

        Unpinned (pinned=False, default): backed by a numpy array. Cheap
        to allocate, usable by numpy-based prep code, but incurs a
        pageable→pinned staging copy on H2D.

        Pinned (pinned=True): backed by cudaMallocHost-allocated memory.
        Enables non_blocking DMA at ~2× unpinned throughput. Used by
        zero3 for weights that will be transferred on the hot path.
        """
        if isinstance(shape, int):
            shape = (shape,)
        if isinstance(shape, list):
            shape = tuple(shape)
        nbx_dt = dtype if isinstance(dtype, NBXDtype) else (
            parse_dtype(str(dtype)) if dtype else NBXDtype.float32)
        numel = math.prod(shape) if shape else 1
        elem = dtype_size(nbx_dt)
        nbytes = numel * elem

        if nbytes == 0:
            return NBXTensor(0, shape, _contiguous_strides(shape), nbx_dt,
                             'cpu', device_idx=0)

        if pinned:
            ptr = DeviceAllocator.malloc_host_pinned(nbytes)
            return NBXTensor(ptr, shape, _contiguous_strides(shape), nbx_dt,
                             'cpu', owns_data=True, device_idx=0,
                             pinned=True)
        else:
            # Numpy backing — keep the array alive via _base.
            import numpy as np
            np_dtype_name = {
                NBXDtype.float32: np.float32,
                NBXDtype.float16: np.float16,
                NBXDtype.bfloat16: np.uint16,   # bf16 stored as uint16 bits
                NBXDtype.float64: np.float64,
                NBXDtype.int64: np.int64,
                NBXDtype.int32: np.int32,
                NBXDtype.int16: np.int16,
                NBXDtype.int8: np.int8,
                NBXDtype.uint8: np.uint8,
                NBXDtype.bool_: np.uint8,
            }.get(nbx_dt, np.float32)
            arr = np.empty(shape, dtype=np_dtype_name)
            return NBXTensor(arr.ctypes.data, shape,
                             _contiguous_strides(shape), nbx_dt, 'cpu',
                             owns_data=True, device_idx=0,
                             base=arr,  # keep numpy alive
                             pinned=False)

    def to_cuda(self, device_idx: int = 0) -> 'NBXTensor':
        """Copy this tensor to a CUDA device. Returns a new GPU NBXTensor.

        Handles CPU→GPU (kind=1 H2D) and GPU→GPU (kind=3 D2D). If the
        tensor is already on the requested CUDA device, returns self.
        Expand views (stride == 0 on a broadcast axis) are materialised
        first so the memcpy does not over-read the backing storage.
        """
        if self._device == 'cuda' and self._device_idx == device_idx:
            return self
        src = self.contiguous() if self.is_expanded() else self
        DeviceAllocator.set_device(device_idx)
        ptr = DeviceAllocator.malloc_cuda(src._nbytes)
        if src._nbytes > 0:
            # kind: 1 = H2D if source is CPU, 3 = D2D if source is GPU.
            kind = 1 if src._device == 'cpu' else 3
            DeviceAllocator.memcpy(ptr, src.data_ptr(), src._nbytes, kind=kind)
        return NBXTensor(ptr, src._shape, src._strides, src._dtype,
                         'cuda', owns_data=True, device_idx=device_idx)

    def to_cuda_async(self, device_idx: int = 0, stream: int = 0) -> 'NBXTensor':
        """Non-blocking variant of to_cuda for zero3 pipelining.

        Enqueues the memcpy on `stream` instead of running synchronously.
        Caller is responsible for sequencing — either via stream_wait_event
        before reading the returned tensor, or via stream_synchronize.

        For H2D to actually overlap with compute on the default stream, the
        source NBXTensor MUST be pinned (self._pinned == True). Otherwise
        the driver falls back to a synchronous bounce-buffer copy and no
        overlap happens.

        Returns a GPU NBXTensor backed by a freshly-malloc'd pointer. The
        allocation itself is synchronous (cudaMalloc); only the fill is
        asynchronous.
        """
        if self._device == 'cuda' and self._device_idx == device_idx:
            return self
        src = self.contiguous() if self.is_expanded() else self
        DeviceAllocator.set_device(device_idx)
        ptr = DeviceAllocator.malloc_cuda(src._nbytes)
        if src._nbytes > 0:
            kind = 1 if src._device == 'cpu' else 3
            DeviceAllocator.memcpy_async(
                ptr, src.data_ptr(), src._nbytes,
                kind=kind, stream=stream)
        return NBXTensor(ptr, src._shape, src._strides, src._dtype,
                         'cuda', owns_data=True, device_idx=device_idx)

    def to_cpu(self, pinned: bool = False) -> 'NBXTensor':
        """Copy this tensor to host memory. Returns a new CPU NBXTensor.

        pinned=True allocates via cudaMallocHost for fast future H2D
        promotion; pinned=False uses numpy backing.
        """
        if self._device == 'cpu' and self._pinned == pinned:
            return self
        dst = NBXTensor.empty_cpu(self._shape, self._dtype, pinned=pinned)
        if self._nbytes > 0:
            # kind: 2 = D2H from GPU, 0 = H2H if we're re-packing CPU.
            kind = 2 if self._device == 'cuda' else 0
            DeviceAllocator.memcpy(dst.data_ptr(), self.data_ptr(),
                                   self._nbytes, kind=kind)
        return dst

    def pin_host(self) -> 'NBXTensor':
        """Promote an unpinned CPU tensor to pinned host memory.

        No-op if the tensor is already pinned. Used by Zero3Strategy to
        accelerate subsequent H2D transfers during block-wise execution.
        Raises on non-CPU tensors — pinning a GPU tensor is nonsensical.
        """
        if self._device != 'cpu':
            raise RuntimeError(
                f"pin_host() called on device={self._device}; pinning only "
                f"applies to CPU-backed NBXTensors")
        if self._pinned:
            return self
        return self.to_cpu(pinned=True)

    # ========================================================================
    # METADATA OPS — pure Python, zero data movement
    # ========================================================================

    def view(self, *shape) -> 'NBXTensor':
        # Match PyTorch's aten::view tolerance: if the source tensor is
        # expanded (stride-0 broadcast) or otherwise non-contiguous, the
        # old path silently re-strode the view as contiguous while the
        # underlying memory still held the unmaterialized broadcast —
        # later kernel launches walked past the end of the backing
        # allocation and got "Pointer argument cannot be accessed from
        # Triton" on the second batch iteration (Janus RoPE bmm). Matches
        # reshape's fallback: contiguous() first, then re-stride.
        if not self.is_contiguous():
            return self.contiguous().view(*shape)
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = tuple(shape[0])
        neg_idx = None
        known = 1
        for i, s in enumerate(shape):
            if s == -1:
                neg_idx = i
            else:
                known *= s
        if neg_idx is not None:
            shape = list(shape)
            shape[neg_idx] = self._numel // known
            shape = tuple(shape)
        return NBXTensor(self._data_ptr, shape, _contiguous_strides(shape),
                         self._dtype, self._device, self._offset, device_idx=self._device_idx, base=self._base if self._base is not None else self, pinned=self._pinned)

    def reshape(self, *shape) -> 'NBXTensor':
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = tuple(shape[0])
        if self.is_contiguous():
            return self.view(*shape)
        return self.contiguous().view(*shape)

    def unsqueeze(self, dim: int) -> 'NBXTensor':
        dim = dim % (self.ndim + 1) if dim >= 0 else dim + self.ndim + 1
        new_shape = list(self._shape)
        new_strides = list(self._strides)
        stride_val = new_strides[dim] * self._shape[dim] if dim < len(self._strides) else 1
        new_shape.insert(dim, 1)
        new_strides.insert(dim, stride_val)
        return NBXTensor(self._data_ptr, tuple(new_shape), tuple(new_strides),
                         self._dtype, self._device, self._offset, device_idx=self._device_idx, base=self._base if self._base is not None else self, pinned=self._pinned)

    def squeeze(self, dim=None) -> 'NBXTensor':
        if dim is not None:
            dim = dim % self.ndim
            if self._shape[dim] != 1:
                return self
            s = list(self._shape)
            st = list(self._strides)
            s.pop(dim)
            st.pop(dim)
        else:
            s, st = [], []
            for sz, sr in zip(self._shape, self._strides):
                if sz != 1:
                    s.append(sz)
                    st.append(sr)
        return NBXTensor(self._data_ptr, tuple(s), tuple(st),
                         self._dtype, self._device, self._offset, device_idx=self._device_idx, base=self._base if self._base is not None else self, pinned=self._pinned)

    def permute(self, *dims) -> 'NBXTensor':
        if len(dims) == 1 and isinstance(dims[0], (list, tuple)):
            dims = tuple(dims[0])
        return NBXTensor(self._data_ptr,
                         tuple(self._shape[d] for d in dims),
                         tuple(self._strides[d] for d in dims),
                         self._dtype, self._device, self._offset, device_idx=self._device_idx, base=self._base if self._base is not None else self, pinned=self._pinned)

    def transpose(self, dim0: int, dim1: int) -> 'NBXTensor':
        dim0, dim1 = dim0 % self.ndim, dim1 % self.ndim
        s = list(self._shape)
        st = list(self._strides)
        s[dim0], s[dim1] = s[dim1], s[dim0]
        st[dim0], st[dim1] = st[dim1], st[dim0]
        return NBXTensor(self._data_ptr, tuple(s), tuple(st),
                         self._dtype, self._device, self._offset, device_idx=self._device_idx, base=self._base if self._base is not None else self, pinned=self._pinned)

    def t(self) -> 'NBXTensor':
        if self.ndim < 2:
            return self
        return self.transpose(0, 1)

    @property
    def T(self) -> 'NBXTensor':
        return self.transpose(-2, -1) if self.ndim >= 2 else self

    def expand(self, *shape) -> 'NBXTensor':
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = tuple(shape[0])
        pad = len(shape) - self.ndim
        old_s = (1,) * pad + self._shape
        old_st = (0,) * pad + self._strides
        new_s, new_st = [], []
        for i, s in enumerate(shape):
            if s == -1:
                new_s.append(old_s[i]); new_st.append(old_st[i])
            elif old_s[i] == 1:
                new_s.append(s); new_st.append(0)
            else:
                new_s.append(s); new_st.append(old_st[i])
        return NBXTensor(self._data_ptr, tuple(new_s), tuple(new_st),
                         self._dtype, self._device, self._offset, device_idx=self._device_idx, base=self._base if self._base is not None else self, pinned=self._pinned)

    def expand_as(self, other) -> 'NBXTensor':
        return self.expand(*other._shape if isinstance(other, NBXTensor) else other.shape)

    def narrow(self, dim: int, start: int, length: int) -> 'NBXTensor':
        dim = dim % self.ndim
        s = list(self._shape)
        s[dim] = length
        new_off = self._offset + start * self._strides[dim]
        return NBXTensor(self._data_ptr, tuple(s), self._strides,
                         self._dtype, self._device, new_off, device_idx=self._device_idx, base=self._base if self._base is not None else self, pinned=self._pinned)

    def select(self, dim: int, index: int) -> 'NBXTensor':
        dim = dim % self.ndim
        new_off = self._offset + index * self._strides[dim]
        s = list(self._shape)
        st = list(self._strides)
        s.pop(dim)
        st.pop(dim)
        return NBXTensor(self._data_ptr, tuple(s), tuple(st),
                         self._dtype, self._device, new_off, device_idx=self._device_idx, base=self._base if self._base is not None else self, pinned=self._pinned)

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> 'NBXTensor':
        start_dim = start_dim % self.ndim
        end_dim = end_dim % self.ndim
        if start_dim == end_dim:
            return self
        flat = 1
        for i in range(start_dim, end_dim + 1):
            flat *= self._shape[i]
        new_s = list(self._shape[:start_dim]) + [flat] + list(self._shape[end_dim + 1:])
        return self.view(*new_s)

    def contiguous(self) -> 'NBXTensor':
        if self.is_contiguous():
            return self
        if self._device == 'cpu':
            return self._contiguous_cpu()
        new = NBXTensor.empty(self._shape, self._dtype, f"cuda:{self._device_idx}")
        n = self._numel
        if n > 0:
            _strided_copy(self, new)
        return new

    def _contiguous_cpu(self) -> 'NBXTensor':
        """CPU-side materialization of a non-contiguous view.

        The shared _strided_copy path launches a Triton kernel over GPU
        pointers, so it cannot reach host memory. This helper performs
        the same strided → contiguous copy in numpy (C-implemented
        `ascontiguousarray` over an `as_strided` view) and preserves the
        source's pinned / unpinned backing. Expand views (stride == 0)
        materialize correctly because numpy replays the same source
        element along zero-stride axes.
        """
        import numpy as np
        dst = NBXTensor.empty_cpu(self._shape, self._dtype, pinned=self._pinned)
        if self._numel == 0:
            return dst
        esz = dtype_size(self._dtype)
        # Byte window touched by the strided view starting at self.data_ptr():
        # last-reachable element offset + one element for esz bytes.
        max_elem_offset = 1 + sum(
            (dim - 1) * st for dim, st in zip(self._shape, self._strides)
            if dim > 0)
        n_bytes = max_elem_offset * esz
        src_bytes = np.ctypeslib.as_array(
            ctypes.cast(self.data_ptr(), ctypes.POINTER(ctypes.c_ubyte)),
            shape=(n_bytes,),
        )
        byte_strides = tuple(s * esz for s in self._strides)
        strided_view = np.lib.stride_tricks.as_strided(
            src_bytes,
            shape=tuple(self._shape) + (esz,),
            strides=byte_strides + (1,),
        )
        contig = np.ascontiguousarray(strided_view)
        ctypes.memmove(dst.data_ptr(), contig.ctypes.data, dst._nbytes)
        return dst

    def unfold(self, dimension: int, size: int, step: int) -> 'NBXTensor':
        dimension = dimension % self.ndim
        n = (self._shape[dimension] - size) // step + 1
        s = list(self._shape)
        st = list(self._strides)
        s[dimension] = n
        s.append(size)
        st[dimension] = self._strides[dimension] * step
        st.append(self._strides[dimension])
        return NBXTensor(self._data_ptr, tuple(s), tuple(st),
                         self._dtype, self._device, self._offset, device_idx=self._device_idx, base=self._base if self._base is not None else self, pinned=self._pinned)

    def as_strided(self, size, stride, storage_offset=None) -> 'NBXTensor':
        off = storage_offset if storage_offset is not None else self._offset
        return NBXTensor(self._data_ptr, tuple(size), tuple(stride),
                         self._dtype, self._device, off, device_idx=self._device_idx, base=self._base if self._base is not None else self, pinned=self._pinned)

    def view_as(self, other) -> 'NBXTensor':
        s = other._shape if isinstance(other, NBXTensor) else other.shape
        return self.view(*s)

    def reshape_as(self, other) -> 'NBXTensor':
        s = other._shape if isinstance(other, NBXTensor) else other.shape
        return self.reshape(*s)

    def movedim(self, src: int, dst: int) -> 'NBXTensor':
        src, dst = src % self.ndim, dst % self.ndim
        dims = list(range(self.ndim))
        dims.pop(src)
        dims.insert(dst, src)
        return self.permute(*dims)

    def __getitem__(self, key):
        """Basic tensor indexing: t[0], t[:, :, 0], t[:, 2:5], etc."""
        if isinstance(key, int):
            return self.select(0, key)
        if isinstance(key, slice):
            start = key.start or 0
            stop = key.stop or self._shape[0]
            return self.narrow(0, start, stop - start)
        if isinstance(key, tuple):
            result = self
            dim_offset = 0
            for dim, k in enumerate(key):
                actual_dim = dim - dim_offset
                if isinstance(k, int):
                    result = result.select(actual_dim, k)
                    dim_offset += 1  # select reduces ndim
                elif isinstance(k, slice):
                    start = k.start or 0
                    stop = k.stop or result._shape[actual_dim]
                    if stop < 0:
                        stop = result._shape[actual_dim] + stop
                    result = result.narrow(actual_dim, start, stop - start)
                # slice(None) = : → no-op on this dim
        return result

    def __setitem__(self, key, value):
        """Indexed write: buffer[..., start:end, :] = source.

        Handles the KV cache pattern: contiguous slice assignment.
        Uses strided_scatter_kernel when the destination view is non-contiguous.
        Guards: dtype cast and device transfer before copy.
        """
        dst = self[key]
        if isinstance(value, NBXTensor):
            # Dtype guard: cast source to destination dtype
            if value._dtype != dst._dtype:
                value = value.to(dst._dtype)
            # Device guard: transfer source to destination device
            if hasattr(value, '_device_idx') and hasattr(dst, '_device_idx') and value._device_idx != dst._device_idx:
                src_contig = value.contiguous()
                tmp = NBXTensor.empty(src_contig._shape, src_contig._dtype, f"cuda:{dst._device_idx}")
                DeviceAllocator.memcpy(tmp.data_ptr(), src_contig.data_ptr(), src_contig._nbytes, kind=3)
                value = tmp
            src = value.contiguous()
            if dst.is_contiguous() and src._nbytes == dst._nbytes:
                DeviceAllocator.memcpy(dst.data_ptr(), src.data_ptr(), src._nbytes)
            elif dst._numel > 0:
                _strided_scatter(src, dst)

    def unbind(self, dim: int = 0) -> tuple:
        dim = dim % self.ndim
        return tuple(self.select(dim, i) for i in range(self._shape[dim]))

    # ========================================================================
    # DATA OPS — cudaMemcpy
    # ========================================================================

    def clone(self) -> 'NBXTensor':
        new = NBXTensor.empty(self._shape, self._dtype, f"cuda:{self._device_idx}")
        n = self._numel
        if n > 0:
            if self.is_contiguous():
                DeviceAllocator.memcpy(new._data_ptr, self.data_ptr(), self._nbytes)
            else:
                _strided_copy(self, new)
        return new

    def detach(self) -> 'NBXTensor':
        return self  # no autograd in triton mode

    def fill_(self, value) -> 'NBXTensor':
        if value == 0 and self._nbytes > 0:
            DeviceAllocator.memset_cuda(self.data_ptr(), 0, self._nbytes)
        # Non-zero fill: needs Triton fill kernel (TODO)
        return self

    def copy_(self, src, non_blocking: bool = False) -> 'NBXTensor':
        if isinstance(src, NBXTensor):
            DeviceAllocator.memcpy(self.data_ptr(), src.data_ptr(), self._nbytes)
        return self

    def item(self) -> float:
        """Read single scalar from GPU via cudaMemcpy D2H."""
        assert self._numel == 1, f"item() requires 1 element, got {self._numel}"
        esz = dtype_size(self._dtype)
        buf = (ctypes.c_char * esz)()
        DeviceAllocator.memcpy(ctypes.addressof(buf), self.data_ptr(), esz, kind=2)

        if self._dtype == NBXDtype.float32:
            return ctypes.cast(buf, ctypes.POINTER(ctypes.c_float))[0]
        elif self._dtype == NBXDtype.float64:
            return ctypes.cast(buf, ctypes.POINTER(ctypes.c_double))[0]
        elif self._dtype == NBXDtype.int32:
            return ctypes.cast(buf, ctypes.POINTER(ctypes.c_int32))[0]
        elif self._dtype == NBXDtype.int64:
            return ctypes.cast(buf, ctypes.POINTER(ctypes.c_int64))[0]
        elif self._dtype == NBXDtype.float16:
            raw = ctypes.cast(buf, ctypes.POINTER(ctypes.c_uint16))[0]
            return struct.unpack('e', struct.pack('H', raw))[0]
        elif self._dtype == NBXDtype.bool_:
            return bool(ctypes.cast(buf, ctypes.POINTER(ctypes.c_uint8))[0])
        return 0.0

    # ========================================================================
    # DTYPE CONVERSION
    # ========================================================================

    def to(self, *args, **kwargs) -> 'NBXTensor':
        target = None
        if args:
            a = args[0]
            if isinstance(a, NBXDtype):
                target = a
            elif isinstance(a, str):
                # Distinguish device string ("cuda:0") from dtype string ("float32")
                if 'cuda' in a or 'cpu' in a:
                    return self  # .to(device) — already on device
                target = parse_dtype(a)
            elif isinstance(a, NBXTensor):
                return self
            else:
                # Check if it's a device-like object (has .type attribute = "cuda"/"cpu")
                s = str(a)
                if 'cuda' in s or 'cpu' in s or 'device' in s:
                    return self  # .to(device) — already on device
                target = parse_dtype(s)
        if 'dtype' in kwargs:
            d = kwargs['dtype']
            target = d if isinstance(d, NBXDtype) else parse_dtype(str(d))
        if target is None or target == self._dtype:
            return self
        # Cast via Triton copy kernel (tl.store auto-converts dtype)
        new = NBXTensor.empty(self._shape, target, f"cuda:{self._device_idx}")
        n = self._numel
        if n > 0:
            from neurobrix.kernels.ops.copy_op import copy_kernel
            import triton
            src = self.contiguous()
            _set_device(src)
            copy_kernel[(triton.cdiv(n, 1024),)](src, new, n, BLOCK_SIZE=1024)
        return new

    def float(self) -> 'NBXTensor':
        return self.to(NBXDtype.float32)

    def half(self) -> 'NBXTensor':
        return self.to(NBXDtype.float16)

    def bfloat16(self) -> 'NBXTensor':
        return self.to(NBXDtype.bfloat16)

    def int(self) -> 'NBXTensor':
        return self.to(NBXDtype.int32)

    def long(self) -> 'NBXTensor':
        return self.to(NBXDtype.int64)

    def type_as(self, other) -> 'NBXTensor':
        target = other._dtype if isinstance(other, NBXTensor) else parse_dtype(str(other.dtype))
        return self.to(target)

    # ========================================================================
    # CREATION HELPERS
    # ========================================================================

    def new_zeros(self, size, **kwargs) -> 'NBXTensor':
        return NBXTensor.zeros(tuple(size) if isinstance(size, list) else size,
                               kwargs.get('dtype', self._dtype),
                               kwargs.get('device', self._device))

    def new_ones(self, size, **kwargs) -> 'NBXTensor':
        # TODO: fill kernel for ones
        return NBXTensor.empty(tuple(size) if isinstance(size, list) else size,
                               kwargs.get('dtype', self._dtype),
                               kwargs.get('device', self._device))

    def new_empty(self, size, **kwargs) -> 'NBXTensor':
        return NBXTensor.empty(tuple(size) if isinstance(size, list) else size,
                               kwargs.get('dtype', self._dtype),
                               kwargs.get('device', self._device))

    # ========================================================================
    # STATIC ASSEMBLY — cat, stack (require data copy)
    # ========================================================================

    @staticmethod
    def cat(tensors: list, dim: int = 0) -> 'NBXTensor':
        # Filter out 0-dim scalars and empty tensors (Gemma2 attention pattern)
        valid = [t for t in tensors if t.ndim > 0 and t._numel > 0]
        if not valid:
            return tensors[0] if tensors else NBXTensor.empty((0,), NBXDtype.float32, 'cuda')
        if len(valid) == 1:
            return valid[0]
        tensors = valid
        dim = dim % tensors[0].ndim
        # Align dtypes — cat_copy_kernel requires all inputs same type
        target_dtype = tensors[0]._dtype
        tensors = [t.to(target_dtype) if t._dtype != target_dtype else t for t in tensors]
        # Align devices — all tensors must be on same GPU
        target_dev = tensors[0]._device_idx
        aligned = []
        for t in tensors:
            if hasattr(t, '_device_idx') and t._device_idx != target_dev:
                src = t.contiguous()
                dst = NBXTensor.empty(src._shape, src._dtype, f"cuda:{target_dev}")
                DeviceAllocator.memcpy(dst.data_ptr(), src.data_ptr(), src._nbytes, kind=3)
                aligned.append(dst)
            else:
                aligned.append(t)
        tensors = aligned
        out_shape = list(tensors[0]._shape)
        out_shape[dim] = sum(t._shape[dim] for t in tensors)
        # Allocate on same device as first input tensor
        DeviceAllocator.set_device(tensors[0]._device_idx)
        out = NBXTensor.empty(tuple(out_shape), target_dtype, tensors[0]._device)

        if dim == 0:
            # Simple case: concat on first dim → contiguous slices
            offset = 0
            for t in tensors:
                tc = t.contiguous()
                if tc._nbytes > 0:
                    DeviceAllocator.memcpy(
                        out._data_ptr + offset * _DTYPE_SIZES[out._dtype],
                        tc.data_ptr(), tc._nbytes)
                offset += tc._numel
        else:
            # General case: use cat_copy_kernel which handles dim offsets
            import triton
            from neurobrix.kernels.ops.cat_op import cat_copy_kernel_4

            dim_size_out = out_shape[dim]
            # Product of dims after cat dim
            dim_prod_post = 1
            for d in range(dim + 1, len(out_shape)):
                dim_prod_post *= out_shape[d]

            # Process tensors in groups of 4 (kernel handles up to 4)
            dim_offset = 0
            for batch_start in range(0, len(tensors), 4):
                batch = tensors[batch_start:batch_start + 4]
                # Pad to 4 with dummy (zero-element) tensors
                contig = [t.contiguous() for t in batch]
                while len(contig) < 4:
                    contig.append(out)  # dummy, total_elements=0

                sizes = [c._shape[dim] if i < len(batch) else 0 for i, c in enumerate(contig)]
                totals = [c._numel if i < len(batch) else 0 for i, c in enumerate(contig)]
                offsets = []
                off = dim_offset
                for s in sizes:
                    offsets.append(off)
                    off += s

                max_total = max(totals)
                if max_total == 0:
                    continue
                BLOCK = 1024
                grid = (triton.cdiv(max_total, BLOCK), len(batch))
                _set_device(out)
                cat_copy_kernel_4[grid](
                    out,
                    contig[0], contig[1], contig[2], contig[3],
                    sizes[0], sizes[1], sizes[2], sizes[3],
                    dim_size_out, dim_prod_post,
                    offsets[0], offsets[1], offsets[2], offsets[3],
                    totals[0], totals[1], totals[2], totals[3],
                    BLOCK_X=BLOCK,
                )
                dim_offset = off

        return out

    @staticmethod
    def stack(tensors: list, dim: int = 0) -> 'NBXTensor':
        return NBXTensor.cat([t.unsqueeze(dim) for t in tensors], dim=dim)

    # ========================================================================
    # REPR
    # ========================================================================

    # ========================================================================
    # PYTHON OPERATORS — delegate to kernel wrappers
    # ========================================================================

    def __add__(self, other):
        from neurobrix.kernels.wrappers import add
        return add(self, other)

    def __radd__(self, other):
        from neurobrix.kernels.wrappers import add
        return add(other, self)

    def __sub__(self, other):
        from neurobrix.kernels.wrappers import sub
        return sub(self, other)

    def __rsub__(self, other):
        from neurobrix.kernels.wrappers import rsub
        return rsub(self, other)

    def __mul__(self, other):
        from neurobrix.kernels.wrappers import mul
        return mul(self, other)

    def __rmul__(self, other):
        from neurobrix.kernels.wrappers import mul
        return mul(other, self)

    def __truediv__(self, other):
        from neurobrix.kernels.wrappers import div
        return div(self, other)

    def __neg__(self):
        from neurobrix.kernels.wrappers import neg
        return neg(self)

    def __repr__(self) -> str:
        return (f"NBXTensor(shape={self._shape}, dtype={self._dtype.name}, "
                f"device={self._device}, ptr=0x{self._data_ptr:x})")
