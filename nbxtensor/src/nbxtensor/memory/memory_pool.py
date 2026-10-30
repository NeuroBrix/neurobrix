"""Triton Memory Pool — arena-based GPU memory allocator.

Pre-allocates one contiguous block per component, sub-allocates inside
via bump pointer. No individual free, no fragmentation.

Zero torch. Uses DeviceAllocator (hardware-agnostic: CUDA/ROCm).
"""

import gc

import numpy as np

from nbxtensor.memory.allocator import DeviceAllocator
from nbxtensor.memory.layout import ARENA_ALIGNMENT, aligned
from nbxtensor.memory.device_transfer import parse_device_idxs


def release_flow_memory(device=None) -> None:
    """Flow-boundary memory release — triton-side mirror of the compiled
    `MemoryManager.release_flow_memory` discipline (R30 naming parity,
    separate implementation: R33-pure, DeviceAllocator only, no torch).

    Consolidates the bare `gc.collect()` calls the triton flow handlers
    ran at stage boundaries. Those calls skipped the sync-before-free
    protection: `gc.collect()` triggers NBXTensor / ComponentArena
    finalizers (`cudaFree`) on cyclic or lingering references, and freeing
    memory an in-flight kernel is still reading is the err-700 class
    documented in `core/memory/manager.py`. Ordering here:

      1. sync every CUDA device named in `device` (the flow's Prism
         primary-device string — may be compound, e.g. "fgp:cuda:0,cuda:1")
         BEFORE any reference can be finalized;
      2. gc.collect() — drop refs, run finalizers;
      3. DeviceAllocator.empty_cache_pool() — return pool-cached blocks to
         the driver (no-op under NBX_ALLOC_POOL=0; individual frees then
         already went straight to cudaFree).
    """
    idxs = parse_device_idxs(device)
    if idxs:
        prev = DeviceAllocator.get_device()
        for i in idxs:
            DeviceAllocator.set_device(i)
            DeviceAllocator.sync_device()
        DeviceAllocator.set_device(prev)
    gc.collect()
    DeviceAllocator.empty_cache_pool()


class ComponentArena:
    """Single contiguous GPU memory block for one component's weights.

    Allocates one big block via DeviceAllocator.malloc_cuda, then
    sub-allocates inside via bump pointer with 256-byte alignment.

    Usage:
        arena = ComponentArena(total_bytes, device_idx)
        ptr1 = arena.alloc(1024)   # sub-allocate 1024 bytes
        ptr2 = arena.alloc(4096)   # sub-allocate 4096 bytes
        arena.free()               # release entire block at once
    """

    ALIGNMENT = ARENA_ALIGNMENT  # the library's one arena layout (memory.layout)

    def __init__(self, total_bytes: int, device_idx: int):
        # Initialize _base_ptr BEFORE the malloc call so __del__ is safe
        # even if malloc_cuda raises (OOM on one device shouldn't leave
        # a half-constructed object whose __del__ then crashes on
        # attribute access). This closes a secondary issue flagged
        # during the zero3 leak investigation (see tests/scratch/
        # zero3_leak_investigation/REPORT.md §0.5).
        self.device_idx = device_idx
        self._base_ptr = 0
        self._total = self._align(total_bytes)
        self._offset = 0

        DeviceAllocator.set_device(device_idx)
        # never pooled: once freed, its bytes go back to the driver whole (DeviceAllocator.malloc_cuda `pooled`)
        self._base_ptr = DeviceAllocator.malloc_cuda(self._total, pooled=False)
        if self._base_ptr == 0 and self._total > 0:
            raise RuntimeError(
                f"ComponentArena: failed to allocate {self._total / 1e9:.2f} GB "
                f"on device {device_idx}")

    def alloc(self, nbytes: int) -> int:
        """Sub-allocate nbytes from the arena. Returns GPU pointer."""
        if nbytes == 0:
            return 0
        aligned = self._align(nbytes)
        if self._offset + aligned > self._total:
            raise RuntimeError(
                f"ComponentArena: out of memory. "
                f"Requested {nbytes} bytes, "
                f"used {self._offset}/{self._total} bytes "
                f"on device {self.device_idx}")
        ptr = self._base_ptr + self._offset
        self._offset += aligned
        return ptr

    def free(self):
        """Release the entire arena block."""
        if self._base_ptr:
            DeviceAllocator.set_device(self.device_idx)
            DeviceAllocator.free_cuda(self._base_ptr)
            self._base_ptr = 0
            self._offset = 0
            self._total = 0

    def reset(self):
        """Reset bump pointer without freeing GPU memory.

        Allows reuse of the same arena for a new component load
        (e.g., lifecycle reload in lazy_sequential strategy).
        """
        self._offset = 0

    @property
    def used(self) -> int:
        return self._offset

    @property
    def capacity(self) -> int:
        return self._total

    def _align(self, n: int) -> int:
        return aligned(n, self.ALIGNMENT)

    def __del__(self):
        # Guard: __init__ may raise before _base_ptr is set (rare but
        # happens when malloc_cuda throws OOM immediately). Let __del__
        # remain a no-op in that case.
        if getattr(self, '_base_ptr', 0):
            self.free()


class HostPinnedArena:
    """One page-locked host block for a component's host-held weights (zero3's block weights, an expert store's
    routed experts), sub-allocated like `ComponentArena`. A pinned allocation per weight cost Qwen3-30B-A3B 18 864
    page-locking calls — about 40 % of a 447 s load (py-spy, V100 rack, 2026-10-27); one block costs one. Each
    weight holds the arena as its `base`, so the block is released when the last of them is."""

    ALIGNMENT = ARENA_ALIGNMENT

    def __init__(self, total_bytes: int):
        """On a backend that page-locks existing memory (`host_register`), the block is ordinary host memory, written
        by the load and page-locked by `pin()` in one call before any transfer reads it: a page-locked allocation
        ran at 1.2 GiB/s on one thread, its threads serialised by the runtime (16 GiB in 13.0 s; 4 or 8 threads,
        16 s), while 16 GiB written by 16 threads and then registered took 4.9 s (V100 rack, 2026-10-30) — and the
        load's own writes are that touch. Elsewhere the block is the backend's page-locked allocation, as before."""
        self._base_ptr = 0
        self._total = aligned(total_bytes, self.ALIGNMENT)
        self._offset = 0
        self._pageable = None          # the host bytes until `pin()` page-locks them (a register-capable backend)
        self._registered = False
        if self._total:
            from nbxtensor.backend import runtime as _rt
            if _rt.device_memory().supports("host_register"):
                self._pageable = np.empty(self._total + self.ALIGNMENT, dtype=np.uint8)
                self._base_ptr = aligned(self._pageable.ctypes.data, self.ALIGNMENT)
            else:
                self._base_ptr = DeviceAllocator.malloc_host_pinned(self._total)

    def pin(self) -> None:
        """Page-lock the block (once; a no-op where it was allocated page-locked). Called by the loader after the
        last weight is written and before any transfer reads one."""
        if self._pageable is not None and not self._registered:
            DeviceAllocator.register_host_pinned(self._base_ptr, self._total)
            self._registered = True

    def alloc(self, nbytes: int) -> int:
        if nbytes == 0:
            return 0
        size = aligned(nbytes, self.ALIGNMENT)
        if self._offset + size > self._total:
            raise RuntimeError(f"HostPinnedArena: {nbytes} bytes asked, {self._offset}/{self._total} used — the "
                               f"sizing pass and the load disagree on a host weight's bytes")
        ptr = self._base_ptr + self._offset
        self._offset += size
        return ptr

    @property
    def used(self) -> int:
        return self._offset

    def __del__(self):
        if getattr(self, "_base_ptr", 0):
            if getattr(self, "_pageable", None) is None:
                DeviceAllocator.free_host_pinned(self._base_ptr)
            elif self._registered:
                DeviceAllocator.unregister_host_pinned(self._base_ptr)
            self._pageable = None
            self._base_ptr = 0
