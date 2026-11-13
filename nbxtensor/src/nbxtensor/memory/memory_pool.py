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
        DeviceAllocator.restore_device(prev)
    gc.collect()
    DeviceAllocator.empty_cache_pool()


def release_weights(weights: dict) -> bool:
    """Release a loaded weight set safely: the library's caches that hold weights (the attention-bias tables, the
    pointer tables that pin experts) dropped; then, when the set is the library's (an NBXTensor in it, or its arenas
    under "_arenas", {device: ComponentArena}), every device holding one synchronised, the dict cleared (the
    finalizers free) and the pool's cached blocks given back — True. False, the dict untouched: another library's
    weights."""
    import sys
    from nbxtensor.tensor import NBXTensor
    for module, release in (("nbxtensor.ops.elementwise_wrappers", "clear_bias_caches"),
                            ("nbxtensor.memory.pointer_tables", "release_pinned_tables")):
        if module in sys.modules:                   # a run that never built one holds none
            getattr(sys.modules[module], release)()
    arenas = weights.get("_arenas")
    if arenas is None and not any(isinstance(t, NBXTensor) for t in weights.values()):
        return False
    devs = {getattr(a, "device_idx", None) for a in (arenas or {}).values()} - {None}
    devs |= {t.device_idx for t in weights.values() if isinstance(t, NBXTensor)}
    for d in sorted(devs):
        DeviceAllocator.device_synchronize(d)
    arenas = None                                   # the dict's references are the last: its clear frees them
    weights.clear()
    if devs:
        DeviceAllocator.empty_cache_pool()
    return True


def join_weights(held: dict, loaded: dict) -> dict:
    """`loaded` with its arenas ("_arenas") merged into `held`'s, to `update` `held` with: every arena of both loads
    stays referenced — a weight of the first load points into its arena — a second arena on a device under a distinct
    key (`release_weights` reads their devices, never their keys)."""
    if "_arenas" not in loaded:
        return loaded
    merged = dict(held.get("_arenas") or {})
    for dev, arena in (loaded.get("_arenas") or {}).items():
        key, n = dev, 1
        while key in merged and merged[key] is not arena:
            key, n = f"{dev}#{n}", n + 1
        merged[key] = arena
    return {**loaded, "_arenas": merged}


def break_aliases(tensors, freed) -> int:
    """Before the tensors of `tensors` (a list or a dict) at the keys `freed` are freed, replace every other entry that
    is a view of one of their allocations (its view chain's root is one of theirs: `NBXTensor.storage_root`) by its
    `contiguous()`; returns how many entries were replaced. None entries, non-tensors and tensors that view nothing are
    left alone. Known limit, kept from the engine's two copies it replaces: a view that is already contiguous is its own
    `contiguous()`, so it still reads the freed memory."""
    from nbxtensor.tensor import NBXTensor
    freed = set(freed)
    get = tensors.get if isinstance(tensors, dict) else tensors.__getitem__
    roots = {id(t.storage_root()) for t in map(get, freed) if isinstance(t, NBXTensor)}
    if not roots:
        return 0
    copied = 0
    for key, t in list(tensors.items() if isinstance(tensors, dict) else enumerate(tensors)):
        if key in freed or not isinstance(t, NBXTensor):
            continue
        root = t.storage_root()
        if root is not t and id(root) in roots:
            tensors[key] = t.contiguous()
            copied += 1
    return copied


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


class WeightArenas:
    """The memory of one weight set: aligned bytes counted per device and for page-locked host memory while the
    application decides where each weight goes, then one `ComponentArena` per device and one `HostPinnedArena`, the
    host block page-locked once its last weight is written. The application decides placement; this holds the
    memory (2026-11-13: the counting, the arenas and the pinning were the engine's loader's own)."""

    def __init__(self):
        self._device_bytes: dict = {}
        self._host_bytes = 0
        self.devices: dict = {}           # {device: ComponentArena}, after `allocate`
        self.host = None                  # the HostPinnedArena, after `allocate`, when any host bytes were reserved

    def reserve(self, device: int, nbytes: int) -> None:
        """Count a weight of `nbytes` stored bytes on `device`, at the arena's alignment."""
        self._device_bytes[device] = self._device_bytes.get(device, 0) + aligned(nbytes, ComponentArena.ALIGNMENT)

    def reserve_host(self, nbytes: int) -> None:
        """Count a weight of `nbytes` stored bytes in page-locked host memory, at the host block's alignment."""
        self._host_bytes += aligned(nbytes, HostPinnedArena.ALIGNMENT)

    def device_bytes(self) -> dict:
        """{device: bytes reserved}."""
        return dict(self._device_bytes)

    def allocate(self) -> "WeightArenas":
        """One arena per device that has bytes reserved, and the host block when host bytes were; refused once."""
        if self.devices or self.host is not None:
            raise RuntimeError("WeightArenas.allocate: already allocated")
        self.devices = {dev: ComponentArena(total, dev) for dev, total in self._device_bytes.items()}
        self.host = HostPinnedArena(self._host_bytes) if self._host_bytes else None
        return self

    def pin_host(self) -> None:
        """Page-lock the host block, once every host weight is written (a no-op without one)."""
        if self.host is not None:
            self.host.pin()


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
