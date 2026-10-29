"""Slab allocators: a run's device allocations carved out of one block per device, and the interceptor that serves
them (moved from the engine's frozen-plan replayer, M2).

A replay recording (the engine's `triton/replay.py`) runs a step once with every allocation it makes served from a slab,
so that the addresses its launches recorded stay valid when they are replayed: the replay issues the same launches in
the same order, so every reuse the recorded run made is safe at replay. `SlabInterceptor` stands in front of the
allocator (`DeviceAllocator.install_interceptor`) and routes: while recording, to the slab of the requested device;
between recordings, to the recycled per-step slab when one is in force; else to the allocator itself, measuring the
watermark a slab will be sized from. Torch-free.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from nbxtensor.memory.allocator import AllocatorInterceptor, DeviceAllocator

ALIGN = 512       # matches typical device allocation granularity

# Slabs with outstanding carved allocations — the free path consults this PERMANENTLY (not only while recording): a
# slab-carved tensor may die on any later run when its arena slot is overwritten, and its finalizer must return the
# range to the slab, never free a slab-interior pointer to the driver.
ACTIVE_SLABS: List["SlabAllocator"] = []


class SlabAllocator:
    """First-fit free-list allocator over ONE device slab.

    The recording run's malloc/free sequence drives carving/reuse; the
    replay issues the same launches in the same order, so every reuse
    the recorded run performed is serialization-safe at replay.
    """

    def __init__(self, nbytes: int, dev_idx: Optional[int]) -> None:
        self.size = int(nbytes)
        self.dev_idx = dev_idx
        self.base = DeviceAllocator.malloc_direct(self.size, dev_idx)
        self.free_list: List[Tuple[int, int]] = [(0, self.size)]
        self.allocs: Dict[int, Tuple[int, int]] = {}  # ptr -> (off, sz)
        self.shortfall = 0  # aligned bytes we could not serve
        # A retired slab (failed recording attempt / invalidated plan)
        # must NOT cudaFree while tensors carved from it are alive:
        # their finalizers free slab-interior pointers, which the seam
        # routes back here. Actual release happens when the last
        # outstanding alloc returns. Probe 2 proved the hazard: an
        # eager release left the text-encoder outputs dangling
        # ("Pointer argument cannot be accessed" + invalid cudaFree of
        # a slab-interior pointer).
        self.retired = False
        ACTIVE_SLABS.append(self)

    def malloc(self, nbytes: int) -> int:
        need = (int(nbytes) + ALIGN - 1) // ALIGN * ALIGN
        for i, (off, sz) in enumerate(self.free_list):
            if sz >= need:
                if sz == need:
                    self.free_list.pop(i)
                else:
                    self.free_list[i] = (off + need, sz - need)
                ptr = self.base + off
                self.allocs[ptr] = (off, need)
                return ptr
        self.shortfall += need
        return 0

    def free(self, ptr: int) -> bool:
        entry = self.allocs.pop(ptr, None)
        if entry is None:
            return False
        off, sz = entry
        self.free_list.append((off, sz))
        self.free_list.sort()
        merged: List[Tuple[int, int]] = []
        for o, s in self.free_list:
            if merged and merged[-1][0] + merged[-1][1] == o:
                merged[-1] = (merged[-1][0], merged[-1][1] + s)
            else:
                merged.append((o, s))
        self.free_list = merged
        if self.retired and not self.allocs:
            self._release_now()
        return True

    def retire(self) -> None:
        """Deferred release: cudaFree only after the last outstanding
        carved allocation has been returned by its owner's finalizer."""
        self.retired = True
        if not self.allocs:
            self._release_now()

    def _release_now(self) -> None:
        if self.base:
            DeviceAllocator.free_direct(self.base)
            self.base = 0
        if self in ACTIVE_SLABS:
            ACTIVE_SLABS.remove(self)


class StepSlab(SlabAllocator):
    """Per-sequence slab RECYCLED at every decode step (P-REPLAY-KV-
    DECODE B1). reset() returns the whole range to the free list so
    the next step's allocations carve IDENTICAL addresses (the decode
    graph's allocation order is structurally constant — census-
    proven); allocations from previous steps become 'stale' and their
    later finalizer frees are absorbed silently (the range is already
    recycled — safe because each step's consumers run before the next
    step begins in the synchronous decode timeline)."""

    def __init__(self, nbytes: int, dev_idx: Optional[int]) -> None:
        super().__init__(nbytes, dev_idx)
        self.stale: set = set()
        self.carve_sizes: List[int] = []   # diagnosis: per-step size seq
        self.prev_carve: Optional[List[int]] = None

    def malloc(self, nbytes: int) -> int:
        """PURE BUMP carving — mid-step frees are absorbed without
        recycling (see free below), so placement is a pure function of
        the allocation-size sequence (measured identical step-to-step)
        and IMMUNE to free-timing nondeterminism (GC-cycle finalizers
        fire at variable points; first-fit reuse turned that into
        per-step placement churn — measured: identical size sequences,
        differing offsets)."""
        self.carve_sizes.append(int(nbytes))
        need = (int(nbytes) + ALIGN - 1) // ALIGN * ALIGN
        off, sz = self.free_list[0]
        if sz < need:
            self.shortfall += need
            return 0
        self.free_list[0] = (off + need, sz - need)
        ptr = self.base + off
        self.allocs[ptr] = (off, need)
        return ptr

    def reset(self) -> None:
        self.prev_carve = self.carve_sizes
        self.carve_sizes = []
        self.stale.update(self.allocs.keys())
        self.allocs.clear()
        self.free_list = [(0, self.size)]

    def free(self, ptr: int) -> bool:
        """Absorb without recycling: the range stays consumed until the
        next reset (bump purity beats intra-step reuse)."""
        if ptr in self.stale:
            self.stale.discard(ptr)
            return True
        if ptr in self.allocs:
            self.allocs.pop(ptr)
            return True
        return False


class SlabInterceptor(AllocatorInterceptor):
    """The allocator's interceptor for slab-served runs. Its state is read and set by the replayer that owns it:

    * `recording` with `slabs` (one per device) or `slab`: every pooled allocation is carved from the slab of the
      REQUESTED device (serving a device-1 tensor from device 0's slab yields a pointer into the wrong card's memory —
      a wrong result or an illegal access, never a clear error); a slab that cannot serve it, or a device without
      one, refuses the recording through `slab_refused(reason)` and the allocator serves it.
    * `stab_slab`: the recycled per-step slab in force; an allocation over it is served by the allocator (address
      stability degrades for that tensor only).
    * `measuring`: the allocator serves, and the aligned bytes are counted per pointer and device (`live`,
      `high_water`, `live_by_dev`, `high_water_by_dev`, `total_measured`) — what a slab is sized from.

    A block outside the pool (`pooled=False`, a weight arena) is never a slab's. A free returns to the slab that
    carved the pointer, if any, before the allocator sees it."""

    def __init__(self) -> None:
        self.measuring = False
        self.recording = False
        self.slab: Optional[SlabAllocator] = None
        self.slabs: Dict[int, SlabAllocator] = {}
        self.stab_slab: Optional["StepSlab"] = None
        # Per-device measurement: one high-water mark cannot size slabs for a run allocating on several cards without
        # sizing each to the SUM.
        self.live_by_dev: Dict[int, int] = {}
        self.high_water_by_dev: Dict[int, int] = {}
        self.measured_dev: Dict[int, int] = {}
        # measuring counters (aligned bytes)
        self.live = 0
        self.high_water = 0
        self.measured: Dict[int, int] = {}  # ptr -> aligned size
        self.total_measured = 0             # cumulative (never decremented)

    def slab_refused(self, reason: str) -> None:
        """A recording the slabs cannot serve: the owner decides (a replayer abandons the plan, by name)."""
        raise NotImplementedError(f"{type(self).__qualname__} does not say what a refused slab does: {reason}")

    def malloc(self, direct, nbytes, dev_idx, pooled):
        if not pooled:
            return direct(nbytes, dev_idx, False)
        if self.recording and (self.slab is not None or self.slabs):
            _dev = dev_idx if dev_idx is not None else DeviceAllocator.get_device()
            _slab = self.slabs.get(_dev, self.slab)
            if _slab is not None:
                ptr = _slab.malloc(nbytes)
                if ptr:
                    return ptr
                self.slab_refused(f"slab exhausted on device {_dev} (short {_slab.shortfall} B)")
            else:
                self.slab_refused(f"no slab recorded for device {_dev}")
            return direct(nbytes, dev_idx, True)
        if self.stab_slab is not None:
            ptr = self.stab_slab.malloc(nbytes)
            if ptr:
                return ptr
        ptr = direct(nbytes, dev_idx, True)
        if self.measuring and ptr:
            need = (int(nbytes) + ALIGN - 1) // ALIGN * ALIGN
            self.measured[ptr] = need
            self.total_measured += need
            self.live += need
            if self.live > self.high_water:
                self.high_water = self.live
            _d = dev_idx if dev_idx is not None else DeviceAllocator.get_device()
            self.measured_dev[ptr] = _d
            self.live_by_dev[_d] = self.live_by_dev.get(_d, 0) + need
            if self.live_by_dev[_d] > self.high_water_by_dev.get(_d, 0):
                self.high_water_by_dev[_d] = self.live_by_dev[_d]
        return ptr

    def free(self, direct, ptr):
        for slab in ACTIVE_SLABS:
            if slab.free(ptr):
                return  # returned to its slab (maybe releasing it)
        if self.measuring:
            need = self.measured.pop(ptr, 0)
            self.live -= need
            _d = self.measured_dev.pop(ptr, None)
            if _d is not None:
                self.live_by_dev[_d] = self.live_by_dev.get(_d, 0) - need
        return direct(ptr)
