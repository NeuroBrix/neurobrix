"""Host-to-device uploads that never make the host wait for the device's queue.

A copy from pageable host memory is synchronous: the runtime returns only once the copy is done, and the copy is
queued behind every kernel already launched, so the host stops feeding the card until its queue drains. Measured
2026-11-13 on a V100 (Qwen2.5-Coder-7B fp16, nsys): ~17 such uploads a decode token — positions, lengths, ids, small
tables of 4 B to 1 KB — each holding the host 165 us. The answer every serious runtime gives is the same (the CUDA
programming guide's "page-locked host memory" section; PyTorch's caching host allocator behind `non_blocking=True`,
which keeps a block until the event recorded behind its copy has completed): copy the bytes into page-locked host
memory, queue an asynchronous copy from there, and reuse the page-locked bytes only once that copy has run.

`upload(dst, src, nbytes)` does that on the stream every kernel is launched on (the legacy stream 0), so stream
order still puts the copy before every later kernel and after every earlier one, exactly as the synchronous copy did.

CONTRACT
  * when the call returns, the source bytes have been read: the caller may change or free them at once;
  * the copy is ordered on stream 0 of the current device like the synchronous copy it replaces;
  * the page-locked ring of a device is rewritten only where every copy that read it has completed (its event);
  * the ring GROWS instead of waiting: an upload that would overwrite a copy still in flight gets a ring twice as
    large (never smaller than the upload), so the ring sizes itself to what is really in flight — no constant;
    the old ring is freed once its last copy has run and the stream is idle (freeing page-locked memory synchronises
    the device). Its bytes are counted by `host_pinned_allocated`;
  * the synchronous copy is kept, read from the facts and never as a silent default:
      - the backend offers no page-locked host memory, asynchronous copy or event/stream queries (Metal's unified
        memory and Level Zero today: their uploads are unchanged);
      - stream 0 is idle (nothing to wait behind: a load-time copy costs the same and never grows the ring);
      - page-locked memory is refused when the ring must grow: said once on the log; the upload then waits for the
        oldest copy in flight (slower, never wrong).
"""
from __future__ import annotations

import collections
import ctypes
import logging
import mmap
import threading
from typing import Dict, Optional

from nbxtensor.backend import runtime as _rt

_log = logging.getLogger(__name__)

#: the runtime entries the staged path needs; a backend whose table lacks one keeps the synchronous copy
NEEDS = ("memcpy_async", "event_create_flags", "event_record", "event_query", "event_sync", "event_destroy",
         "stream_query")
NOT_READY = {"cuda": 600, "hip": 600}   # cudaErrorNotReady (driver_types.h), hipErrorNotReady (hip_runtime_api.h)


def _call(name: str, *args) -> int:
    return getattr(_rt.gpu_runtime(), _rt.backend_table()[name])(*args)


class _Ring:
    """One device's page-locked ring: a bump pointer over `cap` bytes and the copies still reading it, oldest first."""

    def __init__(self, dev: int, ptr: int, cap: int) -> None:
        self.dev, self.ptr, self.cap, self.head = dev, ptr, cap, 0
        self.inflight: "collections.deque" = collections.deque()    # (start, end, event)

    def overlaps(self, a: int, b: int) -> bool:
        return any(s < b and a < e for s, e, _ev in self.inflight)


class _Staging:
    """The per-device rings. One per process (`STAGING`); thread-safe."""

    def __init__(self) -> None:
        self._rings: Dict[int, _Ring] = {}
        self._retired: list = []          # (ring) older rings whose copies may still run
        self._events: Dict[int, list] = {}  # device -> its completed events, reused (an event records only on its own
                                            # device's streams)
        self._lock = threading.Lock()
        self._capable: Optional[bool] = None
        self._said_refused = False
        self.staged = 0                   # counters, read by the tests and the profiles
        self.direct = 0
        self.grown = 0

    # ---- facts --------------------------------------------------------------------------------------------------
    def capable(self) -> bool:
        if self._capable is None:
            table = _rt.backend_table()
            self._capable = (all(k in table for k in NEEDS) and _rt.gpu_backend() in NOT_READY
                             and _rt.device_memory().supports("host_alloc"))
        return self._capable

    def _done(self, rc: int, what: str) -> bool:
        if rc == 0:
            return True
        if rc == NOT_READY[_rt.gpu_backend()]:
            return False
        raise RuntimeError(f"{what} failed rc={rc} (CUDA errors are STICKY: the first checked call after the fault)")

    def stream_idle(self) -> bool:
        return self._done(_call("stream_query", ctypes.c_void_p(0)), "stream query")

    def _event_done(self, ev: int) -> bool:
        return self._done(_call("event_query", ctypes.c_void_p(ev)), "event query")

    # ---- the ring ------------------------------------------------------------------------------------------------
    def _reap(self, ring: _Ring) -> None:
        while ring.inflight and self._event_done(ring.inflight[0][2]):
            self._events.setdefault(ring.dev, []).append(ring.inflight.popleft()[2])

    def _free_retired(self, dev: int) -> None:
        """Free `dev`'s outgrown rings whose copies have all run — called only when `dev`'s stream is idle: freeing
        page-locked memory synchronises the device (cudaFreeHost), which on the hot path is the very wait this module
        removes (measured 2026-11-14 on a V100 with ~25 ms of work queued: cudaFreeHost of 64 KB held the host
        27 ms and returned with the queue empty; cudaMallocHost 0.08 ms; a staged upload 0.03 ms)."""
        from nbxtensor.memory.allocator import DeviceAllocator
        keep = []
        for old in self._retired:
            if old.dev == dev:
                while old.inflight and self._event_done(old.inflight[0][2]):
                    self._events.setdefault(old.dev, []).append(old.inflight.popleft()[2])
            if old.dev != dev or old.inflight:
                keep.append(old)
            else:
                DeviceAllocator.free_host_pinned(old.ptr)
        self._retired = keep

    def _grow(self, dev: int, need: int) -> Optional[_Ring]:
        from nbxtensor.memory.allocator import DeviceAllocator
        old = self._rings.get(dev)
        page = mmap.PAGESIZE
        cap = max(2 * old.cap if old else 0, -(-need // page) * page)
        try:
            ptr = DeviceAllocator.malloc_host_pinned(cap)
        except RuntimeError as exc:
            if not self._said_refused:
                self._said_refused = True
                _log.warning(f"[staging] page-locked memory refused for a {cap}-byte upload ring on device {dev} "
                             f"({exc}): uploads wait for their oldest copy in flight instead (slower, never wrong)")
            return None
        if old is not None:
            self._retired.append(old)         # freed when the stream is next idle (`_free_retired`), never here
        ring = self._rings[dev] = _Ring(dev, ptr, cap)
        self.grown += 1
        return ring

    def _place(self, dev: int, n: int) -> Optional[_Ring]:
        """A ring of `dev` with [head, head + n) free of any copy in flight, head left at that start; None when no
        ring can hold `n` bytes (page-locked memory refused): the caller copies synchronously."""
        ring = self._rings.get(dev)
        if ring is None or n > ring.cap:
            grown = self._grow(dev, n)
            if grown is not None or ring is None or n > ring.cap:
                return grown
        self._reap(ring)
        start = ring.head if ring.head + n <= ring.cap else 0
        if not ring.overlaps(start, start + n):
            ring.head = start
            return ring
        grown = self._grow(dev, n)
        if grown is not None:
            return grown
        while ring.overlaps(start, start + n):           # page-locked memory refused: wait, oldest first
            ev = ring.inflight.popleft()[2]
            self._done(_call("event_sync", ctypes.c_void_p(ev)), "event synchronize")
            self._events.setdefault(dev, []).append(ev)
        ring.head = start
        return ring

    def _event(self, dev: int) -> int:
        free = self._events.get(dev)
        if free:
            return free.pop()
        from nbxtensor.memory.allocator import DeviceAllocator
        return DeviceAllocator.create_event(timing=False)

    # ---- the upload ----------------------------------------------------------------------------------------------
    def upload(self, dst: int, src: int, nbytes: int) -> None:
        """Copy `nbytes` host bytes at `src` to the device pointer `dst` (the current device), stream-ordered."""
        from nbxtensor.memory.allocator import DeviceAllocator
        if nbytes <= 0:
            return
        if not self.capable():
            self.direct += 1
            return DeviceAllocator.memcpy_direct(dst, src, nbytes, 1)
        dev = DeviceAllocator.get_device()
        if self.stream_idle():
            if self._retired:
                with self._lock:
                    self._free_retired(dev)
            self.direct += 1
            return DeviceAllocator.memcpy_direct(dst, src, nbytes, 1)
        with self._lock:
            ring = self._place(dev, nbytes)
            if ring is None:
                self.direct += 1
                return DeviceAllocator.memcpy_direct(dst, src, nbytes, 1)
            at = ring.ptr + ring.head
            ctypes.memmove(at, src, nbytes)
            rc = _call("memcpy_async", ctypes.c_void_p(dst), ctypes.c_void_p(at), ctypes.c_size_t(nbytes),
                       ctypes.c_int(1), ctypes.c_void_p(0))
            if rc != 0:
                raise RuntimeError(f"cudaMemcpyAsync failed rc={rc} (H2D, nbytes={nbytes}) — CUDA errors are STICKY: "
                                   f"the first checked call after the fault, not necessarily the fault itself.")
            ev = self._event(dev)
            rc = _call("event_record", ctypes.c_void_p(ev), ctypes.c_void_p(0))
            if rc != 0:
                raise RuntimeError(f"event record failed rc={rc} after a staged upload")
            ring.inflight.append((ring.head, ring.head + nbytes, ev))
            ring.head += nbytes
            self.staged += 1


STAGING = _Staging()
