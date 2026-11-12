"""Tensors a run has stopped using, kept until the device work that may still read them is done.

A kernel runs on its device after the call that launched it returns: a tensor dropped the moment its last reader is
launched would be freed under that reader. A run keeps what it drops in a `DeferredFrees` and releases it in batches —
when the batch reaches a byte cliff or a tensor count; earlier when a throttled reading of the device's free memory
shows the device within a reserve of its end (with the pool off only: under the pool a freed block stays parked and no
memory comes back to the device); at the end of the run — every device owning one of them synchronised first. While it
lives, a batch is also the allocator's out-of-memory reclaimer (`DeviceAllocator.register_oom_reclaim_hook`).

The thresholds are the caller's policy, handed in; the queue and the order of its frees are the library's.
"""
from __future__ import annotations

import weakref
from typing import Iterable, NamedTuple, Optional

from nbxtensor.memory.allocator import DeviceAllocator


def sync_devices_of(tensors: Iterable) -> None:
    """Synchronise every device that owns one of `tensors` (the current device when none names one), the current device
    restored. Freeing a pointer of device N while N's queue still reads it is the multi-card illegal-access class: the
    current device's synchronisation alone does not cover a batch holding pointers of several cards."""
    devs = sorted({int(d) for d in (getattr(t, "device_idx", None) for t in tensors) if d is not None})
    if not devs:
        DeviceAllocator.sync_device()
        return
    prev = DeviceAllocator.get_device()
    try:
        for d in devs:
            DeviceAllocator.set_device(d)
            DeviceAllocator.sync_device()
    finally:
        DeviceAllocator.set_device(prev)


class Drain(NamedTuple):
    """One batch released by `DeferredFrees.step`: why ("cliff": the bytes or the count; "pressure": the device near its
    end), how many tensors and how many bytes."""
    reason: str
    tensors: int
    nbytes: int


class DeferredFrees:
    """The tensors a run dropped and has not yet freed (`add`), released by batch (`step`, `release`).

    `cliff` bytes or `count` tensors release the batch. With `reserve` > 0 and the allocator's pool off, a batch of at
    least `floor` bytes is also released when the device's free bytes, read at most once every `interval` steps, fall
    under `reserve`. `one_device`: every tensor of the run is on one device, whose queue runs in order — under the pool
    a block handed back is reissued only to work queued after its readers, so a mid-run release needs no
    synchronisation; across devices it always synchronises them. The final `release()` always does. Measured on an
    int4 prefill (2026-09-02, 22.7 s): 397 mid-run synchronisations held the host 15.5 s, and 401 free-memory readings
    cost 1.8 s under the pool, where they have no object."""

    def __init__(self, cliff: int, count: int, floor: Optional[int] = None, reserve: int = 0, interval: int = 1,
                 one_device: bool = True):
        self._held: list = []
        self.nbytes = 0
        self._cliff, self._count = int(cliff), int(count)
        self._floor = int(cliff if floor is None else floor)
        self._reserve, self._interval = int(reserve), int(interval)
        pooled = DeviceAllocator.pool_enabled()
        self._pressure_gate = self._reserve > 0 and not pooled
        self._sync_mid_run = not (one_device and pooled)
        self._last_check = -(1 << 30)
        ref = weakref.ref(self)

        def reclaim(dev_idx, nbytes_requested):          # the allocator's hook: what a held batch frees
            frees = ref()
            return 0 if frees is None else frees.release()

        DeviceAllocator.register_oom_reclaim_hook(reclaim)
        weakref.finalize(self, DeviceAllocator.unregister_oom_reclaim_hook, reclaim)

    def __len__(self) -> int:
        return len(self._held)

    def add(self, t) -> None:
        """Keep `t` (None: nothing) until the batch is released."""
        if t is not None:
            self._held.append(t)
            self.nbytes += t.nbytes()

    def step(self, at: int) -> Optional[Drain]:
        """After step `at` of the run: release the batch when a threshold is reached; what was released, or None."""
        tensors, nbytes = len(self._held), self.nbytes
        if nbytes >= self._cliff or tensors >= self._count:
            reason = "cliff"
        elif self._pressure_gate and nbytes >= self._floor and at - self._last_check >= self._interval:
            self._last_check = at
            if not 0 <= DeviceAllocator.device_free_bytes() < self._reserve:
                return None
            reason = "pressure"
        else:
            return None
        self.release(sync=self._sync_mid_run)
        return Drain(reason, tensors, nbytes)

    def release(self, sync: bool = True) -> int:
        """Free the batch now, after synchronising every device owning one of its tensors (`sync`); the bytes freed."""
        if not self._held:
            return 0
        if sync:
            sync_devices_of(self._held)
        freed, self.nbytes = self.nbytes, 0
        self._held.clear()
        return freed
