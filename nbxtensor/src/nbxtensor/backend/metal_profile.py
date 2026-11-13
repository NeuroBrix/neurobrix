"""Per-kernel GPU time on Metal: a profiling session puts every launch in its OWN encoder and times it at the encoder's
boundaries (the public door: `nbxtensor.profile.kernel_times`).

What the hardware gives (asked of an M4 Pro, 2026-11-13): counter sampling at STAGE boundaries only, never at a
dispatch inside an encoder, and the one counter set `timestamp`. The stream (`metal_device`) normally carries many
launches in one serial encoder, so a session changes the batching: its figures RANK kernels; they are not the
production step's time. The ticks are converted with the device's own CPU/GPU timestamp pairs, sampled when the
session opens and closes.

A command buffer gets one sample buffer of 2 x `dispatches_per_buffer` samples (a launch's start and end at 2i,
2i + 1, i its index in the buffer); its samples are read when the buffer has retired (`metal_device._await`, or the
session's close). The device keeps few sample buffers alive at once (an M4 Pro: 32, measured 2026-11-13), so the
first session takes the ones the device grants, as the process's pool (every later session reuses it: the device
does not give them back when a session ends); when every one is held by a buffer not yet
retired, the session waits for its oldest timed buffer and reuses its samples — a wait of the profiling mode only.
"""
from __future__ import annotations

import struct
import time
from typing import Dict, List, Optional, Tuple

from nbxtensor.profile import KernelTimes

#: the open session, or None: the one flag the stream reads (`metal_device.encoder_for`, `_await`)
SESSION: Optional["Session"] = None

#: how many sample buffers a session asks the device for (it keeps what it is granted: an M4 Pro grants 32)
POOL_ASK = 64

#: a sample Metal could not take reads as this value (MTLCounterErrorValue)
_ERROR_VALUE = 0xFFFFFFFFFFFFFFFF


class Session:
    def __init__(self, device, per_buffer: int) -> None:
        import Metal
        if not device.supportsCounterSampling_(Metal.MTLCounterSamplingPointAtStageBoundary):
            raise RuntimeError(f"ZERO FALLBACK: {device.name()} samples no counter at an encoder's boundary: no "
                               f"per-kernel GPU time on this device")
        sets = [cs for cs in (device.counterSets() or []) if cs.name() == Metal.MTLCommonCounterSetTimestamp]
        if not sets:
            raise RuntimeError(f"ZERO FALLBACK: {device.name()} has no timestamp counter set")
        self.device, self.counters, self.per_buffer = device, sets[0], int(per_buffer)
        self.table = KernelTimes()
        self.free = list(_pool(self))
        self.inflight: Dict[int, Tuple[object, object, List[Optional[str]]]] = {}   # id(buffer) -> (buffer, samples, names)
        self.t0 = time.perf_counter()
        self.cpu0, self.gpu0 = device.sampleTimestamps_gpuTimestamp_(None, None)

    def granted_pool(self) -> list:
        """The sample buffers of 2 x dispatches_per_buffer samples the device grants now, up to POOL_ASK; refused by
        name when it grants none."""
        import Metal
        pool, err = [], None
        while len(pool) < POOL_ASK:
            desc = Metal.MTLCounterSampleBufferDescriptor.alloc().init()
            desc.setCounterSet_(self.counters)
            desc.setStorageMode_(Metal.MTLStorageModeShared)
            desc.setSampleCount_(2 * self.per_buffer)
            samples, err = self.device.newCounterSampleBufferWithDescriptor_error_(desc, None)
            if samples is None:
                break
            pool.append(samples)
        if not pool:
            raise RuntimeError(f"ZERO FALLBACK: a timestamp sample buffer of {2 * self.per_buffer} samples "
                               f"(2 x dispatch.max_dispatches_per_command_buffer) was refused by the device: {err}")
        return pool

    def _samples_for(self, command_buffer):
        held = self.inflight.get(id(command_buffer))
        if held is None:
            if not self.free:                               # every sample buffer held: the oldest timed buffer first
                oldest = next(iter(self.inflight.values()))[0]
                oldest.waitUntilCompleted()
                self.retired(oldest)
            held = self.inflight[id(command_buffer)] = (command_buffer, self.free.pop(), [])
        return held

    def open_encoder(self, command_buffer, kind: str, index: int):
        """The launch's own encoder, its start and end sampled at 2 x index and 2 x index + 1; its row is its kind until
        the launch names it (`name`)."""
        import Metal
        _cb, samples, names = self._samples_for(command_buffer)
        names.extend([None] * (index + 1 - len(names)))
        names[index] = kind
        desc = (Metal.MTLBlitPassDescriptor.blitPassDescriptor() if kind == "blit"
                else Metal.MTLComputePassDescriptor.computePassDescriptor())
        att = desc.sampleBufferAttachments().objectAtIndexedSubscript_(0)
        att.setSampleBuffer_(samples)
        att.setStartOfEncoderSampleIndex_(2 * index)
        att.setEndOfEncoderSampleIndex_(2 * index + 1)
        if kind == "blit":
            return command_buffer.blitCommandEncoderWithDescriptor_(desc)
        return command_buffer.computeCommandEncoderWithDescriptor_(desc)

    def name(self, command_buffer, label: str) -> None:
        """The launch just opened on `command_buffer` is `label` (its kernel's name)."""
        held = self.inflight.get(id(command_buffer))
        if held is not None and held[2]:
            held[2][-1] = label

    def retired(self, command_buffer) -> None:
        """A buffer that has completed: its launches' samples read into the table."""
        held = self.inflight.pop(id(command_buffer), None)
        if held is None:
            return
        _cb, samples, names = held
        self.free.append(samples)
        n = len(names)
        if n == 0:
            return
        raw = bytes(samples.resolveCounterRange_((0, 2 * n)))
        ticks = struct.unpack(f"<{2 * n}Q", raw[:16 * n])
        for i, name in enumerate(names):
            a, b = ticks[2 * i], ticks[2 * i + 1]
            if name is None:
                continue
            if a == _ERROR_VALUE or b == _ERROR_VALUE or b < a:
                self.table.unsampled += 1
                continue
            self.table.add(name, float(b - a))          # ticks, scaled to ms at close

    def close(self) -> KernelTimes:
        for command_buffer, _s, _n in list(self.inflight.values()):
            command_buffer.waitUntilCompleted()
            self.retired(command_buffer)
        cpu1, gpu1 = self.device.sampleTimestamps_gpuTimestamp_(None, None)
        ns_per_tick = (cpu1 - self.cpu0) / (gpu1 - self.gpu0) if gpu1 != self.gpu0 else 1.0
        for row in self.table.rows.values():
            row[1] *= ns_per_tick / 1e6
            row[2] *= ns_per_tick / 1e6
        self.table.wall_s = time.perf_counter() - self.t0
        return self.table


_POOL: Dict[int, list] = {}                      # samples per buffer -> the process's granted sample buffers


def _pool(session: "Session") -> list:
    if session.per_buffer not in _POOL:
        _POOL[session.per_buffer] = session.granted_pool()
    return _POOL[session.per_buffer]


def open_session(device, per_buffer: int, flush) -> KernelTimes:
    """Open the process's one session on `device`: `flush()` commits and waits for the work encoded before it, so
    every launch inside is timed alone. Refused by name when one is open."""
    global SESSION
    if SESSION is not None:
        raise RuntimeError("ZERO FALLBACK: a kernel-time session is already open in this process")
    flush()
    SESSION = Session(device, per_buffer)
    return SESSION.table


def close_session(flush) -> KernelTimes:
    """Every launch of the session retired and read; the flag cleared whatever happens."""
    global SESSION
    session = SESSION
    if session is None:
        raise RuntimeError("ZERO FALLBACK: no kernel-time session is open")
    try:
        flush()
        return session.close()
    finally:
        SESSION = None
