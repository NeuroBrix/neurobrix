"""Weight blocks between the host and one device: a copy stream, a set of host tensors uploaded on it with a readiness
event, the compute stream's wait for that event, and their teardown — moved from the NeuroBrix engine's zero3 strategy,
which keeps which block moves, when, and how much stays resident.

Handles are the allocator's plain ints (`DeviceAllocator.create_stream / create_event`); stream 0 is the default
(compute) stream, and an upload given stream 0 is synchronous and records no event.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

from nbxtensor.memory.allocator import DeviceAllocator as _DA

__all__ = ["open_copy_stream", "upload_block", "wait_for_upload", "drop_upload_events", "close_copy_stream",
           "wait_device"]


def open_copy_stream(device: int) -> int:
    """A new copy stream on `device` (made current), as a handle; raises when the driver cannot create one."""
    _DA.set_device(device)
    return _DA.create_stream()


def upload_block(host: Dict[str, object], device: int, stream: int = 0) -> Tuple[Dict[str, object], Optional[int]]:
    """Each HOST-resident tensor of `host` (name -> tensor; a tensor already on a device is left out) copied to `device`
    — asynchronously on `stream` when one is given, else synchronously — and the readiness event the compute stream
    must wait on before reading any of them (`wait_for_upload`), None when the upload was synchronous.
    Returns (name -> device copy, event)."""
    _DA.set_device(device)
    moved = {}
    for name, t in host.items():
        if not getattr(t, "is_cpu", False):
            continue
        moved[name] = t.to_cuda_async(device, stream=stream) if stream else t.to_cuda(device)
    if not stream:
        return moved, None
    # Ordering only: the event never reaches a timing query, and a timing-enabled event costs 9.6 us to record against
    # 1.6 us for an ordering-only one (measured 2026-09-03).
    event = _DA.create_event(timing=False)
    _DA.record_event(event, stream=stream)
    return moved, event


def wait_for_upload(event: int) -> None:
    """The compute (default) stream waits until the upload that recorded `event` is complete; the event is released."""
    _DA.stream_wait_event(0, event, flags=0)
    _DA.destroy_event(event)


def drop_upload_events(events: Iterable[int]) -> None:
    """Release readiness events nobody will wait on."""
    for event in events:
        _DA.destroy_event(event)


def close_copy_stream(stream: int, events: Iterable[int] = ()) -> None:
    """Release a copy stream (0: none was opened) and the readiness events still recorded on it."""
    drop_upload_events(events)
    if stream:
        _DA.destroy_stream(stream)


def wait_device() -> None:
    """Every operation queued on the current device completes — after a block's copies are dropped, so the next
    allocation may reuse their memory while no read of them is outstanding."""
    _DA.sync_device()
