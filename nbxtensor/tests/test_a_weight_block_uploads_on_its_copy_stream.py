"""`nbxtensor.memory.block_upload`: a block of host weights goes to its device on a copy stream and records ONE
ordering-only readiness event the compute stream waits on; given no stream, the upload is synchronous and records
none; a tensor already on a device is left out; the stream and the events nobody waited on are released.

The card-less half drives the module through a stand-in allocator (the calls it records are the contract); the device
half uploads real bytes and reads them back, on whatever device this host has (skipped without one).

What these tests would do if the code were wrong: an upload on a stream that records no event (the compute stream would
read half-copied weights), an event recorded for a synchronous upload, a device tensor copied again, a wait on the copy
stream instead of the compute stream (0), a stream left open, or bytes that differ after the round trip — RED."""
import numpy as np
import pytest

from nbxtensor.memory import block_upload as BU


class _Allocator:
    def __init__(self):
        self.calls = []
        self.n = 0

    def _handle(self, kind):
        self.n += 1
        self.calls.append((kind, self.n))
        return 100 + self.n

    def set_device(self, d):
        self.calls.append(("set_device", d))

    def create_stream(self):
        return self._handle("create_stream")

    def create_event(self, timing=True):
        self.calls.append(("timing", timing))
        return self._handle("create_event")

    def record_event(self, ev, stream=0):
        self.calls.append(("record_event", ev, stream))

    def stream_wait_event(self, stream, ev, flags=0):
        self.calls.append(("stream_wait_event", stream, ev))

    def destroy_event(self, ev):
        self.calls.append(("destroy_event", ev))

    def destroy_stream(self, s):
        self.calls.append(("destroy_stream", s))

    def sync_device(self):
        self.calls.append(("sync_device",))


class _Host:
    def __init__(self, name, on_host=True):
        self.name, self.is_cpu = name, on_host

    def to_cuda(self, d):
        return ("sync", self.name, d)

    def to_cuda_async(self, d, stream=0):
        return ("async", self.name, d, stream)


@pytest.fixture
def da(monkeypatch):
    fake = _Allocator()
    monkeypatch.setattr(BU, "_DA", fake)
    return fake


def test_an_upload_on_a_stream_records_one_ordering_event_the_compute_stream_waits_on(da):
    stream = BU.open_copy_stream(2)
    assert da.calls[:2] == [("set_device", 2), ("create_stream", 1)] and stream == 101
    moved, ev = BU.upload_block({"w": _Host("w"), "b": _Host("b"), "dev": _Host("dev", on_host=False)}, 2, stream)
    assert moved == {"w": ("async", "w", 2, stream), "b": ("async", "b", 2, stream)}
    assert ("timing", False) in da.calls and ("record_event", ev, stream) in da.calls
    BU.wait_for_upload(ev)
    assert da.calls[-2:] == [("stream_wait_event", 0, ev), ("destroy_event", ev)]


def test_an_upload_without_a_stream_is_synchronous_and_records_nothing(da):
    moved, ev = BU.upload_block({"w": _Host("w")}, 0, 0)
    assert moved == {"w": ("sync", "w", 0)} and ev is None
    assert [c for c in da.calls if c[0] in ("create_event", "record_event")] == []


def test_the_stream_and_its_unwaited_events_are_released(da):
    BU.close_copy_stream(7, [11, 12])
    assert da.calls == [("destroy_event", 11), ("destroy_event", 12), ("destroy_stream", 7)]
    da.calls.clear()
    BU.close_copy_stream(0, [])
    assert da.calls == []
    BU.drop_upload_events([5])
    BU.wait_device()
    assert da.calls == [("destroy_event", 5), ("sync_device",)]


def test_a_block_uploaded_on_this_hosts_device_reads_back_its_bytes():
    from nbxtensor.memory.allocator import DeviceAllocator
    if DeviceAllocator.device_count() < 1:
        pytest.skip("no device on this host")
    from nbxtensor.tensor import NBXTensor
    a = np.arange(4096, dtype=np.float32).reshape(64, 64)
    host = {"w": NBXTensor.from_numpy(a).to_cpu(pinned=True), "b": NBXTensor.from_numpy(a[0] * -1).to_cpu()}
    stream = BU.open_copy_stream(0)
    try:
        moved, ev = BU.upload_block(host, 0, stream)
        assert ev is not None and set(moved) == {"w", "b"} and not moved["w"].is_cpu
        BU.wait_for_upload(ev)
        BU.wait_device()
        assert np.array_equal(moved["w"].numpy(), a) and np.array_equal(moved["b"].numpy(), a[0] * -1)
    finally:
        BU.close_copy_stream(stream)
