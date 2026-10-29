"""The allocator's interceptor (`DeviceAllocator.install_interceptor`) and the slab interceptor built on it
(`nbxtensor.memory.slab.SlabInterceptor`): the five public device operations go through the interceptor in force with
the direct operation passed in, one interceptor at a time; the slab interceptor carves a recording's pooled allocations
from the slab of the requested device, refuses by name what no slab can serve, leaves a block outside the pool to the
allocator, returns a freed carved pointer to its slab, and measures the watermark between recordings. Checked against
direct operations replaced by fakes — no device.

What this test would do if the code were wrong: a public operation that skips the interceptor, a second interceptor
accepted beside the first, a recording allocation served by the wrong device's slab or by the allocator in silence, a
carved pointer handed to the allocator's free -> red."""
import pytest

from nbxtensor.memory import slab as S
from nbxtensor.memory.allocator import AllocatorInterceptor, DeviceAllocator as DA


@pytest.fixture
def fake_direct(monkeypatch):
    """The direct operations replaced by fakes that record their calls; nothing is installed after the test."""
    calls = []
    nxt = {"p": 1 << 20}

    def malloc(nbytes, dev_idx=None, pooled=True):
        calls.append(("malloc", int(nbytes), dev_idx, pooled))
        p = nxt["p"]
        nxt["p"] += 1 << 20
        return p

    monkeypatch.setattr(DA, "malloc_direct", staticmethod(malloc))
    monkeypatch.setattr(DA, "free_direct", staticmethod(lambda ptr: calls.append(("free", ptr))))
    monkeypatch.setattr(DA, "memcpy_direct", staticmethod(lambda d, s, n, k=3: calls.append(("memcpy", d, s, n, k))))
    monkeypatch.setattr(DA, "memset_direct", staticmethod(lambda p, v, n: calls.append(("memset", p, v, n))))
    monkeypatch.setattr(DA, "sync_direct", staticmethod(lambda: calls.append(("sync",))))
    monkeypatch.setattr(DA, "get_device", staticmethod(lambda: 0))
    monkeypatch.setattr(DA, "_interceptor", None)
    monkeypatch.setattr(S, "ACTIVE_SLABS", [])
    return calls


class Seeing(AllocatorInterceptor):
    def __init__(self):
        self.seen = []

    def malloc(self, direct, nbytes, dev_idx, pooled):
        self.seen.append("malloc")
        return direct(nbytes, dev_idx, pooled)

    def free(self, direct, ptr):
        self.seen.append("free")
        return direct(ptr)

    def memcpy(self, direct, dst, src, nbytes, kind):
        self.seen.append("memcpy")
        return direct(dst, src, nbytes, kind)

    def memset(self, direct, ptr, value, nbytes):
        self.seen.append("memset")
        return direct(ptr, value, nbytes)

    def sync(self, direct):
        self.seen.append("sync")
        return direct()


def _all_five():
    p = DA.malloc_cuda(100, 0)
    DA.memcpy(p, 7, 100, 3)
    DA.memset_cuda(p, 0, 100)
    DA.sync_device()
    DA.free_cuda(p)
    return p


def test_without_an_interceptor_every_operation_is_direct(fake_direct):
    p = _all_five()
    assert [c[0] for c in fake_direct] == ["malloc", "memcpy", "memset", "sync", "free"]
    assert fake_direct[-1] == ("free", p)


def test_every_operation_goes_through_the_interceptor_in_force(fake_direct):
    i = Seeing()
    DA.install_interceptor(i)
    DA.install_interceptor(i)                       # the one in force again: a no-op
    _all_five()
    assert i.seen == ["malloc", "memcpy", "memset", "sync", "free"]
    assert [c[0] for c in fake_direct] == ["malloc", "memcpy", "memset", "sync", "free"]
    DA.remove_interceptor(i)
    _all_five()
    assert len(i.seen) == 5, "a removed interceptor still sees the operations"


def test_a_second_interceptor_is_refused_by_name(fake_direct):
    DA.install_interceptor(Seeing())
    with pytest.raises(RuntimeError, match="already installed.*Seeing"):
        DA.install_interceptor(Seeing())
    with pytest.raises(RuntimeError, match="not the allocator interceptor in force"):
        DA.remove_interceptor(Seeing())


class Recorder(S.SlabInterceptor):
    def __init__(self):
        super().__init__()
        self.refused = []

    def slab_refused(self, reason):
        self.refused.append(reason)


def test_a_recording_carves_from_the_slab_of_the_requested_device(fake_direct):
    r = Recorder()
    DA.install_interceptor(r)
    s0, s1 = S.SlabAllocator(1 << 16, 0), S.SlabAllocator(1 << 16, 1)
    r.slabs = {0: s0, 1: s1}
    r.recording = True
    n_direct = len(fake_direct)
    p1 = DA.malloc_cuda(1000, 1)
    p0 = DA.malloc_cuda(1000, 0)
    assert s1.base <= p1 < s1.base + s1.size and s0.base <= p0 < s0.base + s0.size
    assert len(fake_direct) == n_direct, "a carved allocation reached the allocator"
    arena = DA.malloc_cuda(1000, 0, pooled=False)   # a block outside the pool is never a slab's
    assert fake_direct[-1] == ("malloc", 1000, 0, False) and not (s0.base <= arena < s0.base + s0.size)
    DA.free_cuda(p0)
    assert ("free", p0) not in fake_direct, "a carved pointer was freed to the allocator"
    assert s0.free_list == [(0, s0.size)]
    DA.malloc_cuda(1 << 17, 0)                      # more than the slab holds
    assert r.refused and "slab exhausted on device 0" in r.refused[0]
    assert fake_direct[-1] == ("malloc", 1 << 17, 0, True), "the refused allocation is the allocator's"
    r.slabs = {1: s1}
    r.slab = None
    DA.malloc_cuda(64, 0)
    assert r.refused[-1] == "no slab recorded for device 0"


def test_between_recordings_the_watermark_is_measured(fake_direct):
    r = Recorder()
    DA.install_interceptor(r)
    r.measuring = True
    a = DA.malloc_cuda(1000, 0)
    b = DA.malloc_cuda(1, 2)
    assert r.live == 1024 + 512 and r.high_water == 1536 and r.high_water_by_dev == {0: 1024, 2: 512}
    DA.free_cuda(a)
    assert r.live == 512 and r.live_by_dev[0] == 0 and r.high_water == 1536
    assert ("free", a) in fake_direct
    DA.free_cuda(b)
    assert r.total_measured == 1536


def test_a_retired_slab_returns_to_the_allocator_with_its_last_carving(fake_direct):
    r = Recorder()
    DA.install_interceptor(r)
    s = S.SlabAllocator(1 << 16, 0)
    r.slab, r.recording = s, True
    p = DA.malloc_cuda(10, 0)
    r.recording = False
    s.retire()
    assert ("free", s.base) not in fake_direct and s in S.ACTIVE_SLABS
    base = s.base
    DA.free_cuda(p)
    assert ("free", base) in fake_direct and s not in S.ACTIVE_SLABS
