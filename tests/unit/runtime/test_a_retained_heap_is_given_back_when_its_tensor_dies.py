"""Where the device's allocator keeps the heap of a large freed tensor (`memory.compiled_allocator_retention`: MPS's
OVERSIZE heaps), the compiled engine gives the heap back when it frees such a tensor — the device synchronised, its
cache emptied — so what the device holds is the live set the plan priced (Open-Sora-v2's VAE on MPS held 21.2 GB for
3.8 GB live, the Mac, 2026-10-23). A device whose allocator reuses freed memory (CUDA: null) is never synchronised.

What this test would do if the code were wrong: no release on a large free — red at the call count; a release on a
small one, or on CUDA — red at the zero counts; a vendor file stating nothing — red if it is not refused by name; the
freed bytes never waited for, or counted from the start — red at the cliff tests."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from neurobrix.core import device_utils as DU  # noqa: E402
from neurobrix.core.runtime.graph.compiled_sequence import CompiledSequence  # noqa: E402
from neurobrix.core.prism.attention_scratch import vendor_memory_of  # noqa: E402


def _seq(release_from, drain_cliff=None):
    from types import SimpleNamespace
    s = CompiledSequence.__new__(CompiledSequence)
    s._release_from = release_from
    s._drain_cliff = drain_cliff
    s._in_flight = 0
    s.device = torch.device("cpu")
    s.op_resolver = SimpleNamespace(dtype_engine=SimpleNamespace(copy_meter=[0] if drain_cliff is not None else None))
    return s


def _calls(monkeypatch, cached=0):
    seen = []
    monkeypatch.setattr(DU, "device_sync", lambda d=None: seen.append(("sync", d)))
    monkeypatch.setattr(DU, "device_empty_cache", lambda d=None: seen.append(("empty", d)))
    monkeypatch.setattr(DU, "device_cached_bytes", lambda d=None: cached)
    return seen


def test_a_large_free_gives_its_heap_back(monkeypatch):
    seen = _calls(monkeypatch)
    arena = [torch.empty(1 << 20, dtype=torch.uint8), torch.empty(16, dtype=torch.uint8)]
    _seq(1 << 20)._kill(arena, (0, 1))
    assert arena == [None, None] and [k for k, _ in seen] == ["sync", "empty"]


def test_a_small_free_or_a_reusing_allocator_releases_nothing(monkeypatch):
    seen = _calls(monkeypatch)
    _seq(1 << 20)._kill([torch.empty(16, dtype=torch.uint8)], (0,))
    _seq(None)._kill([torch.empty(1 << 21, dtype=torch.uint8)], (0,))
    assert seen == []


def test_the_vendor_files_state_the_retention():
    assert vendor_memory_of("apple", "apple_m4_pro")["compiled_allocator_retention"]["dedicated_heap_min_bytes"] == 1 << 29
    assert vendor_memory_of("nvidia", "volta")["compiled_allocator_retention"] is None


# A temporal stream's own boundary (a state output, `stream.state_out::<k>`) gives the retained heaps back whatever
# the size of what died there: mochi-1-preview's VAE on MPS freed mid-size slices whose heaps stayed part-occupied,
# 19.8 GB of driver memory for 7.0 GB live at its 32nd boundary (the Mac, 2026-10-23 20:17). Never on CUDA (null).
# Injection: `_kill` without the boundary rule -> the first case sees no call, red.
def test_a_stream_boundary_gives_the_heaps_back(monkeypatch):
    seen = _calls(monkeypatch)
    _seq(1 << 20)._kill([torch.empty(16, dtype=torch.uint8)], (0,), "stream.state_out::3")
    assert [k for k, _ in seen] == ["sync", "empty"]


def test_a_stream_boundary_on_a_reusing_allocator_or_another_op_releases_nothing(monkeypatch):
    seen = _calls(monkeypatch)
    _seq(None)._kill([torch.empty(16, dtype=torch.uint8)], (0,), "stream.state_out::3")
    _seq(1 << 20)._kill([torch.empty(16, dtype=torch.uint8)], (0,), "aten.convolution::31")
    assert seen == []


# Such an allocator also frees a buffer only once the device has run the work that read it: a sequence that never waits
# holds every tensor it freed since. mochi-1-preview's VAE on MPS held ~5.7 GB of them at its 33rd convolution and ran
# out of memory on a plan that fitted; a wait after every op ran it at 11 136 MB (the Mac, 2026-11-04 20:05). The
# sequence waits once the bytes it freed since its last wait reach the drain cliff — no cache emptied, the count reset.
# Injection: `_kill` without the cliff -> no sync, red; the count never reset -> a sync at every later free, red.
def test_the_freed_bytes_wait_for_the_device_at_the_cliff(monkeypatch):
    seen = _calls(monkeypatch)
    s = _seq(1 << 20, drain_cliff=1000)
    for _ in range(3):
        s._kill([torch.empty(300, dtype=torch.uint8)], (0,))
    assert seen == []                                          # 900 bytes in flight: under the cliff
    s._kill([torch.empty(300, dtype=torch.uint8)], (0,))
    assert [k for k, _ in seen] == ["sync"] and s._in_flight == 0
    s._kill([torch.empty(300, dtype=torch.uint8)], (0,))
    assert [k for k, _ in seen] == ["sync"]                    # counted from the wait, not from the start


def test_a_heap_release_resets_the_count(monkeypatch):
    seen = _calls(monkeypatch)
    s = _seq(1 << 20, drain_cliff=1 << 22)
    s._kill([torch.empty(300, dtype=torch.uint8)], (0,))
    s._kill([torch.empty(1 << 20, dtype=torch.uint8)], (0,))  # a dedicated heap: sync + empty, the queue drained
    assert [k for k, _ in seen] == ["sync", "empty"] and s._in_flight == 0


# The DtypeEngine's own copies (an operand cast, a wider result stored narrower) die inside their op, out of the arena:
# they are in flight too. The Mac's RUN 33 (2026-11-04 20:27): the drain counted the arena alone, never fired on mochi's
# VAE (816 MB and 1 216 MB of casts a block), and the run died where a wait after every op had held it.
# Injection: the meter not read at the kill -> no sync, red.
def test_the_engines_copies_count_toward_the_cliff(monkeypatch):
    seen = _calls(monkeypatch)
    s = _seq(1 << 20, drain_cliff=1000)
    s.op_resolver.dtype_engine.copy_meter[0] = 900          # a cast the op made and dropped
    s._kill([torch.empty(200, dtype=torch.uint8)], (0,))
    assert [k for k, _ in seen] == ["sync"] and s.op_resolver.dtype_engine.copy_meter == [0]


# A wait returns the frees to the allocator's CACHE, not to the device: MPS kept 4.70 GiB of heaps cached beside 12.56 GiB
# live (under the price) after 1 210 waits on mochi's VAE, and the run died (the Mac's RUN 33b, 2026-11-04 21:23). At the
# cliff, a cache above the priced queue is emptied too; one under it is kept (emptying costs every later allocation).
# Injection: no empty at the cliff -> the first test red; an empty whatever the cache -> the second red.
def test_a_cache_above_the_queue_is_emptied_at_the_cliff(monkeypatch):
    seen = _calls(monkeypatch, cached=5000)
    s = _seq(1 << 20, drain_cliff=1000)
    s._kill([torch.empty(1200, dtype=torch.uint8)], (0,))
    assert [k for k, _ in seen] == ["sync", "empty"]


def test_a_cache_under_the_queue_is_kept(monkeypatch):
    seen = _calls(monkeypatch, cached=500)
    s = _seq(1 << 20, drain_cliff=1000)
    s._kill([torch.empty(1200, dtype=torch.uint8)], (0,))
    assert [k for k, _ in seen] == ["sync"]
