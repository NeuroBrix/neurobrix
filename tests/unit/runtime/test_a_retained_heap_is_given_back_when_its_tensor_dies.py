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
    s = CompiledSequence.__new__(CompiledSequence)
    s._release_from = release_from
    s._drain_cliff = drain_cliff
    s._sync_base = 0
    s.device = torch.device("cpu")
    return s


def _calls(monkeypatch, cached=0, allocated=(0,)):
    """The device's waits and cache empties recorded; the allocator's readings pinned: `allocated` is read in turn
    (its last value repeated)."""
    seen = []
    reads = list(allocated)
    monkeypatch.setattr(DU, "device_sync", lambda d=None: seen.append(("sync", d)))
    monkeypatch.setattr(DU, "device_empty_cache", lambda d=None: seen.append(("empty", d)))
    monkeypatch.setattr(DU, "device_cached_bytes", lambda d=None: cached)
    monkeypatch.setattr(DU, "device_allocated_bytes", lambda d=None: reads.pop(0) if len(reads) > 1 else reads[0])
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
# out of memory on a plan that fitted; a wait after every op ran it at 11 136 MB (the Mac, 2026-11-04 20:05). The frees
# are read on the allocator, which sees them all (a count of the arena's alone missed ~3.3 GB, the Mac's RUN 33d): grown
# by the cliff since the last wait, the device is waited for and the base is read again; a cache then above the cliff is
# emptied (RUN 33b: 4.70 GiB of heaps MPS kept). Injection: no wait at the cliff -> red; the base never read again -> a
# wait at every later kill, red; no empty above the cliff -> red; an empty under it -> red.
def test_the_allocators_growth_waits_at_the_cliff(monkeypatch):
    seen = _calls(monkeypatch, allocated=(900, 1200, 300, 900, 1250))
    s = _seq(1 << 20, drain_cliff=1000)
    s._kill([torch.empty(16, dtype=torch.uint8)], (0,))       # 900 since the base 0: under the cliff
    assert seen == []
    s._kill([torch.empty(16, dtype=torch.uint8)], (0,))       # 1200: a wait, the base read again (300)
    assert [k for k, _ in seen] == ["sync"] and s._sync_base == 300
    s._kill([torch.empty(16, dtype=torch.uint8)], (0,))       # 900 - 300: under the cliff from the new base
    assert [k for k, _ in seen] == ["sync"]
    s._kill([torch.empty(16, dtype=torch.uint8)], (0,))       # 1250 - 300: under still
    assert [k for k, _ in seen] == ["sync"]


def test_a_cache_above_the_cliff_is_emptied_at_the_wait(monkeypatch):
    seen = _calls(monkeypatch, cached=5000, allocated=(1200, 100))
    _seq(1 << 20, drain_cliff=1000)._kill([torch.empty(16, dtype=torch.uint8)], (0,))
    assert [k for k, _ in seen] == ["sync", "empty"]


def test_a_cache_under_the_cliff_is_kept(monkeypatch):
    seen = _calls(monkeypatch, cached=500, allocated=(1200, 100))
    _seq(1 << 20, drain_cliff=1000)._kill([torch.empty(16, dtype=torch.uint8)], (0,))
    assert [k for k, _ in seen] == ["sync"]


def test_a_heap_release_reads_the_base_again(monkeypatch):
    seen = _calls(monkeypatch, allocated=(700,))
    s = _seq(1 << 20, drain_cliff=1 << 22)
    s._kill([torch.empty(1 << 20, dtype=torch.uint8)], (0,))  # a dedicated heap: sync + empty, the base read
    assert [k for k, _ in seen] == ["sync", "empty"] and s._sync_base == 700
