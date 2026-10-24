"""Where the device's allocator keeps the heap of a large freed tensor (`memory.compiled_allocator_retention`: MPS's
OVERSIZE heaps), the compiled engine gives the heap back when it frees such a tensor — the device synchronised, its
cache emptied — so what the device holds is the live set the plan priced (Open-Sora-v2's VAE on MPS held 21.2 GB for
3.8 GB live, the Mac, 2026-10-23). A device whose allocator reuses freed memory (CUDA: null) is never synchronised.

What this test would do if the code were wrong: no release on a large free — red at the call count; a release on a
small one, or on CUDA — red at the zero counts; a vendor file stating nothing — red if it is not refused by name."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from neurobrix.core import device_utils as DU  # noqa: E402
from neurobrix.core.runtime.graph.compiled_sequence import CompiledSequence  # noqa: E402
from neurobrix.core.prism.attention_scratch import vendor_memory_of  # noqa: E402


def _seq(release_from):
    s = CompiledSequence.__new__(CompiledSequence)
    s._release_from = release_from
    s.device = torch.device("cpu")
    return s


def _calls(monkeypatch):
    seen = []
    monkeypatch.setattr(DU, "device_sync", lambda d=None: seen.append(("sync", d)))
    monkeypatch.setattr(DU, "device_empty_cache", lambda d=None: seen.append(("empty", d)))
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
