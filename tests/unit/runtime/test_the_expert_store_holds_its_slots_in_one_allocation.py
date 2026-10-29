"""The Triton expert store holds every slot in ONE device allocation: its footprint is its plan.

Measured 2026-10-14 on Qwen3-Omni-30B-A3B-Instruct (32 GB V100, triton): the store's 2 163 slots
were 6 489 separate 3 MiB cudaMallocs (gate, up, down per slot), the driver rounds every allocation
to its 2 MiB granularity (measured on the rig: a 3 MiB block costs 4 MiB, a 9 MiB block 10 MiB), and
the card carried 6.3 GB no table of the engine knew about — 31.6 GB used for 24.8 GB allocated, the
"untracked" gap of every Omni run. The plan priced the cache at 19 475 MB; the driver held 25 956.

One arena of n_slots x slot_bytes (each projection 256-byte aligned), carved into slot views, is
rounded once: the store's device bytes equal Prism's figure to within one granule, whatever the
granularity of the platform, with no number of the platform written anywhere.

What this test does if the code is wrong: the one-allocation cell fails on the previous store (three
mallocs per slot — red before this change); the carving cell fails if two slots overlap, a slot
leaves the arena, or a projection is not 256-byte aligned; the geometry cell fails if the slots do not
carry the experts' own shape and dtype; the release cell fails if the arena is still referenced once
the store is released.

Run: python -m pytest tests/unit/runtime/test_the_expert_store_holds_its_slots_in_one_allocation.py
"""
from __future__ import annotations

import gc

import pytest

from neurobrix.core.runtime.expert_residency import ExpertResidency
from nbxtensor import tensor as nt
from nbxtensor.tensor import DeviceAllocator as DA, NBXDtype, NBXTensor

HIDDEN, INTER, N_SLOTS = 2048, 768, 7


class _Driver:
    """A device allocator seen from the outside: every malloc_cuda recorded, pointers fake, nothing freed."""

    def __init__(self):
        self.mallocs = []
        self.frees = []
        self._next = 1 << 40

    def malloc_cuda(self, nbytes, dev_idx=None):
        ptr = self._next
        self._next += int(nbytes) + (1 << 21)   # far apart: an overlap is a bug, never a coincidence
        self.mallocs.append((ptr, int(nbytes)))
        return ptr

    def free_cuda(self, ptr):
        self.frees.append(ptr)


@pytest.fixture
def driver(monkeypatch):
    d = _Driver()
    monkeypatch.setattr(DA, "malloc_cuda", staticmethod(d.malloc_cuda))
    monkeypatch.setattr(DA, "free_cuda", staticmethod(d.free_cuda))
    monkeypatch.setattr(DA, "set_device", staticmethod(lambda i: None))
    monkeypatch.setattr(DA, "get_device", staticmethod(lambda: 0))
    monkeypatch.setattr(DA, "create_stream", staticmethod(lambda *a, **k: 7))
    monkeypatch.setattr(DA, "destroy_stream", staticmethod(lambda s: None))
    monkeypatch.setattr(DA, "stream_synchronize", staticmethod(lambda s: None))
    return d


def _like():
    """The experts' geometry: (gate, up, down) bf16 host tensors as the loader hands them."""
    return [NBXTensor.from_raw(0, (INTER, HIDDEN), NBXDtype.bfloat16, device="cpu"),
            NBXTensor.from_raw(0, (INTER, HIDDEN), NBXDtype.bfloat16, device="cpu"),
            NBXTensor.from_raw(0, (HIDDEN, INTER), NBXDtype.bfloat16, device="cpu")]


def _store(driver):
    from neurobrix.triton.expert_store import NBXExpertStore
    return NBXExpertStore(ExpertResidency(N_SLOTS), 0, _like())


def test_the_slots_live_in_one_allocation(driver):
    store = _store(driver)
    proj = INTER * HIDDEN * nt.dtype_size(NBXDtype.bfloat16)
    assert store.slot_bytes == 3 * proj
    assert len(driver.mallocs) == 1, f"{len(driver.mallocs)} device allocations for {N_SLOTS} slots; one arena expected"
    (ptr, nbytes), = driver.mallocs
    assert N_SLOTS * 3 * proj <= nbytes <= N_SLOTS * 3 * (proj + 256)
    assert store.device_bytes == nbytes


def test_the_slots_are_carved_disjoint_and_aligned_inside_the_arena(driver):
    store = _store(driver)
    (ptr, nbytes), = driver.mallocs
    spans = []
    for slot in store.slots:
        for t in slot:
            assert t.data_ptr() % 256 == 0
            assert ptr <= t.data_ptr() and t.data_ptr() + t._nbytes <= ptr + nbytes, "a slot outside the arena"
            spans.append((t.data_ptr(), t.data_ptr() + t._nbytes))
    spans.sort()
    for (a0, a1), (b0, _b1) in zip(spans, spans[1:]):
        assert a1 <= b0, "two projections overlap"
    assert len(spans) == 3 * N_SLOTS


def test_the_slots_carry_the_experts_own_geometry(driver):
    store = _store(driver)
    for slot in store.slots:
        assert [(tuple(t.shape), t._dtype) for t in slot] == [((INTER, HIDDEN), NBXDtype.bfloat16),
                                                              ((INTER, HIDDEN), NBXDtype.bfloat16),
                                                              ((HIDDEN, INTER), NBXDtype.bfloat16)]
        assert all(t._device == "cuda" and t.is_contiguous() for t in slot)


def test_release_lets_the_arena_go(driver):
    store = _store(driver)
    (ptr, _n), = driver.mallocs
    store.release()
    gc.collect()
    assert store.slots == [] and store.device_bytes == 0
    assert driver.frees == [ptr], "the arena is freed once, when the store releases it"
