"""A unified expert store copies a missed expert through ONE host staging buffer per projection and
gives the kernel back the expert's file pages after the copy.

The Mac, 2026-10-14 08:55, Qwen3-30B-A3B-Thinking-2507 native on an M4 Pro: the process's own
footprint stayed flat at 8.6 GB while, during the first forward's expert misses, free memory fell
from 12 GB to 0.6 GB and swap grew 7 GB — memory outside the process, held by the copies from
`torch.frombuffer` views over the private shard maps (55 GB of experts) into MPS slots. The store
now copies through a host staging buffer allocated once per projection and asks the executor to
`madvise(MADV_DONTNEED)` the expert's page-aligned span (`expert_files.drop_pages`).

What this test does if the code is wrong: the staging cell fails when a second miss allocates a
new staging buffer or the bytes do not reach the slot (red before this change: no staging at all);
the release cell fails when the store does not hand each copied view to the release callback (red
before: `TorchExpertStore` took no callback); the pages cell fails when the executor's release does
not advise the page-aligned range of the view's span, or advises on a tensor that is no view.

Run: PYTHONPATH=src python -m pytest tests/unit/runtime/test_a_unified_expert_store_stages_a_missed_expert_and_drops_its_pages.py
"""
from __future__ import annotations

import mmap
import struct

import pytest
import torch

from neurobrix.core.runtime import expert_files as EF
from neurobrix.core.runtime.expert_residency import ExpertResidency
from neurobrix.core.runtime.graph.expert_store import TorchExpertStore

SHAPE = (4, 3)


def _triple(seed):
    g = torch.Generator().manual_seed(seed)
    return tuple(torch.randn(SHAPE, generator=g, dtype=torch.float32) for _ in range(3))


def test_a_missed_expert_is_copied_through_one_staging_buffer_per_projection():
    store = TorchExpertStore(ExpertResidency(2), torch.device("cpu"), _triple(0))
    hosts = {0: _triple(1), 1: _triple(2), 2: _triple(3)}
    on_card, _ = store.acquire("L", [0, 1], lambda e: hosts[e])
    first = store._staging
    assert first is not None and len(first) == 3
    for e in (0, 1):
        for d, s in zip(on_card[e], hosts[e]):
            assert torch.equal(d, s)
    on_card, _ = store.acquire("L", [2], lambda e: hosts[e])        # a second miss: the same staging
    assert store._staging is first
    for d, s in zip(on_card[2], hosts[2]):
        assert torch.equal(d, s)


def test_each_copied_view_is_handed_to_the_release_callback():
    released = []
    store = TorchExpertStore(ExpertResidency(1), torch.device("cpu"), _triple(0), release=released.append)
    hosts = {0: _triple(1), 1: _triple(2)}
    store.acquire("L", [0], lambda e: hosts[e])
    store.acquire("L", [1], lambda e: hosts[e])
    assert [t.data_ptr() for t in released] == [t.data_ptr() for e in (0, 1) for t in hosts[e]]


def test_the_executor_gives_back_the_page_aligned_span_of_a_mapped_view(tmp_path, monkeypatch):
    from neurobrix.core.runtime.graph_executor import GraphExecutor
    path = tmp_path / "shard.bin"
    head = b"\x00" * 1000                                   # the view starts inside a page
    data = struct.pack("<12f", *range(12))
    path.write_bytes(head + data + b"\x00" * 100)
    with open(path, "rb") as f:
        m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY)
    view = torch.frombuffer(m, dtype=torch.float32, count=12, offset=len(head)).view(SHAPE)
    ex = GraphExecutor.__new__(GraphExecutor)
    ex._expert_pages = {view.data_ptr(): (m, len(head), len(data))}
    advised = []
    real = EF.drop_pages
    monkeypatch.setattr(EF, "drop_pages", lambda mm, off, n: advised.append((mm is m, off, n)) or (0, 0))
    ex._release_expert_pages(view)
    assert advised == [(True, len(head), len(data))]
    with pytest.raises(RuntimeError, match="no view of a mapped expert file"):
        ex._release_expert_pages(torch.zeros(SHAPE))
    # the real advice: page-aligned, covering the span, and the mapping still readable afterwards
    start, length = real(m, len(head), len(data))
    assert start == 0 and start + length >= len(head) + len(data)
    assert torch.equal(view, torch.tensor(range(12), dtype=torch.float32).view(SHAPE))
