"""Step 2 of the state engine, the shared half (core/state/pages.py, docs/design/state-engine.md §4.3 and §4.5): pages are
granted as a sequence grows, shared with copy-on-write, and on an asynchronous device a released page returns to the
free list only after the fence past its last use.

What this test does if the code is wrong: the growth cells fail when a sequence takes pages it does not need yet (the
max_len allocation step 2 removes) or when a position maps to the wrong page or slot; the sharing cells fail when a fork
copies pages up front, or when a write into a shared page lands in the other sequence's page; the release cells fail
when an asynchronous device gets a page back before its fence completed (on MPS the command still reading it would see
the next writer's data), or when a synchronous device has to wait for a fence; the exhaustion cell fails when a grant
the pool cannot cover takes part of the pages or is not refused by name.
"""
import pytest

from neurobrix.core.state.pages import PagePool, PageTable, StateExhausted


def test_a_sequence_takes_only_the_pages_its_positions_need():
    pool = PagePool(num_pages=8, tokens_per_page=16, async_release=False)
    t = PageTable(pool)
    assert t.reserve(1) == [0] and pool.free_pages == 7
    assert t.reserve(16) == [], "position 15 is still in the first page"
    assert t.reserve(17) == [1]
    assert t.reserve(50) == [2, 3] and t.length == 50
    assert t.slot(0) == (0, 0) and t.slot(15) == (0, 15) and t.slot(16) == (1, 0) and t.slot(49) == (3, 1)
    with pytest.raises(IndexError):
        t.slot(50)


def test_a_fork_shares_its_pages_and_copies_one_only_on_write():
    pool = PagePool(num_pages=8, tokens_per_page=4, async_release=False)
    cond = PageTable(pool)
    cond.reserve(6)                                  # pages 0, 1
    uncond = cond.fork()
    assert pool.free_pages == 6 and pool.refs(0) == 2 and pool.refs(1) == 2, "a fork copies nothing up front"
    copy = uncond.writable(5)
    assert copy == (1, 2), "the write into the shared page 1 gets its own page, the engine copies 1 -> 2"
    assert uncond.pages == [0, 2] and cond.pages == [0, 1] and pool.refs(1) == 1 and pool.refs(2) == 1
    assert cond.writable(5) is None, "page 1 is now cond's alone: no copy"
    uncond.free()
    assert pool.refs(0) == 1 and pool.free_pages == 6


def test_an_asynchronous_device_gets_a_page_back_only_past_its_fence():
    pool = PagePool(num_pages=4, tokens_per_page=8, async_release=True)
    t = PageTable(pool)
    t.reserve(32)
    assert pool.free_pages == 0
    t.free(fence=7)
    assert pool.free_pages == 0 and pool.pending_pages == 4, "released under fence 7, not yet completed"
    with pytest.raises(StateExhausted, match="4 wait for their release fence"):
        pool.grant(1)
    assert pool.fence_completed(6) == 0
    assert pool.fence_completed(7) == 4 and pool.free_pages == 4
    t2 = PageTable(pool)
    t2.reserve(8)
    t2.free(fence=5)                                 # already completed: free at once
    assert pool.free_pages == 4 and pool.pending_pages == 0


def test_an_asynchronous_release_without_a_fence_is_refused_and_a_synchronous_one_needs_none():
    pool = PagePool(num_pages=2, tokens_per_page=8, async_release=True)
    t = PageTable(pool)
    t.reserve(8)
    with pytest.raises(RuntimeError, match="without a fence"):
        t.free()
    sync = PagePool(num_pages=2, tokens_per_page=8, async_release=False)
    s = PageTable(sync)
    s.reserve(16)
    s.free()
    assert sync.free_pages == 2


def test_a_grant_the_pool_cannot_cover_takes_nothing_and_is_refused_by_name():
    pool = PagePool(num_pages=3, tokens_per_page=8, async_release=False)
    t = PageTable(pool)
    with pytest.raises(StateExhausted, match="ZERO FALLBACK: the state pool has 3 free pages of 8 positions, 4 asked"):
        t.reserve(25)
    assert pool.free_pages == 3 and t.pages == [] and t.length == 0


def test_the_kernels_read_flat_slots_and_a_padded_block_table():
    from neurobrix.core.state.pages import block_table
    pool = PagePool(num_pages=8, tokens_per_page=4, async_release=False)
    a, b = PageTable(pool), PageTable(pool)
    a.reserve(6)                                     # pages 0, 1
    b.reserve(2)                                     # page 2
    assert list(a.slots([0, 3, 4, 5])) == [0, 3, 4, 5] and list(b.slots([0, 1])) == [8, 9]
    rows, width = block_table([a, b])
    assert width == 2 and list(rows) == [0, 1, 2, -1] and rows.itemsize == 4


def test_the_page_size_is_derived_from_the_tile_candidates_the_granule_and_the_layer_bytes():
    from neurobrix.core.state.price import StateUndeclared, page_tokens
    layer = 4 * (64 + 64) * 2                        # kv_heads 4, k/v 64, bf16: 1 KiB a position per layer
    assert page_tokens(layer, [16, 32, 64], granule_bytes=16384, rung_bytes=18 << 30) == 64, "the largest tile covers it"
    assert page_tokens(layer, [16, 32], granule_bytes=65536, rung_bytes=18 << 30) == 64, "doubled to reach the granule"
    assert page_tokens(layer, [16], granule_bytes=2 << 20, rung_bytes=18 << 30) == 2048
    with pytest.raises(ValueError, match="do not all divide"):
        page_tokens(layer, [32, 48], granule_bytes=1, rung_bytes=18 << 30)
    with pytest.raises(StateUndeclared, match="exceeds the rung"):
        page_tokens(1 << 20, [16], granule_bytes=1, rung_bytes=8 << 20)
