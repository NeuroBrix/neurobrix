"""The pool's two size classes and its half rule (Allegro triton on a V100-16GB, gate of 2026-11-01 on 49fbcae6).

The run was refused a 2 531 MB gelu output with 4 429 MB free in six blocks the pool could not return: each sat in a
segment a live tensor pinned. The real trace replayed through the pool's own rules (campaign 2026_11_01_gate_nbxtensor,
trace_allegro/replay.py) named the mechanism: a request takes "the smallest free block that holds it", so a 17 KB scalar
takes a freed 2.5 GB block, splits it and pins the rest alive; a 632 MB request does the same to the next one. Two
rules of torch's caching allocator end it, replayed on three real traces (Allegro 16g, mochi-1-preview 32g, Kokoro):

1. two size classes — a request of at most `POOL_SMALL_BYTES` is served only from small segments of
   `POOL_SMALL_SEGMENT_BYTES`, a larger one only from large segments (kSmallSize / kSmallBuffer);
2. the half rule — in the large class a free block is split only when what is left is at most the request, so a block
   is never pinned by something smaller than its remainder.

With them a wholly free segment may stay parked across a growth (it is the next request's block) and returns to the
driver when a malloc is refused, as before 164e89dc; the pinned remainders that commit flushed around no longer form.

What each test would do if the code were wrong: the scalar carves the parked 2.5 GB block and the second request grows
the driver by another 2.5 GB (peak 5 GB — red); the 632 MB request pins 1.9 GB (red); a request over half the block
refused the split (a second driver growth — red)."""
from nbxtensor.memory.pool import POOL_SMALL_BYTES, POOL_SMALL_SEGMENT_BYTES, SegmentPool, simulate

MB = 1 << 20
BIG = 2531 * MB


def test_a_scalar_never_pins_a_large_block():
    ev = [("alloc", "a", BIG), ("free", "a", 0), ("alloc", "s", 17 * 1024), ("alloc", "b", BIG)]
    sim = simulate(ev)
    assert sim.peak_bytes == BIG + POOL_SMALL_SEGMENT_BYTES, sim


def test_a_request_under_half_a_block_leaves_it_whole():
    ev = [("alloc", "a", BIG), ("free", "a", 0), ("alloc", "m", 632 * MB), ("alloc", "b", BIG)]
    sim = simulate(ev)
    assert sim.peak_bytes == BIG + 632 * MB, sim


def test_a_request_over_half_a_block_splits_it():
    ev = [("alloc", "a", BIG), ("free", "a", 0), ("alloc", "m", 1900 * MB)]
    sim = simulate(ev)
    assert sim.peak_bytes == BIG, sim


def test_the_classes_never_mix():
    """A small request never takes a large free block even when no small segment exists; a large one never takes a
    small free block even when one would hold it."""
    pool = SegmentPool()
    pool.add_free_segment(0, 1 << 40, BIG, request=BIG)
    assert pool.take(0, POOL_SMALL_BYTES) is None
    pool.add_free_segment(0, 1 << 41, POOL_SMALL_SEGMENT_BYTES, request=POOL_SMALL_BYTES)
    assert pool.take(0, POOL_SMALL_BYTES + 1) is None or pool.alloc_size[1 << 40] == BIG
    assert pool.take(0, 1024) == 1 << 41
