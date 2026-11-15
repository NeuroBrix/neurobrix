"""A run's KV cache grows past its floor only into the room above the allocator's reserve; its floor is still granted
from the whole room.

What this test does if the code is wrong: the growth cell fails when the cache takes the reserve again
(Qwen2.5-Coder-7B-Instruct, compiled, lazy_sequential, V100-16GB, 2026-11-15: 8 724 positions, 477 MiB, for a
5-token request; plan and cache filled the 15 845 MB budget and the first 260 MiB prefill matmul ran out of memory);
the floor cell fails when a plan that fits with its request's positions is refused for its growth (Janus-Pro-7B,
lazy on the same card); the negative cell fails when a reserve below zero is priced instead of refused. The solver's
half (which cards pay the reserve, its way to the price): tests/unit/prism/test_the_cache_s_growth_reserve_is_named_by_the_solver.py.
"""
import pytest

from neurobrix.core.state.price import kv_cache_size

KV = {"kind": "kv", "layers": 28, "kv_heads": 4, "k_dim": 128, "v_dim": 128, "window": 32768}
TOKEN = 28 * 4 * 256 * 2
MIB = 1024 * 1024
RESERVE = 3072 * MIB


def test_the_growth_stops_at_the_reserve():
    room = RESERVE + 100 * MIB
    size = kv_cache_size(KV, TOKEN, 8192, 532, False, room, floor_max_tokens=5, growth_reserve_bytes=RESERVE)
    assert size["max_cache_len"] == (100 * MIB) // TOKEN, "the cache grows into the room above the reserve only"
    whole = kv_cache_size(KV, TOKEN, 8192, 532, False, room, floor_max_tokens=5)
    assert whole["max_cache_len"] == 8192 + 532, "without a reserve the package bound is reached, as before"


def test_the_floor_is_granted_from_the_whole_room():
    room = 477 * MIB
    size = kv_cache_size(KV, TOKEN, 8192, 532, False, room, floor_max_tokens=5, growth_reserve_bytes=RESERVE)
    assert size["max_cache_len"] == 5 + 532, "the request's positions, inside the reserve: never refused for growth"
    with pytest.raises(RuntimeError, match="KV cache budget too small"):
        kv_cache_size(KV, TOKEN, 8192, 532, False, TOKEN * 10, floor_max_tokens=5, growth_reserve_bytes=RESERVE)


def test_a_negative_reserve_is_refused_by_name():
    """Below zero the growth would be priced past the room itself: more positions than the remaining bytes hold."""
    with pytest.raises(ValueError, match="growth_reserve_bytes -1"):
        kv_cache_size(KV, TOKEN, 8192, 532, False, 477 * MIB, floor_max_tokens=5, growth_reserve_bytes=-1)
