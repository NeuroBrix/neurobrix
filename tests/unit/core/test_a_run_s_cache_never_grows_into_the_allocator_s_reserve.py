"""A run's KV cache, its floor included, is granted only from the room above the allocator's reserve; where no reserve is
named, from the whole room. A floor inside the reserve is refused by one sentence: positions, cache needed, room.

What this test does if the code is wrong: the growth cell fails when the cache takes the reserve again
(Qwen2.5-Coder-7B-Instruct, compiled, lazy_sequential, V100-16GB, 2026-11-15: 8 724 positions, 477 MiB, for a
5-token request; plan and cache filled the 15 845 MB budget and the first 260 MiB prefill matmul ran out of memory);
the floor cell fails when a plan with no reserve named is refused or shrunk below its request's positions; the
refusal cells fail when a floor is granted inside the reserve again (the same 7B, same plan, 476 MiB above its priced
peak: a 4 358-position cache out of memory twice, 4 458 once in two, 2026-11-15) or the refusal stops saying the
positions, the cache they need and the room; the negative cell fails when a reserve below zero is priced instead of refused. The solver's
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
    """No reserve named (a bounded card, the Triton engine: their capacity holds the band back): the whole room."""
    floor = 5 + 532
    size = kv_cache_size(KV, TOKEN, 8192, 532, False, floor * TOKEN, floor_max_tokens=5)
    assert size["max_cache_len"] == floor, "the request's positions, the room exactly"
    with pytest.raises(RuntimeError, match="KV cache budget too small"):
        kv_cache_size(KV, TOKEN, 8192, 532, False, (floor - 1) * TOKEN, floor_max_tokens=5)


#: the tight plan the 7B ran on (V100-16GB, compiled, lazy_sequential): 476.1 MiB above its priced peak, and the
#: request's floor N + 158 positions (its prompt's margin), each N as the card measured it
TIGHT = int(476.1 * MIB)
MEASURED = [(4096, "ran 2 of 2"), (4200, "out of memory 2 of 2"), (4300, "out of memory 1 of 2")]


@pytest.mark.parametrize("asked,measured", MEASURED)
def test_a_floor_inside_the_reserve_is_refused_by_one_sentence(asked, measured):
    floor = asked + 158
    with pytest.raises(RuntimeError) as refused:
        kv_cache_size(KV, TOKEN, 8192, 158, False, TIGHT, floor_max_tokens=asked, growth_reserve_bytes=RESERVE)
    said = str(refused.value)
    assert said.startswith("ZERO FALLBACK: KV cache budget too small: "), said
    assert f"{floor} positions need {floor * TOKEN / 1e6:.0f} MB of cache, " in said, said
    assert f"{TIGHT / 1e6:.0f} MB left, 0 of them above the reserve of {RESERVE / 1e6:.0f} MB." in said, said


@pytest.mark.parametrize("asked,measured", MEASURED)
def test_a_granted_cache_never_takes_the_reserve(asked, measured):
    floor = asked + 158
    for room in (RESERVE + floor * TOKEN, RESERVE + floor * TOKEN + TOKEN - 1, RESERVE + 2 * floor * TOKEN, 2 * RESERVE):
        size = kv_cache_size(KV, TOKEN, 8192, 158, False, room, floor_max_tokens=asked, growth_reserve_bytes=RESERVE)
        assert floor <= size["max_cache_len"] <= 8192 + 158
        assert size["memory_bytes"] + RESERVE <= room, "a granted plan's cache and the reserve fit the room"
    with pytest.raises(RuntimeError, match="KV cache budget too small"):
        kv_cache_size(KV, TOKEN, 8192, 158, False, RESERVE + floor * TOKEN - 1, floor_max_tokens=asked,
                      growth_reserve_bytes=RESERVE)


def test_a_negative_reserve_is_refused_by_name():
    """Below zero the growth would be priced past the room itself: more positions than the remaining bytes hold."""
    with pytest.raises(ValueError, match="growth_reserve_bytes -1"):
        kv_cache_size(KV, TOKEN, 8192, 532, False, 477 * MIB, floor_max_tokens=5, growth_reserve_bytes=-1)
