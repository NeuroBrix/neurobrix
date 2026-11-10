"""A decoding sequence that outgrows the card's pages spills its oldest pages to host memory (memory/host_spill.py) and
keeps decoding: every step's output equals the same sequence's in a cache whose card holds it all (point 10, the
mid-decode spill). What this test would do if the code were wrong: spilled pages lost or read twice, the card's
positions not shifted (an append over a live page), the host partial left out, a truncation inside the spilled part
keeping stale positions, a prompt past the card's reach spilled before it was written (red)."""
import numpy as np
import pytest

from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.memory.allocator import DeviceAllocator
from nbxtensor.memory.host_spill import HostSpill
from nbxtensor.memory.paged_kv import PagedTritonKVCache
from nbxtensor.tensor import NBXTensor

LAYERS, HK, H, D, PT = 2, 2, 8, 64, 64


def _dev(a):
    return NBXTensor.from_numpy(a).to(device="cuda:0")


def _cache(card_pages, budget=1 << 30):
    spill = HostSpill(LAYERS, HK, D, D, budget) if card_pages else None
    return PagedTritonKVCache(LAYERS, HK, D, D, max_cache_len=PT, dtype=NBXDtype.float16, page_tokens=PT,
                              async_release=False, card_pages=card_pages, host_spill=spill)


def _run(cache, rng_seed, prompt, steps, truncate_at=None):
    rng = np.random.default_rng(rng_seed)
    outs = []
    for i in range(LAYERS):
        kv = rng.standard_normal((2, 1, HK, prompt, D)).astype(np.float16)
        cache.prefill(i, _dev(kv[0]), _dev(kv[1]))
    for s in range(steps):
        if truncate_at is not None and s == steps // 2:
            cache.truncate(truncate_at)
        for i in range(LAYERS):
            q, k, v = (rng.standard_normal(shape).astype(np.float16)
                       for shape in ((1, H, 1, D), (1, HK, 1, D), (1, HK, 1, D)))
            outs.append(cache.decode(i, _dev(q), _dev(k), _dev(v), D ** -0.5).numpy().astype(np.float32))
    return outs


@pytest.fixture(autouse=True)
def _accelerator():
    if DeviceAllocator.device_count() == 0:          # only a machine without an accelerator skips; any other error fails
        pytest.skip("no accelerator on this machine")


@pytest.mark.parametrize("prompt, steps, card_pages", [(100, 200, 2), (64, 300, 3), (10, 150, 1)])
def test_every_step_equals_the_card_that_holds_it_all(prompt, steps, card_pages):
    spilled_cache = _cache(card_pages)
    got = _run(spilled_cache, prompt, prompt, steps)
    want = _run(_cache(0), prompt, prompt, steps)
    assert spilled_cache.spilled > 0 and spilled_cache.spilled % PT == 0
    assert spilled_cache.get_cache_len() == prompt + steps
    for g, w in zip(got, want):
        assert np.allclose(g, w, rtol=0, atol=2e-3)


def test_a_truncation_inside_the_spilled_part_keeps_only_the_prefix():
    a = _cache(2)
    got = _run(a, 7, 70, 160, truncate_at=40)          # spilled 64 by step 80, the prefix 40 ends inside it
    want = _run(_cache(0), 7, 70, 160, truncate_at=40)
    assert a.spilled == 40 or a.spilled % PT == 0          # the kept prefix moved the spill point, then it spilled on
    for g, w in zip(got, want):
        assert np.allclose(g, w, rtol=0, atol=2e-3)


def test_past_the_reach_is_refused_by_name():
    with pytest.raises(RuntimeError, match="past the card's reach"):
        _run(_cache(1), 3, 130, 1)                                   # a prompt longer than the card's pages
    with pytest.raises(RuntimeError, match="host budget"):
        _run(_cache(1, budget=LAYERS * HK * 2 * D * 4 * PT), 3, 60, 200)  # one page of host budget
    c = _cache(1)
    _run(c, 5, 60, 100)
    with pytest.raises(RuntimeError, match="not parked"):
        c.tier_length()


def test_a_spilled_step_is_never_replayed():
    """A spilled step runs host code (the host partial): a recorded replay could not reproduce it."""
    c = _cache(1)
    _run(c, 9, 60, 10)
    assert c.spilled > 0 and c.replay_key(0) is None
