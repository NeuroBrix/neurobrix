"""A served session's prefill chunk (exactly C rows) attends every cached key, and Prism prices that attention at its
DEAREST key length (core/prism/served_attention.py), never only at the chunk's own C x C.

Before (2026-11-12): the served Qwen3-Coder-30B-A3B on one V100-32GB ran out of memory in an agent's turn: its plan
priced the decoder's attention at the walk's C x C (over the math budget there: the flash route, no scores), while a
3 650-row turn under the budget took the math route and wrote 1.6 GB of scores and their scaled copy.

What this test does if the code is wrong:
1. against a brute force over every key length C <= tk <= keys of a route-like price (the math route's scores below a
   bound, nothing above it) plus the keys and values that length gathers: a price that misses the dearest length is
   red, and so is one that is not exact on this price;
2. the price is never below the base price at the shape the walk asked;
3. a price that states no scores bounds is refused by name;
4. the solver: the served decoder's attention at C rows over the session's keys, on the pinned V100-16GB, is priced
   at least at the math route's cost at the edge of its budget (red on main: `_attention_scratch` names no
   component and prices C x C only); another component, or a request with no cached keys, keeps the base price.

Run: python -m pytest tests/unit/prism/test_a_served_chunk_attention_is_priced_at_its_dearest_shape.py
"""
import pytest


def _route_like(per_cell, bound, item):
    """A price shaped like the Triton route: the math route's scores (and one copy) while they fit the bound, else
    nothing (the flash route)."""
    def scratch(b, h, tq, tk, D, Dv, Hk=None, mask=False, budget=None):
        scores = b * h * tq * tk * 4
        return 2 * scores if scores <= (bound if budget is None else budget) else 0
    scratch.scores_bounds = lambda D, budget=None: [bound if budget is None else budget]
    return scratch


@pytest.mark.parametrize("C,keys,bound", [(64, 300, 4 * 2 * 64 * 200), (32, 64, 4 * 2 * 32 * 50), (16, 500, 10 ** 9),
                                          (48, 96, 4 * 2 * 47 * 96), (64, 300, 4 * 2 * 64 * 63)])
def test_the_dearest_shape_is_found(C, keys, bound):
    from neurobrix.core.prism.served_attention import over_the_cached_keys
    b, h, hk, D, item = 1, 2, 1, 8, 2
    base = _route_like(b * h * 4, bound, item)
    served = over_the_cached_keys(base, keys, item)
    brute = max(base(b, h, C, k, D, D, Hk=hk) + b * hk * k * 2 * D * item for k in range(C, keys + 1))
    assert served(b, h, C, C, D, D, Hk=hk, mask=False) == brute
    assert served(b, h, C, C, D, D, Hk=hk, mask=False) >= base(b, h, C, C, D, D, Hk=hk)


def test_a_price_without_bounds_is_refused():
    from neurobrix.core.prism.served_attention import over_the_cached_keys
    with pytest.raises(ValueError, match="states no scores bounds"):
        over_the_cached_keys(lambda *a, **k: 0, 100, 2)


def test_a_zero_budget_prices_no_scores():
    from neurobrix.core.prism.served_attention import over_the_cached_keys
    b, h, hk, D, item, C, keys = 1, 2, 1, 8, 2, 64, 300
    base = _route_like(b * h * 4, 4 * 2 * 64 * 200, item)
    served = over_the_cached_keys(base, keys, item, lambda D: 0)
    assert served(b, h, C, C, D, D, Hk=hk, mask=False) == b * hk * keys * 2 * D * item


def test_the_served_decoder_is_priced_over_the_cached_keys(monkeypatch):
    from tests.unit.prism._pinned_machine import profile, V100_16GB
    from neurobrix.core.prism import PrismSolver
    from nbxtensor.ops.tiling_sizes import sdpa_scores_bytes
    s = PrismSolver()
    s._mode = "triton"
    # the device's own math budget (a card whose matrix unit takes no flash at this head dim)
    monkeypatch.setattr("neurobrix.core.prism.served_attention.served_chunk_scores_budget", lambda *a: None)
    prof = profile(V100_16GB)
    base = s._attention_scratch(prof, "float16")
    bound = base.scores_bounds(64)[0]
    b, h, C, D, keys = 1, 32, 2048, 64, 2048 * 8
    s._kv_decoder = "model"
    s._input_config = type("IC", (), {"prefill_keys": keys})()
    served = s._attention_scratch(prof, "float16", "model")
    # the math route at the edge of its budget: C rows against the most keys it takes
    b, C = 1, 1024
    keys_at_edge = bound // sdpa_scores_bytes(b, h, C, 1)
    assert C < keys_at_edge < keys
    edge = base(b, h, C, int(keys_at_edge), D=D, Dv=D, Hk=h)
    assert edge > base(b, h, C, C, D=D, Dv=D, Hk=h)
    assert served(b, h, C, C, D=D, Dv=D, Hk=h, mask=False) >= edge
    # another component, or no cached keys: the base price
    assert s._attention_scratch(prof, "float16", "vision") is base
    s._input_config = type("IC", (), {"prefill_keys": None})()
    assert s._attention_scratch(prof, "float16", "model") is base
