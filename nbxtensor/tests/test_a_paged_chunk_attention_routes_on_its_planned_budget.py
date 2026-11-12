"""A prefill chunk's attention over the paged cache routes on the budget its cache carries from the caller's plan
(`PagedTritonKVCache.chunk_scores_budget` -> `scaled_dot_product_attention_wrapper(scores_budget_bytes=)`), never on a
reading of its own; None keeps the device's budget.

What this test does if the code is wrong: `chunk_attention` drops the cache's budget (the wrapper would route on the
device's: the math route, its full scores) — red; a budget of 0 is not passed as 0 — red; the route at budget 0 is not
the flash where the matrix unit takes it — red. Card-less: the gather, the mask and the wrapper are stood in for.

Run: python -m pytest nbxtensor/tests/test_a_paged_chunk_attention_routes_on_its_planned_budget.py
"""
import pytest


@pytest.mark.parametrize("budget", [0, None, 12345])
def test_the_chunk_passes_its_planned_budget(monkeypatch, budget):
    from nbxtensor.memory import paged_kv as PK
    from nbxtensor.ops import attention_wrappers as AW
    seen = {}

    class Cache:
        spilled = 0
        page_tokens = 16
        chunk_scores_budget = budget

        def gather_pages(self, layer, which, keys, heads, dtype):
            return (which, keys)

    class Q:
        shape = (1, 4, 8, 64)
        _dtype = "float16"
        _device_idx = 0

    monkeypatch.setattr(PK, "chunk_mask", lambda *a: "mask")
    monkeypatch.setattr(AW, "scaled_dot_product_attention_wrapper",
                        lambda q, k, v, **kw: seen.update(kw) or "out")
    assert PK.chunk_attention(Cache(), 0, Q(), 24, 4, 0.125, 0.0, {}) == "out"
    assert "scores_budget_bytes" in seen and seen["scores_budget_bytes"] == budget


def test_a_zero_budget_routes_to_the_flash_where_the_unit_takes_it():
    from nbxtensor.ops import launch_keys as LK
    assert LK.sdpa_route(1, 32, 128, 65536, 128, 128, 2 ** 31, 128, 16, unit_flash=True) == ("math", 0)
    assert LK.sdpa_route(1, 32, 128, 65536, 128, 128, 0, 128, 16, unit_flash=True) == ("flash", 0)


def test_the_last_chunk_is_completed_to_the_smallest_rung_that_holds_it():
    """`chunk_rows(rungs=)`: full chunks of `rows`, then the turn's remainder completed to the smallest rung that
    holds it — red when it is completed to `rows` (a one-row continuation ran the whole chunk), when it runs its own
    row count (a row count the caller's ladder does not hold), or when a remainder no rung holds passes."""
    from nbxtensor.memory.kv_session import chunk_rows
    ids = list(range(100, 111))                                   # 11 positions
    padded = chunk_rows(ids, 3, 4, True)
    assert [(s, real, len(c)) for s, real, c, _ in padded] == [(3, 4, 4), (7, 4, 4)]
    assert chunk_rows(ids, 2, 4, True)[-1][2] == [110, 110, 110, 110]            # no rungs: completed to `rows`
    laddered = chunk_rows(ids, 2, 4, True, rungs=[2, 4])
    assert laddered == [(2, 4, [102, 103, 104, 105], [2, 3, 4, 5]), (6, 4, [106, 107, 108, 109], [6, 7, 8, 9]),
                        (10, 1, [110, 110], [10, 10])]                           # 1 row -> the 2-row rung
    assert chunk_rows(ids, 8, 4, True, rungs=[2, 4]) == [(8, 3, [108, 109, 110, 110], [8, 9, 10, 10])]
    assert chunk_rows(ids, 10, 128, False, rungs=[64, 128]) == [(10, 1, [110] * 64, list(range(64)))]
    with pytest.raises(ValueError, match="no rung holds it"):
        chunk_rows(ids, 0, 8, True, rungs=[2, 4])
