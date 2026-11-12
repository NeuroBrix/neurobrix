"""The attention a served session's prefill chunk runs over the cached keys, priced at its dearest key length.

Why it exists. A served turn is prefilled in chunks of exactly C rows (nbxtensor `kv_session.chunk_rows`: the last
one completed with copies of its last row), and each chunk's attention reads every key the session has cached
(nbxtensor `paged_kv.chunk_attention`: the keys and values gathered as whole pages, [B, heads, keys, D] in the query's
dtype, beside the graph's own) — C query rows against any key length from C up to the session's reach. The Triton
engine's route is not monotonic in the key length: the math route writes every score, then its scaled copy and its
softmax, while the fp32 scores fit the executing device's budget, and a chunked or flash route writes less beyond it.
A chunk priced at its own C x C (over the budget there: no scores priced) ran a served Qwen3-Coder-30B-A3B out of
memory when a turn of 3 650 rows took the math route under the budget: 1.6 GB of scores, then their scaled copy, beside
a card the experts' cache had filled to the price (2026-11-12).

Contract.
- In: `scratch(b, h, tq, tk, D=, Dv=, Hk=, mask=, budget=) -> bytes`, the route-aware price of ONE attention (the
  solver's, `attention_scratch.triton_scratch`), carrying `scratch.scores_bounds(D, budget) -> [bytes]`, the fp32 scores bound
  of every device the route may run on; the session's cached keys `keys`; the gathered keys' item size; `budget_of(D)`,
  the math route's budget the plan states for the chunk's attention at head dim D (`KVCachePlan.
  chunk_scores_budget_bytes`: 0, the flash route, writes no scores) or None, each device's own.
- Out: a callable of the same signature. Asked at a chunk's C rows (the walk's query rows), it answers the dearest key
  length C <= tk <= keys, plus the keys and values that length gathers: within the math route the price grows with
  the key length, so on each device it is dearest at the most keys the math route takes (C x tk under the bound); past
  that edge and up to the reach the other routes are asked at the first key length past it and at the reach.
- Never cheaper than `scratch` at the shape the walk asked, on the route the plan states: that shape is one of the
  candidates (on the flash route, which the plan may state for the chunk, that is the gathered keys alone).
- Torch-free (Prism imports it).
"""
from __future__ import annotations

from typing import Callable, Iterable, Optional, Set


def _edge_keys(per_row_key: int, bound: int, rows: int, keys: int) -> Iterable[int]:
    """The key lengths C <= tk <= keys where one device's price may be dearest: the most keys its math route takes at
    `rows` query rows (scores of `per_row_key` bytes a row and a key, under `bound`), and the first length past it."""
    if per_row_key <= 0:
        return
    edge = int(bound) // (int(per_row_key) * int(rows))
    for tk in (edge, edge + 1):
        if rows <= tk <= keys:
            yield tk


def over_the_cached_keys(scratch: Callable, keys: int, gathered_itemsize: int,
                         budget_of: Callable = lambda D: None) -> Callable:
    """`scratch` priced at the dearest key length a served chunk's attention over `keys` cached keys can take (the
    module's contract). Refused by name when `scratch` states no scores bounds."""
    bounds_of = getattr(scratch, "scores_bounds", None)
    if bounds_of is None:
        raise ValueError("ZERO FALLBACK: the attention price states no scores bounds — a served chunk's attention "
                         "over the cached keys cannot be priced at its dearest key length")
    keys, item = int(keys), int(gathered_itemsize)
    from nbxtensor.ops.tiling_sizes import sdpa_scores_bytes
    memo = {}

    def served(b, h, tq, tk, D, Dv, Hk, mask):           # the walk states every argument (`_attention_scratch_of`)
        rows, hk = int(tq), int(Hk or h)
        if keys < rows:
            raise ValueError(f"ZERO FALLBACK: a served chunk of {rows} rows over {keys} cached keys — the session's "
                             f"reach is shorter than its own chunk")
        at = (int(b), int(h), rows, int(tk), int(D), int(Dv), hk, bool(mask))
        if at not in memo:
            lengths: Set[int] = {int(tk), keys}
            per_row_key = sdpa_scores_bytes(int(b), int(h), 1, 1)
            budget = budget_of(int(D))
            for bound in bounds_of(int(D), budget):
                lengths.update(_edge_keys(per_row_key, int(bound), rows, keys))
            # the walk counts the graph's own keys and values; the chunk gathers the cache's beside them
            memo[at] = max(int(scratch(b, h, rows, k, D=D, Dv=Dv, Hk=hk, mask=mask, budget=budget))
                           + int(b) * hk * k * (int(D) + int(Dv)) * item for k in lengths)
        return memo[at]

    return served


def served_chunk_scores_budget(profile, dtype: str, head_dim: int, mode: str) -> Optional[int]:
    """The math route's budget a served chunk's attention over the cached keys routes on (`KVCachePlan.
    chunk_scores_budget_bytes`): 0 on the Triton engines where every card of the profile runs the matrix unit's flash at
    this head dim and dtype (`launch_keys.unit_flash_takes`) — the flash writes no scores, and a math route's reserve is
    held beside the session for its whole life, taken from what speeds its decode (measured 2026-11-12, Qwen3-Coder-
    30B-A3B on one V100-32GB: the reserve at its 65 664 cached keys, 4.1 GB, took the experts' cache from 20.9 to 15.3 GB
    and the decode from 0.053 to 0.092 s a token); None elsewhere: the device's own budget, priced at its dearest key
    length."""
    from neurobrix.core.prism.host_footprint import engine_of
    if engine_of(mode) != "triton" or profile is None or not profile.devices:
        return None
    from nbxtensor.dtype.nbx_dtype import NBXDtype
    from nbxtensor.ops import launch_keys as _lk
    from neurobrix.core.prism.attention_scratch import triton_routes
    t = NBXDtype[str(dtype)]
    takes = all(_lk.unit_flash_takes(int(head_dim), t, t, t, unit) for _b, _r, _c, unit in triton_routes(profile))
    return 0 if takes else None


def over_the_session_keys(scratch: Callable, input_config, profile, dtype: str, is_decoder: bool) -> Callable:
    """`scratch` as the served decoder's attention runs it: priced at its dearest key length over the session's keys
    (`over_the_cached_keys`) when the request states them (`InputConfig.prefill_keys`, the served plan's) and the
    component is the flow's KV decoder, on the route the plan states for the chunk (`served_chunk_scores_budget`, per
    head dim); else `scratch`. The gathered keys and values are in the query's dtype: the run dtype."""
    keys = int(getattr(input_config, "prefill_keys", 0) or 0)
    if scratch is None or not keys or not is_decoder:
        return scratch
    from nbxtensor.dtype import itemsize as _itemsize
    return over_the_cached_keys(scratch, keys, _itemsize.itemsize(str(dtype)),
                                lambda D: served_chunk_scores_budget(profile, str(dtype), int(D), "triton"))


def served_chunk_route(profile, dtypes, decoder, kv_plan, mode: str) -> Optional[int]:
    """The route budget a served plan states for its chunk's attention (`KVCachePlan.chunk_scores_budget_bytes`), at
    the query's dtype — the decoder's run dtype (`dtypes`, the solve's resolution; the gathered keys are widened to it)
    and the cache's key head dim. A decoder whose run dtype the solve did not resolve is refused by name."""
    if decoder not in (dtypes or {}):
        raise RuntimeError(f"ZERO FALLBACK: no run dtype is resolved for the decoder {decoder!r}: its served chunk's "
                           f"attention route cannot be planned")
    return served_chunk_scores_budget(profile, str(dtypes[decoder]), int(kv_plan.k_head_dim), mode)
