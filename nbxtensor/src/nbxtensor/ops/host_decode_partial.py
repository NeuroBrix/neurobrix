"""Decode attention over the pages of a sequence spilled to host memory (point 10, the mid-decode spill): computed where
those pages live, never shipped back over the link — FastDecode (arXiv 2403.11421), NEO (MLSys 2025), llama.cpp
`--no-kv-offload` compute attention next to the KV for the same reason (moving the cache is the bottleneck).

The result is a PARTIAL in the split kernel's own form (ops/paged_decode_attn.py `paged_decode_attn_split_kernel`):
per query row the unnormalised acc = sum exp(s - m) v, the running max m and the sum l, natural exp, fp32. It joins the
card's split partials in the same log-sum-exp combine (`flash_decode_reduce_kernel`): FlashDecoding's split-KV, exact.
Only the query (heads x D values) comes down and (heads x (D_v + 2)) values go up per layer and step.

Host math, numpy, torch-free; `merge_partials` is the reduce's arithmetic on the host, for tests and for a cache with no
card part at all.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

Partial = Tuple[np.ndarray, np.ndarray, np.ndarray]


def host_decode_partial(q: np.ndarray, k: np.ndarray, v: np.ndarray, n_tokens: int, groups: int,
                        scale: float) -> Partial:
    """q [H, D] (the step's query rows, rounded as the card reads them) · k [H_kv, cap, D] / v [H_kv, cap, D_v] the
    spilled positions in order, float32, in the attention's own layout (memory/host_spill.py writes them so once per
    page: a per-step transpose and fp16 -> fp32 conversion cost 7x the math, measured 2026-11-10 on 131 072 positions)
    · n_tokens the positions held (<= cap) · groups = H / H_kv -> (acc [H, D_v], m [H], l [H]) float32.
    No position (n_tokens == 0): m = -inf, l = 0, acc = 0, neutral in the merge."""
    heads, d = q.shape
    kv_heads, cap, d_k = k.shape
    d_v = v.shape[2]
    if heads != kv_heads * groups:
        raise ValueError(f"host_decode_partial: {heads} query heads are not {kv_heads} KV heads x {groups}")
    if d_k != d or v.shape[:2] != k.shape[:2] or k.dtype != np.float32 or v.dtype != np.float32:
        raise ValueError(f"host_decode_partial: q {q.shape} against k {k.shape} {k.dtype} / v {v.shape} {v.dtype} "
                         f"(float32, [H_kv, positions, D])")
    if not 0 <= n_tokens <= cap:
        raise ValueError(f"host_decode_partial: {n_tokens} positions but the host pages hold {cap}")
    if n_tokens == 0:
        return (np.zeros((heads, d_v), np.float32), np.full(heads, -np.inf, np.float32), np.zeros(heads, np.float32))
    from nbxtensor.backend.cpu.pool import parallel_for
    qg = q.astype(np.float32, copy=False).reshape(kv_heads, groups, d)          # query head h reads KV head h // groups
    sc = np.float32(scale)

    def piece(a: int, b: int) -> Partial:
        # the positions [a, b) of every head: one partial, merged with the others below
        s = np.matmul(qg, k[:, a:b].transpose(0, 2, 1)) * sc                    # [H_kv, groups, b - a]
        m = s.max(axis=2)
        p = np.exp(s - m[..., None])
        return np.matmul(p, v[:, a:b]), m, p.sum(axis=2)                        # acc [H_kv, groups, D_v]

    # the positions cut over the pool's threads (the application's: NeuroBrix sets them from its hardware profile's
    # compute threads), inline under the threshold measured for this kind of work (backend/cpu/pool.py)
    parts = parallel_for(n_tokens, piece, "host_decode_partial", probe=lambda n: _probe(n, kv_heads, groups, d, d_v))
    acc, m, l = parts[0]
    for a2, m2, l2 in parts[1:]:                                                # fixed order: deterministic
        mn = np.maximum(m, m2)
        c1, c2 = np.exp(m - mn), np.exp(m2 - mn)
        acc, l, m = acc * c1[..., None] + a2 * c2[..., None], l * c1 + l2 * c2, mn
    return acc.reshape(heads, d_v), m.reshape(heads), l.reshape(heads)


def _probe(n: int, kv_heads: int, groups: int, d: int, d_v: int):
    """The work of `n` spilled positions, for the pool's threshold measurement: a callable of (start, stop)."""
    rng = np.random.default_rng(0)
    q = rng.standard_normal((kv_heads, groups, d)).astype(np.float32)
    k = rng.standard_normal((kv_heads, n, d)).astype(np.float32)
    v = rng.standard_normal((kv_heads, n, d_v)).astype(np.float32)

    def run(a: int, b: int):
        p = np.exp(np.matmul(q, k[:, a:b].transpose(0, 2, 1)))
        return np.matmul(p, v[:, a:b])
    return run


def merge_partials(parts: Sequence[Optional[Partial]]) -> np.ndarray:
    """The combine of `flash_decode_reduce_kernel` on the host, in the given (fixed) order: [H, D_v] = acc / l, a row
    with no position at all (l = 0) giving 0 as the reduce does."""
    acc = m = l = None
    for part in parts:
        if part is None:
            continue
        a, ms, ls = (np.asarray(x, np.float64) for x in part)
        if acc is None:
            acc, m, l = a.copy(), ms.copy(), ls.copy()
            continue
        mn = np.maximum(m, ms)
        safe = np.where(np.isneginf(mn), 0.0, mn)
        c_old = np.where(np.isneginf(m), 0.0, np.exp(m - safe))
        c_new = np.where(np.isneginf(ms), 0.0, np.exp(ms - safe))
        acc, l, m = acc * c_old[:, None] + a * c_new[:, None], l * c_old + ls * c_new, mn
    if acc is None:
        raise ValueError("merge_partials: no partial to merge")
    return acc / np.where(l == 0.0, 1.0, l)[:, None]
