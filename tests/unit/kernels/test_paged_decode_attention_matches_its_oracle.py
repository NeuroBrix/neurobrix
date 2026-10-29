"""The paged KV append and decode read (state engine step 2, kernels/ops/paged_decode_attn.py) against a float64 oracle.

The kernels are born under the correctness-oracle gate class (an oracle proves we are right; a byte gate only that we
repeat ourselves). What each test would do if the code were wrong:
  1. the read through a SHUFFLED page table equals the float64 attention over the sequence's own positions: a kernel
     that ignored the table (read pages in order) or mis-addressed a tile inside a page reads other positions — red;
  2. pages the table does not name are filled with NaN: a read past a sequence's length or through a -1 entry turns the
     output NaN — red;
  3. a batch of different lengths in one launch: a kernel that read the longest length for every row reads NaN pages —
     red;
  4. every declared tile candidate (each divides the page) gives the oracle's answer; a tile that does not divide the
     page is refused by name before any launch;
  5. three identical calls give identical bytes (the fixed-order reduce);
  6. the append writes each new position exactly at the slot the mapping names, and nowhere else (the untouched slots
     keep their NaN sentinel).
"""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

from nbxtensor.tensor import NBXTensor
from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.memory.allocator import DeviceAllocator
from nbxtensor.ops import wrappers as W
from nbxtensor.ops.paged_decode_attn import PAGED_DECODE_TILE_CANDIDATES

# fp16 K/V, fp32 accumulation over a few thousand positions: a correct kernel sits at a few 1e-4 relative (the
# contiguous vector decode kernel's oracle bound, tests/unit/kernels/test_decode_vec_oracle.py).
BOUND = 2e-03

_NP = {NBXDtype.float16: (np.float16, 2), NBXDtype.float32: (np.float32, 4), NBXDtype.int32: (np.int32, 4)}


def _has_gpu() -> bool:
    try:
        DeviceAllocator.set_device(0)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_gpu(), reason="needs a device")


def _d2h(t):
    dt, sz = _NP[t._dtype]
    n = t.numel()
    buf = (ctypes.c_char * (n * sz))()
    DeviceAllocator.memcpy(ctypes.addressof(buf), t.data_ptr(), n * sz, kind=2)
    return np.frombuffer(bytes(buf), dtype=dt).copy()


def _t(x):
    return NBXTensor.from_numpy(np.ascontiguousarray(x))


def _ref(q, ks, vs, scale):
    """float64 decode attention of each sequence over its own positions; q [B, H, D], ks[b] [H_kv, T_b, D]."""
    out = []
    for b in range(q.shape[0]):
        k, v = ks[b].astype(np.float64), vs[b].astype(np.float64)
        g = q.shape[1] // k.shape[0]
        K, V = np.repeat(k, g, axis=0), np.repeat(v, g, axis=0)
        s = np.einsum("hd,htd->ht", q[b].astype(np.float64), K) * scale
        e = np.exp(s - s.max(-1, keepdims=True))
        out.append(np.einsum("ht,htd->hd", e / e.sum(-1, keepdims=True), V))
    return np.stack(out)


def _paged(lens, h_kv, d, page_tokens, seed):
    """Sequences of the given lengths in a pool whose pages are granted in shuffled order; every page no sequence holds,
    and every position past a sequence's length in its last page, is NaN."""
    rng = np.random.default_rng(seed)
    pages_per = [-(-n // page_tokens) for n in lens]
    num_pages = sum(pages_per) + 3                      # spare pages no table names
    order = rng.permutation(num_pages)
    kpool = np.full((num_pages, h_kv, page_tokens, d), np.nan, dtype=np.float16)
    vpool = np.full((num_pages, h_kv, page_tokens, d), np.nan, dtype=np.float16)
    width = max(pages_per)
    table = np.full((len(lens), width), -1, dtype=np.int32)
    ks, vs, i = [], [], 0
    for b, n in enumerate(lens):
        k = rng.standard_normal((h_kv, n, d)).astype(np.float16)
        v = rng.standard_normal((h_kv, n, d)).astype(np.float16)
        ks.append(k); vs.append(v)
        for p in range(pages_per[b]):
            page = int(order[i]); i += 1
            table[b, p] = page
            lo, hi = p * page_tokens, min((p + 1) * page_tokens, n)
            kpool[page, :, : hi - lo] = k[:, lo:hi]
            vpool[page, :, : hi - lo] = v[:, lo:hi]
    return kpool, vpool, table, ks, vs


def _read(lens, h, h_kv, d, page_tokens, block_n, seed=0):
    kpool, vpool, table, ks, vs = _paged(lens, h_kv, d, page_tokens, seed)
    q = np.random.default_rng(seed + 1).standard_normal((len(lens), h, d)).astype(np.float16)
    out = W.paged_decode_attention(_t(q.reshape(len(lens), h, 1, d)), _t(kpool), _t(vpool), _t(table),
                                   _t(np.asarray(lens, dtype=np.int32)), 1.0 / np.sqrt(d), max(lens), block_n)
    got = _d2h(out).astype(np.float64).reshape(len(lens), h, d)
    ref = _ref(q, ks, vs, 1.0 / np.sqrt(d))
    return got, ref


@pytest.mark.parametrize("lens,h,h_kv,d,page_tokens", [
    ([1000], 32, 8, 128, 64),            # GQA 4, a partial last page
    ([4165], 32, 8, 128, 128),           # a long row, several splits
    ([7, 300, 1025], 16, 2, 64, 64),     # a batch of different lengths, 8:1 GQA
    ([64, 128], 20, 20, 64, 64),         # MHA, whole pages only
    ([513], 32, 8, 127, 64),             # odd head dim
])
def test_the_paged_read_is_the_attention_over_the_sequences_own_positions(lens, h, h_kv, d, page_tokens):
    for block_n in PAGED_DECODE_TILE_CANDIDATES:
        if page_tokens % block_n:
            continue
        got, ref = _read(lens, h, h_kv, d, page_tokens, block_n)
        assert np.isfinite(got).all(), f"tile {block_n}: a NaN page or position was read"
        err = float(np.abs(got - ref).max() / np.abs(ref).max())
        assert err < BOUND, f"tile {block_n}: relative error {err:.2e} against float64"


def test_a_tile_that_does_not_divide_the_page_is_refused_before_any_launch():
    kpool, vpool, table, _, _ = _paged([40], 8, 64, 48, 0)
    q = _t(np.zeros((1, 8, 1, 64), dtype=np.float16))
    with pytest.raises(ValueError, match="straddle two pages"):
        W.paged_decode_attention(q, _t(kpool), _t(vpool), _t(table), _t(np.asarray([40], dtype=np.int32)),
                                 0.125, 40, 32)


def test_three_identical_reads_give_identical_bytes():
    kpool, vpool, table, _, _ = _paged([2000, 77], 8, 128, 64, 3)
    q = _t(np.random.default_rng(9).standard_normal((2, 32, 1, 128)).astype(np.float16))
    args = (_t(kpool), _t(vpool), _t(table), _t(np.asarray([2000, 77], dtype=np.int32)), 0.088, 2000, 32)
    outs = [_d2h(W.paged_decode_attention(q, *args)).tobytes() for _ in range(3)]
    assert outs[0] == outs[1] == outs[2]


def test_the_append_writes_exactly_the_slots_the_mapping_names():
    num_pages, h_kv, page_tokens, d = 6, 4, 32, 64
    pool = _t(np.full((num_pages, h_kv, page_tokens, d), np.nan, dtype=np.float16))
    rng = np.random.default_rng(5)
    slots = np.array([5 * 32 + 31, 0, 2 * 32 + 7, 3 * 32], dtype=np.int32)   # out of order, across pages
    new = rng.standard_normal((len(slots), h_kv, d)).astype(np.float32)   # cast to the pool's fp16 on store
    W.paged_kv_append(_t(new), pool, _t(slots))
    got = _d2h(pool).reshape(num_pages, h_kv, page_tokens, d)
    written = np.zeros((num_pages, page_tokens), dtype=bool)
    for i, s in enumerate(slots):
        p, pos = divmod(int(s), page_tokens)
        np.testing.assert_array_equal(got[p, :, pos], new[i].astype(np.float16))
        written[p, pos] = True
    by_slot = got.transpose(0, 2, 1, 3)                    # [page, position, head, D]
    assert np.isnan(by_slot[~written]).all(), "the append wrote a slot the mapping does not name"


def test_a_launch_sized_at_an_earlier_length_still_reads_every_position():
    """A replayed decode re-launches the kernel with the arguments its recorded step had: the length the host passed
    then (`max_seq_len`) is stale while the device's `seq_lens` has grown. The split count only moves every
    PAGED_DECODE_SPLIT_POSITIONS positions, so the read must size its splits from the device lengths — sized from the
    stale figure, the newest positions fell past the last split and a replayed TinyLlama stopped attending to them
    (2026-10-28). Red on a kernel that takes the split length from the host."""
    from nbxtensor.ops.paged_decode_attn import PAGED_DECODE_SPLIT_POSITIONS, PAGED_DECODE_MAX_SPLITS
    import math
    block_n, page_tokens, h, h_kv, d = 64, 64, 8, 2, 64

    def split_of(n):
        return max(1, min(math.ceil(max(n, 1) / PAGED_DECODE_SPLIT_POSITIONS), PAGED_DECODE_MAX_SPLITS))

    def covered(n):                         # what a split length sized at `n` covers
        s = split_of(n)
        return s * math.ceil(math.ceil(n / s) / block_n) * block_n

    T = 200                                    # a plan recorded early, one split, re-launched as the length grows
    stale = next(n for n in range(T - 1, 0, -1) if split_of(n) == split_of(T) and covered(n) < T)
    kpool, vpool, table, ks, vs = _paged([T], h_kv, d, page_tokens, 3)
    q = np.random.default_rng(4).standard_normal((1, h, d)).astype(np.float16)
    out = W.paged_decode_attention(_t(q.reshape(1, h, 1, d)), _t(kpool), _t(vpool), _t(table),
                                   _t(np.asarray([T], dtype=np.int32)), 1.0 / np.sqrt(d), stale, block_n)
    got = _d2h(out).astype(np.float64).reshape(1, h, d)
    ref = _ref(q, ks, vs, 1.0 / np.sqrt(d))
    err = float(np.abs(got - ref).max() / np.abs(ref).max())
    assert err < BOUND, f"sized at {stale} for {T} positions: relative error {err:.2e} — the newest positions unread"
