"""The paged decode's autotune key moves exactly when the replayed decode's bucket does: both are the splits the longest
sequence makes (`paged_decode_attn.paged_decode_splits`).

The key was the length's own bucket (`bucket_of("N", ...)`, exact below 64 positions, 16 or 32 apart above): a decode
step replayed in the bucket of its split count met, at the verify pass, a launch the certificate keyed at another
length — another configuration, other bytes — and the plan was rejected; TinyLlama's decode stopped replaying, 25 -> 135
ms a token (2026-11-04).

What this test would do if the code were wrong: the key back on the length's bucket — the key changes between two
lengths of one split (red at the first assert); the replay bucket computed by another rule — red at the second."""
from __future__ import annotations

import types

from nbxtensor.dtype.nbx_dtype import NBXDtype as D
from nbxtensor.ops import launch_keys as LK
from nbxtensor.ops.paged_decode_attn import PAGED_DECODE_MAX_SPLITS, PAGED_DECODE_SPLIT_POSITIONS, paged_decode_splits


def _key(tk):
    (_q, key), _reduce = LK.paged_decode_launches(32, 4, tk, 64, 64, D.float16, D.float16, 64)
    return key


def test_the_key_is_constant_inside_a_split_and_moves_at_its_boundary():
    for tk in range(1, 3 * PAGED_DECODE_SPLIT_POSITIONS):
        same = paged_decode_splits(tk) == paged_decode_splits(tk + 1)
        assert (_key(tk) == _key(tk + 1)) == same, tk
    assert _key(10 ** 6)[0] == PAGED_DECODE_MAX_SPLITS


def test_the_replay_bucket_is_the_keys_split():
    from neurobrix.triton.paged_kv_cache import PagedTritonKVCache
    for length in (0, 1, 63, 255, 256, 257, 4000, 70000):
        cache = types.SimpleNamespace(pool=object(), tables=[types.SimpleNamespace(pages=[0] * 2000)],
                                      page_tokens=64, length=length, batch=1, _uid=7, generation=0)
        k = PagedTritonKVCache.replay_key(cache, 0)
        assert k is not None and k[1] == _key(length + 1)[0], (length, k)
