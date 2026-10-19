"""The Triton attention interceptor's decode mode starts at the cache's length (mirror of the compiled wrapper):
openaudio's first cached token on the Triton engine was rotated at position 0 (2026-10-19, 4 % off the oracle at step 1,
the compiled step bit-identical) because `set_decode_mode` left the positional offset at 0 and the loop moved it only
from the second step on. What this test would do if the code were wrong: the offset read 0 after the prefill (red)."""
from __future__ import annotations

import pytest


def test_decode_mode_starts_at_the_caches_length():
    pytest.importorskip("triton")
    from neurobrix.triton.kv_cache import TritonAttentionInterceptor

    class _Cache:
        num_kv_heads = 1
        def get_cache_len(self):
            return 18
    it = TritonAttentionInterceptor.__new__(TritonAttentionInterceptor)
    it.cache, it._is_prefill, it._call_count, it._position_offset = _Cache(), True, 3, 0
    it.set_decode_mode()
    assert it._is_prefill is False and it._call_count == 0
    assert it.get_cache_len() == 18 and it._position_offset == 18
