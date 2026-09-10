"""Two decode contexts on one LM: the attention interceptors of both engines
swap their per-sequence state (cache, prefill flag, call count, position
offset) between branches, and a new branch is an empty cache with the same
geometry. VibeVoice's next-token diffusion keeps its CFG negative context
as a second branch (the decoder-plan lever, 2026-09-06)."""
from __future__ import annotations

from neurobrix.triton.kv_cache import KVBranch, TritonAttentionInterceptor


class _Cache:
    def __init__(self, tag):
        self.tag = tag
        self.num_kv_heads = 4
        self.num_layers = 2

    def clone_empty(self):
        return _Cache(self.tag + "'")

    def get_seq_len(self):
        return 0


def test_branches_swap_the_interceptor_state_and_keep_each_others():
    it = TritonAttentionInterceptor(cache=_Cache("pos"), num_heads=8)
    pos = it.branch_state()
    assert pos.cache.tag == "pos" and pos.is_prefill is True
    it._is_prefill = False
    it._call_count = 7
    it._position_offset = 42
    neg = it.new_branch()
    assert neg.cache.tag == "pos'" and neg.is_prefill is True and neg.position_offset == 0
    it.use_branch(neg)
    assert it.cache.tag == "pos'" and it._is_prefill is True and it._call_count == 0 and it._position_offset == 0
    assert pos.is_prefill is False and pos.call_count == 7 and pos.position_offset == 42   # saved on the way out
    it._is_prefill = False
    it._position_offset = 3
    it.use_branch(pos)
    assert it.cache.tag == "pos" and it._position_offset == 42 and it._call_count == 7
    assert neg.position_offset == 3 and neg.is_prefill is False


def test_compiled_wrapper_has_the_same_surface():
    from neurobrix.core.runtime.graph import kv_cache_wrapper as K
    assert hasattr(K.KVCacheAttentionWrapper, "branch_state") and hasattr(K.KVCacheAttentionWrapper, "use_branch")
    assert hasattr(K.KVCacheAttentionWrapper, "new_branch") and K.KVBranch.__slots__[:4] == KVBranch.__slots__
