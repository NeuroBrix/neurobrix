"""Prompt-aware KV sizing — S1 finding TINYLLAMA-KVCAP (fixed 2026-09-01).

The run-mode KV budget (`max_tokens + prompt_margin`) was blind to the
actual prompt: any prompt beyond ~`prompt_margin` tokens overflowed at
prefill on BOTH engines while the model's window had plenty of room
(TinyLlama: 1,740-token prompt vs a 640-slot cache under a 2,048
window). Pins:

  1. Triton (the paged cache since the state engine's step 2): the
     pools, allocated AT the prefill where the prompt length is known,
     hold prompt + decode budget when the plan size cannot, bounded by
     the model window; short prompts hold exactly the plan size; a
     growth reallocates the pools and retires the replay plans recorded
     against the old addresses (the recorded-replay contract).
  2. A prompt at/beyond the window raises the loud model-limit error.
  3. Compiled: the factory raises the growth CEILING to the window
     while the initial allocation stays at the plan size — the
     wrapper's existing on-demand growth does the rest.
"""
from __future__ import annotations

import pytest

from neurobrix.triton.dtype import NBXDtype
from neurobrix.triton.paged_kv_cache import PagedTritonKVCache
import nbxtensor.memory.allocator as _nbx_allocator  # the module the moved names live in (M2 cut of tensor.py)


def _paged(**kw):
    kw.setdefault("page_tokens", 16)
    kw.setdefault("async_release", False)
    return PagedTritonKVCache(**kw)


def _cache(plan_len=640, window=2048, budget=512):
    return _paged(num_layers=2, num_kv_heads=2, k_head_dim=8, v_head_dim=8,
                  max_cache_len=plan_len, dtype=NBXDtype.float16,
                  window_ceiling=window, decode_budget=budget)


def test_short_prompt_allocates_exactly_the_plan_size() -> None:
    c = _cache()
    assert c.capacity_for(13) == 640          # 13+512 <= 640
    assert c.capacity_for(128) == 640         # boundary: 128+512 == 640


def test_long_prompt_sizes_to_need_bounded_by_window() -> None:
    c = _cache()
    assert c.capacity_for(600) == 1112        # 600+512, inside window
    assert c.capacity_for(1740) == 2048       # min(1740+512, window)


def test_window_is_the_models_own_limit() -> None:
    c = _cache()
    with pytest.raises(RuntimeError, match="context window"):
        c.capacity_for(2048)
    with pytest.raises(RuntimeError, match="context window"):
        c.capacity_for(5000)


def test_no_window_grows_to_need() -> None:
    c = _cache(window=0)
    assert c.capacity_for(1740) == 2252       # 1740+512, unbounded
    assert c.capacity_for(13) == 640


def test_no_budget_falls_back_to_minimal_fit() -> None:
    c = _cache(budget=0)
    assert c.capacity_for(13) == 640
    assert c.capacity_for(700) == 701         # first_len+1, minimal


def test_compiled_factory_raises_ceiling_keeps_initial() -> None:
    """Path-1 composition: window > plan -> max_len=window,
    initial=plan (allocation preserved); no window -> untouched."""
    from neurobrix.core.module.cache.factory import StateCacheFactory

    class _Plan:
        pass

    class _KVPlan:
        num_layers, num_kv_heads = 2, 2
        k_head_dim, v_head_dim = 8, 8
        max_cache_len = 640
        dtype = "float16"
        initial_cache_len = 0
        memory_bytes = 1 << 30

    class _Ctx:
        plan = _Plan()

    _Ctx.plan.kv_cache_plan = _KVPlan()
    lm = {"num_layers": 2, "num_heads": 2, "hidden_size": 16,
          "max_position_embeddings": 2048}
    # the card the cache executes on (its vendor file gives the page size and release rule)
    w = StateCacheFactory.create(_Ctx(), lm, "cuda", "float16", hardware=("nvidia", "volta"))
    assert w.cache.config.max_cache_len == 2048
    assert w.cache.config.initial_cache_len == 640

    _Ctx.plan.kv_cache_plan = _KVPlan()
    lm_nowin = {"num_layers": 2, "num_heads": 2, "hidden_size": 16}
    w2 = StateCacheFactory.create(_Ctx(), lm_nowin, "cuda", "float16", hardware=("nvidia", "volta"))
    assert w2.cache.config.max_cache_len == 640
    assert w2.cache.config.initial_cache_len == 0


# ---------------------------------------------------------------------------
# Serve-mode initial size + grow-by-replacement (2026-09-03, the serve
# prefill lever): Prism's serve plan sets `initial_cache_len = max_tokens +
# margin` and `max_cache_len` = the whole remaining VRAM. The triton cache
# allocated the ceiling up front (10.8 GB pinned by the first request on a
# 32 GB card, every later prefill at the memory edge); it now allocates the
# initial size and replaces a warm layer's buffers when a longer request
# arrives — the compiled cache's initial + growth semantics (R30).
# ---------------------------------------------------------------------------

def test_serve_initial_size_is_the_base_not_the_ceiling() -> None:
    c = _paged(num_layers=2, num_kv_heads=2, k_head_dim=8, v_head_dim=8,
               max_cache_len=110_000, dtype=NBXDtype.float16,
               window_ceiling=262_144, decode_budget=512,
               initial_cache_len=640)
    assert c.capacity_for(13) == 640              # short prompt: the plan's initial size
    assert c.capacity_for(8_300) == 8_812         # 8300 + 512: the need, not the ceiling
    assert c.capacity_for(120_000) == 120_512     # beyond the plan: the window still bounds


def test_run_mode_initial_zero_keeps_the_plan_size() -> None:
    c = _cache()                                     # initial_cache_len defaults to 0
    assert c.initial_cache_len == 0
    assert c.capacity_for(13) == 640


def _gpu() -> bool:
    nbx = pytest.importorskip("nbxtensor.tensor")
    set_device = _nbx_allocator.DeviceAllocator.set_device   # a renamed symbol raises here, loudly
    try:
        set_device(0)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _gpu(), reason="needs a CUDA device")
def test_warm_pools_grow_for_a_longer_request() -> None:
    import numpy as np
    from nbxtensor.tensor import NBXTensor
    c = _paged(num_layers=1, num_kv_heads=2, k_head_dim=8, v_head_dim=8,
               max_cache_len=4096, dtype=NBXDtype.float16,
               window_ceiling=8192, decode_budget=16, initial_cache_len=64)
    k = NBXTensor.from_numpy(np.ones((1, 2, 20, 8), np.float16))
    c.prefill(0, k, k)                                # request 1: 20 + 16 <= 64 → 64 positions
    pools1 = c.k_pools[0]
    assert c.capacity == 64 and c.regrowths == 0
    c.clear()
    k2 = NBXTensor.from_numpy(np.ones((1, 2, 300, 8), np.float16))
    c.prefill(0, k2, k2)                              # request 2: 300 + 16 > 64 → the pools grow
    assert c.capacity == 316 and c.regrowths == 1 and c.generation == 1
    assert c.k_pools[0].data_ptr() != pools1.data_ptr()
    pools2 = c.k_pools[0]
    c.clear()
    c.prefill(0, k, k)                                # request 3: 20 fits 316 → kept
    assert c.k_pools[0] is pools2 and c.regrowths == 1


@pytest.mark.skipif(not _gpu(), reason="needs a CUDA device for the layer buffers")
def test_replacement_retires_the_old_generation_replay_plans() -> None:
    """The plans dict is keyed by the FULL `replay.signature()` tuple (the
    owner contributions sit nested at sig[-2]); a buffer replacement bumps
    the cache generation once per request and retires every plan a
    sequence recorded under the previous generation — frozen plans AND
    pending VERIFY plans (slab retired, captured graph destroyed, entry
    gone) — while another cache's plans survive. Keys are built by the
    real `signature()` on a stub sequence, not hand-assembled."""
    from neurobrix.triton import replay as R

    class _Slab:
        def __init__(self): self.retired = False
        def retire(self): self.retired = True

    class _Graph:
        def __init__(self): self.destroyed = False
        def destroy(self): self.destroyed = True

    class _Owner:  # the registration contract an interceptor exposes
        def __init__(self, cache): self.cache = cache
        def replay_signature(self, funcs):
            return ("kv_decode_paged", 1, 1, self.cache._uid, self.cache.generation)
        def replay_advance(self): pass
        def replay_restore(self): pass
        def intercept(self, *a, **k): pass

    class _Seq:  # the fields signature() reads
        def __init__(self, owner):
            self._op_interceptors = {"aten::scaled_dot_product_attention": owner.intercept}
            self._op_uid_interceptors = {}
            self._ops = []
            self._replay_has_nondet = False
            self._num_weights, self._num_inputs = 0, 0
            self._arena = []
            self._symbol_resolver = None
            self._activations_fp16_safe = False

    c = _paged(num_layers=1, num_kv_heads=2, k_head_dim=8, v_head_dim=8,
               max_cache_len=4096, dtype=NBXDtype.float16,
               window_ceiling=8192, decode_budget=16, initial_cache_len=64)
    other = _paged(num_layers=1, num_kv_heads=2, k_head_dim=8, v_head_dim=8,
                   max_cache_len=4096, dtype=NBXDtype.float16, initial_cache_len=64)
    seq = _Seq(_Owner(c))
    sig_mine = R.signature(seq)
    assert sig_mine is not None and ("kv_decode_paged", 1, 1, c._uid, 0) in sig_mine[-2]
    sig_theirs = R.signature(_Seq(_Owner(other)))
    frozen = R.FrozenPlan.__new__(R.FrozenPlan)
    frozen.slab, frozen.graph = _Slab(), _Graph()
    pending = R.FrozenPlan.__new__(R.FrozenPlan)
    pending.slab, pending.graph = _Slab(), None
    theirs = R.FrozenPlan.__new__(R.FrozenPlan)
    theirs.slab, theirs.graph = _Slab(), None
    # a second bucket of MINE, still pending its verify pass
    sig_mine_512 = tuple(list(sig_mine[:-2]) + [(("kv_decode_paged", 2, 1, c._uid, 0),), sig_mine[-1]])
    seq.__dict__["_replay_plans"] = {sig_mine: frozen, sig_mine_512: ("VERIFY", pending),
                                     sig_theirs: theirs}
    R._PLAN_SEQS.add(seq)
    if not _gpu():
        pytest.skip("needs a CUDA device for the layer buffers")
    import numpy as np
    from nbxtensor.tensor import NBXTensor
    k = NBXTensor.from_numpy(np.ones((1, 2, 20, 8), np.float16))
    c.prefill(0, k, k)
    c.clear()
    k2 = NBXTensor.from_numpy(np.ones((1, 2, 300, 8), np.float16))
    c.prefill(0, k2, k2)                                     # growth → generation 1
    assert c.generation == 1
    plans = seq.__dict__["_replay_plans"]
    assert sig_mine not in plans and sig_mine_512 not in plans   # both buckets of the old generation retired
    assert frozen.slab.retired and frozen.graph is None
    assert pending.slab.retired                                  # the VERIFY plan's slab too
    assert sig_theirs in plans and not theirs.slab.retired       # another cache: untouched
    # the new generation signs differently: no stale key can match
    assert ("kv_decode_paged", 1, 1, c._uid, 1) in R.signature(seq)[-2]


def test_sequence_teardown_retires_all_plans() -> None:
    from neurobrix.triton import replay as R

    class _Slab:
        def __init__(self): self.retired = False
        def retire(self): self.retired = True

    class _Seq: pass
    seq = _Seq()
    p1 = R.FrozenPlan.__new__(R.FrozenPlan); p1.slab, p1.graph = _Slab(), None
    p2 = R.FrozenPlan.__new__(R.FrozenPlan); p2.slab, p2.graph = _Slab(), None
    seq.__dict__["_replay_plans"] = {("a",): p1, ("b",): ("VERIFY", p2), ("c",): "UNREPLAYABLE"}
    R._PLAN_SEQS.add(seq)
    assert R.retire_sequence_plans(seq) == 3
    assert p1.slab.retired and p2.slab.retired and not seq.__dict__["_replay_plans"]
    assert seq not in R._PLAN_SEQS


def test_warm_request_sizing_inputs_follow_the_request() -> None:
    """A warm memo hit refreshes decode_budget / window_ceiling from THIS
    request before clear() (gardien 2026-09-03): the replacement rule then
    measures the need with the current --max-tokens, not the first
    request's. Pinned on the cache object the flow mutates."""
    c = _paged(num_layers=1, num_kv_heads=2, k_head_dim=8, v_head_dim=8,
               max_cache_len=4096, dtype=NBXDtype.float16,
               window_ceiling=8192, decode_budget=16, initial_cache_len=64)
    assert c.capacity_for(40) == 64                    # 40 + 16 fits the initial size
    c.decode_budget, c.window_ceiling = 512, 8192        # what the memo-hit branch now does
    c.clear()
    assert c.capacity_for(40) == 552                   # 40 + 512: the request's own budget
