"""A streamed LLM's plan — segments, what stays beside them, their constants AND the KV cache —
fits the usable part of the rung it was cut against.

Three defects, one plan. On main (fa6e13d2), the Mac's profile, host idle at 18 186 MB free:

* the post-scoring KV check judged `layer_streaming` against the SUM of every device's raw
  capacity, not the figure its segments were cut against, and subtracted the KV estimate from a
  streamed segment peak that never contained it. The cache was then sized from memory the rung
  does not grant:
      DeepSeek-Coder-V2-Lite, serve   13 598.8 + KV 3 846.4 = 17 445 MB
      Qwen3-Coder-30B-A3B,   serve   11 966.3 + KV 8 394.4 = 20 361 MB
  against a 16 384 MB rung, 15 073.3 usable, 17 276.7 capacity — past the device itself;
* the segment reserve was the RUN-mode KV estimate while a serve plan demands two turns
  (`_kv_min_tokens`), so closing the first defect alone REFUSED Qwen3 in serve mode;
* the reserve's dtype was `getattr(self, "_target_dtype_str", "float16")` — invented when
  absent — and after the fp32 fallback `solve` updated a local, not the attribute, so the
  re-evaluated strategies reserved an fp32 cache at 2 bytes.

The machine is built (register 102): the Mac's profile and its idle reading, no door.
"""
from __future__ import annotations

import pytest

from neurobrix.core.prism import InputConfig, PrismSolver
from neurobrix.nbx import NBXContainer
from tests.unit.prism._pinned_machine import (APPLE_M4_PRO, V100_16GB, container_root, no_door,
                                              pin_host, pin_shared_card, profile)

MB = 1024 * 1024
LLMS = ["DeepSeek-Coder-V2-Lite-Instruct", "Qwen3-Coder-30B-A3B-Instruct"]


def _solve(monkeypatch, model, serve):
    no_door(monkeypatch)
    pin_host(monkeypatch, 24576, 18186, "the Mac, idle")
    s = PrismSolver()
    seen, evals = {}, []
    real_mem, real_eval = s._compute_memory, s._evaluate_all_strategies

    def spy_mem(*a, **k):
        out = real_mem(*a, **k)
        seen.update(out)
        return out

    def spy_eval(*a, **k):
        evals.append(a)
        return real_eval(*a, **k)

    s._compute_memory = spy_mem
    s._evaluate_all_strategies = spy_eval
    c = NBXContainer.load(str(container_root(model)))
    p = s.solve_smart(c, profile(APPLE_M4_PRO), InputConfig(batch_size=1), serve_mode=serve,
                      mode="triton")
    return p, s, seen, c, evals


@pytest.mark.parametrize("serve", [False, True], ids=["run", "serve"])
@pytest.mark.parametrize("model", LLMS)
def test_segments_resident_constants_and_cache_fit_the_usable_rung(monkeypatch, model, serve):
    p, s, seen, c, _ = _solve(monkeypatch, model, serve)
    assert p.strategy == "layer_streaming", (
        f"{model} ({'serve' if serve else 'run'}) planned {p.strategy!r}; this cell judges a "
        f"streamed plan — a refusal here is the second defect back")
    parts = s._layer_stream_partitions
    dev = s._prepare_devices(profile(APPLE_M4_PRO))[0]
    # Precondition, so this oracle never has to copy the solver's subtraction rule: the LM is the
    # streamed component, whose segment peak never held the cache; what stays beside it is whole.
    assert s._lm_component_name in parts, (s._lm_component_name, list(parts))
    beside = sum(m.total_bytes for n, m in seen.items() if n not in parts)
    held = (sum(q.peak_resident_bytes for q in parts.values()) + beside
            + s._layer_stream_constant_bytes + p.kv_cache_plan.memory_bytes) / MB
    assert held <= s._usable_mb(dev) + 1e-6, (
        f"{model}: segments + resident + constants + KV = {held:.1f} MB over the usable "
        f"{s._usable_mb(dev):.1f} MB of the rung the segments were cut against")


def test_a_served_llm_on_a_shared_card_sizes_its_cache_on_the_rung_not_the_capacity(monkeypatch):
    """The KV check's non-streaming branches judged against the SUM of raw capacity. On a card
    another process holds 3 000 MB of, the rung sits well under the capacity, and a served cache —
    which takes every byte of `remaining` — was sized from the difference.

    The bound is independent of the solver's own cost arithmetic: whatever else it holds, every
    resident component holds at least its activations (the LM's without the KV estimate the check
    sizes itself), so cache + those activations must fit the rung."""
    no_door(monkeypatch)
    pin_host(monkeypatch, 257530, 200000, "an idle rack host")
    pin_shared_card(monkeypatch, 16151, 3000, 306, "a V100-16GB another process holds 3 000 MB of")
    s = PrismSolver()
    c = NBXContainer.load(str(container_root("DeepSeek-Coder-V2-Lite-Instruct")))
    p = s.solve_smart(c, profile(V100_16GB), InputConfig(batch_size=1), serve_mode=True, mode="triton")
    # The branch under test is the KV check's GENERAL one (sum of the strategy's device rungs);
    # lazy_sequential is what wins here. One device, so the sum over devices IS this card's rung —
    # a multi-card host (where the sum overstates the LM's own card) is not covered by this cell.
    # Seen failing with raw capacity injected into that branch (2026-09-24, injection round 4).
    assert p.strategy == "lazy_sequential", p.strategy
    dev = s._prepare_devices(profile(V100_16GB))[0]
    rung = s._effective_capacity_mb(dev)
    assert rung < dev.capacity_mb - 2000, ("precondition: rung well under capacity", rung, dev.capacity_mb)
    kv_est = s._estimate_kv_cache_bytes(c, s._target_dtype_str)
    floor = sum(m.activation_bytes - (kv_est if n == s._lm_component_name else 0)
                for n, m in p.component_memory.items())
    held = (floor + p.kv_cache_plan.memory_bytes) / MB
    assert held <= rung + 1e-6, (f"{p.strategy}: cache {p.kv_cache_plan.memory_bytes / MB:.1f} MB + "
                                 f"activations {floor / MB:.1f} MB = {held:.1f} over the {rung:.1f} rung")


def test_the_reserve_refuses_to_invent_a_dtype(monkeypatch):
    """With no dtype decided, the reserve is REFUSED — it was sized at an invented float16."""
    _, s, _, c, evals = _solve(monkeypatch, LLMS[0], False)
    sorted_comps, comp_mem, devices, shard_sizes, prof, container = evals[0][1:7]
    del s._target_dtype_str
    with pytest.raises(RuntimeError, match="ZERO FALLBACK"):
        s._try_layer_streaming(sorted_comps, comp_mem, devices, shard_sizes, prof, container)


def test_after_the_fp32_fallback_the_reserve_and_the_plan_read_fp32(monkeypatch):
    """Force the fallback path: the first evaluation yields nothing, the fallback applies. Every
    KV figure taken after it must be taken at float32."""
    no_door(monkeypatch)
    pin_host(monkeypatch, 24576, 18186, "the Mac, idle")
    s = PrismSolver()
    calls = {"n": 0}
    real_eval = s._evaluate_all_strategies

    def first_empty(*a, **k):
        calls["n"] += 1
        return [] if calls["n"] == 1 else real_eval(*a, **k)

    dtypes = []
    real_est, real_min, real_plan = s._estimate_kv_cache_bytes, s._kv_min_bytes, s._compute_kv_cache_plan
    s._evaluate_all_strategies = first_empty
    s._try_fp32_fallback = lambda *a, **k: True
    s._estimate_kv_cache_bytes = lambda c, d: (dtypes.append(("estimate", calls["n"], d)), real_est(c, d))[1]
    s._kv_min_bytes = lambda c, d: (dtypes.append(("minimum", calls["n"], d)), real_min(c, d))[1]
    s._compute_kv_cache_plan = lambda c, d, r: (dtypes.append(("plan", calls["n"], d)), real_plan(c, d, r))[1]
    s.solve_smart(NBXContainer.load(str(container_root(LLMS[0]))), profile(APPLE_M4_PRO),
                  InputConfig(batch_size=1), mode="triton")
    after = [(what, d) for what, n, d in dtypes if n >= 2]
    assert any(w == "minimum" for w, _ in after) and any(w == "plan" for w, _ in after), dtypes
    assert all(d == "float32" for _, d in after), after
