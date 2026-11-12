"""A served decode cache and the Triton hot loop's drain queue never share a byte of the card.

On the Triton engine a card holding its components WHOLE leaves the hot loop's dead-tensor queue the room beside them
(`_assign_drain_cliffs`), and a served cache takes every byte the KV check leaves. Sized against the whole rung, the
cache took the room the queue had been promised, and the queue's cliff was then set beside the largest component alone,
without the cache: deepseek-moe-16b-chat served on one V100-32GB, 31 376 MB held whole + a session reach of 2 126
positions (930 MB) + a 725 MB cliff on a 32 195 MB card — request 2 out of memory at 31 858 MB live, the conversation
0 of 5 turns (2026-11-12, since 0.6.8 at least).

What this test does if the code is wrong: it fails when what the plan holds on the cache's card — its whole components,
or an expert store's dense part and cache, the cache as planned, and the drain cliff the card is given — exceeds the
card's rung (seen red on main 74ad36bab: deepseek-moe-16b-chat 31 246 + 454 + 722 = 32 422 MB over 31 701; DeepSeek-Coder-V2-Lite
31 050 + 524 + 763 = 32 337 over 31 574).

Run: python -m pytest tests/unit/prism/test_a_served_cache_and_the_drain_never_share_a_byte.py
"""
from __future__ import annotations

import pytest

from neurobrix.core.prism import InputConfig, PrismSolver
from neurobrix.nbx import NBXContainer
from tests.unit.prism._pinned_machine import (V100_32GB, container_root, no_door, pin_host, pin_no_calibration,
                                              pin_shared_card, profile)

MB = 1024 * 1024


#: the Triton runtime's growth the rack's engine store measured for each model on Volta (MB, 2026-11-12): the room it
#: leaves is what lets the model be held whole — under the vendor's unmeasured default (911 MB) it is not
GROWTH = {"deepseek-moe-16b-chat": 494.2, "DeepSeek-Coder-V2-Lite-Instruct": 620.8}


@pytest.mark.parametrize("model", sorted(GROWTH))
def test_the_cache_the_components_and_the_drain_fit_the_cards_rung(monkeypatch, model):
    from neurobrix.core.runtime import runtime_growth
    monkeypatch.setattr(runtime_growth, "load", lambda m, engine, arch: GROWTH.get(m) if engine == "triton" else None)
    no_door(monkeypatch)
    pin_no_calibration(monkeypatch)
    pin_host(monkeypatch, 257530, 200000, "an idle rack host")
    pin_shared_card(monkeypatch, 32501, 0, 306, "an idle V100-32GB")
    s = PrismSolver()
    seen = {}
    real = s._compute_memory

    def spy(*a, **k):
        out = real(*a, **k)
        seen.update(out)
        return out
    s._compute_memory = spy
    c = NBXContainer.load(str(container_root(model)))
    p = s.solve_smart(c, profile(V100_32GB), InputConfig(batch_size=1), serve_mode=True, mode="triton")
    dev = s._prepare_devices(profile(V100_32GB))[0]
    lm = s._lm_component_name
    kv_est = s._estimate_kv_cache_bytes(c, s._target_dtype_str)
    held = 0
    for name, a in p.components.items():
        if str(a.device) != dev.device_string:
            continue
        m = seen[name]
        if a.strategy in ("expert_store", "zero3"):
            held += m.activation_bytes + m.overhead_bytes + (a.resident_weight_mb + a.expert_cache_mb) * MB
        else:
            held += m.total_bytes
        if name == lm:
            held -= kv_est                      # the cache is counted once, as planned, below
    cliff = (p.deferred_drain_cliffs or {}).get(dev.device_string, 0)
    total = (held + p.kv_cache_plan.memory_bytes + cliff) / MB
    assert total <= s._rung_part_mb(dev) + 1e-6, (
        f"{model} served on one V100-32GB: held {held / MB:.0f} + cache {p.kv_cache_plan.memory_bytes / MB:.0f} "
        f"+ drain cliff {cliff / MB:.0f} = {total:.0f} MB over the card's {s._rung_part_mb(dev):.0f} MB "
        f"({p.strategy}, {p.components[lm].strategy})")
