"""An expert store leaves its card ONE drain cliff: the profile's, priced once inside the usable part.

The Triton hot loop's dead-tensor queue is priced inside `_usable_mb` (net of the profile's cliff) and the plan gives
each card the cliff its held components leave (`_assign_drain_cliffs`). An expert store was priced there at its whole
weights — every routed expert, which stays in pinned host memory or in the container's pages — so its card got a cliff
of 0 (Qwen3-Omni-30B 32g triton, 2026-10-13); the grant then set a second cliff aside, and the served KV check, sized
against the usable part, took the store's floor: DeepSeek-Coder-V2-Lite and Qwen3-Coder-30B-A3B served on the Mac's
idle profile descended the ladder (16 384 -> 8 192) and streamed (d364e04fe, bisected 2026-10-15).

What this test does if the code is wrong: the cliff cell fails when a store's card drains at less than the profile's
cliff (the store priced at its whole weights), and the rung cell fails when a served store is not held at the first
rung (the cliff set aside twice).

Run: python -m pytest tests/unit/prism/test_an_expert_store_leaves_its_card_one_drain_cliff.py
"""
from __future__ import annotations

import pytest

from neurobrix.core.config.loader import get_vendor_config
from neurobrix.core.prism.memory_budget import deferred_drain_policy_of
from tests.unit.prism._pinned_machine import APPLE_M4_PRO, profile
from tests.unit.prism.test_a_streamed_llm_keeps_its_cache_inside_the_rung import LLMS, _solve


@pytest.mark.parametrize("model", LLMS)
def test_a_served_store_is_held_at_the_first_rung_and_its_card_keeps_the_profiles_cliff(monkeypatch, model):
    p, s, seen, c, _ = _solve(monkeypatch, model, True)
    lm = p.components[s._lm_component_name]
    assert lm.strategy == "expert_store", (model, p.strategy, lm.strategy, p.unified_rungs_tried)
    dev = s._prepare_devices(profile(APPLE_M4_PRO))[0]
    spec = APPLE_M4_PRO["devices"][0]
    cliff = deferred_drain_policy_of(get_vendor_config(spec["brand"], spec["architecture"])["memory"])[0]
    assert lm.deferred_drain_cliff_bytes == cliff, (
        f"{model}: the store's card drains at {lm.deferred_drain_cliff_bytes} B, not the profile's {cliff} B — "
        f"the store priced at its whole weights left no room beside it")
    refused = [r for r in (p.unified_rungs_tried or []) if r[1] == "refused"]
    assert not refused, f"{model}: the served plan descended the ladder: {p.unified_rungs_tried}"
