"""S3 (Prism's half, the supervisor 2026-10-28 06:12 point 2): an expert store's home cache is priced by the routing's
reuse curve — a measured fact of the model (`expert_topology.routing_reuse`, the artifact's profile.json) — its misses
crossing the host link at (1 - hit(slots per layer)) x top_k x expert bytes per store layer per token; the home card
holds whole blocks only where that lowers the bytes missed. Without the curve the plan is not timed, said by name — no
guess. The Dell's curve for Qwen3-30B-A3B (2026-10-28): 8 -> 0.30, 16 -> 0.53, 32 -> 0.74, 64 -> 0.93, 96 -> 0.96; with
blocks held on the home unpriced its 4 V100 decode went 381 -> 468 ms. No card."""
import pytest

import neurobrix.core.prism.expert_topology as ET
from neurobrix.core.prism import PrismSolver
from neurobrix.core.prism.loader import load_profile
from tests.unit.prism._pinned_machine import pin_dedicated_card, pin_host
from tests.unit.prism.test_a_component_over_the_rung_streams_inside_itself import _input_config, no_door

MODEL = "Qwen3-30B-A3B-Thinking-2507"
CURVE = [(8.0, 0.30), (16.0, 0.53), (32.0, 0.74), (64.0, 0.93), (96.0, 0.96)]


def _solve(monkeypatch, curve):
    no_door(monkeypatch)
    pin_host(monkeypatch, 257530, 200000, "an idle rack host")
    pin_dedicated_card(monkeypatch, 16151, 306, "the rack's V100-16GB cards as measured")
    monkeypatch.setattr(ET, "routing_reuse", lambda cache_path, comp: curve)
    c, ic = _input_config(MODEL, "triton")
    s = PrismSolver()
    return s.solve_smart(c, load_profile("c4140-4xv100-16GB-nvlink"), ic, mode="triton", replan=False), s


def test_the_hit_rate_reads_the_curve():
    assert ET.hit_rate(CURVE, 0) == 0.0
    assert ET.hit_rate(CURVE, 4) == pytest.approx(0.15)
    assert ET.hit_rate(CURVE, 24) == pytest.approx(0.635)
    assert ET.hit_rate(CURVE, 500) == 0.96


def test_with_the_curve_the_store_is_timed_and_the_home_keeps_the_cache_that_misses_least(monkeypatch):
    p, s = _solve(monkeypatch, CURVE)
    store = p.components["model"]
    assert isinstance(p.estimated_s, float), p.selection_reason[-300:]
    topo = s._expert_topology["model"]
    held = {b for c in store.expert_cards.values() for b in range(c["blocks"][0], c["blocks"][1] + 1)}
    left = [l for l in topo.layers if l.block not in held]
    slot_mb = left[0].expert_bytes / 2 ** 20 * s._expert_mult["model"]
    slots = store.expert_cache_mb / slot_mb / len(left)
    # at the Dell's 444 slots over 14 layers (~32 a layer) the curve gives 0.74; holding blocks on the home must not
    # leave it with fewer slots a layer than the curve says pay for themselves: no home block below 16 slots a layer
    assert slots >= 16 or store.device not in store.expert_cards, (slots, store.expert_cards)


def test_without_the_curve_the_plan_says_what_it_lacks(monkeypatch):
    p, _ = _solve(monkeypatch, None)
    assert p.estimated_s is None and "routing's reuse curve" in p.selection_reason, p.selection_reason[-300:]
    assert p.components["model"].device not in p.components["model"].expert_cards
