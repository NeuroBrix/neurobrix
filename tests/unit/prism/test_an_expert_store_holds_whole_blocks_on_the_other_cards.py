"""S3 (Prism's half): an expert store on a profile with other cards holds whole blocks' routed experts in those cards'
memory (`ComponentAllocation.expert_cards`), each card under its usable rung, the home cache serving only the blocks left
to it, and the plan re-timed with the held blocks' experts crossing no host link. Before: Qwen3-30B-A3B on the rack's
4 V100-16GB (c4140-4xv100-16GB-nvlink) served every routed expert from card 0's 3.7 GB cache, cards 1-3 idle, estimated
160 s (compiled) / 203 s (triton) (the Mac's probe on stable 58b39d7e, 2026-10-28 01:2x). One card: the plan as before.
No card."""
import pytest

from neurobrix.core.prism import PrismSolver
from neurobrix.core.prism.loader import load_profile
from tests.unit.prism._pinned_machine import V100_16GB, pin_dedicated_card, pin_host, profile
from tests.unit.prism.test_a_component_over_the_rung_streams_inside_itself import _input_config, no_door

MODEL = "Qwen3-30B-A3B-Thinking-2507"
BEFORE_S = {"compiled": 160.0, "triton": 203.0}


def _solve(monkeypatch, prof, mode):
    no_door(monkeypatch)
    pin_host(monkeypatch, 257530, 200000, "an idle rack host")
    pin_dedicated_card(monkeypatch, 16151, 306, "the rack's V100-16GB cards as measured")
    c, ic = _input_config(MODEL, mode)
    s = PrismSolver()
    return s.solve_smart(c, prof, ic, mode=mode, replan=False), s


@pytest.mark.parametrize("mode", ["compiled", "triton"])
def test_four_cards_hold_whole_blocks_of_experts_and_the_plan_is_faster(monkeypatch, mode):
    prof = load_profile("c4140-4xv100-16GB-nvlink")
    p, s = _solve(monkeypatch, prof, mode)
    store = p.components["model"]
    assert store.strategy == "expert_store", store.strategy
    cards = store.expert_cards
    assert len(cards) >= 2 and store.device not in cards, cards
    devs = {d.device_string: d for d in s._prepare_devices(prof)}
    reserve = s._live_activation_mb(p.component_memory["model"])
    for card, held in cards.items():
        assert held["cache_mb"] + reserve <= s._usable_mb(devs[card]), (card, held)
    blocks = [b for c in cards.values() for b in range(c["blocks"][0], c["blocks"][1] + 1)]
    assert len(blocks) == len(set(blocks)), cards                     # no block on two cards
    assert p.estimated_s is not None and p.estimated_s < BEFORE_S[mode], (p.estimated_s, p.selection_reason[-300:])


def test_one_card_keeps_one_store(monkeypatch):
    p, _ = _solve(monkeypatch, profile(V100_16GB), "compiled")
    a = p.components.get("model")
    assert a is not None and not a.expert_cards, (p.strategy, a and a.expert_cards)


def test_the_cards_travel_with_the_plan(tmp_path):
    from neurobrix.core.prism.solver import ComponentAllocation, ExecutionPlan
    cards = {"cuda:1": {"blocks": [40, 47], "cache_mb": 9000.0, "experts": 1024}}
    alloc = ComponentAllocation(name="model", devices=["cuda:0"], dtype="float16", memory_mb=1.0, architecture="volta",
                                vendor="nvidia", strategy="expert_store", expert_cards=cards)
    plan = ExecutionPlan(components={"model": alloc}, target_dtype="float16", total_memory_mb=1.0, component_memory={},
                         strategy="lazy_sequential")
    plan.save(tmp_path / "plan.json")
    assert ExecutionPlan.load(tmp_path / "plan.json").components["model"].expert_cards == cards
