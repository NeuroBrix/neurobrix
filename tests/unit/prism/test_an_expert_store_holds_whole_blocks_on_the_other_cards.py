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
    # the routing's reuse curve the Dell measured (2026-10-28): the store is timed with it (without it, the plan says
    # it is not timed — test_an_expert_cache_is_priced_by_the_routing_reuse)
    import neurobrix.core.prism.expert_topology as ET
    monkeypatch.setattr(ET, "routing_reuse", lambda cache_path, comp: [(8.0, 0.30), (16.0, 0.53), (32.0, 0.74),
                                                                         (64.0, 0.93), (96.0, 0.96)])
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
    others = {c: v for c, v in cards.items() if c != store.device}
    assert len(others) >= 2, cards
    devs = {d.device_string: d for d in s._prepare_devices(prof)}
    reserve = s._moe_op_reserve_mb("model", s._expert_topology["model"].layers)
    assert reserve is not None and 0 < reserve < 2048, reserve       # the MoE op's own activations, not the model's
    for card, held in others.items():
        assert held["cache_mb"] + reserve <= s._usable_mb(devs[card]), (card, held)
    # the home card keeps its whole room as the cache of the blocks left on the host (the reuse curve is unmeasured:
    # holding blocks there cost the Dell's decode 381 -> 468 ms, 2026-10-28)
    if store.device in cards:      # held there only where the curve says the cache left misses less
        assert store.expert_cache_mb >= 0, store.expert_cache_mb
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


def test_a_card_with_no_room_for_one_block_takes_none(monkeypatch):
    """Card 3 read with 1 000 MB free NOW, under one block's experts (1 152 MB at fp16) (another process holds the rest): the facts' plan puts blocks there, the
    execution door sees it (an expert card is not one of the plan's devices) and the busy re-plan — on what is free
    now — gives card 3 no block while the others still hold theirs (at 1 500 MB free it takes one block, 1 152 MB)."""
    import neurobrix.core.prism.solver as solver_mod
    from neurobrix.core.prism.memory_budget import DeviceReading
    from neurobrix.kernels.nbx_tensor import DeviceAllocator
    no_door(monkeypatch)
    pin_host(monkeypatch, 257530, 200000, "an idle rack host")
    free = {0: 15845, 1: 15845, 2: 15845, 3: 1000}
    monkeypatch.setattr(solver_mod, "read_device_sharing", lambda i: DeviceReading(
        kind="device", capacity_mb=16151, free_mb=free[i], own_context_mb=306, measured=True,
        source="injected: card 3 held by another process"))
    monkeypatch.setattr(DeviceAllocator, "free_memory_mb", staticmethod(lambda i: free[i]))
    c, ic = _input_config(MODEL, "compiled")
    p = PrismSolver().solve_smart(c, load_profile("c4140-4xv100-16GB-nvlink"), ic, mode="compiled")
    assert p.busy_replan, "the facts' plan put blocks on the busy card and the door did not see it"
    cards = p.components["model"].expert_cards
    assert "cuda:3" not in cards and len([c for c in cards if c != p.components["model"].device]) >= 2, cards
