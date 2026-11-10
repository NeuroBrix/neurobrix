"""A routed language model on two V100-32GB holds its experts on the cards, served or run, and a card the profile
does not describe is refused by name (2026-11-10, the owner's order on the slowness of Qwen3-Coder-30B-A3B served:
6.7 s a token, every block streamed from the host at every step).

Three defects made that plan: the two-card profile carried no runtime facts and the untouched second card was
budgeted its whole pool (the plan overflowed it once its context existed, the door re-planned to a fallback); a
served session's KV cache was sized against the card's budget while the expert store is granted from its usable part
(the cache took the store's floor); weight_sharding sized its split against the raw budget and stacked the head on a
card already holding half the model (out of memory in its first prefill).

What these tests would do if the code were wrong: the served plan comes back `zero3` with no expert held on a card —
RED; the short run comes back `weight_sharding` with a card over its usable part — RED; a card without its runtime
facts plans as if its whole pool were free — RED (no refusal)."""
import copy

import pytest

from neurobrix.core.prism import InputConfig, PrismSolver
from neurobrix.core.prism.loader import load_profile
from neurobrix.core.prism.solver import ProfileFactMissing
from neurobrix.nbx import NBXContainer
from tests.unit.prism._pinned_machine import (V100_16GB, container_root, pin_dedicated_card, pin_host,
                                              pin_no_calibration, profile)

MODEL = "Qwen3-Coder-30B-A3B-Instruct"
PROFILE = "v100-32g-x2-nvlink"           # the committed profile: two V100-SXM2-32GB, cards 2 and 3


def _machine(monkeypatch):
    pin_host(monkeypatch, 257530, 200000, "an idle rack host")
    pin_dedicated_card(monkeypatch, 32501, 306, "a dedicated V100-32GB as its runtime reads it")
    pin_no_calibration(monkeypatch)


def _solve(config, serve):
    c = NBXContainer.load(str(container_root(MODEL)))
    return PrismSolver().solve_smart(c, load_profile(PROFILE), config, serve_mode=serve, mode="compiled")


def test_a_served_moe_holds_its_experts_on_the_cards(monkeypatch):
    _machine(monkeypatch)
    p = _solve(InputConfig(batch_size=2, dtype="float16"), serve=True)
    model = p.components["model"]
    assert model.strategy == "expert_store", (p.strategy, model.strategy, model.resident_weight_mb)
    assert model.expert_cards, "no block's experts held on a card: every routed weight crosses the host link"
    assert model.resident_weight_mb > 0


def test_a_short_run_never_overfills_a_card(monkeypatch):
    """The CLI's request of 2026-11-10 03:23 (17 prompt tokens, 16 decoded): weight_sharding planned 28 854 MB a card
    and the head beside, and ran out of memory on the first card. Whatever strategy wins, every card's plan stays
    within the part one component may hold."""
    _machine(monkeypatch)
    s = PrismSolver()
    c = NBXContainer.load(str(container_root(MODEL)))
    p = s.solve_smart(c, load_profile(PROFILE), InputConfig(batch_size=1, seq_len=17, max_tokens=16,
                                                            decode_tokens=16, dtype="float16"), mode="compiled")
    assert p.strategy != "weight_sharding", f"weight_sharding, whose split this request overfills: {p.selection_reason}"
    model = p.components["model"]
    assert model.strategy == "expert_store" and model.expert_cards, (p.strategy, model.strategy, model.expert_cards)


def test_a_card_whose_runtime_facts_the_profile_lacks_is_refused_by_name(monkeypatch):
    spec = copy.deepcopy(V100_16GB)
    for d in spec["devices"]:
        d.pop("runtime_total_mb", None)
        d.pop("runtime_context_mb", None)
    pin_host(monkeypatch, 257530, 200000, "an idle rack host")
    pin_dedicated_card(monkeypatch, 16151, 0, "a V100-16GB this process has not touched yet")
    c = NBXContainer.load(str(container_root("TinyLlama-1.1B-Chat-v1.0")))
    with pytest.raises(ProfileFactMissing, match="runtime_context_mb"):
        PrismSolver().solve_smart(c, profile(spec), InputConfig(batch_size=1, dtype="float16"), mode="compiled")


def test_a_served_cache_takes_what_the_models_residency_leaves(monkeypatch):
    """Decided between the agents 2026-11-10 04:37: the served cache is sized after the model's residency, never
    before it. Sized first, the window (244 500 tokens) left the home card's blocks on the host: 0.9 s a token served,
    0.33 s run. What this would do if the code were wrong: the home card's experts get a 72 MB cache — RED."""
    _machine(monkeypatch)
    p = _solve(InputConfig(batch_size=2, dtype="float16"), serve=True)
    model = p.components["model"]
    held_mb = model.expert_cache_mb + sum(c["cache_mb"] for k, c in (model.expert_cards or {}).items() if k == "cuda:2")
    assert held_mb > 10000, (model.expert_cache_mb, model.expert_cards)     # the home card's room went to the experts
    kv = p.kv_cache_plan
    assert kv is not None and kv.max_cache_len < 244500, kv           # below what the window took when sized first
