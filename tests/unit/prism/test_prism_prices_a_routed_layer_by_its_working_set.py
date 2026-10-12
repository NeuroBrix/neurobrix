"""A fused MoE op is priced by what a token reads from it, not by its whole layer.

The Mac's M4 profile (2026-10-12, 3634e4fe): Qwen3-30B-A3B-Thinking-2507 native was refused with
3 768 MB usable because the unsplittable `moe_fused::block.0` op priced all 128 experts of its
layer (128 x 9 MB = 1 152 MB), while a token reads top-8 x 9 MB = 72 MB. Under the `expert_store`
strategy the routed component's footprint is its dense part plus an expert cache of at least the
widest `top_k` experts (`PrismSolver._expert_store_budget`); a call routing to more experts than
the cache holds is served in runs of cached experts by both engines, so one layer's full set is
never required on the card.

What these tests would do if the code were wrong: a floor priced as the whole layer refuses the
first case (red); a cache not capped at the routed experts over-reserves the card (red); a stacked
slab, which has no per-expert host tensor for the store to copy, planned as a store (red).
"""
from __future__ import annotations

from neurobrix.core.prism.expert_topology import ExpertTopology, RoutedLayer
from neurobrix.core.prism.solver import ComponentAllocation, PrismSolver

MB = 1024 * 1024

#: Qwen3-30B-A3B's routed layers (https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507/blob/main/config.json:
#: 48 layers, 128 experts, 8 per token); one expert 3 x 2048 x 768 x 2 B = 9 MB in fp16.
LAYERS, EXPERTS, TOP_K, EXPERT = 48, 128, 8, 3 * 2048 * 768 * 2
#: Dense part and usable room of the Mac's refused case (the dense figure of the working-set test).
DENSE_MB, USABLE_MB = 2642, 3768


def _solver(stacked=False, layers=LAYERS):
    s = PrismSolver.__new__(PrismSolver)
    s._expert_topology = {"lm": ExpertTopology("lm", [
        RoutedLayer(op_uid=f"m{i}", block=i, n_experts=EXPERTS, top_k=TOP_K, norm_topk_prob=True,
                    expert_bytes=EXPERT, shared_bytes=0, stacked=stacked) for i in range(layers)])}
    return s


def _weight_mb(s):
    return DENSE_MB + s._expert_topology["lm"].routed_bytes / MB


def test_the_old_whole_layer_price_refuses_and_the_working_set_plans():
    s = _solver()
    weight_mb = _weight_mb(s)
    whole_layer_mb = EXPERTS * EXPERT / MB
    assert whole_layer_mb == 1152 and TOP_K * EXPERT / MB == 72
    # the old price: the fused op holds its whole layer beside the dense part — over the room
    assert DENSE_MB + whole_layer_mb > USABLE_MB
    plan = s._expert_store_budget("lm", weight_mb, USABLE_MB, 1.0)
    assert plan is not None, "the routed working set (dense + top_k experts) fits and was refused"
    dense, cache = plan
    assert dense == DENSE_MB
    assert cache == USABLE_MB - DENSE_MB                 # the rest of the room, floored
    assert cache >= TOP_K * EXPERT / MB                  # at least what one token reads
    assert cache < whole_layer_mb                        # below one layer: served in runs


def test_below_the_top_k_floor_it_stays_zero3():
    s = _solver()
    assert s._expert_store_budget("lm", _weight_mb(s), DENSE_MB + 71, 1.0) is None
    assert s._expert_store_budget("lm", _weight_mb(s), DENSE_MB + 72, 1.0) == (DENSE_MB, 72.0)


def test_where_every_weight_fits_it_stays_zero3_and_the_cache_is_capped_at_the_experts():
    s = _solver(layers=2)
    w = _weight_mb(s)
    assert s._expert_store_budget("lm", w, w, 1.0) is None          # every weight resident: zero3
    dense, cache = s._expert_store_budget("lm", w, w - 1, 1.0)
    assert cache <= 2 * 1152 and dense + cache <= w


def test_the_device_cost_multiplier_scales_both_figures():
    s = _solver()
    w = _weight_mb(s) * 2.0
    assert s._expert_store_budget("lm", w, 2 * DENSE_MB + 143, 2.0) is None
    assert s._expert_store_budget("lm", w, 2 * DENSE_MB + 144, 2.0) == (2 * DENSE_MB, 144.0)


def test_a_stacked_slab_or_a_dense_component_is_never_a_store():
    s = _solver(stacked=True)
    assert s._expert_store_budget("lm", _weight_mb(s), USABLE_MB, 1.0) is None
    assert _solver()._expert_store_budget("other", 10_000.0, 5_000.0, 1.0) is None


def test_the_plan_carries_the_strategy_and_the_cache_across_a_save(tmp_path):
    from neurobrix.core.prism.solver import ExecutionPlan
    a = ComponentAllocation(name="lm", devices=["zero3:cuda:0"], dtype="float16", memory_mb=57000.0,
                            architecture="", vendor="", strategy="expert_store",
                            resident_weight_mb=2642.0, expert_cache_mb=1126.0)
    plan = ExecutionPlan(components={"lm": a}, target_dtype="float16", total_memory_mb=0.0,
                         strategy="lazy_sequential", component_memory={})
    plan.save(tmp_path / "plan.json")
    back = ExecutionPlan.load(tmp_path / "plan.json").components["lm"]
    assert (back.strategy, back.resident_weight_mb, back.expert_cache_mb) == ("expert_store", 2642.0, 1126.0)
