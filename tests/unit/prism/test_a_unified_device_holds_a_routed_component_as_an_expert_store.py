"""On a unified device a routed component is held as an expert store, priced at its working set.

The Mac's M4 Pro (2026-10-12): Qwen3-30B-A3B-Thinking-2507 (48 routed layers x 128 experts of 9 MB,
8 per token) was never placed whole on mps:0 — 58 GB of weights — and zero3 frees nothing on one
pool, so the plan fell to layer_streaming: segments each holding whole layers of 128 experts,
re-read from disk on every token, or refused where one layer's 1 152 MB exceeded a segment. The
discrete card's `expert_store` (3ee5ee57) was never reached: `_zero3_resident_budgets` skipped a
unified device, and only a zero3 component was refined to a store.

Here `_place_component` holds such a component on the unified device as an expert store (Strategy
2.5): its dense part and the widest layer's top_k experts beside its live activations, against the
usable part of the rung (`_expert_store_floor`, the discrete grant's own reading). The plan names it
`expert_store`, the KV check and the plan's device peak price it at that working set, never at
its whole weights, and `_zero3_resident_budgets` grants the rest of the room as its expert cache.

What these tests would do if the code were wrong: the old placement returns None for the routed
component (red); a plan that keeps pricing the whole fused op prices 58 GB on the device (red); a
cache granted beyond the room, or not granted at all, fails the grant figures (red); a stacked slab
or a discrete card placed as a unified store fails the inertia cells (red).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from neurobrix.core.prism.expert_topology import ExpertTopology, RoutedLayer
from neurobrix.core.prism.loader import load_profile
from neurobrix.core.prism.solver import (
    ComponentMemory, DeviceState, PrismSolver, _device_is_unified,
)
from neurobrix.core.prism.structure import DeviceBrand, DeviceSpec

MB = 1024 * 1024

#: Qwen3-30B-A3B (https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507/blob/main/config.json:
#: 48 layers, 128 experts, 8 per token, hidden 2048, moe_intermediate 768); one expert
#: 3 x 2048 x 768 x 2 B = 9 MB in bf16.
LAYERS, EXPERTS, TOP_K, EXPERT = 48, 128, 8, 3 * 2048 * 768 * 2
#: The dense part and the activations of the M4's explain-plan (3ee5ee57, NBX_PRISM_BUDGET_MB=16384):
#: dense 2 939.4 MB of the 58 235.4 MB model, activations 3 101 MB.
DENSE_MB, ACT_MB = 2940, 3101
ROUTED_MB = LAYERS * EXPERTS * EXPERT // MB          # 55 296
FLOOR_MB = TOP_K * EXPERT // MB                       # 72: one layer's top_k experts


class _Container:
    """The dtype door's declared components; no cache path (fgp finds no block structure)."""
    from pathlib import Path as _P
    cache_path = _P("/nonexistent-prism-fixture")

    def get_neural_components(self):
        return [SimpleNamespace(name="model", get_dominant_dtype=lambda: "bfloat16")]


def _expert_names(layer):
    return tuple(f"block.{layer}.ffn.expert.{e}.{p}.weight"
                 for p in ("ffn_gate", "up", "down") for e in range(EXPERTS))


def _topology(stacked=False):
    return {"model": ExpertTopology("model", [
        RoutedLayer(op_uid=f"m{i}", block=i, n_experts=EXPERTS, top_k=TOP_K, norm_topk_prob=True,
                    expert_bytes=EXPERT, shared_bytes=0, stacked=stacked,
                    expert_weight_names=() if stacked else _expert_names(i)) for i in range(LAYERS)])}


def _indexed(tmp_path, dtype="bfloat16"):
    """A container whose weights index stores every routed expert in `dtype` — what the plan reads
    the experts' stored dtype from (`_refuse_unheld_expert_files`)."""
    import json
    comp = tmp_path / "components" / "model"
    comp.mkdir(parents=True, exist_ok=True)
    tensors = {n: {"shard": "shard_000.safetensors", "dtype": dtype}
               for i in range(LAYERS) for n in _expert_names(i)}
    (comp / "weights_index.json").write_text(json.dumps({"tensors": tensors}))
    return SimpleNamespace(cache_path=tmp_path)


def _model_mem():
    return ComponentMemory(component_name="model", weight_bytes=(DENSE_MB + ROUTED_MB) * MB,
                           activation_bytes=ACT_MB * MB, overhead_bytes=0)


def _solver(stacked=False):
    s = PrismSolver.__new__(PrismSolver)          # no machine probing
    from neurobrix.core.config.system import PRISM_DEFAULTS
    s.oom_reserve_mb = PRISM_DEFAULTS["oom_reserve_mb"]
    s._mode = "compiled"
    s._expert_topology = _topology(stacked)
    return s


def _device(device_string, unified):
    profile = load_profile("a10-24g")             # a fixture in the tree, not this machine's file
    profile.devices[0].unified_memory = unified
    spec = DeviceSpec(index=0, name="dev", memory_mb=24576, compute_capability="0.0",
                      supports_dtypes=["bfloat16"], architecture="apple_silicon", brand=DeviceBrand.APPLE)
    # the M4 Pro's budget pinned at the 16 384 MB rung (NBX_PRISM_BUDGET_MB)
    dev = DeviceState(device_string=device_string, capacity_mb=18186.0, spec=spec,
                      recommended_mb=18186.0, budget_mb=16384.0)
    return profile, dev


def _place(s, profile, dev, mem=None):
    return s._place_component(_Container(), "model", mem or _model_mem(), [dev], {}, profile)


def test_the_old_price_holds_the_whole_layer_and_the_store_places_the_working_set():
    s = _solver()
    profile, dev = _device("mps:0", unified=True)
    usable = s._usable_mb(dev)
    # the old price: every expert of every layer, or a segment holding a layer's 128 experts
    assert DENSE_MB + ROUTED_MB + ACT_MB > usable
    assert EXPERTS * EXPERT / MB == 1152 and FLOOR_MB == 72
    result = _place(s, profile, dev)
    assert result is not None, (
        f"the routed component's working set (dense {DENSE_MB} + top_k {FLOOR_MB} + activations "
        f"{ACT_MB} MB) fits {usable:.0f} MB usable on the unified device and was not placed")
    device_string, _ = result
    assert device_string == "mps:0", f"placed on {device_string!r}: a unified store is held on its device"
    assert s._unified_expert_stores == {"model": DENSE_MB + FLOOR_MB}


def test_the_plan_names_the_store_and_prices_its_working_set_not_its_weights():
    s = _solver()
    profile, dev = _device("mps:0", unified=True)
    s._lm_component_name = None
    s._place_component(_Container(), "model", _model_mem(), [dev], {}, profile)
    comp_mem = {"model": _model_mem()}
    allocs = {"model": ("mps:0", {})}
    # the KV check: the store's working set beside its activations, never its 58 GB of weights
    held = s._resident_bytes_for_kv_check("lazy_sequential", allocs, comp_mem, profile)
    assert held == (ACT_MB + DENSE_MB + FLOOR_MB) * MB
    plan = s._build_plan(allocs, comp_mem, [dev], {"model": "bfloat16"}, profile, "lazy_sequential")
    a = plan.components["model"]
    assert a.strategy == "expert_store"
    assert a.memory_mb == ACT_MB + DENSE_MB + FLOOR_MB
    # another strategy's allocation of the same name is not a store
    other = s._build_plan(allocs, comp_mem, [dev], {"model": "bfloat16"}, profile, "single_gpu")
    assert other.components["model"].strategy == "single_gpu"


def test_the_rest_of_the_room_is_granted_as_the_expert_cache(tmp_path):
    s = _solver()
    profile, dev = _device("mps:0", unified=True)
    s._lm_component_name = "model"
    s._target_dtype_str = "bfloat16"
    kv_mb = 600
    s._estimate_kv_cache_bytes = lambda c, d: kv_mb * MB      # priced inside the activations
    s._get_component_dtype = lambda c, n: "bfloat16"
    s._loads_on_demand = lambda strat: False                  # the SUM: no phases
    s._place_component(_Container(), "model", _model_mem(), [dev], {}, profile)
    # a unified store streams no window: its blocks' sizes do not enter the grant
    s._parse_blocks = lambda c, n: {"blocks": {}, "block_sizes": {0: 1300.0}}
    comp_mem = {"model": _model_mem()}
    plan = s._build_plan({"model": ("mps:0", {})}, comp_mem, [dev], {"model": "bfloat16"}, profile,
                         "lazy_sequential")
    plan.kv_cache_plan = SimpleNamespace(memory_bytes=kv_mb * MB)
    s._zero3_resident_budgets(_indexed(tmp_path), plan, [dev], profile)
    a = plan.components["model"]
    room = s._usable_mb(dev) - ACT_MB                         # the KV as planned = as priced; no window
    assert a.resident_weight_mb == DENSE_MB
    assert a.expert_cache_mb == int(room - DENSE_MB)
    assert FLOOR_MB <= a.expert_cache_mb < EXPERTS * EXPERT / MB * LAYERS
    assert a.memory_mb == ACT_MB + DENSE_MB + a.expert_cache_mb
    assert plan.total_memory_mb == a.memory_mb
    # the plan's device peak (the host footprint's unified term) is the same working set
    s._phase_peak = lambda c, totals, outliving, owner: sum(totals.values())
    assert s._peak_loaded_bytes(None, plan) == int((ACT_MB + DENSE_MB + a.expert_cache_mb) * MB)


def test_a_room_below_the_floor_after_the_kv_cache_is_refused_by_name(tmp_path):
    s = _solver()
    profile, dev = _device("mps:0", unified=True)
    s._lm_component_name = "model"
    s._target_dtype_str = "bfloat16"
    s._estimate_kv_cache_bytes = lambda c, d: 0
    s._get_component_dtype = lambda c, n: "bfloat16"
    s._loads_on_demand = lambda strat: False
    s._place_component(_Container(), "model", _model_mem(), [dev], {}, profile)
    comp_mem = {"model": _model_mem()}
    plan = s._build_plan({"model": ("mps:0", {})}, comp_mem, [dev], {"model": "bfloat16"}, profile,
                         "lazy_sequential")
    room = s._usable_mb(dev) - ACT_MB
    plan.kv_cache_plan = SimpleNamespace(memory_bytes=int((room - DENSE_MB - FLOOR_MB + 1) * MB))
    with pytest.raises(RuntimeError, match="below its dense part and the widest top_k experts"):
        s._zero3_resident_budgets(_indexed(tmp_path), plan, [dev], profile)


def test_a_stacked_slab_on_a_unified_device_is_not_a_store():
    s = _solver(stacked=True)
    profile, dev = _device("mps:0", unified=True)
    result = _place(s, profile, dev)
    assert result is None or not getattr(s, "_unified_expert_stores", {})


def test_a_discrete_card_keeps_zero3_and_refines_it_with_the_plan():
    """The inertia proof: Strategy 2.5 is the unified device's; a discrete card still places zero3
    (the Dell's `expert_store` is chosen later, from it, by `_zero3_resident_budgets`)."""
    s = _solver()
    profile, dev = _device("cuda:0", unified=False)
    assert not _device_is_unified("cuda:0", profile)
    result = _place(s, profile, dev)
    assert result is not None and result[0] == "zero3:cuda:0"
    assert not getattr(s, "_unified_expert_stores", {})


def test_a_component_that_fits_whole_is_held_whole():
    s = _solver()
    profile, dev = _device("mps:0", unified=True)
    small = ComponentMemory(component_name="model", weight_bytes=4000 * MB, activation_bytes=1000 * MB,
                            overhead_bytes=0)
    assert _place(s, profile, dev, small) == ("mps:0", {})
    assert not getattr(s, "_unified_expert_stores", {})


def test_the_topology_names_the_expert_weights_a_store_reads_from_its_files():
    from neurobrix.core.prism.expert_topology import read_expert_topology
    names = [f"block.0.ffn.expert.{i}.{p}.weight" for p in ("ffn_gate", "up", "down") for i in range(4)]
    graph = {"tensors": {f"param::{n}": {"is_parameter": True, "shape": [4, 8], "dtype": "bfloat16",
                                         "weight_name": n} for n in names},
             "ops": {"m": {"op_type": "custom::moe_fused", "attributes": {
                 "num_experts": 4, "top_k": 2, "norm_topk_prob": True,
                 **{f"expert_{r}_weight_ids": [f"param::block.0.ffn.expert.{i}.{p}.weight" for i in range(4)]
                    for r, p in (("gate", "ffn_gate"), ("up", "up"), ("down", "down"))}}}},
             "execution_order": ["m"]}
    t = read_expert_topology("lm", graph, None, 1.0)
    assert sorted(t.layers[0].expert_weight_names) == sorted(names)


def test_the_compiled_load_of_a_unified_store_passes_only_its_dense_part_through_the_host():
    """host_footprint.py: the compiled load passes a component through the host whole (read, then pinned
    at the plan's width). A unified expert store's routed experts are mapped from the container's files,
    never loaded whole — Qwen3-30B-A3B on the M4 priced 115 288 MB of loading for a 15 073 MB plan."""
    from neurobrix.core.prism.host_footprint import host_footprint
    alloc = SimpleNamespace(device="mps:0", shard_map={"s0": "mps:0"}, dtype="bfloat16", strategy="expert_store")
    plan = SimpleNamespace(components={"model": alloc}, component_memory={"model": _model_mem()},
                           loading_mode="lazy")
    keys = {"model": {"dense.w": 100 * MB, "e0": 50 * MB, "e1": 50 * MB}}
    args = (plan, keys, {"model": {"s0": 200 * MB}}, "compiled", None, {"bfloat16": 2},
            lambda k: False, {"model": {"bfloat16"}})
    whole = host_footprint(*args)["transient_bytes"]
    assert whole == 400 * MB                                  # read + pinned, every weight
    assert host_footprint(*args, file_backed={"model": {"e0", "e1"}})["transient_bytes"] == 200 * MB
    with pytest.raises(ValueError, match="absent from the weights index"):
        host_footprint(*args, file_backed={"model": {"e0", "e9"}})


def test_the_solver_names_the_file_backed_experts_of_a_unified_store_only():
    s = _solver()
    names = {f"e{i}" for i in range(3)}
    s._expert_topology["model"] = ExpertTopology("model", [RoutedLayer(
        op_uid="m0", block=0, n_experts=EXPERTS, top_k=TOP_K, norm_topk_prob=True, expert_bytes=EXPERT,
        shared_bytes=0, expert_weight_names=tuple(names))])
    profile, _ = _device("mps:0", unified=True)
    plan = SimpleNamespace(components={"model": SimpleNamespace(device="mps:0", strategy="expert_store")})
    assert s._file_backed_experts(plan, profile) == {"model": names}
    plan.components["model"].strategy = "lazy_sequential"
    assert s._file_backed_experts(plan, profile) == {}
    profile, _ = _device("cuda:0", unified=False)
    plan = SimpleNamespace(components={"model": SimpleNamespace(device="cuda:0", strategy="expert_store")})
    assert s._file_backed_experts(plan, profile) == {}
