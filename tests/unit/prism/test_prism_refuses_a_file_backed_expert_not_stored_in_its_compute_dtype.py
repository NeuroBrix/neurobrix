"""Prism refuses, at plan time and by name, a unified expert store whose experts are not stored in
the dtype their engine computes them in.

A unified expert store (0a90722c) binds each routed expert to a view of its STORED bytes and copies
them into cache slots unconverted, so the load refuses an expert stored in another dtype than the
one its engine holds it in (core/runtime/expert_files.py `bind_views`). Prism checked nothing of the
kind: the plan was accepted and the run died at load. The plan now reads the same rule
(`expert_files.held_dtype`, which both engines' views call) from the same two sources — the stored
dtype from the component's weights index, the compute dtype from the plan's per-component dtype —
under the engine the plan is for.

What these tests would do if the code were wrong: with no plan-time check, `_zero3_resident_budgets`
grants the store and returns (the refusal cells go red); a check that compared against a constant
or ignored the engine would refuse fp16-stored experts under the triton engine at an fp32 plan, or
accept them under the compiled one (the engine cell goes red either way); a check that also ran on a
discrete card, whose experts the loader converts into pinned host memory, refuses it (red).
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from neurobrix.core.runtime.expert_files import held_dtype
from nbxtensor.dtype.nbx_dtype import NBXDtype
from tests.unit.prism.test_a_unified_device_holds_a_routed_component_as_an_expert_store import (
    _Container, _device, _indexed, _model_mem, _solver,
)


def _plan_store(tmp_path, *, stored, plan_dtype, mode, device="mps:0", unified=True):
    """Plan the Qwen3-30B-A3B-sized routed component as the unified store fixture does, its experts
    stored `stored` and the plan's dtype `plan_dtype`, under `mode`, then grant it."""
    s = _solver()
    s._mode = mode
    profile, dev = _device(device, unified=unified)
    s._lm_component_name = "model"
    s._target_dtype_str = plan_dtype
    s._estimate_kv_cache_bytes = lambda c, d: 0
    s._get_component_dtype = lambda c, n: plan_dtype
    s._loads_on_demand = lambda strat: False
    placed = s._place_component(_Container(), "model", _model_mem(), [dev], {}, profile)
    assert placed is not None
    s._parse_blocks = lambda c, n: {"blocks": {}, "block_sizes": {0: 1300.0}}
    plan = s._build_plan({"model": placed}, {"model": _model_mem()}, [dev], {"model": plan_dtype},
                         profile, "lazy_sequential")
    plan.kv_cache_plan = SimpleNamespace(memory_bytes=0)
    s._zero3_resident_budgets(_indexed(tmp_path, stored), plan, [dev], profile)
    return plan.components["model"]


@pytest.mark.parametrize("mode", ["compiled", "triton", "triton_sequential"])
def test_experts_stored_in_the_compute_dtype_are_held_as_a_store(tmp_path, mode):
    a = _plan_store(tmp_path, stored="bfloat16", plan_dtype="bfloat16", mode=mode)
    assert a.strategy == "expert_store" and a.expert_files and a.expert_cache_mb > 0


@pytest.mark.parametrize("mode", ["compiled", "triton"])
def test_experts_stored_bf16_under_an_fp16_plan_are_refused_by_name(tmp_path, mode):
    with pytest.raises(RuntimeError) as e:
        _plan_store(tmp_path, stored="bfloat16", plan_dtype="float16", mode=mode)
    msg = str(e.value)
    assert "'model'" in msg and "stored bfloat16 held as float16" in msg, msg
    assert "block.0.ffn.expert.0." in msg and "converts none" in msg, msg


def test_the_rule_is_the_engines_not_a_constant(tmp_path):
    """fp16 experts under an fp32 plan: the compiled loader converts every floating weight to the
    placement dtype (a view cannot), the triton loader keeps a half weight half under fp32
    (`stored_dtype_in_compute`) — so the compiled plan is refused and the triton plan is not."""
    with pytest.raises(RuntimeError, match="stored float16 held as float32"):
        _plan_store(tmp_path, stored="float16", plan_dtype="float32", mode="compiled")
    a = _plan_store(tmp_path, stored="float16", plan_dtype="float32", mode="triton")
    assert a.strategy == "expert_store" and a.expert_files


def test_the_plan_reads_the_rule_the_load_reads():
    """`held_dtype` is what both engines' views call (graph_executor `_expert_file_views`); on the
    triton engine it IS the loader's `stored_dtype_in_compute`, pair by pair."""
    from neurobrix.triton.weight_loader import stored_dtype_in_compute
    floats = [NBXDtype.float16, NBXDtype.bfloat16, NBXDtype.float32, NBXDtype.float64]
    for st in floats + [NBXDtype.int8]:
        for c in floats:
            assert held_dtype(st, c, "triton") == stored_dtype_in_compute(st, c)
            assert held_dtype(st, c, "compiled") == (c if st in floats else st)
    with pytest.raises(RuntimeError, match="no engine named"):
        held_dtype(NBXDtype.float16, NBXDtype.float16, "metal")


def test_an_index_that_cannot_be_read_is_refused(tmp_path):
    s = _solver()
    alloc = SimpleNamespace(device="mps:0", dtype="bfloat16")
    with pytest.raises(RuntimeError, match="weights index cannot be read"):
        s._refuse_unheld_expert_files(SimpleNamespace(cache_path=tmp_path), "model", alloc)
    with pytest.raises(RuntimeError, match="weights index cannot be read"):
        s._refuse_unheld_expert_files(None, "model", alloc)


def test_an_expert_the_index_does_not_record_is_refused(tmp_path):
    s = _solver()
    container = _indexed(tmp_path)
    path = tmp_path / "components" / "model" / "weights_index.json"
    tensors = json.loads(path.read_text())["tensors"]
    tensors.pop("block.3.ffn.expert.7.up.weight")
    path.write_text(json.dumps({"tensors": tensors}))
    with pytest.raises(RuntimeError, match=r"'block\.3\.ffn\.expert\.7\.up\.weight' has no stored dtype"):
        s._refuse_unheld_expert_files(container, "model", SimpleNamespace(device="mps:0", dtype="bfloat16"))


def test_a_discrete_cards_store_is_not_held_to_it(tmp_path):
    """On a discrete card the experts are LOADED into pinned host memory, converted by the loader
    like any weight: the stored dtype may differ from the compute dtype there."""
    a = _plan_store(tmp_path, stored="bfloat16", plan_dtype="float16", mode="compiled",
                    device="cuda:0", unified=False)
    assert not a.expert_files
    assert a.strategy in ("zero3", "expert_store")
