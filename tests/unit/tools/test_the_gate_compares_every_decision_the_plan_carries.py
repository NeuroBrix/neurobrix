"""The gate's plan differential compares every decision the plan carries.

2026-10-16: the state engine's step 1 (e83b3d1e2) moved whisper's KV cache plan from 576 to 448 positions; its gate
scoped 0 cells and it landed without one, because tools/gate_scope.yml `plan_fields` named no `kv_cache` (nor
`deferred_drain_cliffs`, `layer_stream_chunks`, `expert_working_set`, `cpu_ram_mb`). Every key core/prism/solver.
plan_record can emit is a compared field or a named explanation; and the plan views the scope caches are keyed by the
fields they hold, so a change to the list never reuses a view without the new field.
"""
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
import gate_scope as GS  # noqa: E402
from neurobrix.core.prism.solver import plan_record  # noqa: E402


def _every_field_plan():
    """A plan with every optional section of the record present."""
    alloc = NS(devices=["cuda:0"], dtype="float16", sharded=False, strategy="zero3", resident_weight_mb=1.0,
               attention_scores_budget_bytes=1)
    mem = NS(weight_bytes=1, activation_bytes=1, overhead_bytes=1, peak_op_uid="op", activation_profiled=True)
    kv = NS(max_cache_len=448, memory_bytes=1, dtype="float16", num_layers=4, num_kv_heads=20, k_head_dim=64,
            v_head_dim=64)
    return NS(components={"c": alloc}, component_memory={"c": mem}, strategy="single_gpu", loading_mode="eager",
              target_dtype="float16", selection_reason="r", candidates=[("single_gpu", 1.0)],
              rejected=[("zero3", 0.5, "w")], total_memory_mb=1.0, cpu_ram_mb=1, device_window_mb=1.0,
              deferred_drain_cliffs={"cuda:0": 1}, runtime_op_tiling={}, component_tiling={}, kv_cache_plan=kv,
              cfg_split_components=["c"], layer_stream_chunks={"c": [{"a": 1}]}, host_footprint={"total_bytes": 1},
              unified_rungs_tried=[(1, 2)], expert_working_set={"c": 1})


def test_every_key_of_the_plan_record_is_compared_or_an_explanation():
    planner = GS.load_cfg()["planner"]
    fields, explained = set(planner["plan_fields"]), set(planner["explanation_fields"])
    assert not fields & explained, f"both compared and explanation: {sorted(fields & explained)}"
    keys = set(plan_record(_every_field_plan()))
    unseen = keys - fields - explained
    assert not unseen, f"plan decisions the gate does not compare: {sorted(unseen)}"


def test_a_cached_plan_view_is_keyed_by_the_fields_it_holds(tmp_path, monkeypatch):
    monkeypatch.setattr(GS, "SCRATCH", tmp_path)
    cfg = {"package": "neurobrix", "planner": {"env": {}, "plan_fields": ["strategy"]}}
    tree = NS(sha="abc", path=tmp_path)
    argv = ["run", "--model", "m"]
    old_cmd = ["-m", "neurobrix", *argv, "--hardware", "hw", "--explain-plan", "--json"]
    # the view a run with the shorter field list cached, under the key that list made
    (tmp_path / "plans").mkdir()
    for key in (hashlib.sha256(json.dumps([tree.sha, old_cmd]).encode()).hexdigest()[:24],
                hashlib.sha256(json.dumps([tree.sha, old_cmd, ["strategy"]]).encode()).hexdigest()[:24]):
        (tmp_path / "plans" / f"{key}.json").write_text(json.dumps({"plan": {"strategy": "s"}}))
    class Ran(Exception):
        pass
    def run(*a, **k):
        raise Ran()
    monkeypatch.setattr(GS.subprocess, "run", run)
    assert GS.explain_plan(tree, "hw", argv, cfg) == {"plan": {"strategy": "s"}}
    cfg["planner"]["plan_fields"] = ["strategy", "kv_cache"]
    with pytest.raises(Ran):
        GS.explain_plan(tree, "hw", argv, cfg)
