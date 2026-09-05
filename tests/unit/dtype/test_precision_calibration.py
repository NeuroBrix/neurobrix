"""Precision calibration — the DtypeEngine fp32-island detector.

A per-artifact record of the largest magnitude every op produced on the
conservative reference path (a whole request, every pass); the engine derives
its fp32 islands from it for the compute dtype at hand. No model name, no
module name, no hand-written list anywhere. Lever record:
validation_outputs/precision_census_2026_09_05/RECORD.md.
"""
import json
import math

import pytest
import torch

from neurobrix.core.dtype import calibration as cal


def _dag():
    """x --mm::0--> big --view::0--> v --mm::1--> y --add::0--> z ; small --exp::0--> e
    mm::0's output is over range; view::0 aliases it; mm::1 READS it; add::0
    reads mm::1's (in-range) output; exp::0 is far from all of it."""
    ops = {
        "aten.mm::0": {"op_type": "aten::mm", "parent_module": "block.0.ffn.down",
                       "input_tensor_ids": ["input::x", "param::w0"], "output_tensor_ids": ["aten.mm::0::out_0"]},
        "aten.view::0": {"op_type": "aten::view", "parent_module": "block.0.ffn.down",
                         "input_tensor_ids": ["aten.mm::0::out_0"], "output_tensor_ids": ["aten.view::0::out_0"]},
        "aten.mm::1": {"op_type": "aten::mm", "parent_module": "block.1.attn.q",
                       "input_tensor_ids": ["aten.view::0::out_0", "param::w1"], "output_tensor_ids": ["aten.mm::1::out_0"]},
        "aten.add::0": {"op_type": "aten::add", "parent_module": "block.1",
                        "input_tensor_ids": ["aten.mm::1::out_0", "input::x"], "output_tensor_ids": ["aten.add::0::out_0"]},
        "aten.exp::0": {"op_type": "aten::exp", "parent_module": "block.1",
                        "input_tensor_ids": ["input::x"], "output_tensor_ids": ["aten.exp::0::out_0"]},
        "aten.add_::0": {"op_type": "aten::add_", "parent_module": "block.1",
                         "input_tensor_ids": ["aten.view::0::out_0"], "output_tensor_ids": ["aten.add_::0::out_0"]},
    }
    return {"component_name": "t", "ops": ops, "execution_order": list(ops),
            "tensors": {}, "input_tensor_ids": ["input::x"], "output_tensor_ids": ["aten.add::0::out_0"]}


def test_bound_is_the_dtype_max_over_the_headroom():
    assert cal.island_bound(torch.float16, headroom_bits=2) == pytest.approx(65504.0 / 4)
    assert cal.island_bound(torch.float16, headroom_bits=0) == pytest.approx(65504.0)
    assert cal.island_bound(torch.bfloat16, headroom_bits=2) > 1e37
    assert cal.island_bound(torch.float32, headroom_bits=2) > 1e37


def test_islands_pin_the_producer_and_every_reader_of_an_over_range_value():
    max_abs = {"aten.mm::0": 1.4e5, "aten.view::0": 1.4e5, "aten.mm::1": 12.0,
               "aten.add::0": 13.0, "aten.exp::0": 3.0, "aten.add_::0": 1.4e5}
    pinned = cal.islands_from_calibration(_dag(), max_abs, bound=16376.0)
    # the producer, and the compute op that READS the over-range value
    assert "aten.mm::0" in pinned and "aten.mm::1" in pinned
    # a view carries no kernel: never pinned; an in-place op neither
    assert "aten.view::0" not in pinned and "aten.add_::0" not in pinned
    # in-range ops downstream and elsewhere are untouched
    assert "aten.add::0" not in pinned and "aten.exp::0" not in pinned


def test_islands_are_empty_when_nothing_exceeds_and_on_wide_dtypes():
    max_abs = {u: 100.0 for u in _dag()["ops"]}
    assert cal.islands_from_calibration(_dag(), max_abs, bound=16376.0) == frozenset()
    max_abs["aten.mm::0"] = 1.4e5
    assert cal.islands_from_calibration(_dag(), max_abs, bound=cal.island_bound(torch.bfloat16, 2)) == frozenset()


def test_a_non_finite_reference_value_is_no_island():
    """±inf and NaN are representable in fp16 (a -inf mask fill, log(0) in a
    position bucket): they never pin, and a record that stored inf for an op
    (finite / non-finite not yet told apart) never pins either."""
    max_abs = {u: 1.0 for u in _dag()["ops"]}
    max_abs["aten.exp::0"] = math.inf
    assert "aten.exp::0" not in cal.islands_from_calibration(_dag(), max_abs, bound=16376.0)
    max_abs["aten.mm::0"] = math.nan
    assert cal.islands_from_calibration(_dag(), max_abs, bound=16376.0) == frozenset()

def test_an_unrecorded_op_is_treated_as_in_range():
    pinned = cal.islands_from_calibration(_dag(), {"aten.mm::1": 2e5}, bound=16376.0)
    assert pinned == frozenset({"aten.mm::1", "aten.add::0"})


def test_graph_signature_changes_with_the_trace_not_with_the_values():
    d = _dag()
    s1 = cal.graph_signature(d)
    d2 = _dag(); d2["ops"]["aten.mm::1"]["op_type"] = "aten::addmm"
    assert s1 != cal.graph_signature(d2)
    assert s1 == cal.graph_signature(_dag())


def test_record_round_trip_keeps_non_finite_values(tmp_path):
    rec = cal.CalibrationRecord.build(
        model_name="m", component="t", dag=_dag(),
        max_abs={"aten.mm::0": 1.4e5, "aten.exp::0": math.inf},
        stimulus={"prompt": "p", "steps": 2}, passes=3, reference="conservative",
        non_finite=["aten.exp::0"])
    path = tmp_path / "t.json"
    rec.save(path)
    back = cal.CalibrationRecord.load(path)
    assert back.max_abs["aten.mm::0"] == 1.4e5 and math.isinf(back.max_abs["aten.exp::0"])
    assert back.non_finite == ["aten.exp::0"]
    assert back.graph_signature == cal.graph_signature(_dag())
    assert back.matches(_dag())
    other = _dag(); other["ops"]["aten.mm::1"]["op_type"] = "aten::addmm"
    assert not back.matches(other)
    assert json.loads(path.read_text())["format"] == cal.FORMAT


def test_store_paths_are_per_model_and_component(tmp_path, monkeypatch):
    monkeypatch.setattr(cal, "STORE_ROOT", tmp_path)
    assert cal.store_path("Sana_1600M", "transformer") == tmp_path / "Sana_1600M" / "transformer.json"
    assert cal.load_record("Sana_1600M", "transformer") is None
    rec = cal.CalibrationRecord.build("Sana_1600M", "transformer", _dag(), {"aten.mm::0": 1.0},
                                      stimulus={}, passes=1, reference="conservative")
    rec.save(cal.store_path("Sana_1600M", "transformer"))
    assert cal.load_record("Sana_1600M", "transformer").max_abs == {"aten.mm::0": 1.0}


def test_census_accumulates_the_max_over_passes_without_host_sync():
    census = cal.RangeCensus()
    census.observe("a", torch.tensor([1.0, -3.0]))
    census.observe("a", torch.tensor([2.0, 0.5]))
    census.observe("b", (torch.tensor([7.0]), torch.tensor([9], dtype=torch.int64)))  # tuple output, ints ignored
    census.observe("c", torch.tensor([5], dtype=torch.int64))                        # integer-only: not recorded
    census.observe("d", torch.tensor([float("inf"), -2.0, float("nan")]))           # finite max 2, non-finite seen
    census.observe("e", torch.tensor([], dtype=torch.float32))                       # empty: not recorded
    out = census.finalize()
    assert out["a"] == 3.0 and out["b"] == 7.0 and out["d"] == 2.0
    assert "c" not in out and "e" not in out
    assert census.non_finite_ops() == ["d"]
    assert census.passes == 0  # passes are counted by the sequence, at run boundaries
    census.pass_done(); census.pass_done()
    assert census.passes == 2


def test_bind_freezes_the_signature_before_the_executor_rewrites_its_graph():
    """The executor rewrites its graph after the contract is resolved; the
    record must carry the identity seen at resolve time or every later
    load would report a stale record and stay conservative (2026-09-05)."""
    dag = _dag()
    census = cal.RangeCensus()
    census.bind(dag, None)
    frozen = census.signature
    dag["ops"]["aten.mm::1"]["op_type"] = "aten::clone"      # a later in-place rewrite
    rec = cal.CalibrationRecord.build("m", "t", dag, {"aten.mm::0": 1.0}, stimulus={},
                                      passes=1, reference="conservative",
                                      graph_signature=census.signature)
    assert rec.graph_signature == frozen and rec.matches(_dag()) and not rec.matches(dag)
