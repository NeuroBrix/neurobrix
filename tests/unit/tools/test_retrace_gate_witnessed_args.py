"""The retrace gate's argument-level class of the closed defect.

A shape argument whose old value — a symbol, or an integer — claimed a trace
value that contradicts the extent the op's own output witnessed, replaced by
that witnessed integer, is the closed defect (D-TRACE-SYMBOLIC-DIMS-FOREIGN-INT)
at the argument level: admitted. Any other difference between two records of
one op stays beyond the annotation and the gate refuses it.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import retrace_zoo as R  # noqa: E402

S0 = {"type": "symbol", "id": "s0", "trace": 23}
OLD = {"op_uid": "aten.expand::2", "op_type": "aten.expand", "input_tensor_ids": ["u::out_0"], "output_tensor_ids": ["aten.expand::2::out_0"],
       "attributes": {"args": [{"type": "tensor", "tensor_id": "u::out_0"}, {"type": "list", "value": [1, S0, S0]}], "kwargs": {}, "size": [1, S0, S0]}}
TENSORS = {"aten.expand::2::out_0": {"shape": [1, 23, 34], "symbolic_shape": {"dims": [1, S0, 34], "concrete": [1, 23, 34]}}}


def _new(**size_slot):
    n = copy.deepcopy(OLD)
    v = size_slot.get("value", 34)
    n["attributes"]["size"][2] = v
    n["attributes"]["args"][1]["value"][2] = v
    return n


def test_a_false_symbol_replaced_by_the_witnessed_extent_is_the_closed_defect():
    sites = R.witnessed_arg_changes(OLD, _new(), TENSORS)
    assert sites is not None and len(sites) == 2
    assert {s["path"] for s in sites} == {"size.2", "args.1.value.2"}
    assert all(s["old"] == S0 and s["new"] == 34 for s in sites)


def test_a_bare_integer_that_contradicted_the_extent_is_the_same_class():
    old = copy.deepcopy(OLD)
    old["attributes"]["size"][2] = 23
    old["attributes"]["args"][1]["value"][2] = 23
    assert len(R.witnessed_arg_changes(old, _new(), TENSORS)) == 2


def test_a_replacement_that_is_not_the_witnessed_extent_is_refused():
    assert R.witnessed_arg_changes(OLD, _new(value=35), TENSORS) is None


def test_a_symbol_whose_trace_already_matched_is_not_a_correction():
    """s0 (trace 23) → 23 at a position that witnessed 23 changes nothing the
    old contradicted: not the class (and not a difference the old was wrong about)."""
    tensors = {"aten.expand::2::out_0": {"shape": [1, 23, 23], "symbolic_shape": {"dims": [1, S0, S0], "concrete": [1, 23, 23]}}}
    assert R.witnessed_arg_changes(OLD, _new(value=23), tensors) is None


def test_the_reverse_direction_is_refused():
    """An integer turned into a symbol is another kind of change."""
    assert R.witnessed_arg_changes(_new(), OLD, TENSORS) is None


def test_a_difference_outside_the_attributes_is_refused():
    n = _new(); n["input_tensor_ids"] = ["other::out_0"]
    assert R.witnessed_arg_changes(OLD, n, TENSORS) is None


def test_a_structural_attribute_difference_is_refused():
    n = _new(); n["attributes"]["size"] = [1, S0]
    assert R.witnessed_arg_changes(OLD, n, TENSORS) is None


def test_a_mixed_change_is_refused_as_a_whole():
    n = _new(); n["attributes"]["kwargs"] = {"implicit": True}
    assert R.witnessed_arg_changes(OLD, n, TENSORS) is None


# ── the symbolized class: an integer the corrected rule now derives from the input ──
PAD = {"type": "add", "left": {"type": "mul", "left": {"type": "symbol", "id": "s1", "trace": 1}, "right": 76800, "trace": 76800}, "right": 20, "trace": 76820}
VIEW_OLD = {"op_uid": "aten.view::32", "op_type": "aten.view", "input_tensor_ids": ["aten.reflection_pad1d::0::out_0"], "output_tensor_ids": ["aten.view::32::out_0"],
            "attributes": {"args": [{"type": "tensor", "tensor_id": "aten.reflection_pad1d::0::out_0"}, {"type": "list", "value": [1, 76820]}], "kwargs": {}, "shape": [1, 76820]}}
VIEW_TENSORS = {"aten.reflection_pad1d::0::out_0": {"shape": [1, 1, 76820], "symbolic_shape": {"dims": [1, 1, PAD], "concrete": [1, 1, 76820]}},
                "aten.view::32::out_0": {"shape": [1, 76820], "symbolic_shape": {"dims": [1, PAD], "concrete": [1, 76820]}}}


def _view_new(expr=PAD):
    n = copy.deepcopy(VIEW_OLD)
    n["attributes"]["shape"][1] = expr
    n["attributes"]["args"][1]["value"][1] = expr
    return n


def test_an_integer_argument_now_derived_from_the_inputs_dim_is_symbolized():
    sites = R.witnessed_arg_changes(VIEW_OLD, _view_new(), VIEW_TENSORS)
    assert sites is not None and len(sites) == 2 and all(s["kind"] == "symbolized" for s in sites)


def test_a_symbol_the_inputs_do_not_carry_is_refused():
    """A value-matched guess (a symbol with the right trace that no input carries) is not derived."""
    guess = {"type": "symbol", "id": "s9", "trace": 76820}
    assert R.witnessed_arg_changes(VIEW_OLD, _view_new(guess), VIEW_TENSORS) is None


def test_a_symbolization_with_another_trace_is_refused():
    wrong = copy.deepcopy(PAD); wrong["trace"] = 76800
    tensors = copy.deepcopy(VIEW_TENSORS); tensors["aten.reflection_pad1d::0::out_0"]["symbolic_shape"]["dims"][2] = wrong
    assert R.witnessed_arg_changes(VIEW_OLD, _view_new(wrong), tensors) is None


def test_trace_time_provenance_is_not_a_difference():
    """The card a trace ran on (and its memory figures) is provenance, not semantics."""
    a = dict(OLD, device="cuda:2"); b = dict(OLD, device="cuda:0")
    stripped = lambda o: {k: v for k, v in o.items() if k not in R.PROVENANCE_KEYS}
    assert stripped(a) == stripped(b)
    assert R.PROVENANCE_KEYS == {"device", "memory_info"}
