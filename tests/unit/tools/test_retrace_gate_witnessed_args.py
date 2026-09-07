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
    assert R.PROVENANCE_KEYS >= {"device", "memory_info"}


def test_a_device_argument_recorded_by_an_op_is_provenance_too():
    a = {"op_uid": "aten._to_copy::0", "attributes": {"args": [], "kwargs": {"dtype": {"type": "dtype", "value": "torch.float32"}, "device": {"type": "device", "value": "cuda:2"}},
                                                        "dtype": {"type": "dtype", "value": "torch.float32"}, "device": {"type": "device", "value": "cuda:2"}}, "device": "cuda:2"}
    b = copy.deepcopy(a)
    for d in (b["attributes"]["kwargs"]["device"], b["attributes"]["device"]):
        d["value"] = "cuda:0"
    b["device"] = "cuda:0"
    assert R.scrub_provenance(a) == R.scrub_provenance(b)
    c = copy.deepcopy(b); c["attributes"]["kwargs"]["dtype"]["value"] = "torch.float16"
    assert R.scrub_provenance(a) != R.scrub_provenance(c), "a dtype is semantics"


# ── the re-expressed class: the same extent spelled by the corrected rules' algebra ──
S1 = {"type": "symbol", "id": "s1", "trace": 112}; S2 = {"type": "symbol", "id": "s2", "trace": 80}
def _fd(x, k, tr): return {"type": "floordiv", "left": x, "right": k, "trace": tr}
def _mul(x, y, tr): return {"type": "mul", "left": x, "right": y, "trace": tr}
RAW = _mul(S1, S2, 8960)                                    # H*W
WIN = _mul(_mul(_fd(S1, 8, 14), _fd(S2, 8, 10), 140), 64, 8960)   # (H//8)*(W//8)*64


def test_eval_and_equivalence_of_dim_expressions():
    env = {"s1": 112, "s2": 80}
    assert R.eval_dim(RAW, env) == 8960 and R.eval_dim(WIN, env) == 8960
    assert R.equivalent_dims(RAW, WIN)                       # equal at 112/80, 224/160, 336/240
    assert R.equivalent_dims(S1, _mul(_fd(S1, 8, 14), 8, 112))   # (H//8)*8 == H on multiples of 8
    assert not R.equivalent_dims(RAW, _mul(_fd(S1, 8, 14), _fd(S2, 8, 10), 140))   # H*W != windows


def test_a_re_expressed_view_argument_is_admitted():
    old = {"op_uid": "aten.view::21", "op_type": "aten.view", "input_tensor_ids": ["v::out_0"], "output_tensor_ids": ["aten.view::21::out_0"],
           "attributes": {"args": [{"type": "tensor", "tensor_id": "v::out_0"}, {"type": "list", "value": [RAW, 180]}], "kwargs": {}, "shape": [RAW, 180]}}
    new = copy.deepcopy(old); new["attributes"]["shape"][0] = WIN; new["attributes"]["args"][1]["value"][0] = WIN
    tensors = {"v::out_0": {"shape": [140, 64, 180], "symbolic_shape": {"dims": [_mul(_fd(S1, 8, 14), _fd(S2, 8, 10), 140), 64, 180], "concrete": [140, 64, 180]}},
               "aten.view::21::out_0": {"shape": [8960, 180], "symbolic_shape": {"dims": [WIN, 180], "concrete": [8960, 180]}}}
    sites = R.witnessed_arg_changes(old, new, tensors)
    assert sites is not None and len(sites) == 2 and all(x["kind"] == "re-expressed" for x in sites)


def test_a_re_expression_that_is_not_equivalent_is_refused():
    other = _mul(_fd(S1, 8, 14), _fd(S2, 8, 10), 140)        # windows, not tokens: trace 140 ≠ 8960
    bad = copy.deepcopy(other); bad["trace"] = 8960            # a lie about the trace
    old = {"op_uid": "aten.view::21", "op_type": "aten.view", "input_tensor_ids": ["v::out_0"], "output_tensor_ids": ["aten.view::21::out_0"],
           "attributes": {"args": [{"type": "tensor", "tensor_id": "v::out_0"}, {"type": "list", "value": [RAW, 180]}], "kwargs": {}, "shape": [RAW, 180]}}
    new = copy.deepcopy(old); new["attributes"]["shape"][0] = bad; new["attributes"]["args"][1]["value"][0] = bad
    tensors = {"aten.view::21::out_0": {"shape": [8960, 180], "symbolic_shape": {"dims": [bad, 180], "concrete": [8960, 180]}}}
    assert R.witnessed_arg_changes(old, new, tensors) is None


def test_an_integer_symbolized_to_the_outputs_dim_is_admitted():
    win = _mul(_fd(S1, 8, 14), _fd(S2, 8, 10), 140)
    old = {"op_uid": "aten.view::6", "op_type": "aten.view", "input_tensor_ids": ["m::out_0"], "output_tensor_ids": ["aten.view::6::out_0"],
           "attributes": {"args": [{"type": "tensor", "tensor_id": "m::out_0"}, {"type": "list", "value": [140, 64, 180]}], "kwargs": {}, "shape": [140, 64, 180]}}
    new = copy.deepcopy(old); new["attributes"]["shape"][0] = win; new["attributes"]["args"][1]["value"][0] = win
    tensors = {"m::out_0": {"shape": [8960, 180], "symbolic_shape": {"dims": [RAW, 180], "concrete": [8960, 180]}},
               "aten.view::6::out_0": {"shape": [140, 64, 180], "symbolic_shape": {"dims": [win, 64, 180], "concrete": [140, 64, 180]}}}
    sites = R.witnessed_arg_changes(old, new, tensors)
    assert sites is not None and all(x["kind"] == "symbolized" for x in sites)


def test_a_vendor_minus_one_may_become_the_derived_output_dim():
    win = _mul(_fd(S1, 8, 14), _fd(S2, 8, 10), 140)
    old = {"op_uid": "aten.view::23", "op_type": "aten.view", "input_tensor_ids": ["w::out_0"], "output_tensor_ids": ["aten.view::23::out_0"],
           "attributes": {"args": [{"type": "tensor", "tensor_id": "w::out_0"}, {"type": "list", "value": [-1, 8, 8, 180]}], "kwargs": {}, "shape": [-1, 8, 8, 180]}}
    new = copy.deepcopy(old); new["attributes"]["shape"][0] = win; new["attributes"]["args"][1]["value"][0] = win
    tensors = {"w::out_0": {"shape": [140, 64, 180], "symbolic_shape": {"dims": [win, 64, 180], "concrete": [140, 64, 180]}},
               "aten.view::23::out_0": {"shape": [140, 8, 8, 180], "symbolic_shape": {"dims": [win, 8, 8, 180], "concrete": [140, 8, 8, 180]}}}
    sites = R.witnessed_arg_changes(old, new, tensors)
    assert sites is not None and all(x["kind"] == "symbolized" for x in sites)
    wrong = copy.deepcopy(win); wrong["trace"] = 150                    # not the witnessed extent
    new2 = copy.deepcopy(old); new2["attributes"]["shape"][0] = wrong; new2["attributes"]["args"][1]["value"][0] = wrong
    t2 = copy.deepcopy(tensors); t2["aten.view::23::out_0"]["symbolic_shape"]["dims"][0] = wrong
    assert R.witnessed_arg_changes(old, new2, t2) is None


# ── the batch split restored: [1, σ·s, D] → [σ, s, D] ────────────────────────────────
B = {"type": "symbol", "id": "s0", "trace": 1}; SEQ = {"type": "symbol", "id": "s1", "trace": 7}
FOLD = {"type": "mul", "left": B, "right": SEQ, "trace": 7}


def test_a_folded_batch_split_back_is_admitted():
    old = {"op_uid": "aten.view::2", "op_type": "aten.view", "input_tensor_ids": ["m::out_0"], "output_tensor_ids": ["aten.view::2::out_0"],
           "attributes": {"args": [{"type": "tensor", "tensor_id": "m::out_0"}, {"type": "list", "value": [1, FOLD, 1280]}], "kwargs": {}, "shape": [1, FOLD, 1280]}}
    new = copy.deepcopy(old)
    for lst in (new["attributes"]["shape"], new["attributes"]["args"][1]["value"]):
        lst[0], lst[1] = B, SEQ
    tensors = {"m::out_0": {"shape": [7, 1280], "symbolic_shape": {"dims": [FOLD, 1280], "concrete": [7, 1280]}},
               "aten.view::2::out_0": {"shape": [1, 7, 1280], "symbolic_shape": {"dims": [B, SEQ, 1280], "concrete": [1, 7, 1280]}}}
    sites = R.witnessed_arg_changes(old, new, tensors)
    assert sites is not None and len(sites) == 2 and all(x["kind"] == "batch-split-restored" for x in sites)


def test_the_fold_itself_is_refused():
    """The reverse direction — a symbol turned into a literal 1 — is the regression, never admitted."""
    new_is_old = {"op_uid": "aten.view::2", "op_type": "aten.view", "input_tensor_ids": ["m::out_0"], "output_tensor_ids": ["aten.view::2::out_0"],
                  "attributes": {"args": [{"type": "tensor", "tensor_id": "m::out_0"}, {"type": "list", "value": [B, SEQ, 1280]}], "kwargs": {}, "shape": [B, SEQ, 1280]}}
    folded = copy.deepcopy(new_is_old)
    for lst in (folded["attributes"]["shape"], folded["attributes"]["args"][1]["value"]):
        lst[0], lst[1] = 1, FOLD
    tensors = {"aten.view::2::out_0": {"shape": [1, 7, 1280], "symbolic_shape": {"dims": [1, FOLD, 1280], "concrete": [1, 7, 1280]}}}
    assert R.witnessed_arg_changes(new_is_old, folded, tensors) is None


def test_a_vendor_module_rename_is_naming_not_semantics():
    a = dict(OLD, parent_module="final_layer.norm_final"); b = dict(OLD, parent_module="final_layer.final_norm")
    assert R.scrub_provenance(a) == R.scrub_provenance(b)


def test_a_flatten_that_regains_its_batch_factor_is_admitted():
    old = {"op_uid": "aten.view::15", "op_type": "aten.view", "input_tensor_ids": ["h::out_0"], "output_tensor_ids": ["aten.view::15::out_0"],
           "attributes": {"args": [{"type": "tensor", "tensor_id": "h::out_0"}, {"type": "list", "value": [SEQ, 1536]}], "kwargs": {}, "shape": [SEQ, 1536]}}
    new = copy.deepcopy(old); new["attributes"]["shape"][0] = FOLD; new["attributes"]["args"][1]["value"][0] = FOLD
    tensors = {"h::out_0": {"shape": [1, 23, 1536], "symbolic_shape": {"dims": [B, SEQ, 1536], "concrete": [1, 23, 1536]}},
               "aten.view::15::out_0": {"shape": [23, 1536], "symbolic_shape": {"dims": [FOLD, 1536], "concrete": [23, 1536]}}}
    sites = R.witnessed_arg_changes(old, new, tensors)
    assert sites is not None and all(x["kind"] == "batch-factor-restored" for x in sites) and len(sites) == 2


def test_consumer_lists_are_verified_against_the_new_graphs_ops():
    g = {"ops": [{"op_uid": "aten.mul::0", "input_tensor_ids": ["w", "x"], "output_tensor_ids": ["y"]}],
         "tensors": {"w": {"consumer_op_uids": ["aten.mul::0"]}, "x": {"consumer_op_uids": ["aten.add::9"]}, "y": {}}}
    assert R.derived_consumers_consistent(g) == 1


def test_a_batch_counted_twice_is_corrected_to_once():
    twice = _mul(B, _mul(B, 1500, 1500), 1500); once = _mul(B, 1500, 1500)
    assert R.equivalent_modulo_unit_factors(twice, once) and not R.equivalent_dims(twice, once)
    old = {"op_uid": "aten.view::11", "op_type": "aten.view", "input_tensor_ids": ["e::out_0"], "output_tensor_ids": ["aten.view::11::out_0"],
           "attributes": {"args": [{"type": "tensor", "tensor_id": "e::out_0"}, {"type": "list", "value": [twice, 1280]}], "kwargs": {}, "shape": [twice, 1280]}}
    new = copy.deepcopy(old); new["attributes"]["shape"][0] = once; new["attributes"]["args"][1]["value"][0] = once
    tensors = {"aten.view::11::out_0": {"shape": [1500, 1280], "symbolic_shape": {"dims": [once, 1280], "concrete": [1500, 1280]}}}
    sites = R.witnessed_arg_changes(old, new, tensors)
    assert sites is not None and all(x["kind"] == "unit-factor-corrected" for x in sites)
    # a change in a non-unit symbol is never this class
    other = _mul(B, 1600, 1600)
    assert not R.equivalent_modulo_unit_factors(once, other)


def test_a_vendor_minus_one_restored_and_a_slice_end_to_the_end_are_admitted():
    tp = _fd(S1, 8, 14)
    old = {"op_uid": "aten.view::21", "op_type": "aten.view", "input_tensor_ids": ["p::out_0"], "output_tensor_ids": ["aten.view::21::out_0"],
           "attributes": {"args": [{"type": "tensor", "tensor_id": "p::out_0"}, {"type": "list", "value": [1, tp, 750]}], "kwargs": {}, "shape": [1, tp, 750]}}
    new = copy.deepcopy(old)
    for lst in (new["attributes"]["shape"], new["attributes"]["args"][1]["value"]):
        lst[0], lst[2] = B, -1
    tensors = {"p::out_0": {"shape": [1, 14, 750], "symbolic_shape": {"dims": [B, tp, 750], "concrete": [1, 14, 750]}},
               "aten.view::21::out_0": {"shape": [1, 14, 750], "symbolic_shape": {"dims": [B, tp, 750], "concrete": [1, 14, 750]}}}
    sites = R.witnessed_arg_changes(old, new, tensors)
    assert sites is not None and {x["kind"] for x in sites} == {"symbolized", "inference-restored"}
    old_s = {"op_uid": "aten.slice::8", "op_type": "aten.slice", "input_tensor_ids": ["q::out_0"], "output_tensor_ids": ["aten.slice::8::out_0"],
             "attributes": {"args": [{"type": "tensor", "tensor_id": "q::out_0"}, {"type": "scalar", "value": 2}, {"type": "scalar", "value": 0}, tp], "kwargs": {}, "dim": 2, "start": 0, "end": tp}}
    new_s = copy.deepcopy(old_s); new_s["attributes"]["args"][3] = {"type": "scalar", "value": R.INT64_MAX}; new_s["attributes"]["end"] = R.INT64_MAX
    t2 = {"q::out_0": {"shape": [1, 8, 14], "symbolic_shape": {"dims": [1, 8, tp], "concrete": [1, 8, 14]}}, "aten.slice::8::out_0": {"shape": [1, 8, 14], "symbolic_shape": {"dims": [1, 8, tp], "concrete": [1, 8, 14]}}}
    sites = R.witnessed_arg_changes(old_s, new_s, t2)
    assert sites is not None and all(x["kind"] == "slice-end-to-the-end" for x in sites)


def test_an_expression_of_unit_symbols_only_may_become_the_literal_it_was_worth():
    B2 = {"type": "symbol", "id": "s2", "trace": 1}
    fold = _mul(_mul(B, 50, 50), _mul(B2, 33, 33), 1650)
    old = {"op_uid": "aten.view::2", "op_type": "aten.view", "input_tensor_ids": ["r::out_0"], "output_tensor_ids": ["aten.view::2::out_0"],
           "attributes": {"args": [{"type": "tensor", "tensor_id": "r::out_0"}, {"type": "list", "value": [fold, 640]}], "kwargs": {}, "shape": [fold, 640]}}
    new = copy.deepcopy(old); new["attributes"]["shape"][0] = 1650; new["attributes"]["args"][1]["value"][0] = 1650
    tensors = {"aten.view::2::out_0": {"shape": [1650, 640], "symbolic_shape": {"dims": [1650, 640], "concrete": [1650, 640]}}}
    sites = R.witnessed_arg_changes(old, new, tensors)
    assert sites is not None and all(x["kind"] == "unit-only-literalized" for x in sites)
    # a real symbol turned into its trace value is still refused
    old2 = copy.deepcopy(old); old2["attributes"]["shape"][0] = _mul(SEQ, 50, 1150); old2["attributes"]["args"][1]["value"][0] = _mul(SEQ, 50, 1150)
    new2 = copy.deepcopy(new); new2["attributes"]["shape"][0] = 1150; new2["attributes"]["args"][1]["value"][0] = 1150
    t2 = {"aten.view::2::out_0": {"shape": [1150, 640], "symbolic_shape": {"dims": [1150, 640], "concrete": [1150, 640]}}}
    assert R.witnessed_arg_changes(old2, new2, t2) is None


def test_a_scalar_argument_bound_to_a_symbol_follows_the_namespace_remap():
    """An `arange` whose end is a symbol carries `symbol_id`, not `id`: renamed with the rest."""
    op = {"op_uid": "aten.arange::4", "attributes": {"args": [{"type": "symbol", "symbol_id": "s0", "trace_value": 23}],
                                                     "end": {"type": "symbol", "symbol_id": "s0", "trace_value": 23}}}
    out = R.rewrite_symbols(op, {"s0": "s1", "s1": "s0"})
    assert out["attributes"]["end"]["symbol_id"] == "s1" and out["attributes"]["args"][0]["symbol_id"] == "s1"
    assert R.rewrite_symbols({"type": "symbol", "id": "s0", "trace": 23}, {"s0": "s1"})["id"] == "s1"
