"""cse (core/optim/passes/cse.py): a chain the builder recorded once per layer merges link by link from its creators.

TinyLlama's decoder builds its causal mask in every layer: ones(s, s) -> tril -> where(mask, 0, -inf), 22 times, each
read by its layer's attention; at decode that was 110 launches a token (Metal profile, 2026-11-13). The planner left
them all: it skipped nullary ops and keyed each op on its raw input ids, so no tril ever shared an input.

What this test does if the code were wrong:
- the chain cell fails when the three layers' creators, trils and wheres are not merged onto the first layer's;
- the shape cell fails when creators of DIFFERENT symbolic shapes merge;
- the empty / draw cell fails when an `empty` or a random draw is value-numbered;
- the barrier cell fails when a mutation barrier lets an expression computed before it be reused after it, or when
  the aliases decided before it are forgotten (a chain split by a barrier must still read its kept creator).
"""
from neurobrix.core.optim.passes.cse import plan_cse


def _sym(name):
    return {"type": "symbol", "id": name}


def _t(tid):
    return {"type": "tensor", "tensor_id": tid}


def _op(uid, op_type, ins, attrs=None):
    return uid, {"op_uid": uid, "op_type": op_type, "input_tensor_ids": list(ins),
                 "output_tensor_ids": [f"{uid}::out_0"], "attributes": attrs or {}}


def _mask_layer(i, sym="s1"):
    ones, tril, z, ninf, where = (f"ones{i}", f"tril{i}", f"zero{i}", f"ninf{i}", f"where{i}")
    return [
        _op(ones, "aten::ones", [], {"args": [{"type": "list", "value": [_sym(sym), _sym(sym)]}],
                                     "kwargs": {"dtype": {"type": "dtype", "value": "torch.bool"}}}),
        _op(tril, "aten::tril", [f"{ones}::out_0"], {"args": [_t(f"{ones}::out_0")]}),
        _op(z, "aten::scalar_tensor", [], {"args": [0.0]}),
        _op(ninf, "aten::scalar_tensor", [], {"args": [float("-inf")]}),
        _op(where, "aten::where", [f"{tril}::out_0", f"{z}::out_0", f"{ninf}::out_0"],
            {"args": [_t(f"{tril}::out_0"), _t(f"{z}::out_0"), _t(f"{ninf}::out_0")]}),
        _op(f"attn{i}", "aten::scaled_dot_product_attention", [f"q{i}", f"k{i}", f"v{i}", f"{where}::out_0"]),
    ]


def _dag(*layers, extra=()):
    ops = [o for layer in layers for o in layer] + list(extra)
    return {"ops": dict(ops), "execution_order": [u for u, _ in ops], "tensors": {}, "output_tensor_ids": ["final"]}


def _dropped(plan):
    return {m["drop"]: m["keep"] for m in (plan or {"merges": []})["merges"]}


def test_three_layers_masks_merge_onto_the_first_from_their_creators():
    plan = _dropped(plan_cse(_dag(_mask_layer(0), _mask_layer(1), _mask_layer(2))))
    for i in (1, 2):
        assert plan[f"ones{i}"] == "ones0" and plan[f"tril{i}"] == "tril0" and plan[f"where{i}"] == "where0"
        assert plan[f"zero{i}"] == "zero0" and plan[f"ninf{i}"] == "ninf0"
        assert f"attn{i}" not in plan                       # each layer's attention reads other q/k/v: kept
    assert len(plan) == 10


def test_creators_of_different_shapes_never_meet():
    plan = _dropped(plan_cse(_dag(_mask_layer(0, "s1"), _mask_layer(1, "s2"))))
    assert "ones1" not in plan and "tril1" not in plan and "where1" not in plan


def test_an_empty_and_a_draw_are_never_value_numbered():
    extra = [_op("e0", "aten::empty", [], {"args": [[4]]}), _op("e1", "aten::empty", [], {"args": [[4]]}),
             _op("r0", "aten::randn", [], {"args": [[4]]}), _op("r1", "aten::randn", [], {"args": [[4]]})]
    plan = _dropped(plan_cse(_dag(extra=extra)))
    assert "e1" not in plan and "r1" not in plan


def test_a_barrier_clears_what_is_available_but_keeps_the_aliases_decided():
    before = _mask_layer(0) + _mask_layer(1)
    barrier = [_op("cp", "aten::copy", ["x", "y"])]
    after = [_op("tril9", "aten::tril", ["ones1::out_0"], {"args": [_t("ones1::out_0")]})]
    plan = _dropped(plan_cse(_dag(before, barrier, after)))
    assert plan["ones1"] == "ones0" and plan["tril1"] == "tril0"
    assert "tril9" not in plan                               # tril0 was computed before the barrier: not reusable
