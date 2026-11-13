"""cse (core/optim/passes/cse.py): constants recorded with the same bytes are one value, and so are views of one value.

TinyLlama's container records its RoPE tables once per layer (`block.N.attn.rotary_embed.cos_cached` and `sin_cached`,
44 constants of the same bytes); each layer slices its own and gathers the step's rows from it (`aten::index`), so a
decode step gathered the same rows 44 times, then widened them 44 times for the rope (Metal launch list, 2026-11-13:
88 launches a token). The planner keyed every op on its raw input ids, so no two gathers ever met.

What this test does if the code were wrong:
- the merge cell fails when the second layer's gather is not dropped onto the first's, or when a VIEW is dropped (views
  are keyed as their twin, never removed: removing them was the measured byte change of 2026-08-19);
- the bytes cell fails when constants of different bytes, or of the same bytes under another dtype or shape, meet;
- the barrier cell fails when the equality of constants and views outlives a mutation barrier (a write may have reached
  one of them through a view).
"""
from neurobrix.core.optim.passes.cse import plan_cse


def _t(tid):
    return {"type": "tensor", "tensor_id": tid}


def _op(uid, op_type, ins, attrs=None):
    return uid, {"op_uid": uid, "op_type": op_type, "input_tensor_ids": list(ins),
                 "output_tensor_ids": [f"{uid}::out_0"], "attributes": attrs or {"args": [_t(i) for i in ins]}}


def _const(tid, data, dtype="bfloat16", shape=(2048, 64)):
    return tid, {"tensor_id": tid, "constant": True, "constant_data": data, "dtype": dtype, "shape": list(shape),
                 "is_parameter": True}


def _layer(i):
    table = f"param::block.{i}.cos_cached"
    return [
        _op(f"slice{i}", "aten::slice", [table], {"args": [_t(table), 0, 0, {"type": "symbol", "id": "s1"}]}),
        _op(f"index{i}", "aten::index", [f"slice{i}::out_0", "input::position_ids"],
            {"args": [_t(f"slice{i}::out_0"), {"type": "tensor_tuple", "tensor_ids": ["input::position_ids"]}]}),
        _op(f"unsq{i}", "aten::unsqueeze", [f"index{i}::out_0"], {"args": [_t(f"index{i}::out_0"), 1]}),
        _op(f"mul{i}", "aten::mul", [f"q{i}", f"unsq{i}::out_0"]),
    ]


def _dag(tables, *layers, extra=()):
    ops = [o for layer in layers for o in layer] + list(extra)
    return {"ops": dict(ops), "execution_order": [u for u, _ in ops], "tensors": dict(tables),
            "output_tensor_ids": ["final"]}


def _dropped(plan):
    return {m["drop"]: m["keep"] for m in (plan or {"merges": []})["merges"]}


def _same_tables(n, data="QUJD"):
    return [_const(f"param::block.{i}.cos_cached", data) for i in range(n)]


def test_the_layers_gathers_from_identical_tables_merge_onto_the_first():
    plan = plan_cse(_dag(_same_tables(3), _layer(0), _layer(1), _layer(2)))
    dropped = _dropped(plan)
    assert dropped == {"index1": "index0", "index2": "index0"}       # the gathers merge; slices, views, muls stay
    m = next(m for m in plan["merges"] if m["drop"] == "index1")
    assert m["alias"] == [["index1::out_0", "index0::out_0"]]


def test_constants_of_other_bytes_dtype_or_shape_never_meet():
    for t1 in (_const("param::block.1.cos_cached", "QUJE"),
               _const("param::block.1.cos_cached", "QUJD", dtype="float16"),
               _const("param::block.1.cos_cached", "QUJD", shape=(1024, 128))):
        tables = [_const("param::block.0.cos_cached", "QUJD"), t1]
        assert _dropped(plan_cse(_dag(tables, _layer(0), _layer(1)))) == {}


def test_the_equality_of_constants_and_views_ends_at_a_barrier():
    barrier = [_op("cp", "aten::copy", ["slice0::out_0", "y"])]
    plan = _dropped(plan_cse(_dag(_same_tables(2), _layer(0), barrier, _layer(1))))
    assert "index1" not in plan
