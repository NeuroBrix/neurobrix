"""The drift-site detector names the FIRST op, in the oracle's order, whose
per-op window deviates beyond the bound between two engines' dumps."""
from __future__ import annotations

import json

from neurobrix.core.dtype import drift


def _dump(path, records):
    with open(path, "w") as fh:
        for r in records:
            fh.write(json.dumps({"engine": "x", "record": r}) + "\n")


def _rec(comp, tid, uid, head, last=None, op="aten.mm"):
    return {"component": comp, "tid": tid, "op_uid": uid, "op_type": op, "dtype": "fp16",
            "shape": [1, 4], "head10": head, "last_pos10": last if last is not None else head, "l2_norm": 1.0}


def test_first_site_in_oracle_order_and_cascade_behind_it(tmp_path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    _dump(a, [_rec("model", "t0", "aten.mm::0", [1.0, 2.0, 3.0]),
              _rec("model", "t1", "aten.add::0", [10.0, 20.0, 30.0]),
              _rec("model", "t2", "aten.mm::1", [0.5, 0.5, 0.5]),
              _rec("lm_head", "t0", "aten.mm::0", [7.0, 7.0, 7.0])])       # uids restart per component
    _dump(b, [_rec("model", "t0", "aten.mm::0", [1.0, 2.0, 3.0001]),      # within the bound
              _rec("model", "t2", "aten.mm::1", [0.5, 0.5, -0.5]),        # cascade (after the site)
              _rec("model", "t1", "aten.add::0", [10.0, 20.0, 33.0]),     # the site: 3/33 = 9 %
              _rec("lm_head", "t0", "aten.mm::0", [7.0, 7.0, 7.0])])
    rep = drift.detect(a, b, bound=0.02, top=3)
    assert rep.ops_a == 4 and rep.matched == 4 and rep.missing_in_b == 0
    assert rep.first is not None and rep.first.op_uid == "aten.add::0" and rep.first.component == "model"
    assert rep.first.index == 1                                             # the oracle's order, not the file's
    assert rep.over_bound == 2 and rep.top[0].op_uid == "aten.mm::1"        # the largest is the cascade
    assert rep.first_same_dtype is rep.first and rep.policy_sites == 0
    text = drift.describe(rep)
    assert "DRIFT SITE model/aten.add::0" in text
    d = rep.to_dict()
    assert d["first"]["rel_dev"] > 0.02 and json.dumps(d)


def test_no_site_when_within_the_bound_and_missing_ops_are_counted(tmp_path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    _dump(a, [_rec("model", "t0", "aten.mm::0", [1.0, 2.0]), _rec("model", "t1", "aten.add::0", [3.0, 4.0])])
    _dump(b, [_rec("model", "t0", "aten.mm::0", [1.0, 2.0])])
    rep = drift.detect(a, b)
    assert rep.first is None and rep.missing_in_b == 1 and rep.over_bound == 0
    assert "no drift site" in drift.describe(rep)


def test_a_dtype_disagreement_is_a_policy_site_not_a_kernel_site(tmp_path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    ra = _rec("model", "t0", "aten.mm::0", [1.0, 1.0]); ra["dtype"] = "torch.float16"
    rb = _rec("model", "t0", "aten.mm::0", [1.0, 1.1]); rb["dtype"] = "fp32"           # the engine kept fp32
    ra2 = _rec("model", "t1", "aten.add::0", [2.0, 2.0]); rb2 = _rec("model", "t1", "aten.add::0", [2.0, 2.5])
    _dump(a, [ra, ra2]); _dump(b, [rb, rb2])
    rep = drift.detect(a, b, bound=0.02)
    assert rep.first.op_uid == "aten.mm::0" and rep.policy_sites == 1
    assert rep.first_same_dtype.op_uid == "aten.add::0"
    assert "KERNEL DRIFT SITE model/aten.add::0" in drift.describe(rep)


def test_an_integer_origin_is_a_discrete_decision_named_with_the_float_deviation_before_it(tmp_path):
    """openaudio-s1-mini, 2026-09-06: the first over-bound op was a slice of int64 codes in the
    quantizer (rel_dev 0.34) while every float op before it stayed under 0.01 — a code that
    flipped on a sub-bound distance, not a kernel to read. The report says so and names the
    largest float deviation before the flip."""
    a = tmp_path / "a.jsonl"; b = tmp_path / "b.jsonl"
    rows_a = [{"component": "q", "tid": "aten.mm::0::out_0", "op_uid": "aten.mm::0", "op_type": "aten::mm", "dtype": "torch.float16", "shape": [4], "head10": [1.0, 2.0, 3.0, 4.0], "l2_norm": 5.477},
              {"component": "q", "tid": "aten.argmin::0::out_0", "op_uid": "aten.argmin::0", "op_type": "aten::argmin", "dtype": "torch.int64", "shape": [4], "head10": [3, 1, 2, 0], "l2_norm": 3.74}]
    rows_b = [{"component": "q", "tid": "aten.mm::0::out_0", "op_uid": "aten.mm::0", "op_type": "aten::mm", "dtype": "fp16", "shape": [4], "head10": [1.005, 2.0, 3.0, 4.0], "l2_norm": 5.478},
              {"component": "q", "tid": "aten.argmin::0::out_0", "op_uid": "aten.argmin::0", "op_type": "aten::argmin", "dtype": "int64", "shape": [4], "head10": [3, 2, 2, 0], "l2_norm": 4.12}]
    a.write_text("\n".join(json.dumps(r) for r in rows_a) + "\n"); b.write_text("\n".join(json.dumps(r) for r in rows_b) + "\n")
    rep = drift.detect(str(a), str(b), bound=0.02)
    assert rep.first.op_uid == "aten.argmin::0" and rep.origin_class == "discrete"
    assert rep.float_before is not None and rep.float_before.op_uid == "aten.mm::0" and rep.float_before.rel_dev < 0.02
    assert "DISCRETE" in drift.describe(rep)
    d = rep.to_dict()
    assert d["origin_class"] == "discrete" and d["float_before"]["op_uid"] == "aten.mm::0"


def test_a_same_dtype_arithmetic_origin_is_a_kernel_site(tmp_path):
    a = tmp_path / "a.jsonl"; b = tmp_path / "b.jsonl"
    ra = {"component": "e", "tid": "aten.glu::0::out_0", "op_uid": "aten.glu::0", "op_type": "aten::glu", "dtype": "torch.float16", "shape": [4], "head10": [1.0, 2.0, 3.0, 4.0], "l2_norm": 5.477}
    rb = {**ra, "dtype": "fp16", "head10": [1.1, 2.0, 3.0, 4.0], "l2_norm": 5.5}
    a.write_text(json.dumps(ra) + "\n"); b.write_text(json.dumps(rb) + "\n")
    rep = drift.detect(str(a), str(b), bound=0.02)
    assert rep.origin_class == "kernel" and rep.first_same_dtype.op_uid == "aten.glu::0" and rep.float_before is None
