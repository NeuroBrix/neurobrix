"""A view of weights alone (a linear's `aten::t`) is never a seam value: each piece that reads it makes it from the
weight it loads, a piece that does not read it does not make it, and the partitioner counts that weight for the
reading op. HAT-S at 2048 px on a 16 GB V100 (2026-10-22): the cut fell between a linear's weight transpose and its
addmm, the view crossed the seam, and the next piece read it at an address its own piece had freed with its weights.

Injection (seen red, then restored green, 2026-10-22): `weight_views` returning {} -> every assertion's case red."""
from __future__ import annotations

from neurobrix.core.prism.layer_partition import LayerPartitioner, Segment, build_segment_graph


def _graph():
    t = lambda tid, shape, p=False: {"tensor_id": tid, "shape": shape, "dtype": "float32", "is_parameter": p,
                                     **({"weight_name": tid[7:]} if p else {})}
    tensors = {"input::x": t("input::x", [2, 8]), "param::w": t("param::w", [4, 8], True),
               "aten.t::0::out_0": t("aten.t::0::out_0", [8, 4]), "aten.relu::0::out_0": t("aten.relu::0::out_0", [2, 8]),
               "aten.mm::0::out_0": t("aten.mm::0::out_0", [2, 4])}
    op = lambda uid, typ, ins, outs: {"op_uid": uid, "op_type": typ, "input_tensor_ids": ins, "output_tensor_ids": outs,
                                      "attributes": {"args": [{"type": "tensor", "tensor_id": i} for i in ins]}}
    ops = {"aten.t::0": op("aten.t::0", "aten::t", ["param::w"], ["aten.t::0::out_0"]),
           "aten.relu::0": op("aten.relu::0", "aten::relu", ["input::x"], ["aten.relu::0::out_0"]),
           "aten.mm::0": op("aten.mm::0", "aten::mm", ["aten.relu::0::out_0", "aten.t::0::out_0"], ["aten.mm::0::out_0"])}
    return {"tensors": tensors, "ops": ops, "execution_order": ["aten.t::0", "aten.relu::0", "aten.mm::0"],
            "input_tensor_ids": ["input::x"], "output_tensor_ids": ["aten.mm::0::out_0"]}


def _seg(i, first, last, n):
    return Segment(index=i, first_op=first, last_op=last, op_count=n, weight_names=set(), weight_bytes=0,
                   live_bytes_at_exit=0)


def test_the_reading_piece_makes_the_view_and_the_seam_does_not_carry_it():
    g = _graph()
    first = build_segment_graph(g, _seg(0, "aten.t::0", "aten.relu::0", 2))
    second = build_segment_graph(g, _seg(1, "aten.mm::0", "aten.mm::0", 1))
    assert "aten.t::0::out_0" not in first["output_tensor_ids"], first["output_tensor_ids"]
    assert "aten.t::0" not in first["execution_order"], first["execution_order"]     # nothing there reads it
    assert second["execution_order"] == ["aten.t::0", "aten.mm::0"], second["execution_order"]
    assert "aten.t::0::out_0" not in second["segment_input_names"], second["segment_input_names"]
    assert "param::w" in second["tensors"]


def test_the_reading_op_is_priced_with_the_weight_it_views():
    assert LayerPartitioner(_graph())._op_weight_names("aten.mm::0") == {"w"}
