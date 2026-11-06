"""An op whose precision wrapper casts an operand to its compute width holds the copy while it runs
(`runtime_widths.ExecutedWidths.casts`): the profiler prices it for a component held whole, and the layer partition's op
curves price it too — at the partitioner's own symbols, so a sliced stretch's pass prices its slice's copy.

Sana_1600M_4Kpx_BF16 native on the M4 Pro (the Mac's Metal gate, 2026-11-06): the VAE's sliced stretch upsample::2..
conv::69 priced at 12.9 GB in 2 slices of 79 rows; the run died at conv::62 when the DtypeEngine cast its fp32 input to
the compute width (`dtype/engine.py` `_cast`: 4.00 GiB asked beside 15.02 GiB). The solver also dropped the casts when
it kept the request's widths for the partitioner (`dict(widths)`).

What this test does if the code were wrong: the cast left off the op's figure, or sized at the request where a pass runs
a slice — red."""
from __future__ import annotations

from neurobrix.core.prism.layer_partition import LayerPartitioner
from neurobrix.core.prism.runtime_widths import ExecutedWidths


def _t(shape, dtype="float32", **kw):
    return {"shape": list(shape), "dtype": dtype, **kw}


def _op(uid, op_type, ins, outs):
    return {"op_uid": uid, "op_type": op_type, "input_tensor_ids": list(ins), "output_tensor_ids": list(outs),
            "attributes": {"args": [{"type": "tensor", "tensor_id": t} for t in ins]}}


def _graph():
    T = {"input::x": _t([1, 8, "s0", 64], is_input=True), "a": _t([1, 8, "s0", 64]), "c": _t([1, 8, "s0", 64])}
    O = [_op("relu::0", "aten::relu", ["input::x"], ["a"]), _op("gelu::0", "aten::gelu", ["a"], ["c"])]
    return {"tensors": T, "ops": {o["op_uid"]: o for o in O}, "execution_order": [o["op_uid"] for o in O],
            "input_tensor_ids": ["input::x"], "output_tensor_ids": ["c"], "torch_dtype": "float32",
            "symbols": {"s0": {"trace_value": 16}}}


def _widths(cast):
    w = ExecutedWidths({"input::x": 4, "a": 4, "c": 4})
    w.casts = {"gelu::0": [("a", 2)]} if cast else {}
    return w


def test_the_cast_copy_is_on_the_op_s_figure_at_the_partitioner_s_symbols():
    g = _graph()
    i = g["execution_order"].index("gelu::0")
    for rows in (64, 16):
        plain = LayerPartitioner(g, {}, symbol_map={"s0": rows}, compute_dtype_bytes=2, widths=_widths(False))
        cast = LayerPartitioner(g, {}, symbol_map={"s0": rows}, compute_dtype_bytes=2, widths=_widths(True))
        assert cast.op_peak_curve()[i] == plain.op_peak_curve()[i] + 8 * rows * 64 * 2
        assert cast.op_peak_curve()[0] == plain.op_peak_curve()[0]
