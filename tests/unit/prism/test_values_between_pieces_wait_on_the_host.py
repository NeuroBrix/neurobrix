"""A component whose activations ALONE are over the budget, because a long-lived value (a skip) is live beside the
largest ops, is cut when the values between pieces wait on the host (`LayerPartitioner._greedy_swapped`,
`Partition.swapped`): a piece holds its weights, the inputs it reads and what it produces; the skip is parked
until the piece that reads it. The chain below (bytes in KB, fp32):

    a = neg(x)  4 | b = neg(a)  8 | c = neg(b)  8 | d = neg(c)  4 | z = mul(d, a)  4

Whole, c's op holds a + b + c = 20 KB: over an 18 KB budget, and no cut between ops helps ("activations alone").
Parked, x..b is one piece (a and b), c's op one (its input b 8 + c 8), d's one, z's one (d 4 + the parked skip a 4 + z 4) — at most 16 KB.

What each test would do if the code were wrong: the inputs not counted -> the 4-op piece b..z is accepted at a figure
under its real 20 KB (the piece assertion red); the skip counted as live in every piece (no parking) -> refused
(red); an op over the budget alone not named -> the refusal assertion red."""
from __future__ import annotations

from neurobrix.core.prism.layer_partition import LayerPartitioner

KB = 1024


def _t(n, **kw):
    return {"shape": [n], "dtype": "float32", **kw}


def _op(uid, op_type, ins, outs):
    return {"op_uid": uid, "op_type": op_type, "input_tensor_ids": list(ins), "output_tensor_ids": list(outs),
            "attributes": {"args": [{"type": "tensor", "tensor_id": t} for t in ins]}}


def _graph():
    T = {"input::x": _t(16, is_input=True), "a": _t(1024), "b": _t(2048), "c": _t(2048), "d": _t(1024), "z": _t(1024)}
    O = [_op("neg::0", "aten::neg", ["input::x"], ["a"]), _op("neg::1", "aten::neg", ["a"], ["b"]),
         _op("neg::2", "aten::neg", ["b"], ["c"]), _op("neg::3", "aten::neg", ["c"], ["d"]),
         _op("mul::0", "aten::mul", ["d", "a"], ["z"])]
    return {"tensors": T, "ops": {o["op_uid"]: o for o in O}, "execution_order": [o["op_uid"] for o in O],
            "input_tensor_ids": ["input::x"], "output_tensor_ids": ["z"], "torch_dtype": "float32"}


def test_whole_the_skip_makes_the_activations_alone_over_the_budget():
    part = LayerPartitioner(_graph(), {}).partition(18 * KB)
    assert not part.fits and "activations alone" in part.refusal, part.refusal


def test_parked_every_piece_holds_only_what_it_reads_and_makes():
    part = LayerPartitioner(_graph(), {}, swap_values=True).partition(18 * KB)
    assert part.fits and part.swapped, part.refusal
    assert part.peak_resident_bytes <= 18 * KB
    assert part.peak_resident_bytes == 16 * KB, part.peak_resident_bytes      # c's op: b (input) + c
    spans = [(s.first_op, s.last_op) for s in part.segments]
    assert ("neg::2", "neg::2") in spans and ("mul::0", "mul::0") in spans, spans    # c alone; the product with the parked skip


def test_an_op_over_the_budget_alone_is_named():
    part = LayerPartitioner(_graph(), {}, swap_values=True).partition(15 * KB)
    assert not part.fits and "'neg::2'" in part.refusal and "on its own" in part.refusal, part.refusal
