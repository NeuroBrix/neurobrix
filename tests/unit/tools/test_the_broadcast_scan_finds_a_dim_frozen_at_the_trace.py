"""The static broadcast scan finds a dim that agrees with its neighbour only at the trace values.

Built from the Wan VAE encoder's defect (regression matrix, 2026-09-26): a view folds the time
expression into the channel dim (T'*384) while the norm's expand keeps a literal 384; at the trace
T' = 1 and both are 384. The scan must name that op and say it breaks when TIME moves — not when
the batch moves; a correctly symbolic pair must pass; two symbols sharing one name move together.
On a scan that evaluated only the trace, or moved every symbol at once, these fail.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import symbolic_broadcast_scan as S  # noqa: E402

SYM = lambda i, t: {"type": "symbol", "id": i, "trace": t}  # noqa: E731


def _graph(dims_a, dims_b, symbols):
    return {"symbolic_context": {"symbols": symbols},
            "tensors": {"a": {"symbolic_shape": {"dims": dims_a}}, "b": {"symbolic_shape": {"dims": dims_b}}},
            "ops": {"aten.div::62": {"op_type": "aten::div", "input_tensor_ids": ["a", "b"],
                                     "parent_module": "encoder.mid.attn.0.norm"}}}


SYMBOLS = {"s0": {"name": "batch", "trace_value": 1}, "s1": {"name": "time", "trace_value": 1},
           "s2": {"name": "height", "trace_value": 14}}


def test_the_folded_time_dim_is_found_and_named():
    folded = {"type": "mul", "left": SYM("s1", 1), "right": 384}
    g = _graph([SYM("s0", 1), folded, SYM("s2", 14)], [SYM("s0", 1), 384, SYM("s2", 14)], SYMBOLS)
    found, _ = S.scan_graph(g)
    assert [f["op"] for f in found] == ["aten.div::62"]
    assert found[0]["breaks_when"] == ["time"]


def test_a_symbolic_pair_passes():
    g = _graph([SYM("s0", 1), 384, SYM("s2", 14)], [SYM("s0", 1), 1, SYM("s2", 14)], SYMBOLS)
    assert S.scan_graph(g)[0] == []


def test_two_symbols_of_one_name_move_together():
    symbols = {"s1": {"name": "seq_len", "trace_value": 23}, "s3": {"name": "seq_len", "trace_value": 23}}
    g = _graph([SYM("s1", 23), 64], [SYM("s3", 23), 64], symbols)
    assert S.scan_graph(g)[0] == []
