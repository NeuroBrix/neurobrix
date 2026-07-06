"""Unit test for the compiled seq_len scalar-promotion offset policy.

Regression guard for the synthetic-sum fuzzy-match bug: when a graph has
two seq_len symbols (e.g. trace values 14 and 1), the promotion pass
synthesizes a combined symbol `_sum_sA_sB` (trace 15). The +1 offset
tolerance — designed for single-tracer-symbol `seq_len + 1` patterns —
also applied to that synthetic sum, so a STRUCTURAL model constant 16
(the rotate-half half-width of a 32-wide per-axis RoPE chunk) was
promoted to `sum + 1`. At runtime the slice end rebound to
runtime_seq + 1, widening the rotated q/k chunk (48 instead of 32) and
crashing the first RoPE mul in compiled mode while the sequential
oracle (exact-match patching) passed.

Policy under test:
  - real tracer symbols: offset 0..1 allowed (seq_len + 1 slicing);
  - synthetic `_sum_*` symbols: EXACT match only (offset 0), which is
    the concatenated-sequence pattern the sums exist for and re-inherits
    the weight-dims collision guard checked at synthesis time.

Runnable:  PYTHONPATH=src python3 -m pytest tests/unit/runtime/test_seq_promotion_sum_offset.py -v
"""
from __future__ import annotations

from neurobrix.core.runtime.graph.compiled_sequence import CompiledSequence


def _make_dag(slice_ends):
    """Minimal DAG: two seq_len symbols (traces 14 and 9 → sum 23) and one
    aten::slice per requested end value."""
    ops = {}
    tensors = {
        # One weight so the collision filter has a realistic dims set;
        # none of its dims equal 14, 9, 10, 15, 23 or 24.
        "param::w": {"weight_name": "w", "shape": [2304, 96]},
    }
    for i, end in enumerate(slice_ends):
        uid = f"aten.slice::{i}"
        ops[uid] = {
            "op_uid": uid,
            "op_type": "aten::slice",
            "input_tensor_ids": [f"t{i}"],
            "output_tensor_ids": [f"{uid}::out_0"],
            "attributes": {
                "args": [
                    {"type": "tensor", "tensor_id": f"t{i}"},
                    {"type": "scalar", "value": 3},
                    {"type": "scalar", "value": 0},
                    {"type": "scalar", "value": end},
                ],
                "kwargs": {},
            },
        }
        tensors[f"t{i}"] = {"shape": [1, 1, 5, 64]}
    dag = {
        "symbolic_context": {
            "symbols": {
                "s9": {"name": "seq_len", "trace_value": 14},
                "s11": {"name": "seq_len", "trace_value": 9},
            }
        },
        "tensors": tensors,
        "ops": ops,
    }
    return dag, tensors, ops


def _run_promotion(slice_ends):
    dag, tensors, ops = _make_dag(slice_ends)
    seq = object.__new__(CompiledSequence)
    seq.dag = dag
    seq._promote_seq_len_scalars_to_symbolic(tensors, ops)
    return ops


def _end_arg(ops, i):
    return ops[f"aten.slice::{i}"]["attributes"]["args"][3]


def test_sum_symbol_requires_exact_match():
    """sum trace = 14 + 9 = 23: end=23 promotes (exact), end=24 must NOT
    promote via sum+1 (that fuzz is what captured structural constants)."""
    ops = _run_promotion([23, 24])

    exact = _end_arg(ops, 0)
    assert exact.get("type") == "symbol", f"exact sum match must promote: {exact}"
    assert exact["symbol_id"].startswith("_sum_")
    assert exact["offset"] == 0

    fuzzy = _end_arg(ops, 1)
    assert fuzzy.get("type") == "scalar", (
        f"sum + 1 fuzzy match must NOT promote (structural-constant "
        f"false positive): {fuzzy}"
    )
    assert fuzzy["value"] == 24


def test_single_symbol_keeps_plus_one_tolerance():
    """Real tracer symbols keep the seq_len and seq_len+1 promotions."""
    ops = _run_promotion([14, 15])

    exact = _end_arg(ops, 0)
    assert exact.get("type") == "symbol" and exact["symbol_id"] == "s9"
    assert exact["offset"] == 0

    plus_one = _end_arg(ops, 1)
    assert plus_one.get("type") == "symbol" and plus_one["symbol_id"] == "s9"
    assert plus_one["offset"] == 1


def test_structural_constant_between_sums_untouched():
    """A value matching neither symbol nor exact sum stays concrete."""
    ops = _run_promotion([16])
    arg = _end_arg(ops, 0)
    assert arg.get("type") == "scalar" and arg["value"] == 16
