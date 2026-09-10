"""A literal nested inside a dim expression that became the dim expression with that literal as
its trace — PixArt-XL-2's transformer views, 2026-09-07: `s3 · 4096` → `s3 · (((h−2)//2+1) ·
((w−2)//2+1))`, the patch count the old tracer had frozen — is the closed defect, judged at the
enclosing dim the op's output carries; a nested change whose enclosing dim changes its trace is not."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import retrace_zoo as R  # noqa: E402

S3 = {"type": "symbol", "id": "s3", "trace": 2}
PATCHES = {"type": "mul",
           "left": {"type": "add", "left": {"type": "floordiv", "left": {"type": "add", "left": {"type": "symbol", "id": "s4", "trace": 128}, "right": -2, "trace": 126}, "right": 2, "trace": 63}, "right": 1, "trace": 64},
           "right": {"type": "add", "left": {"type": "floordiv", "left": {"type": "add", "left": {"type": "symbol", "id": "s5", "trace": 128}, "right": -2, "trace": 126}, "right": 2, "trace": 63}, "right": 1, "trace": 64},
           "trace": 4096}
OLD_DIM = {"type": "mul", "left": S3, "right": 4096, "trace": 8192}
NEW_DIM = {"type": "mul", "left": S3, "right": PATCHES, "trace": 8192}


def _op(dim):
    return {"op_uid": "aten.view::21", "op_type": "aten::view", "input_tensor_ids": ["x"], "output_tensor_ids": ["aten.view::21::out_0"],
            "attributes": {"args": [{"type": "tensor", "tensor_id": "x"}, {"type": "list", "value": [dim, 1152]}], "kwargs": {}}}


def test_nested_literal_symbolized_at_the_enclosing_dim():
    tensors_new = {"aten.view::21::out_0": {"symbolic_shape": {"dims": [NEW_DIM, 1152], "concrete": [8192, 1152]}}, "x": {}}
    sites = R.witnessed_arg_changes(_op(OLD_DIM), _op(NEW_DIM), tensors_new, {})
    assert sites and [s["kind"] for s in sites] == ["symbolized"]
    assert sites[0]["old"] == OLD_DIM and sites[0]["new"] == NEW_DIM


def test_a_nested_expression_with_another_trace_is_not_admitted():
    other = dict(NEW_DIM, right=dict(PATCHES, trace=4000))          # the literal 4096 is not its trace
    tensors_new = {"aten.view::21::out_0": {"symbolic_shape": {"dims": [other, 1152], "concrete": [8192, 1152]}}, "x": {}}
    assert R.witnessed_arg_changes(_op(OLD_DIM), _op(other), tensors_new, {}) is None


def test_a_nested_change_of_symbol_is_not_admitted():
    other = dict(NEW_DIM, left={"type": "symbol", "id": "s9", "trace": 2})
    tensors_new = {"aten.view::21::out_0": {"symbolic_shape": {"dims": [other, 1152], "concrete": [8192, 1152]}}, "x": {}}
    assert R.witnessed_arg_changes(_op(OLD_DIM), _op(other), tensors_new, {}) is None
