"""The chain analyser judged "never carried" by size ARGUMENTS alone: a graph whose only op is an
embedding carries `seq_len` in its output shape and names it nowhere, so canary-qwen's
`embed_tokens` read as frozen and the census dropped the whole model (2026-09-21). A dimension
that lives in an operation's output shape is carried.

Shapes: the trace prime 23 for the sequence; 2048 the embedding width — chosen so the literal
23 appears nowhere as a weight extent and the verdict cannot lean on the parameter-extent excuse.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

TOOL = Path(__file__).resolve().parents[3] / "tools" / "where_the_symbol_chain_breaks.py"


def _tool():
    spec = importlib.util.spec_from_file_location("where_the_symbol_chain_breaks", TOOL)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _sym(sid, trace):
    return {"type": "symbol", "id": sid, "trace": trace}


def _embedding_only_graph(tmp_path, output_carries: bool):
    out_dims = [_sym("s0", 1), _sym("s1", 23), 2048] if output_carries else [1, 23, 2048]
    d = {
        "tensors": {
            "input::input": {"shape": [1, 23], "symbolic_shape": {"dims": [_sym("s0", 1), _sym("s1", 23)]}},
            "param::weight": {"is_parameter": True, "shape": [32000, 2048]},
            "aten.embedding::0::out_0": {"shape": [1, 23, 2048], "symbolic_shape": {"dims": out_dims}},
        },
        "ops": {"aten.embedding::0": {"op_type": "aten::embedding",
                                      "input_tensor_ids": ["param::weight", "input::input"],
                                      "output_tensor_ids": ["aten.embedding::0::out_0"],
                                      "input_shapes": [[32000, 2048], [1, 23]], "output_shapes": [[1, 23, 2048]],
                                      "attributes": {"args": [{"type": "tensor", "tensor_id": "param::weight"},
                                                              {"type": "tensor", "tensor_id": "input::input"}], "kwargs": {}}}},
        "execution_order": ["aten.embedding::0"],
        "symbolic_context": {"symbols": {"s0": {"name": "batch", "trace_value": 1, "source": "input::input::dim_0"},
                                         "s1": {"name": "seq_len", "trace_value": 23, "source": "input::input::dim_1"}},
                             "expressions": {}},
    }
    p = tmp_path / "graph.json"
    p.write_text(json.dumps(d))
    return p


def _row(rows, sid):
    return next(r for r in rows if r.get("symbol") == sid)


def test_a_dimension_carried_by_an_output_shape_is_carried(tmp_path):
    rows = _tool().analyse(_embedding_only_graph(tmp_path, output_carries=True))
    r = _row(rows, "s1")
    assert r["never_carried"] is False, r
    assert r["shape_carriers"] == 1, r


def test_a_dimension_no_shape_and_no_argument_carries_is_never_carried(tmp_path):
    rows = _tool().analyse(_embedding_only_graph(tmp_path, output_carries=False))
    r = _row(rows, "s1")
    assert r["never_carried"] is True, r
