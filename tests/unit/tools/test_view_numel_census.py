"""The census names a view whose inferred slot breaks the element count off the trace,
and stays silent on a sound one.

granite-speech-3.3-8b, 2026-09-26: [1, 31, 4096] -> [1, 31, 32, 128] with the head count
recorded as `s0 + s1` reproduced the trace and died at 209 tokens. The cells below are that
graph in miniature (RED on a census that only checked the trace point) and its corrected
form (literal 32), plus the LLM flatten `[s0, s1, H] -> [-1, H]` whose slot is `s0 * s1`
and is sound, plus a corrupted-at-trace site, which is counted and never evaluated.

Run: python -m pytest tests/unit/tools/test_view_numel_census.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

TOOL = Path(__file__).resolve().parents[3] / "tools" / "view_numel_census.py"


def _tool():
    spec = importlib.util.spec_from_file_location("view_numel_census", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


S0 = {"type": "symbol", "id": "s0", "trace": 1}
S1 = {"type": "symbol", "id": "s1", "trace": 31}
SUM = {"type": "add", "left": S0, "right": S1, "trace": 32}
PROD = {"type": "mul", "left": S0, "right": S1, "trace": 31}
SYMBOLS = {"s0": {"name": "batch", "trace_value": 1}, "s1": {"name": "seq_len", "trace_value": 31}}


def _graph(in_dims, in_shape, out_dims, out_shape, target):
    return {
        "symbolic_context": {"symbols": SYMBOLS, "expressions": {}},
        "tensors": {"x": {"shape": in_shape, "symbolic_shape": {"dims": in_dims}},
                    "y": {"shape": out_shape, "symbolic_shape": {"dims": out_dims}}},
        "ops": {"aten.view::0": {"op_uid": "aten.view::0", "op_type": "aten::view",
                                 "input_tensor_ids": ["x"], "output_tensor_ids": ["y"],
                                 "attributes": {"args": [{"type": "tensor", "tensor_id": "x"},
                                                         {"type": "list", "value": target}]}}},
    }


def test_the_granite_head_count_as_a_sum_is_named():
    m = _tool()
    g = _graph([S0, S1, 4096], [1, 31, 4096], [S0, S1, SUM, 128], [1, 31, 32, 128], [S0, S1, -1, 128])
    one = m.census_one(g)
    assert one["evaluated"] == 1 and len(one["misbound"]) == 1
    site = one["misbound"][0]
    assert site["slot"] == 2 and site["element_count_implies"] == 32 and site["off_trace_out"][2] == 66


def test_the_literal_head_count_is_sound():
    m = _tool()
    g = _graph([S0, S1, 4096], [1, 31, 4096], [S0, S1, 32, 128], [1, 31, 32, 128], [S0, S1, -1, 128])
    assert m.census_one(g) == {"evaluated": 1, "misbound": [], "corrupted": 0, "unevaluable": 0}


def test_the_flatten_whose_slot_is_the_product_is_sound():
    m = _tool()
    g = _graph([S0, S1, 4096], [1, 31, 4096], [PROD, 4096], [31, 4096], [-1, 4096])
    assert m.census_one(g)["misbound"] == []


def test_a_batch_frozen_to_one_in_the_target_is_named():
    m = _tool()
    g = _graph([S0, S1, 32, 128], [1, 31, 32, 128], [1, S1, 4096], [1, 31, 4096], [1, S1, -1])
    one = m.census_one(g)
    assert len(one["misbound"]) == 1 and one["misbound"][0]["element_count_implies"] == 12288


def test_a_site_corrupted_at_the_trace_is_counted_not_evaluated():
    m = _tool()
    g = _graph([S0, S1, 4096], [1, 31, 4096], [S0, S1, SUM, 128], [1, 31, 40, 128], [S0, S1, -1, 128])
    assert m.census_one(g) == {"evaluated": 0, "misbound": [], "corrupted": 1, "unevaluable": 0}


def test_a_census_over_zero_graphs_refuses(tmp_path):
    m = _tool()
    import pytest
    with pytest.raises(SystemExit, match="zero graphs"):
        m.census(tmp_path)
