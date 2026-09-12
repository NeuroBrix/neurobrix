"""The census must name a blind axis, and must not name a deliberate one.

It answers "where would a symbolic-shape defect be INVISIBLE", which is a larger
set than "where is one". An axis traced at a value where two rules agree cannot be
cleared by the trace-point check, however many invariants pass — register 29.

Run: PYTHONPATH=tools python -m pytest tests/unit/tools/test_symbol_collision_census.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

TOOL = Path(__file__).resolve().parents[3] / "tools" / "symbol_collision_census.py"


def _tool():
    spec = importlib.util.spec_from_file_location("symbol_collision_census", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _graph(tmp_path, symbols, params=()):
    tensors = {f"param::w{i}": {"is_parameter": True, "shape": list(s)}
               for i, s in enumerate(params)}
    d = {"ops": {}, "execution_order": [], "tensors": tensors,
         "symbolic_context": {"symbols": symbols, "expressions": {}}}
    p = tmp_path / "graph.json"
    p.write_text(json.dumps(d))
    return p


def test_a_time_axis_at_one_is_named():
    """The live CogVideoX-5b-I2V case: 3*s and s+2 both give 3 at s=1."""
    m = _tool()
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = _graph(Path(d), {"s1": {"name": "time", "trace_value": 1,
                                    "source": "input::args::dim_2"}})
        rows, accepted = m.census_one(p)
    assert len(rows) == 1 and rows[0]["name"] == "time"
    assert any(c == "arithmetic" for c, _ in rows[0]["flags"])
    assert accepted == []


def test_the_batch_axis_is_counted_not_listed():
    """A deliberate choice is not a finding — but it is never silently dropped."""
    m = _tool()
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = _graph(Path(d), {"s0": {"name": "batch", "trace_value": 1},
                             "s6": {"name": "batch", "trace_value": 2}})
        rows, accepted = m.census_one(p)
    assert rows == [], "the project requires batch to stay symbolic at 1 and at 2"
    assert len(accepted) == 2, "and it must still be counted"


def test_all_shows_the_deliberate_class_for_re_examination():
    m = _tool()
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = _graph(Path(d), {"s0": {"name": "batch", "trace_value": 1}})
        rows, accepted = m.census_one(p, show_all=True)
    assert len(rows) == 1 and accepted == []


def test_an_axis_colliding_with_a_weight_extent_is_named():
    """real-esrgan: height traced at 64, which is also a parameter extent."""
    m = _tool()
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = _graph(Path(d),
                   {"s1": {"name": "height", "trace_value": 64}},
                   params=[(64, 3, 3, 3)])
        rows, _ = m.census_one(p)
    assert len(rows) == 1
    assert any(c == "weight-extent" for c, _ in rows[0]["flags"])


def test_a_safe_axis_is_not_named():
    """23 is the project's sequence trace value precisely because it collides with
    nothing — prime, and not a parameter extent here."""
    m = _tool()
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = _graph(Path(d), {"s1": {"name": "seq_len", "trace_value": 23}},
                   params=[(4096, 4096)])
        rows, _ = m.census_one(p)
    assert rows == []


def test_a_census_over_zero_graphs_refuses(tmp_path, monkeypatch, capsys):
    """An instrument that examined nothing must not read as one that found nothing."""
    m = _tool()
    monkeypatch.setattr(sys, "argv", ["census", "--root", str(tmp_path)])
    assert m.main() == 1
    assert "not a clean census" in capsys.readouterr().err
