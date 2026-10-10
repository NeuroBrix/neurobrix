"""The census keys no launch below the range a graph's own algebra admits (chatterbox, 2026-10-09).

Chatterbox's vocoder length is (2 * (prompt + speech tokens) - 320) * 480: at 1..3 speech tokens it
is negative, and a site enumerated from 1 recorded keys with M = -1920 that certify refused
("negative dimensions are not allowed"). The site now starts at the first binding as full as its
top, and a dim that resolves below zero is refused by name instead of keyed.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import derived_census as D  # noqa: E402


def _graph():
    sym = {"type": "symbol", "id": "s1", "trace": 23}
    length = {"type": "add", "left": sym, "right": -3, "trace": 20}
    return {"symbolic_context": {"symbols": {"s1": {"name": "seq_len", "trace_value": 23,
                                                    "source": "input::tokens::dim_1"}}},
            "tensors": {
                "input::tokens": {"shape": [1, 23], "is_input": True, "input_name": "tokens",
                                  "symbolic_shape": {"dims": [1, sym]}},
                "op::out_0": {"shape": [1, 20], "symbolic_shape": {"dims": [1, length]}},
                "op::empty": {"shape": [0, 4], "symbolic_shape": {"dims": [0, 4]}},
            },
            "ops": {}, "execution_order": [], "output_tensor_ids": ["op::out_0"]}


def test_the_site_starts_at_the_first_full_binding(monkeypatch):
    monkeypatch.setattr(D, "raw_graph", lambda model, comp: _graph())
    # 1..3 leave the length empty or negative; 4 is as full as the top (the always-empty
    # buffer stays legitimate)
    assert D.first_full_binding("m", "c", 1, 64, lambda n: {"tokens": [1, n]}) == 4


def test_a_negative_extent_is_refused_by_name():
    shape = D._shape_fn(_graph(), "c", {"s1": 2}, {}, {})
    with pytest.raises(D.NegativeExtent, match="op::out_0"):
        shape("op::out_0")
    assert shape("op::empty") == [0, 4]
