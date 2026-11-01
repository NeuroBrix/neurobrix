"""A replayed plan is keyed by every symbol's bound value (deepseek-moe-16b-chat triton, gate of 2026-11-01): its
rotary table is `arange(nbx_rope_len + 1)`, `nbx_rope_len` bound per decode step from position_ids' last element; the
signature read a `resolved` the Triton resolver never had, so a plan recorded at step N (a table of N + 1 rows) was
replayed at step N + 1, indexed at row N + 1: index_select out of range, CUresult 710.

What this test would do if the code were wrong: two runs whose `nbx_rope_len` differs give the same signature (red). No
card."""
from types import SimpleNamespace

from neurobrix.triton import replay
from neurobrix.triton.symbols import SymbolResolver


def _seq(rope_len: int):
    res = SymbolResolver({"symbols": {
        "s1": {"name": "seq_len", "trace_value": 23, "source": "input::input_ids::dim_1"},
        "nbx_rope_len": {"name": "nbx_rope_len", "trace_value": 22, "source": "input::position_ids::val_-1"}}})
    res._bind("s1", 1)
    res._bind("nbx_rope_len", rope_len)
    return SimpleNamespace(dag={"component_name": "model"}, _is_multi_device=False, _replay_ineligible=False,
                           _op_uid_interceptors={}, _op_interceptors={}, _ops=[SimpleNamespace(op_type="aten::mm")],
                           _num_weights=0, _num_inputs=0, _arena=[], _symbol_resolver=res)


def test_two_steps_whose_rope_length_differs_are_two_plans():
    a, b = replay.signature(_seq(22)), replay.signature(_seq(23))
    assert a is not None and b is not None
    assert a != b, "a plan recorded at one rotary length would replay at the next step's position"
    assert replay.signature(_seq(22)) == a
