"""The Triton engine's fused MoE call (`triton/moe.execute_moe_fused`) hands the library's op
(`nbxtensor.ops.moe_fused.fused_moe`) its arguments as given and the engine's switches as read at the call:
NBX_MOE_VEC=0 refuses the int4 one-token band, NBX_MOEV_DSPLIT=0 its split down projection, unset or "1" keeps each;
NBX_Z3_TRITON_DIAG=1 asks the memory log — all three read once, at import; NBX_MOE_DIAG unset attaches no dump, "1" dumps every call, another value
only the calls whose key contains it.

What this test would do if the code were wrong: a switch read once at import (the cell flips it between two calls), a
switch inverted, an argument dropped or renamed on the way, a dump attached with the diagnostic off or a filter that
lets another key through — RED on the recorded call. No card: the op is replaced by a recorder.
"""
from __future__ import annotations

import pytest

pytest.importorskip("triton")


@pytest.fixture
def recorded(monkeypatch):
    import neurobrix.triton.moe as M
    seen = []
    monkeypatch.setattr(M, "fused_moe", lambda *a, **k: seen.append((a, k)) or "out")
    for name, default in (("_INT4_BAND", True), ("_SPLIT_DOWN", True), ("_LOG_MEMORY", False)):
        monkeypatch.setattr(M, name, default)
    return M, seen


def test_the_arguments_reach_the_op_as_given(recorded):
    M, seen = recorded
    store, gs, h, g, u, d, ti, tw = (object() for _ in range(8))
    assert M.execute_moe_fused(gs, h, g, u, d, 4, 16, norm_topk_prob=False, cache_key="k", topk_indices=ti,
                               topk_weights=tw, weights_per_call=True, expert_store=store, expert_layer="L3") == "out"
    (a, k), = seen
    assert a == (gs, h, g, u, d, 4, 16), a
    assert {n: k[n] for n in ("norm_topk_prob", "topk_indices", "topk_weights", "weights_per_call", "store", "layer",
                              "label")} == {"norm_topk_prob": False, "topk_indices": ti, "topk_weights": tw,
                                            "weights_per_call": True, "store": store, "layer": "L3", "label": "k"}, k


@pytest.mark.parametrize("value,taken", [(None, True), ("1", True), ("0", False)])
@pytest.mark.parametrize("switch,const,arg", [("NBX_MOE_VEC", "_INT4_BAND", "int4_band"),
                                              ("NBX_MOEV_DSPLIT", "_SPLIT_DOWN", "split_down")])
def test_a_switch_is_read_once_at_import_and_reaches_the_op(recorded, monkeypatch, switch, const, arg, value, taken):
    import importlib
    M, seen = recorded
    monkeypatch.delenv(switch, raising=False)
    if value is not None:
        monkeypatch.setenv(switch, value)
    fresh = importlib.reload(M)
    try:
        assert getattr(fresh, const) is taken, (switch, value)
        monkeypatch.setattr(fresh, "fused_moe", lambda *a, **k: seen.append((a, k)) or "out")
        fresh.execute_moe_fused(None, None, [], [], [], 2, 4)
        assert seen[-1][1][arg] is taken
    finally:
        monkeypatch.undo()
        importlib.reload(M)


def test_the_memory_log_is_asked_by_its_switch(recorded, monkeypatch):
    M, seen = recorded
    M.execute_moe_fused(None, None, [], [], [], 2, 4)
    monkeypatch.setattr(M, "_LOG_MEMORY", True)
    M.execute_moe_fused(None, None, [], [], [], 2, 4)
    assert [k["log_memory"] for _, k in seen] == [False, True]


@pytest.mark.parametrize("diag,key,dumps", [("", "block.3", False), ("1", "block.3", True),
                                            ("block.3", "seq_block.3", True), ("block.4", "seq_block.3", False)])
def test_the_dump_is_attached_and_filtered_by_its_switch(recorded, monkeypatch, diag, key, dumps):
    M, seen = recorded
    monkeypatch.setattr(M, "_MOE_DIAG_ENV", diag)
    read = []
    monkeypatch.setattr(M, "host_values", lambda t: read.append(t) or __import__("numpy").zeros(3))
    M.execute_moe_fused(None, None, [], [], [], 2, 4, cache_key=key)
    dump = seen[0][1]["dump"]
    if not diag:
        assert dump is None, "a dump attached with the diagnostic off"
        return
    tensor = type("T", (), {"shape": (1, 3), "dtype": "fp16"})()
    dump("gate_out", tensor)
    assert read == ([tensor] if dumps else []), (diag, key, read)
