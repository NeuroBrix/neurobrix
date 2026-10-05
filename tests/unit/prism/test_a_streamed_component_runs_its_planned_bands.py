"""A streamed component runs the plan's op-level tiling in its pieces.

MEASURED 2026-10-05, SANA-Video_2B_720p on a 16 GB V100, triton certified-only: the plan streamed
the transformer AND banded its `aten.convolution::3` (`runtime_op_tiling`); the census keyed the
bands, the piece launched the whole conv (11, 6720, 21, 42) — a refused key. The runtime
registers op-level tiling on the component's executor, whose `run` the streaming strategy
replaces with a walk over its pieces; nothing carried the interceptors to the pieces.

On the CPU, real `GraphExecutor` + real `LayerStreamingStrategy` (the graph of
test_a_streamed_component_returns_what_it_would_whole), compiled and sequential:
  * an op_uid interceptor registered on the base BEFORE the install runs in the piece holding
    its op, and the streamed answer is the whole answer;
  * one registered AFTER the install reaches the piece too;
  * a proxy-handing group split across two pieces is refused by name.

Injection (seen red, then restored green): `_forward_op_level_tiling` not called -> the
interceptor never ran in either engine — RED.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_a_streamed_component_returns_what_it_would_whole as W  # noqa: E402

MODES = ["compiled", "sequential"]


def _recording_mm(calls):
    def _mm(a, b, *args, **kwargs):
        calls.append(tuple(a.shape))
        return torch.mm(a, b)
    return _mm


def _streamed_base(mode, before=None):
    base = W._executor(mode)
    if before:
        base.register_op_uid_interceptors(before)
    from types import SimpleNamespace
    from neurobrix.core.optim.passes.normalize import graph_fingerprint, normalize_for_branch
    cut = normalize_for_branch(base._dag, base.mode, base.family, declared_moe=None)
    order = cut["execution_order"]
    split = order.index("aten.view::0")
    ctx = SimpleNamespace(
        component_executors={W.COMPONENT: base},
        layer_segments={W.COMPONENT: [[order[0], order[split - 1]], [order[split], order[-1]]]},
        layer_graphs={W.COMPONENT: graph_fingerprint(cut)}, layer_moe={})
    assert W._Strategy(ctx, "layer_streaming").install_for_executor(W.COMPONENT, base) is True
    return base


@pytest.mark.parametrize("mode", MODES)
def test_a_band_registered_before_the_install_runs_in_its_piece(mode):
    embeds = torch.randn(W.B, W.T_TRACE, W.D)
    whole = W._executor(mode)
    whole.load_weights(None, W.COMPONENT)
    want = whole.run({"inputs_embeds": embeds})
    calls = []
    base = _streamed_base(mode, before={"aten.mm::0": _recording_mm(calls)})
    got = base.run({"inputs_embeds": embeds})
    assert calls == [(W.B * W.T_TRACE, W.D)], f"the planned interceptor ran {calls} in the pieces"
    assert set(got) == set(want)
    for k in want:
        assert torch.equal(got[k], want[k]), k


@pytest.mark.parametrize("mode", MODES)
def test_a_band_registered_after_the_install_reaches_its_piece(mode):
    calls = []
    base = _streamed_base(mode)
    base.register_op_uid_interceptors({"aten.mm::0": _recording_mm(calls)})
    base.run({"inputs_embeds": torch.randn(W.B, 7, W.D)})
    assert calls == [(W.B * 7, W.D)], calls


def test_a_group_split_across_pieces_is_refused_by_name():
    with pytest.raises(RuntimeError, match=r"split across pieces \[0, 1\]"):
        _streamed_base("compiled", before=None).register_op_uid_interceptors(
            {"aten.mul::0": lambda *a, **k: None, "aten.mm::0": lambda *a, **k: None},
            groups=[("aten.mul::0", "aten.mm::0")])


def test_a_planned_band_no_piece_holds_is_refused_by_name():
    """A band the plan tiles on an op no piece of the streamed graph holds would silently run
    whole — the failure this forward exists to end.

    Injection (seen red, then restored green, 2026-10-05): the `nowhere` refusal disabled."""
    with pytest.raises(RuntimeError, match=r"no piece of the streamed graph holds"):
        _streamed_base("compiled").register_op_uid_interceptors(
            {"aten.absent::0": lambda *a, **k: None}, planned=["aten.absent::0"])
