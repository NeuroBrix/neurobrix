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


@pytest.mark.parametrize("mode", MODES)
def test_the_forward_is_the_executor_s_hook_not_a_replaced_method(mode):
    """The strategy follows the base's registrations through the explicit hook
    (`GraphExecutor.follow_interceptor_registrations`), never by assigning over the executor's
    own `register_*` methods.

    Injections (seen red, then restored green — STATE.md of the op_tiler campaign, 2026-10-05):
      A. the follower loop removed from `register_op_uid_interceptors` -> red here and in
         test_a_band_registered_after_the_install_reaches_its_piece;
      B. the instance assignment put back in `_forward_op_level_tiling` -> red here."""
    base = _streamed_base(mode)
    assert not {"register_op_uid_interceptors", "register_triton_interceptors"} & set(vars(base))
    assert "layer_streaming:" + W.COMPONENT in base._op_uid_followers
    calls = []
    base.register_op_uid_interceptors({"aten.mm::0": _recording_mm(calls)})
    base.run({"inputs_embeds": torch.randn(W.B, 5, W.D)})
    assert calls == [(W.B * 5, W.D)], calls


def test_an_executor_without_the_hook_is_refused_by_name():
    from types import SimpleNamespace
    from neurobrix.core.strategies.layer_streaming import LayerStreamingStrategy
    ex = SimpleNamespace(register_op_uid_interceptors=lambda *a, **k: None)
    with pytest.raises(RuntimeError, match=r"'c'.*follow_interceptor_registrations"):
        LayerStreamingStrategy._follow(ex, "c", op_uid=lambda *a, **k: None)


@pytest.mark.parametrize("mode", MODES)
def test_an_op_type_interceptor_registered_after_the_install_reaches_the_pieces(mode):
    """The autoregressive flow registers its KV cache by op TYPE (`register_op_interceptors`,
    `core/flow/autoregressive.py`) on the component's executor, after the strategy is installed.
    Only the op_uid and triton registrations followed the base to its pieces, so a streamed
    decode on the compiled engine ran with no KV cache. MEASURED 2026-10-08 on the M4 Pro,
    TinyLlama-1.1B native, greedy, 'The capital of France is': whole -> ' Paris.', forced
    layer_streaming -> 'The' and eleven newlines, and ' Paris.' again with NBX_KV_RECOMPUTE=1.
    canary-qwen-2.5b, GLM-4.1V-9B and MiniCPM-o-4_5 streamed native gave garbage the same way."""
    calls = []
    base = _streamed_base(mode)
    base.register_op_interceptors({"aten::mm": _recording_mm(calls)})
    base.run({"inputs_embeds": torch.randn(W.B, 3, W.D)})
    assert calls == [(W.B * 3, W.D)], calls


@pytest.mark.parametrize("mode", MODES)
def test_an_op_type_interceptor_registered_before_the_install_runs_in_the_pieces(mode):
    calls = []
    base = W._executor(mode)
    base.register_op_interceptors({"aten::mm": _recording_mm(calls)})
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
    base.run({"inputs_embeds": torch.randn(W.B, 4, W.D)})
    assert calls == [(W.B * 4, W.D)], calls
