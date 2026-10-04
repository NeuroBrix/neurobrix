"""A component streamed in pieces answers its flow exactly as it answers whole: the same
declared outputs under the same keys, and — through `get_hidden_states` — the same tensor the
whole run keeps beyond its outputs.

MEASURED on the Mac 2026-10-04 (tree 81eb2d46), `Janus-Pro-7B --mode image --triton
--certified-only`, planned `layer_streaming` ('language_model' in 4 pieces, rung 8 192 MB): a
certified-key miss on gen_head's `aten.addmm::0`, bf16, M_BUCKET 25, N 4096, K 4096. The same
request whole runs clean with M = 1.

The cause, read in the code: the image-AR session asks the language model's executor to protect
its pre-head hidden (`enable_hidden_states_capture`: the first input of the last `aten::mm`, the
(B*T, 4096) view before the text head) and reads it back after the run (`get_hidden_states`).
Streamed, that executor is a BASE whose `run` walks the pieces: it runs no op, the tid it
protected lives inside the last piece and never crosses a seam, so no piece kept it and the
base's reader found nothing. The torch session refuses on None; the triton session fell to its
output scan's "first output" and took the declared output — the text-vocab LOGITS
(B, T, 102400) — as the hidden. gen_head took their last row (1, 1, 102400) as `x` and viewed it
as (-1, 4096): M = 102400 / 4096 = 25. Not the prompt length.

What this file holds, on the CPU, with the real `GraphExecutor` and the real
`LayerStreamingStrategy` over a graph shaped like that language model (an embedding-fed body, a
norm, a text head through `aten::mm`, logits declared as the output), in the two torch engines:

  * the streamed base's `get_hidden_states` is the whole executor's, value for value, at two
    sequence lengths (the second far from the trace) — a decode step reads its own run;
  * the streamed run returns the whole run's declared outputs under the whole run's keys;
  * the triton engines' reader serves the same capture (the arena itself needs a card: the
    reader is held on the state the strategy leaves, and `tensors_of_last_run` on the state each
    triton engine leaves);
  * the triton session no longer turns "no hidden" into "the first output".

Seen RED on a-streamed-plan-states-its-window (7978444f) before the fix: the base's
`get_hidden_states` returned None in both engines.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import neurobrix.core.runtime  # noqa: F401  (pre-resolve the cfg<->runtime import cycle)
from neurobrix.core.optim.passes.normalize import graph_fingerprint, normalize_for_branch
from neurobrix.core.runtime.graph_executor import GraphExecutor
from neurobrix.core.strategies.layer_streaming import LayerStreamingStrategy

B, T_TRACE, D, V = 2, 5, 4, 8
COMPONENT = "language_model"
HIDDEN_TID = "aten.view::0::out_0"          # the pre-head (B*T, D) view: the last mm's input

S0 = {"type": "symbol", "id": "s0", "trace": B}
S1 = {"type": "symbol", "id": "s1", "trace": T_TRACE}
BT = {"type": "mul", "left": S0, "right": S1, "trace": B * T_TRACE}


def _t(tid, shape, sym, **kw):
    d = {"tensor_id": tid, "shape": shape, "dtype": "float32", "device": "cpu",
         "is_parameter": False, "is_input": False, "weight_name": None, "input_name": None,
         "output_name": None, "symbolic_shape": {"dims": sym, "concrete": shape}}
    d.update(kw)
    return d


def _ref(tid):
    return {"type": "tensor", "tensor_id": tid}


def _op(uid, typ, ins, outs, args):
    return {"op_uid": uid, "op_type": typ, "input_tensor_ids": ins, "output_tensor_ids": outs,
            "attributes": {"args": args, "kwargs": {}}}


def _graph():
    """An embedding-fed body (`inputs_embeds` -> two scaled blocks), the final norm, the text
    head through `aten::mm`, and the LOGITS as the one declared output — the shape of an
    image-AR language model whose head is inside its graph."""
    x = "input::inputs_embeds"
    tensors = {
        x: _t(x, [B, T_TRACE, D], [S0, S1, D], is_input=True, input_name="inputs_embeds"),
        "param::layers.0.w": _t("param::layers.0.w", [D], [D], is_parameter=True,
                                weight_name="layers.0.w"),
        "param::layers.1.w": _t("param::layers.1.w", [D], [D], is_parameter=True,
                                weight_name="layers.1.w"),
        "param::head.weight": _t("param::head.weight", [V, D], [V, D], is_parameter=True,
                                 weight_name="head.weight"),
        "aten.mul::0::out_0": _t("aten.mul::0::out_0", [B, T_TRACE, D], [S0, S1, D]),
        "aten.add::0::out_0": _t("aten.add::0::out_0", [B, T_TRACE, D], [S0, S1, D]),
        "aten.mul::1::out_0": _t("aten.mul::1::out_0", [B, T_TRACE, D], [S0, S1, D]),
        HIDDEN_TID: _t(HIDDEN_TID, [B * T_TRACE, D], [BT, D]),
        "aten.t::0::out_0": _t("aten.t::0::out_0", [D, V], [D, V]),
        "aten.mm::0::out_0": _t("aten.mm::0::out_0", [B * T_TRACE, V], [BT, V]),
        "aten._unsafe_view::0::out_0": _t("aten._unsafe_view::0::out_0", [B, T_TRACE, V],
                                          [S0, S1, V], output_name="logits"),
    }
    ops = {
        "aten.mul::0": _op("aten.mul::0", "aten::mul", [x, "param::layers.0.w"],
                           ["aten.mul::0::out_0"], [_ref(x), _ref("param::layers.0.w")]),
        "aten.add::0": _op("aten.add::0", "aten::add", ["aten.mul::0::out_0", x],
                           ["aten.add::0::out_0"], [_ref("aten.mul::0::out_0"), _ref(x)]),
        "aten.mul::1": _op("aten.mul::1", "aten::mul",
                           ["aten.add::0::out_0", "param::layers.1.w"], ["aten.mul::1::out_0"],
                           [_ref("aten.add::0::out_0"), _ref("param::layers.1.w")]),
        "aten.view::0": _op("aten.view::0", "aten::view", ["aten.mul::1::out_0"], [HIDDEN_TID],
                            [_ref("aten.mul::1::out_0"), {"type": "list", "value": [BT, D]}]),
        "aten.t::0": _op("aten.t::0", "aten::t", ["param::head.weight"], ["aten.t::0::out_0"],
                         [_ref("param::head.weight")]),
        "aten.mm::0": _op("aten.mm::0", "aten::mm", [HIDDEN_TID, "aten.t::0::out_0"],
                          ["aten.mm::0::out_0"], [_ref(HIDDEN_TID), _ref("aten.t::0::out_0")]),
        "aten._unsafe_view::0": _op("aten._unsafe_view::0", "aten::_unsafe_view",
                                    ["aten.mm::0::out_0"], ["aten._unsafe_view::0::out_0"],
                                    [_ref("aten.mm::0::out_0"),
                                     {"type": "list", "value": [S0, S1, V]}]),
    }
    return {
        "component_name": COMPONENT, "format": "tensor_dag", "version": "0.1",
        "torch_dtype": "float32", "tensors": tensors, "ops": ops,
        "execution_order": list(ops), "input_tensor_ids": [x],
        "output_tensor_ids": ["aten._unsafe_view::0::out_0"],
        "symbolic_context": {"symbols": {
            "s0": {"name": "batch", "trace_value": B, "source": f"{x}::dim_0"},
            "s1": {"name": "seq_len", "trace_value": T_TRACE, "source": f"{x}::dim_1"}},
            "expressions": {}},
    }


torch.manual_seed(0)
WEIGHTS = {"layers.0.w": torch.randn(D), "layers.1.w": torch.randn(D),
           "head.weight": torch.randn(V, D)}


class _CpuExecutor(GraphExecutor):
    """The real executor; only where its weights come from is replaced (no container on disk).
    `unload_weights` is the real one: it is what drops a piece's run state."""

    def load_weights(self, nbx_path, component, *args, **kwargs):
        self._weights = dict(WEIGHTS)
        self._weights_loaded = True

    @staticmethod
    def non_block_keys(nbx_path, component):
        return set()

    def load_flow_read_weights(self, nbx_path, component, shard_map=None, keys=None):
        return 0


class _Strategy(LayerStreamingStrategy):
    def _nbx_path(self, component_name):
        return "<no container: weights come from the test>"


def _executor(mode):
    ex = _CpuExecutor(family="llm", vendor="nvidia", arch="volta", device="cpu",
                      dtype="float32", mode=mode)
    ex.load_graph_from_dict(_graph())
    return ex


def _streamed_base(mode):
    base = _executor(mode)
    cut = normalize_for_branch(base._dag, base.mode, base.family, declared_moe=None)
    order = cut["execution_order"]
    split = order.index("aten.view::0")              # the head's piece starts at the view
    ctx = SimpleNamespace(
        component_executors={COMPONENT: base},
        layer_segments={COMPONENT: [[order[0], order[split - 1]],
                                    [order[split], order[-1]]]},
        layer_graphs={COMPONENT: graph_fingerprint(cut)},
        layer_moe={},
    )
    strategy = _Strategy(ctx, "layer_streaming")
    assert strategy.install_for_executor(COMPONENT, base) is True
    return base


def _session_read(ex, embeds):
    """What both LM sessions do at prefill: protect, run, read the hidden back."""
    ex.enable_hidden_states_capture()
    out = ex.run({"inputs_embeds": embeds})
    return out, ex.get_hidden_states(expected_hidden_dim=D, expected_batch_size=B)


@pytest.mark.parametrize("mode", ["sequential", "compiled"])
def test_the_streamed_base_reads_the_hidden_the_whole_executor_reads(mode):
    whole = _executor(mode)
    whole.load_weights(None, COMPONENT)
    streamed = _streamed_base(mode)

    for seq_len in (T_TRACE, 3 * T_TRACE + 2):       # the trace length, and one far from it
        embeds = torch.randn(B, seq_len, D)
        whole_out, whole_hidden = _session_read(whole, embeds)
        streamed_out, streamed_hidden = _session_read(streamed, embeds)

        assert whole_hidden is not None and tuple(whole_hidden.shape) == (B, seq_len, D)
        assert streamed_hidden is not None, (
            f"{mode}, T={seq_len}: the streamed base read no hidden — the tid it protects lives "
            f"in a piece that never kept it. The torch session refuses here; the triton one "
            f"handed the head the logits (Janus-Pro-7B: gen_head addmm at M = 25).")
        assert tuple(streamed_hidden.shape) == (B, seq_len, D), streamed_hidden.shape
        assert torch.equal(streamed_hidden, whole_hidden), (
            f"{mode}, T={seq_len}: the streamed hidden differs from the whole one")

        assert sorted(streamed_out) == sorted(whole_out), (
            f"{mode}: the streamed run returns {sorted(streamed_out)}, whole returns "
            f"{sorted(whole_out)}")
        for key in whole_out:
            assert torch.equal(streamed_out[key], whole_out[key]), key


def test_a_protected_tid_no_piece_produces_is_refused_by_name():
    base = _streamed_base("sequential")
    base.protect_tensor_id("aten.nothing::0::out_0")
    with pytest.raises(RuntimeError, match=r"aten\.nothing::0::out_0"):
        base.run({"inputs_embeds": torch.randn(B, T_TRACE, D)})


@pytest.mark.parametrize("mode", ["triton", "triton_sequential"])
def test_the_triton_reader_serves_what_the_pieces_left(mode):
    """The triton engines' `get_hidden_states` on a streamed base reads the capture the
    strategy rebuilt from the pieces, before its own (absent) arena. Stand-in tensors: the
    reader only reads `.shape`, `.ndim` and `.view`."""
    base = GraphExecutor(family="llm", vendor="nvidia", arch="volta", device="cpu",
                         dtype="float32", mode="sequential")
    base.load_graph_from_dict(_graph())
    base.mode = mode
    flat = torch.arange(B * 7 * D, dtype=torch.float32).reshape(B * 7, D)
    logits = torch.zeros(B, 7, V)
    base._pieces_capture = {HIDDEN_TID: flat, "aten._unsafe_view::0::out_0": logits}
    hidden = base.get_hidden_states(expected_hidden_dim=D, expected_batch_size=B)
    assert hidden is not None and tuple(hidden.shape) == (B, 7, D)
    assert torch.equal(hidden, flat.view(B, 7, D))


@pytest.mark.parametrize("mode", ["triton", "triton_sequential"])
def test_a_piece_is_read_where_its_triton_engine_keeps_its_run(mode):
    ex = GraphExecutor(family="llm", vendor="nvidia", arch="volta", device="cpu",
                       dtype="float32", mode=mode)
    kept = object()
    if mode == "triton":
        ex._triton_seq = SimpleNamespace(
            gather_outputs=lambda ids: {t: kept for t in ids if t == HIDDEN_TID})
    else:
        ex._tritonseq_captured = {HIDDEN_TID: kept}
    assert ex.tensors_of_last_run([HIDDEN_TID, "absent"]) == {HIDDEN_TID: kept}
    assert ex.tensors_of_last_run([]) == {}, (
        "an empty request must not read the whole arena (gather_outputs([]) reads every output)")


def test_the_triton_session_does_not_take_the_first_output_as_the_hidden():
    from neurobrix.triton.session import TritonLMSession
    session = TritonLMSession.__new__(TritonLMSession)
    session.hidden_dim = 4096
    logits = SimpleNamespace(shape=(2, 23, 102400))
    assert session._extract_hidden({"logits": logits}) is None, (
        "an output of the vocabulary's width was returned as the hidden — the head then views "
        "(1, 1, 102400) as (25, 4096)")
    hidden = SimpleNamespace(shape=(2, 23, 4096))
    assert session._extract_hidden({"logits": logits, "h": hidden}) is hidden


def test_a_protected_tid_a_piece_produced_and_did_not_keep_is_refused_by_name(monkeypatch):
    """The other half of the door: the piece produces the tid but its run did not keep it."""
    base = _streamed_base("sequential")
    monkeypatch.setattr(_CpuExecutor, "tensors_of_last_run", lambda self, tids: {})
    base.enable_hidden_states_capture()
    with pytest.raises(RuntimeError, match=r"did not keep"):
        base.run({"inputs_embeds": torch.randn(B, T_TRACE, D)})
