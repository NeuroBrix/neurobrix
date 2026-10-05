"""A stretch of a streamed component whose activations alone are over the rung runs in slices of a
token axis, and answers what it answers whole.

The case: SANA-Video_2B_720p's transformer on the 16 GB V100 profile, triton and
triton-sequential, at the 4 096 / 6 144 / 8 192 rungs — refused after the guidance split ("activations
alone peak at ... This needs a smaller batch or context"), its linear-attention blocks holding the
whole video's tokens at once. Prism now plans such a stretch in PASSES along the axis
(core/prism/chunked_region.py `plan_region`): a contraction over the axis (the linear attention's
K^T V) is summed over the slices and fed whole to the next pass, every other S-carrying tensor is
computed one slice at a time. The plan records each stretch (`Partition.chunks`,
`Plan.layer_stream_chunks`); the strategy runs it as one piece (`ChunkedPiece`).

Held here, on the CPU, with the real `GraphExecutor` and the real `LayerStreamingStrategy`, in the
two torch engines, over a graph shaped like that block (a seam from the piece before it, a K^T V
contraction over the tokens, its product with Q, a residual):

  * the sliced piece returns the whole run's output within float32 summation tolerance, at the
    trace length and at a length far from it whose last slice is shorter;
  * the partitioner refuses on the activations without the token split and places with it,
    recording the stretch, its slice, its passes and a peak under the budget;
  * a stretch the plan names that is not one of the plan's pieces is refused by name.

Injections (seen red, then restored green — STATE.md of the op_tiler campaign):
  A. `ChunkedPiece.run` keeps the last slice's contraction instead of the sum -> the value test red.
  B. `plan_region` does not lift a contraction's readers to a later pass -> the value test red
     (the product reads a partial K^T V) — and the pass count no longer matches the plan.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import neurobrix.core.runtime  # noqa: F401  (pre-resolve the cfg<->runtime import cycle)
from neurobrix.core.optim.passes.normalize import graph_fingerprint, normalize_for_branch
from neurobrix.core.prism.chunked_region import TokenAxis, plan_region
from neurobrix.core.prism.layer_partition import LayerPartitioner
from neurobrix.core.runtime.graph_executor import GraphExecutor
from neurobrix.core.strategies.layer_streaming import LayerStreamingStrategy

B, T_TRACE, D = 2, 6, 4
COMPONENT = "transformer"
S0 = {"type": "symbol", "id": "s0", "trace": B}
S1 = {"type": "symbol", "id": "s1", "trace": T_TRACE}
X = "input::hidden_states"


def _t(tid, shape, sym, **kw):
    d = {"tensor_id": tid, "shape": shape, "dtype": "float32", "device": "cpu",
         "is_parameter": False, "is_input": False, "weight_name": None, "input_name": None,
         "output_name": None, "symbolic_shape": {"dims": sym, "concrete": shape}}
    d.update(kw)
    return d


def _ref(tid):
    return {"type": "tensor", "tensor_id": tid}


def _s(v):
    return {"type": "scalar", "value": v}


def _op(uid, typ, args):
    ins = [a["tensor_id"] for a in args if isinstance(a, dict) and a.get("type") == "tensor"]
    return {"op_uid": uid, "op_type": typ, "input_tensor_ids": ins,
            "output_tensor_ids": [uid + "::out_0"], "attributes": {"args": args, "kwargs": {}}}


def _graph():
    """h = x*wp (the piece before) | q = h*wq; kv = q^T @ h (over the tokens); y = q @ kv;
    z = y + h; out = z*wo — a linear-attention block fed by a seam."""
    bsd, bds, bdd = [B, T_TRACE, D], [B, D, T_TRACE], [B, D, D]
    sbsd, sbds, sbdd = [S0, S1, D], [S0, D, S1], [S0, D, D]
    tensors = {X: _t(X, bsd, sbsd, is_input=True, input_name="hidden_states")}
    for w in ("wp", "wq", "wo"):
        tensors["param::" + w] = _t("param::" + w, [D], [D], is_parameter=True, weight_name=w)
    ops = {}

    def add(uid, typ, args, shape, sym, **kw):
        ops[uid] = _op(uid, typ, args)
        tensors[uid + "::out_0"] = _t(uid + "::out_0", shape, sym, **kw)
        return uid + "::out_0"

    h = add("aten.mul::0", "aten::mul", [_ref(X), _ref("param::wp")], bsd, sbsd)
    q = add("aten.mul::1", "aten::mul", [_ref(h), _ref("param::wq")], bsd, sbsd)
    qt = add("aten.transpose::0", "aten::transpose", [_ref(q), _s(1), _s(2)], bds, sbds)
    kv = add("aten.bmm::0", "aten::bmm", [_ref(qt), _ref(h)], bdd, sbdd)
    y = add("aten.bmm::1", "aten::bmm", [_ref(q), _ref(kv)], bsd, sbsd)
    z = add("aten.add::0", "aten::add", [_ref(y), _ref(h)], bsd, sbsd)
    add("aten.mul::2", "aten::mul", [_ref(z), _ref("param::wo")], bsd, sbsd, output_name="sample")
    return {
        "component_name": COMPONENT, "format": "tensor_dag", "version": "0.1",
        "torch_dtype": "float32", "tensors": tensors, "ops": ops,
        "execution_order": list(ops), "input_tensor_ids": [X],
        "output_tensor_ids": ["aten.mul::2::out_0"],
        "symbolic_context": {"symbols": {
            "s0": {"name": "batch", "trace_value": B, "source": f"{X}::dim_0"},
            "s1": {"name": "seq_len", "trace_value": T_TRACE, "source": f"{X}::dim_1"}},
            "expressions": {}},
    }


torch.manual_seed(0)
WEIGHTS = {w: torch.randn(D) for w in ("wp", "wq", "wo")}
FIRST, LAST = "aten.mul::1", "aten.mul::2"


class _CpuExecutor(GraphExecutor):
    """The real executor; only where its weights come from is replaced (no container on disk).
    Weights are taken by reference from the lender when it holds them, as `_borrow` does."""

    loads = 0

    def load_weights(self, nbx_path, component, *args, **kwargs):
        lender = getattr(getattr(self, "_borrow_from", None), "_weights", None)
        if lender:
            self._weights = dict(lender)
        else:
            type(self).loads += 1
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
    ex = _CpuExecutor(family="image", vendor="nvidia", arch="volta", device="cpu",
                      dtype="float32", mode=mode)
    ex.load_graph_from_dict(_graph())
    return ex


def _passes():
    return len(plan_region(*_region())[0].passes)


def _region():
    ax = TokenAxis(_graph(), "s1")
    return ax, ax.index[FIRST], ax.index[LAST]


def _streamed(mode, chunk):
    base = _executor(mode)
    cut = normalize_for_branch(base._dag, base.mode, base.family, declared_moe=None)
    order = cut["execution_order"]
    ctx = SimpleNamespace(
        component_executors={COMPONENT: base},
        layer_segments={COMPONENT: [[order[0], order[0]], [FIRST, LAST]]},
        layer_graphs={COMPONENT: graph_fingerprint(cut)},
        layer_moe={},
        layer_chunks={COMPONENT: [chunk]},
    )
    return _Strategy(ctx, "layer_streaming")


def _chunk(slice_):
    return {"first_op": FIRST, "last_op": LAST, "symbol": "s1", "slice": slice_,
            "count": -(-T_TRACE // slice_), "passes": _passes(), "peak_bytes": 0}


def test_the_stretch_plans_two_passes_the_contraction_between_them():
    plan, why = plan_region(*_region())
    assert plan is not None, why
    assert len(plan.passes) == 2 and plan.contractions == ["aten.bmm::0::out_0"], plan
    assert "aten.bmm::1" in plan.passes[1].ops and "aten.bmm::1" not in plan.passes[0].ops


@pytest.mark.parametrize("mode", ["sequential", "compiled"])
def test_the_sliced_stretch_returns_what_the_whole_one_does(mode):
    whole = _executor(mode)
    whole.load_weights(None, COMPONENT)
    strategy = _streamed(mode, _chunk(2))
    for seq_len in (T_TRACE, 4 * T_TRACE + 1):      # the trace, and far from it, a short last slice
        x = torch.randn(B, seq_len, D)
        want = whole.run({"hidden_states": x})
        got = strategy.execute_component(COMPONENT, "loop", {"hidden_states": x})
        assert sorted(got) == sorted(want) == ["sample"]
        assert tuple(got["sample"].shape) == (B, seq_len, D)
        torch.testing.assert_close(got["sample"], want["sample"], rtol=1e-5, atol=1e-5,
                                   msg=lambda m: f"{mode}, T={seq_len}: {m}")


def test_a_planned_stretch_that_is_not_a_piece_is_refused_by_name():
    chunk = dict(_chunk(2), first_op="aten.transpose::0")
    strategy = _streamed("sequential", chunk)
    with pytest.raises(RuntimeError, match=r"aten\.transpose::0.*not one of its pieces"):
        strategy.execute_component(COMPONENT, "loop", {"hidden_states": torch.randn(B, 5, D)})


def test_a_plan_priced_on_other_passes_is_refused():
    strategy = _streamed("sequential", dict(_chunk(2), passes=_passes() + 1))
    with pytest.raises(RuntimeError, match=r"priced .* pass"):
        strategy.execute_component(COMPONENT, "loop", {"hidden_states": torch.randn(B, 5, D)})


def _partitioner(token_split, seq):
    sizes = {w: D * 4 for w in WEIGHTS}
    return LayerPartitioner(_graph(), sizes, symbol_map={"s0": B, "s1": seq},
                            compute_dtype_bytes=4, token_split=token_split)


def test_the_partitioner_slices_the_stretch_only_when_the_activations_refuse():
    seq = 4096
    whole_peak = max(_partitioner(False, seq).live_activation_curve())
    budget = whole_peak * 3 // 4 + 3 * D * 4          # under the whole stretch, over a slice of it
    plain = _partitioner(False, seq).partition(budget)
    assert not plain.fits and "activations alone" in plain.refusal, plain.refusal
    part = _partitioner(True, seq).partition(budget)
    assert part.fits, part.refusal
    assert len(part.chunks) == 1, part.chunks
    c = part.chunks[0]
    assert c["symbol"] == "s1" and 1 <= c["slice"] < seq and c["count"] == -(-seq // c["slice"])
    assert c["passes"] == 2 and c["peak_bytes"] <= budget
    assert part.peak_resident_bytes <= budget
    # the stretch is one piece of its own
    assert [c["first_op"], c["last_op"]] in [[s.first_op, s.last_op] for s in part.segments]


class _Holder:
    def __init__(self):
        self.registered = []

    def register_op_uid_interceptors(self, interceptors, groups=(), planned=()):
        self.registered.append((sorted(interceptors), list(planned)))


def test_a_planned_band_inside_a_sliced_stretch_is_refused_and_one_outside_reaches_the_holder():
    """Prism drops a sliced stretch's op-level tiling (`OpLevelTilingPlan.drop_ops`): a planned band
    inside it reaching the piece is a plan the budget was not accepted under. An unplanned
    interceptor inside (an in-place reuse proven on the whole graph) is not run on the passes; one
    outside the stretch goes to the piece's own executor.

    Injection (seen red, then restored green, 2026-10-05): the refusal's `hit` emptied -> the
    planned band is accepted."""
    from neurobrix.core.strategies.chunked_piece import ChunkedPiece
    holder = _Holder()
    piece = ChunkedPiece(_graph(), holder, _chunk(2), lambda g, lender: None)
    with pytest.raises(RuntimeError, match=r"plan tiles \['aten\.bmm::1'\] inside the stretch"):
        piece.register_op_uid_interceptors({"aten.bmm::1": print}, planned=["aten.bmm::1"])
    piece.register_op_uid_interceptors({"aten.bmm::1": print, "aten.mul::0": print},
                                       planned=["aten.mul::0"])
    assert holder.registered == [(["aten.mul::0"], ["aten.mul::0"])], holder.registered


def test_drop_ops_removes_every_planned_entry_touching_the_stretch():
    """Injection (seen red, then restored green, 2026-10-05): residual chains kept whole -> the
    chain through the stretch's op survives."""
    from neurobrix.core.module.tiling_engine import OpLevelTilingPlan
    p = OpLevelTilingPlan("c")
    p.fusion_pairs = [("a", "b", {}), ("c", "d", {})]
    p.tiled_ops = [("b", {}), ("e", {})]
    p.residual_chains = [{"fork_uid": "f", "merge_uid": "g", "chain_uids": ["b"]},
                         {"fork_uid": "h", "merge_uid": "i", "chain_uids": []}]
    p.conv3d_chunks = ["b", "j"]
    assert p.drop_ops({"b"}) == 4
    assert [e[0] for e in p.fusion_pairs] == ["c"] and [e[0] for e in p.tiled_ops] == ["e"]
    assert [r["fork_uid"] for r in p.residual_chains] == ["h"] and p.conv3d_chunks == ["j"]
    assert p.drop_ops({"zzz"}) == 0
