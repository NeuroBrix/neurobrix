"""A streamed piece receives every input its ops read — including a COMPUTABLE buffer, which no
seam carries and no weight file holds.

The defect (the Mac, 2026-09-29 14:20): Sana_1600M_4Kpx_BF16, triton-sequential, planned
`layer_streaming` in 24 pieces, died in piece 0:

    [triton-sequential] Failed at aten.add::0 (aten::add): AttributeError: 'NoneType' object has
    no attribute '_dtype' | None args at positions [1] of 2

`aten.add::0` is the patch embed plus `param::patch_embed.pos_embed`, a parameter marked
`is_computable` (a `sincos_2d_pos_embed` at the request's resolution): absent from
`weights_index.json` by construction, and not a seam input (a parameter never is). An executor
computes it in `load_weights` from its runtime resolution and its component handler — both given
by the runtime and the factory to the COMPONENT's executor only. A piece is built by the strategy,
never reached by either, so its `_compute_computable_buffers` returned at "no runtime resolution"
and the op read None.

The cell runs the real `LayerStreamingStrategy.execute_component` over pieces the real
`build_segment_graph` cut, on real `GraphExecutor`s (compiled, CPU); only the two things that need
a card or a container are stubbed — the shard read (there is no weight file: the buffer is not in
one) and each piece's op execution (it reports what the piece holds when it runs). Two requests at
two resolutions, the second far from the first: a piece outlives a request, so a resolution copied
once at build would pass the first and fail the second.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from neurobrix.core.optim.passes.normalize import graph_fingerprint, normalize_for_branch
from neurobrix.core.runtime.graph_executor import GraphExecutor, output_key
from neurobrix.core.strategies.layer_streaming import LayerStreamingStrategy

COMPONENT = "transformer"
EMBED_DIM = 8
VAE_SCALE = 32            # what the component handler answers, as Sana's does
POS = "param::pe.pos_embed"


def _tensor(tid, shape, **extra):
    base = {"tensor_id": tid, "shape": shape, "dtype": "float32", "device": "cpu",
            "producer_op_uid": None, "output_index": None, "consumer_op_uids": [],
            "is_parameter": False, "is_input": False, "weight_name": None,
            "input_name": None, "output_name": None}
    base.update(extra)
    return base


def _op(uid, op_type, ins, out):
    return {"op_uid": uid, "op_type": op_type, "input_tensor_ids": ins,
            "output_tensor_ids": [out], "attributes": {
                "args": [{"type": "tensor", "tensor_id": t} for t in ins], "kwargs": {}}}


def _graph():
    """x + pos_embed (the computable buffer), then * w — two pieces, the buffer in the first."""
    spec = {"method": "sincos_2d_pos_embed",
            "params": {"embed_dim": EMBED_DIM, "base_size": 2, "interpolation_scale": 1.0,
                       "patch_size": 1},
            "shape_source": {"grid_h": "runtime.latent_h", "grid_w": "runtime.latent_w"},
            "traced_shape": [1, 4, EMBED_DIM]}
    tensors = {
        "input::x": _tensor("input::x", [1, 4, EMBED_DIM], is_input=True, input_name="x",
                            consumer_op_uids=["aten.add::0"]),
        POS: _tensor(POS, [1, 4, EMBED_DIM], is_parameter=True, weight_name="pe.pos_embed",
                     consumer_op_uids=["aten.add::0"], is_computable=True,
                     computation_spec=spec),
        "aten.add::0::out_0": _tensor("aten.add::0::out_0", [1, 4, EMBED_DIM],
                                      producer_op_uid="aten.add::0", output_index=0,
                                      consumer_op_uids=["aten.mul::0"]),
        "param::w": _tensor("param::w", [EMBED_DIM], is_parameter=True, weight_name="w",
                            consumer_op_uids=["aten.mul::0"]),
        "aten.mul::0::out_0": _tensor("aten.mul::0::out_0", [1, 4, EMBED_DIM],
                                      producer_op_uid="aten.mul::0", output_index=0,
                                      output_name="out"),
    }
    ops = {"aten.add::0": _op("aten.add::0", "aten::add", ["input::x", POS],
                              "aten.add::0::out_0"),
           "aten.mul::0": _op("aten.mul::0", "aten::mul", ["aten.add::0::out_0", "param::w"],
                              "aten.mul::0::out_0")}
    return {"component_name": COMPONENT, "format": "tensor_dag", "version": "0.1",
            "torch_dtype": "float32", "input_names": ["x"], "tensors": tensors, "ops": ops,
            "execution_order": ["aten.add::0", "aten.mul::0"],
            "input_tensor_ids": ["input::x"], "output_tensor_ids": ["aten.mul::0::out_0"],
            "computable_buffers": {"pe.pos_embed": spec}}


def _strategy():
    graph = _graph()
    base = GraphExecutor(family="image", vendor="nvidia", arch="volta", device="cpu",
                         dtype="float32", mode="compiled")
    base.load_graph_from_dict(graph)
    base._component_name = COMPONENT
    # The factory attaches the handler to the COMPONENT's executor; nothing else gets one.
    base._component_handler = types.SimpleNamespace(get_latent_scale=lambda: VAE_SCALE)
    pkg = types.SimpleNamespace(cache_path="/nonexistent-container")
    ctx = types.SimpleNamespace(
        component_executors={COMPONENT: base},
        layer_segments={COMPONENT: [["aten.add::0", "aten.add::0"],
                                    ["aten.mul::0", "aten.mul::0"]]},
        layer_graphs={COMPONENT: graph_fingerprint(
            normalize_for_branch(graph, base.mode, base.family, declared_moe=None))},
        layer_moe={}, runtime_package=pkg)
    return LayerStreamingStrategy(ctx, "layer_streaming"), base


def _instrument(strategy, seen):
    """Stub the shard read (no container: the buffer is in no weight file) and each piece's op
    execution — which records, for every parameter its ops read, what the piece holds when it
    runs. Everything else (`load_weights`, constants, computable buffers, the loop) is real."""
    pieces = strategy._build_segment_executors(COMPONENT)
    strategy._segment_executors[COMPONENT] = pieces

    for piece in pieces:
        def read_shards(nbx_path, component, shard_map, only=None, _p=piece):
            _p._weights = {"w": np.ones(EMBED_DIM, dtype=np.float32)} \
                if "param::w" in _p._dag["tensors"] else {}

        def run(inputs, _p=piece):
            sub = _p._dag
            for op_uid in sub["execution_order"]:
                for tid in sub["ops"][op_uid]["input_tensor_ids"]:
                    if (sub["tensors"].get(tid) or {}).get("is_parameter"):
                        seen[(sub["segment_index"], tid)] = _p._weights.get(tid[len("param::"):])
            # Keyed as every engine keys a run's outputs (`output_key`): the component's own
            # output by its name. Keyed by tid, the stand-in answered a contract no engine has,
            # and the strategy's whole-run answer could not find 'out' (merge-queue-17).
            return {output_key(sub["tensors"].get(t), t): np.zeros(1)
                    for t in sub["output_tensor_ids"]}

        piece._load_weights_native = read_shards
        piece.run = run
    return pieces


@pytest.mark.parametrize("height,width", [(64, 64), (256, 128)])
def test_a_streamed_piece_holds_the_computable_buffer_its_op_reads(height, width):
    strategy, base = _strategy()
    seen = {}
    _instrument(strategy, seen)

    # First request at the trace-sized grid, so a piece built with a stale resolution would pass it.
    base.set_runtime_resolution(64, 64)
    strategy.execute_component(COMPONENT, inputs={"x": np.zeros((1, 4, EMBED_DIM))})
    # The request under test — what `RuntimeExecutor` does per run: it sets the COMPONENT's executor.
    base.set_runtime_resolution(height, width)
    seen.clear()
    strategy.execute_component(COMPONENT, inputs={"x": np.zeros((1, 4, EMBED_DIM))})

    assert (0, POS) in seen, "piece 0 never ran the op that reads the buffer"
    held = seen[(0, POS)]
    assert held is not None, (
        f"piece 0 ran aten.add::0 without {POS}: a computable buffer is computed from the "
        f"component's runtime resolution and handler, and the piece never received them "
        f"(the Mac's Sana_1600M_4Kpx_BF16 failure)")
    tokens = (height // VAE_SCALE) * (width // VAE_SCALE)
    assert tuple(held.shape) == (1, tokens, EMBED_DIM), (
        f"piece 0 holds a buffer for another resolution: {tuple(held.shape)}, the request "
        f"{height}x{width} makes {tokens} tokens")
    # And the ordinary parameter of the next piece still arrives as it did.
    assert seen[(1, "param::w")] is not None
