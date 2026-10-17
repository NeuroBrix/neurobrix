"""An encoder whose graph couples the positions of its spatial extent — an attention whose keys span
height x width, a group norm, a softmax or a reduction over them — is never cut into spatial tiles.

A spatial tile is the untiled encoder only where every op is local in space: on a tile, each
position of a coupling op reads the tile instead of the whole extent, and the latents differ
everywhere (diffusers says it of its own tiled encode: "The end result of tiled encoding is
different from non-tiled encoding because each tile uses a different encoder"). Wan2.2-I2V-A14B at
17 frames on a 16 GB V100 (2026-10-16): its encoder's mid-block attention spans each frame's 60 x 104
latent positions; tiled in 352-pixel tiles, the I2V condition latent stood 0.54% from the untiled
one (2.8% on a seam), the transformer's first projection 10x the untiled run's error, and the
step-1 velocity 3.4% from the float32 vendor against 0.3% for the untiled plan.

On the old solver the causal encoder WITH a spatial attention is tiled exactly like the one
without, and `spatial_coupling` does not exist.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from neurobrix.core.prism import InputConfig, PrismSolver
from neurobrix.core.prism.solver import ComponentMemory

GB = 1024 ** 3
RUNG_MB = 16384


def _sym(i, trace):
    return {"type": "symbol", "id": i, "trace": trace}


def _op(kind, left, right, trace):
    return {"type": kind, "left": left, "right": right, "trace": trace}


T_IN, H_IN, W_IN = 9, 112, 176
B, T, H, W = _sym("s0", 1), _sym("s1", T_IN), _sym("s2", H_IN), _sym("s3", W_IN)
T_OUT = _op("add", _op("floordiv", _op("add", T, -1, T_IN - 1), 4, 2), 1, 3)
H8, W8 = _op("floordiv", H, 8, H_IN // 8), _op("floordiv", W, 8, W_IN // 8)


def _tensor(tid, concrete, dims):
    return {"shape": concrete, "symbolic_shape": {"dims": dims}}


def _encoder_graph(mid_op=None):
    """A causal video encoder [1, 3, T, H, W] -> [1, 32, 1 + (T-1)//4, H/8, W/8]; `mid_op` adds one
    op at the latent grid (the mid block), as the Wan / CogVideoX / Allegro encoders carry."""
    tensors = {
        "x": _tensor("x", [1, 3, T_IN, H_IN, W_IN], [B, 3, T, H, W]),
        "z": _tensor("z", [1, 32, 3, H_IN // 8, W_IN // 8], [B, 32, T_OUT, H8, W8]),
    }
    ops, order = {}, []
    hw = _op("mul", H8, W8, (H_IN // 8) * (W_IN // 8))
    if mid_op == "attention":       # per-frame attention: frames are the batch, H*W the tokens
        for n in ("q", "k", "v"):
            tensors[n] = _tensor(n, [3, 1, hw["trace"], 384], [T_OUT, 1, hw, 384])
        ops["attn"] = {"op_type": "aten::_scaled_dot_product_efficient_attention",
                       "input_tensor_ids": ["q", "k", "v"], "attributes": {"args": []}}
    elif mid_op == "group_norm":
        tensors["m"] = _tensor("m", [1, 32, 3, H_IN // 8, W_IN // 8], [B, 32, T_OUT, H8, W8])
        ops["gn"] = {"op_type": "aten::native_group_norm", "input_tensor_ids": ["m"],
                     "attributes": {"args": []}}
    elif mid_op == "channel_norm":  # Wan's RMS norm: over channels, per position — local in space
        tensors["m"] = _tensor("m", [1, 32, 3, H_IN // 8, W_IN // 8], [B, 32, T_OUT, H8, W8])
        ops["norm"] = {"op_type": "aten::linalg_vector_norm", "input_tensor_ids": ["m"],
                       "attributes": {"args": [{"type": "tensor", "tensor_id": "m"},
                                               {"type": "scalar", "value": 2.0},
                                               {"type": "list", "value": [1]},
                                               {"type": "scalar", "value": True}]}}
    elif mid_op == "spatial_softmax":
        tensors["m"] = _tensor("m", [3, 1, hw["trace"]], [T_OUT, 1, hw])
        ops["sm"] = {"op_type": "aten::_softmax", "input_tensor_ids": ["m"],
                     "attributes": {"args": [{"type": "tensor", "tensor_id": "m"},
                                             {"type": "scalar", "value": -1},
                                             {"type": "scalar", "value": False}]}}
    order = list(ops)
    return {"input_tensor_ids": ["x"], "output_tensor_ids": ["z"], "tensors": tensors,
            "symbolic_context": {"symbols": {
                "s0": {"name": "batch", "trace_value": 1, "source": "x::dim_0"},
                "s1": {"name": "time", "trace_value": T_IN, "source": "x::dim_2"},
                "s2": {"name": "height", "trace_value": H_IN, "source": "x::dim_3"},
                "s3": {"name": "width", "trace_value": W_IN, "source": "x::dim_4"}}},
            "ops": ops, "execution_order": order}


@pytest.mark.parametrize("mid_op,coupled", [
    ("attention", True), ("group_norm", True), ("spatial_softmax", True),
    ("channel_norm", False), (None, False)])
def test_the_coupling_is_read_from_the_graph(mid_op, coupled):
    from neurobrix.core.prism.profiler import spatial_coupling
    found = spatial_coupling(_encoder_graph(mid_op))
    assert (found is not None) == coupled, (mid_op, found)
    if coupled:
        assert found[0] in _encoder_graph(mid_op)["ops"], found


def test_a_graph_whose_spatial_extent_is_frozen_cannot_be_read():
    from neurobrix.core.prism.profiler import spatial_coupling
    g = _encoder_graph(None)
    g["tensors"]["x"]["symbolic_shape"]["dims"][-1] = W_IN       # width frozen at its trace value
    uid, why = spatial_coupling(g)
    assert uid == "" and "not symbols" in why, why


def _decide_files(tmp_path, mid_op):
    comp = tmp_path / "components" / "enc"
    comp.mkdir(parents=True)
    (comp / "graph.json").write_text(json.dumps(_encoder_graph(mid_op)))
    (comp / "profile.json").write_text(json.dumps({"config": {"block_out_channels": [1, 2, 3, 4]}}))


def _solver(activation_bytes=80 * GB):
    s = PrismSolver()
    s._input_config = InputConfig(batch_size=1, height=480, width=832, num_frames=17,
                                  temporal_compression=4)
    mem = ComponentMemory(component_name="enc", weight_bytes=int(0.2 * GB),
                          activation_bytes=int(activation_bytes), overhead_bytes=0)
    return s, mem


def _decide(tmp_path, mid_op, activation_bytes=80 * GB):
    _decide_files(tmp_path, mid_op)
    s, mem = _solver(activation_bytes)
    return s, s._spatial_component_tiling(SimpleNamespace(_cache_path=tmp_path), "enc", mem, RUNG_MB)


def test_a_coupled_encoder_past_the_card_is_not_tiled_and_prism_descends(tmp_path):
    """Hocine, 2026-10-16 18:40 / 18:42: Prism chooses only exact cuts, and refuses only at the bottom of the ladder.
    A coupled encoder whose untiled activations exceed the card gets no tile — never a cut into another function —
    and no refusal here: the plan descends the exact strategies (the refusal is `_fail_error`'s alone)."""
    s, spec = _decide(tmp_path, "attention")
    assert spec is None, f"a coupled encoder was tiled: {spec}"
    assert s._tiling_exactness["enc"]["exact"] is False


def test_a_coupled_encoder_within_the_card_is_not_tiled_and_the_plan_says_why(tmp_path):
    # over the tile budget (0.40 of the rung), within the rung: the tile is declined, the encoder placed untiled
    s, spec = _decide(tmp_path, "attention", activation_bytes=int(0.7 * RUNG_MB * 1024 * 1024))
    assert spec is None, f"an encoder whose attention spans the spatial extent was tiled: {spec}"
    why = s._tiling_declined.get("enc", "")
    assert "attn" in why and "not the untiled component" in why, why
    rec = s._tiling_exactness["enc"]
    assert rec["exact"] is False and "attn" in (rec["op"] or ""), rec


def test_a_vendor_whose_reference_tiles_declares_it_and_keeps_its_tile(tmp_path, monkeypatch):
    """The single exception: the vendor's own reference tiles (Allegro's VAE has no untiled encode), declared by Forge
    from the registry with its source (`components/<c>/vendor_tiling.json`, its clip graph's digest beside it) — and
    then the cut is the VENDOR's: the spec names its method, never a tile size Prism chose."""
    import hashlib
    _decide_files(tmp_path, "attention")
    comp = tmp_path / "components" / "enc"
    decl = {"format": "vendor_tiling", "version": 1, "method": "mask_sum", "when": "always",
            "clip_graph_sha256": hashlib.sha256((comp / "graph.json").read_bytes()).hexdigest(),
            "tile_in": [8, 128, 128], "stride_in": [4, 96, 96], "tile_out": [2, 16, 16], "stride_out": [1, 12, 12],
            "in_axes": [2, 3, 4], "out_axes": [2, 3, 4], "out_scale": [[1, 4], [1, 8], [1, 8]],
            "source": ["autoencoder_kl_allegro.py:804"]}
    (comp / "vendor_tiling.json").write_text(json.dumps(decl))
    s, mem = _solver()
    monkeypatch.setattr(s, "_get_component_dtype", lambda c, n: "float32")
    monkeypatch.setattr(s, "_activation_widths", lambda *a, **k: None)
    spec = s._spatial_component_tiling(SimpleNamespace(_cache_path=tmp_path), "enc", mem, RUNG_MB)
    assert spec is not None and spec.get("vendor") == "mask_sum", f"a declared vendor-tiled encoder lost its tile: {spec}"
    assert s._tiling_exactness["enc"]["exact"] is True and "allegro" in s._tiling_exactness["enc"]["why"]
    decl["clip_graph_sha256"] = "0" * 64
    (comp / "vendor_tiling.json").write_text(json.dumps(decl))
    s, mem = _solver()
    from neurobrix.core.prism.temporal_stream import StreamDeclarationError
    with pytest.raises(StreamDeclarationError, match="re-derive"):
        s._spatial_component_tiling(SimpleNamespace(_cache_path=tmp_path), "enc", mem, RUNG_MB)


def test_an_encoder_local_in_space_is_still_tiled(tmp_path):
    """The decline is the coupling, nothing else: the same encoder with a per-position channel norm
    keeps its spatial tile (the causal-encoder rule, test_a_causal_encoder_is_tiled_in_space_...)."""
    s, spec = _decide(tmp_path, "channel_norm")
    assert spec is not None and spec["downscale"] is True and "t_tile" not in spec, spec
    assert s._tiling_exactness["enc"] == {"exact": True, "why": "no op couples space"}


def _decoder_graph():
    """A video decoder [1, 16, T, h, w] -> [1, 3, T, 8h, 8w] whose mid block attends over every latent position of a
    frame, as the Wan decoder does: the decode-side tiling path (the upscale branch)."""
    h, w = _sym("s2", 14), _sym("s3", 22)
    hw = _op("mul", h, w, 14 * 22)
    tensors = {
        "z": _tensor("z", [1, 16, 3, 14, 22], [B, 16, T, h, w]),
        "x": _tensor("x", [1, 3, 3, 112, 176], [B, 3, T, _op("mul", h, 8, 112), _op("mul", w, 8, 176)]),
    }
    for n in ("q", "k", "v"):
        tensors[n] = _tensor(n, [3, 1, 14 * 22, 384], [T, 1, hw, 384])
    ops = {"attn": {"op_type": "aten::_scaled_dot_product_efficient_attention",
                    "input_tensor_ids": ["q", "k", "v"], "attributes": {"args": []}}}
    return {"input_tensor_ids": ["z"], "output_tensor_ids": ["x"], "tensors": tensors, "ops": ops,
            "execution_order": list(ops)}


def _decide_decoder(tmp_path, activation_bytes):
    comp = tmp_path / "components" / "dec"
    comp.mkdir(parents=True)
    (comp / "graph.json").write_text(json.dumps(_decoder_graph()))
    (comp / "profile.json").write_text(json.dumps({"config": {"block_out_channels": [1, 2, 3, 4]}}))
    s = PrismSolver()
    s._input_config = InputConfig(batch_size=1, height=480, width=832, num_frames=17, temporal_compression=4,
                                  vae_scale=8)
    mem = ComponentMemory(component_name="dec", weight_bytes=int(0.2 * GB), activation_bytes=int(activation_bytes),
                          overhead_bytes=0)
    return s, s._spatial_component_tiling(SimpleNamespace(_cache_path=tmp_path), "dec", mem, RUNG_MB)


def test_a_coupled_decoder_is_never_tiled_either(tmp_path):
    """The decode side takes the same rule: the Wan VAE decoder's mid-block attention spans the frame, and the plans
    that tiled the encoder tiled the decoder too."""
    s, spec = _decide_decoder(tmp_path, int(0.7 * RUNG_MB * 1024 * 1024))
    assert spec is None, f"a decoder whose attention spans the frame was tiled: {spec}"
    assert s._tiling_exactness["dec"]["exact"] is False
    s2, spec2 = _decide_decoder(tmp_path / "b", 80 * GB)       # past the card: still no tile, no refusal here
    assert spec2 is None and s2._tiling_exactness["dec"]["exact"] is False
