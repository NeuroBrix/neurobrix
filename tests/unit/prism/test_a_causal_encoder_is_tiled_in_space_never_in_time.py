"""A causal video encoder (t_out = 1 + (t_in - 1) // r) over its budget is tiled in SPACE, the whole
clip in every tile — never in time.

Prism declined every causal encoder: a temporal tile taken mid-clip would re-apply the clip-start
semantics, and that stays true. But the Wan encoders traced as one causal pass (2026-09-26) hold
8.5 GiB in their first activation at 81 frames of 352x832, and the decline killed the run on a
16 GB card. A spatial tile carrying the whole clip is exact in time — it is the vendor's own tiled
encode for these VAEs (diffusers AutoencoderKLWan.tiled_encode).

On the old solver the causal-spec test fails (None); on a solver that tiled time for this class
the no-temporal-key assertion fails.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from neurobrix.core.prism import InputConfig, PrismSolver
from neurobrix.core.prism.profiler import temporal_causal_downscale_ratio, temporal_downscale_ratio
from neurobrix.core.prism.solver import ComponentMemory

GB = 1024 ** 3
RUNG_MB = 16384


def _sym(i, trace):
    return {"type": "symbol", "id": i, "trace": trace}


def _op(kind, left, right, trace):
    return {"type": kind, "left": left, "right": right, "trace": trace}


def _encoder_graph(causal: bool):
    t_in, h_in, w_in = 9, 112, 176
    b, t, h, w = _sym("s0", 1), _sym("s1", t_in), _sym("s2", h_in), _sym("s3", w_in)
    if causal:
        t_out = _op("add", _op("floordiv", _op("add", t, -1, t_in - 1), 4, 2), 1, 3)
    else:
        t_out = _op("floordiv", t, 4, t_in // 4)
    return {
        "input_tensor_ids": ["x"], "output_tensor_ids": ["z"],
        "tensors": {
            "x": {"shape": [1, 3, t_in, h_in, w_in], "symbolic_shape": {"dims": [b, 3, t, h, w]}},
            "z": {"shape": [1, 32, 3, h_in // 8, w_in // 8],
                  "symbolic_shape": {"dims": [b, 32, t_out, _op("floordiv", h, 8, h_in // 8),
                                              _op("floordiv", w, 8, w_in // 8)]}},
        },
        "ops": {}, "execution_order": [],
    }


def test_the_causal_map_is_read_from_the_graph():
    assert temporal_causal_downscale_ratio(_encoder_graph(True)) == (4, 2)
    assert temporal_downscale_ratio(_encoder_graph(True)) is None        # still not linear
    assert temporal_causal_downscale_ratio(_encoder_graph(False)) is None


@pytest.fixture()
def causal_encoder(tmp_path):
    comp = tmp_path / "components" / "enc"
    comp.mkdir(parents=True)
    (comp / "graph.json").write_text(json.dumps(_encoder_graph(True)))
    (comp / "profile.json").write_text(json.dumps({"config": {"block_out_channels": [1, 2, 3, 4]}}))
    return SimpleNamespace(_cache_path=tmp_path)


def _decide(container, frames, activation_bytes):
    s = PrismSolver()
    s._input_config = InputConfig(batch_size=1, height=352, width=832, num_frames=frames,
                                  temporal_compression=4)
    mem = ComponentMemory(component_name="enc", weight_bytes=int(0.3 * GB),
                          activation_bytes=activation_bytes, overhead_bytes=0)
    return s._spatial_component_tiling(container, "enc", mem, RUNG_MB)


def test_a_causal_encoder_over_its_budget_is_tiled_in_space_with_the_whole_clip(causal_encoder):
    spec = _decide(causal_encoder, 81, 80 * GB)
    assert spec is not None, "the causal encoder was declined"
    assert spec["downscale"] is True and spec["scale_factor"] == 8, spec
    assert "t_tile" not in spec, f"a causal encoder was tiled in time: {spec}"
    assert spec["tiled_activation_bytes"] <= RUNG_MB * 0.40 * 1024 * 1024, spec
    assert spec["tile_size"] % 8 == 0 and spec["tile_size"] < 832, spec


def test_a_causal_encoder_off_its_frame_lattice_or_within_budget_is_left_alone(causal_encoder):
    assert _decide(causal_encoder, 88, 80 * GB) is None      # (88 - 1) % 4 != 0
    assert _decide(causal_encoder, 81, 1 * GB) is None       # fits untiled


def test_a_causal_encoder_is_sized_at_the_requests_pixels_not_its_latent():
    """The tiling above only triggers if the plan SEES the overflow: a causal encoder's
    time/height/width bind to the request's pixel extents, as a linear encoder's always did. On
    the old binding this binds the latent extents (21, 44, 104) and the Wan encoder was sized at
    1.1 GB for 81 frames of 352x832."""
    from neurobrix.core.prism.profiler import ActivationProfiler
    g = _encoder_graph(True)
    g["symbolic_context"] = {"symbols": {
        "s0": {"name": "batch", "trace_value": 1}, "s1": {"name": "time", "trace_value": 9},
        "s2": {"name": "height", "trace_value": 112}, "s3": {"name": "width", "trace_value": 176}}}
    m = ActivationProfiler(g).build_symbol_map(
        InputConfig(batch_size=1, height=352, width=832, num_frames=81, temporal_compression=4))
    assert (m["s1"], m["s2"], m["s3"]) == (81, 352, 832), m


def test_an_encode_alias_without_a_block_list_takes_its_ratio_from_its_graph(tmp_path):
    """Wan's `vae_encoder` is an encode alias of the VAE: its profile carries no block list (the
    configuration lives on the decode component). The ratio is then the graph's traced in/out
    extent; on the old solver the spec is None (declined for want of a config)."""
    comp = tmp_path / "components" / "enc"
    comp.mkdir(parents=True)
    (comp / "graph.json").write_text(json.dumps(_encoder_graph(True)))
    (comp / "profile.json").write_text(json.dumps({"config": {"_class_name": "AutoencoderKLWan"}}))
    spec = _decide(SimpleNamespace(_cache_path=tmp_path), 81, 80 * GB)
    assert spec is not None and spec["scale_factor"] == 8 and "t_tile" not in spec, spec
