"""An encoder tiled in space with the whole clip in every tile asks no frame lattice — and a tiling
decline says why, instead of "no tiling fits".

Allegro-TI2V at its confirmation request (720x1280, 88 frames, 2026-10-03) was refused on both
V100 classes: its `vae_encoder` (319 500 MB of activations) got no tile, and the refusal said
"no tiling fits" beside an arithmetic that said 39 tiles fit. Two causes, in that order:

* its container's time map is ((s0*s1 + 1)//2 + 1)//2 — the trace froze the batch at a literal 1
  (`aten.view::1` sized [1, s0*s1, 128, s2, s3]) and folded the batch symbol into time. Neither
  temporal class reads a time map in two symbols. That is the container's defect (Forge re-emits
  it); the engine's part is to NAME it, which the refusal did not.
* with the batch a symbol, the map is ((t+1)//2 + 1)//2 — what a stride-2 slice `[::2]` records.
  As integers that is 1 + (t-1)//4, the causal class, which is tiled in space with the whole clip
  in every tile. Its gate still asked `(num_frames - 1) % 4`, a frame lattice no tile of that
  class depends on, and refused the encoder at 88 frames.

Injections, each seen RED: the frame gate restored to `(num_frames - 1) % t_ratio` -> the two
88-frame cases; the decline recorder silenced -> the naming cases; the refusal back to "no tiling
fits" -> the refusal case.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from neurobrix.core.prism import InputConfig, PrismSolver
from neurobrix.core.prism.profiler import temporal_causal_downscale_ratio, temporal_downscale_ratio
from neurobrix.core.prism.solver import ComponentMemory

MB = 1024 ** 2
ACTIVATION = 319_500 * MB          # the planner's figure for Allegro-TI2V's encoder at 88x720x1280


def _sym(i, trace):
    return {"type": "symbol", "id": i, "trace": trace}


def _op(kind, left, right, trace):
    return {"type": kind, "left": left, "right": right, "trace": trace}


def _encoder(tmp_path, form):
    """[B, 3, T, H, W] -> [B, T', 4, H/8, W/8] (the latent's time on axis 1, as Allegro's encode
    writes it), T' by `form`:
      slice   ((t+1)//2 + 1)//2                — two stride-2 slices
      linear  t//4
      folded  ((s0*t + 1)//2 + 1)//2, batch 1  — the frozen-batch trace"""
    t_in, h_in, w_in = 20, 112, 176
    b, t, h, w = _sym("s0", 1), _sym("s1", t_in), _sym("s2", h_in), _sym("s3", w_in)
    ceil4 = lambda x: _op("floordiv", _op("add", _op("floordiv", _op("add", x, 1, 21), 2, 10), 1, 11), 2, 5)
    t_out, b_out = {"slice": (ceil4(t), b), "linear": (_op("floordiv", t, 4, 5), b),
                    "folded": (ceil4(_op("mul", b, t, 20)), 1)}[form]
    graph = {
        "input_tensor_ids": ["x"], "output_tensor_ids": ["z"],
        "tensors": {
            "x": {"shape": [1, 3, t_in, h_in, w_in], "symbolic_shape": {"dims": [b, 3, t, h, w]}},
            "z": {"shape": [1, 5, 4, h_in // 8, w_in // 8],
                  "symbolic_shape": {"dims": [b_out, t_out, 4, _op("floordiv", h, 8, 14),
                                              _op("floordiv", w, 8, 22)]}},
        },
        "ops": {}, "execution_order": [],
    }
    comp = tmp_path / form / "components" / "enc"
    comp.mkdir(parents=True)
    (comp / "graph.json").write_text(json.dumps(graph))
    (comp / "profile.json").write_text(json.dumps({"config": {"block_out_channels": [1, 2, 3, 4]}}))
    return SimpleNamespace(_cache_path=tmp_path / form), graph


def _decide(container, frames, rung_mb):
    s = PrismSolver()
    s._input_config = InputConfig(batch_size=1, height=720, width=1280, num_frames=frames,
                                  temporal_compression=4)
    mem = ComponentMemory(component_name="enc", weight_bytes=244 * MB, activation_bytes=ACTIVATION,
                          overhead_bytes=0)
    return s, s._spatial_component_tiling(container, "enc", mem, rung_mb)


def test_the_slice_form_reads_as_the_causal_class(tmp_path):
    _, g = _encoder(tmp_path, "slice")
    assert temporal_causal_downscale_ratio(g) == (4, 1)
    assert temporal_downscale_ratio(g) is None


@pytest.mark.parametrize("rung_mb", [32768, 16384])
def test_a_slice_form_encoder_at_88_frames_is_tiled_in_space_with_the_whole_clip(tmp_path, rung_mb):
    container, _ = _encoder(tmp_path, "slice")
    _, spec = _decide(container, 88, rung_mb)
    assert spec is not None, "an encoder whose time is never tiled was refused for its frame count"
    assert spec["downscale"] is True and "t_tile" not in spec and spec["t_axis_out"] == 1, spec
    assert spec["tile_size"] % 8 == 0 and spec["tile_size"] < 720, spec
    assert spec["tiled_activation_bytes"] <= rung_mb * 0.40 * MB, spec


def test_a_linear_encoder_off_its_frame_lattice_is_still_declined_and_says_so(tmp_path):
    """The linear class tiles TIME: its tiles land at frame / 4, so 89 frames stays a decline."""
    container, _ = _encoder(tmp_path, "linear")
    s, spec = _decide(container, 89, 32768)
    assert spec is None, spec
    assert "4 frames" in s._tiling_declined["enc"], s._tiling_declined


def test_a_frozen_batch_time_map_is_declined_by_name(tmp_path):
    container, _ = _encoder(tmp_path, "folded")
    s, spec = _decide(container, 88, 32768)
    assert spec is None, spec
    why = s._tiling_declined.get("enc", "")
    assert "(s0*s1)" in why and "s1 alone" in why, why


def test_the_refusal_says_why_the_encoder_got_no_tile(tmp_path):
    container, _ = _encoder(tmp_path, "folded")
    s, _ = _decide(container, 88, 32768)
    s._strategies_tried, s._layer_streaming_declined = ["cpu_streaming"], None
    s._host_device_overflow = {"cpu_streaming": ("cuda:0", 28_639.0, {"enc": 415_350.0})}
    comp = SimpleNamespace(weight_mb=244.0, activation_mb=319_500.0, total_mb=319_744.0,
                           weight_bytes=244 * MB, activation_bytes=ACTIVATION, overhead_bytes=0)
    card = SimpleNamespace(device_string="cuda:0", capacity_mb=31_130.0, recommended_mb=None,
                           host_memory=None)
    with pytest.raises(RuntimeError) as exc:
        s._fail_error([("enc", comp)], [card])
    msg = str(exc.value)
    assert "no tiling fits" not in msg, msg
    assert "enc's activations (415,350 MB untiled; no tile: no axis of its output z" in msg, msg
