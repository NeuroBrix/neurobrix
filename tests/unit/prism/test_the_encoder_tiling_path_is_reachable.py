"""The 5-D ENCODER tiling path could not run: every plan that reached it raised.

`_spatial_component_tiling` hands a DOWNSAMPLING component (a video VAE encoder: pixels ->
latent) to `_encode_component_tiling(graph, profile, mem, budget_bytes, ...)` — about thirty lines
BEFORE `budget_bytes` was assigned, so the call raised `UnboundLocalError`. And
`_encode_component_tiling` wrote `"budget_rung_mb": int(rung_mb)` into its spec from a `rung_mb`
it never received — a `NameError` one step further in. Found by a static read (Pyright), never by
a run: nothing reached the branch.

Two cells reach it, and they reach different depths:

* REAL CONTAINER, the branch's entry. Every 5-D downsampler in this cache: Allegro-TI2V,
  CogVideoX-5b-I2V and Wan2.1-VACE-1.3B `vae_encoder`. Before the fix all three RAISED on
  entry, including the ones that should simply be declined. None is in the D2 class the tail
  was written for (measured 2026-09-24):
    - Allegro-TI2V: TILED, because its container was rewritten. The old trace froze the batch at
      a literal 1 (`aten.view::1` sized [1, s0*s1, 128, s2, s3]), folding the batch symbol into
      time — ((b*t + 1)//2 + 1)//2, no time map in one symbol — and the encoder was declined
      with that expression named. Forge rewrote the container on 2026-10-04 23:20:41 (graph
      c1861e72 -> f587fc4b, campaigns/2026_10_04_frozen_scan/STATE.md line 213): the cache's
      vae_encoder graph holds no `s0*s1` any more, and its time map is ((s1 + 1)//2 + 1)//2 in s1
      alone. A down-map of one symbol, tiled in space with the whole clip in every tile — the
      same spec as Wan2.1-VACE (tile 264, overlap 40, t_axis_out 1, no t_tile, rung 16 384).
      The time-map test below reads that from the container, so a stale container is named.
    - CogVideoX-5b-I2V: traced at t = 1, so it has no temporal ratio to read: declined.
    - Wan2.1-VACE: causal ((t - 1)//4 + 1). Since 2026-09-26 that class is tiled in space with
      the whole clip in every tile, and since 2026-10-04 at any frame count — 88 included,
      which is off its 4k+1 lattice and was declined for it, though no tile of it is taken in
      time.
* SYNTHETIC ENCODER, the tail. A minimal graph with the one property the gate asks for — a
  linear-down temporal map t -> t//4 and an 8x spatial ratio — written to disk as a container.
  This is the only way to reach the spec the tail builds (and its `rung_mb`), because no cached
  container is in that class. The scenario is constructed on purpose, not found.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from neurobrix.core.prism import InputConfig, PrismSolver
from neurobrix.core.prism.solver import ComponentMemory
from tests.unit.prism._pinned_machine import container_root
RUNG_MB = 16384
GB = 1024 ** 3
REAL_ENCODERS = ["Allegro-TI2V", "CogVideoX-5b-I2V", "Wan2.1-VACE-1.3B-diffusers"]


def _decide(container, comp, activation_bytes):
    s = PrismSolver()
    # 720 x 1280, 88 frames: on every lattice these encoders declare (8 spatial, 4 temporal).
    s._input_config = InputConfig(batch_size=1, height=720, width=1280, num_frames=88,
                                  temporal_compression=4)
    mem = ComponentMemory(component_name=comp, weight_bytes=int(0.3 * GB),
                          activation_bytes=activation_bytes, overhead_bytes=0)
    return s._spatial_component_tiling(container, comp, mem, RUNG_MB)


@pytest.mark.parametrize("model,tiled", [("Allegro-TI2V", True), ("CogVideoX-5b-I2V", False),
                                         ("Wan2.1-VACE-1.3B-diffusers", True)])
def test_a_real_encoder_reaching_the_branch_gets_a_decision_not_a_crash(model, tiled):
    container = SimpleNamespace(_cache_path=container_root(model))
    spec = _decide(container, "vae_encoder", 80 * GB)
    if tiled:
        assert spec is not None and "t_tile" not in spec, (
            f"{model}'s encoder is tiled in space with the whole clip: {spec}")
        assert {k: spec[k] for k in SPACE_TILED} == SPACE_TILED_AT_THE_RUNG, (model, spec)
    else:
        assert spec is None, f"{model}'s encoder has no time map the engine tiles: {spec}"


#: The spec of a down-map encoder tiled in space with the whole clip in every tile, at 720 x 1280
#: on the 16 GB rung: what Wan2.1-VACE and the rewritten Allegro-TI2V both receive.
SPACE_TILED_AT_THE_RUNG = {"tile_size": 264, "overlap": 40, "t_axis_out": 1, "downscale": True,
                           "budget_rung_mb": RUNG_MB}
SPACE_TILED = tuple(SPACE_TILED_AT_THE_RUNG)


def _symbols(expr):
    if isinstance(expr, dict):
        own = {expr["id"]} if expr.get("type") == "symbol" else set()
        return own.union(*(_symbols(v) for k, v in expr.items() if k in ("left", "right")))
    return set()


@pytest.mark.parametrize("model", ["Allegro-TI2V", "Wan2.1-VACE-1.3B-diffusers"])
def test_a_tiled_encoder_s_time_map_is_one_symbol_never_the_batch(model):
    """The reason a tiled encoder is tiled, read from the container: its output's time axis
    (`t_axis_out`) is a map of ONE symbol, and not the batch's. Allegro-TI2V's pre-23:20 graph
    failed here (its time folded the batch: s0*s1)."""
    graph = json.loads((container_root(model) / "components" / "vae_encoder" / "graph.json")
                       .read_text())
    (inp,), (out,) = graph["input_tensor_ids"], graph["output_tensor_ids"]
    batch = _symbols(graph["tensors"][inp]["symbolic_shape"]["dims"][0])
    t_axis = SPACE_TILED_AT_THE_RUNG["t_axis_out"]
    time = _symbols(graph["tensors"][out]["symbolic_shape"]["dims"][t_axis])
    assert len(time) == 1 and not time & batch, (model, time, batch)


def _sym(i, trace):
    return {"type": "symbol", "id": i, "trace": trace}


@pytest.fixture()
def linear_down_encoder(tmp_path):
    """A container holding one component whose graph says: [B,3,T,H,W] -> [B,16,T//4,H//8,W//8]."""
    t_in, h_in, w_in = 20, 112, 176
    b, t, h, w = _sym("s0", 1), _sym("s1", t_in), _sym("s2", h_in), _sym("s3", w_in)

    def div(x, n, trace):
        return {"type": "floordiv", "left": x, "right": n, "trace": trace}

    graph = {
        "input_tensor_ids": ["x"],
        "output_tensor_ids": ["z"],
        "tensors": {
            "x": {"shape": [1, 3, t_in, h_in, w_in], "symbolic_shape": {"dims": [b, 3, t, h, w]}},
            "z": {"shape": [1, 16, t_in // 4, h_in // 8, w_in // 8],
                  "symbolic_shape": {"dims": [b, 16, div(t, 4, t_in // 4),
                                              div(h, 8, h_in // 8), div(w, 8, w_in // 8)]}},
        },
        "ops": {}, "execution_order": [],
    }
    comp = tmp_path / "components" / "enc"
    comp.mkdir(parents=True)
    (comp / "graph.json").write_text(json.dumps(graph))
    (comp / "profile.json").write_text(json.dumps({"config": {"block_out_channels": [1, 2, 3, 4]}}))
    return SimpleNamespace(_cache_path=tmp_path)


def test_a_linear_down_encoder_over_its_budget_is_tiled_on_the_rung(linear_down_encoder):
    spec = _decide(linear_down_encoder, "enc", 80 * GB)
    assert spec is not None, "the D2 gate declined the one class it was written for"
    assert spec["downscale"] is True and spec["scale_factor"] == 8, spec
    assert spec["budget_rung_mb"] == RUNG_MB, spec
    assert spec["tiled_activation_bytes"] <= RUNG_MB * 0.40 * 1024 * 1024, spec


def test_a_linear_down_encoder_that_fits_untiled_is_left_alone(linear_down_encoder):
    assert _decide(linear_down_encoder, "enc", 1 * GB) is None
