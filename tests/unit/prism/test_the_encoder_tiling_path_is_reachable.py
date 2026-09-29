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
  was written for, and the decline is right for each (measured 2026-09-24):
    - Allegro: temporal map ((b*t + 1)//2 + 1)//2. It is a ceil, not linear-down (21 frames
      give 6, not 5), and it folds the batch symbol into time.
    - CogVideoX-5b-I2V: traced at t = 1, so it has no temporal ratio to read.
    - Wan2.1-VACE: causal ((t - 1)//4 + 1).
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


@pytest.mark.parametrize("model", REAL_ENCODERS)
def test_a_real_encoder_reaching_the_branch_gets_a_decision_not_a_crash(model):
    container = SimpleNamespace(_cache_path=container_root(model))
    assert _decide(container, "vae_encoder", 80 * GB) is None, (
        f"{model}'s encoder is outside the linear-down class and must be declined")


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
