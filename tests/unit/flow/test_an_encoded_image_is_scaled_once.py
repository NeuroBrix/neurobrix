"""An image-to-video condition is the vae_encoder component's output, scaled ONCE.

The vae_encoder component's traced graph ends with `mode * scaling_factor` (invert-aware) and the frames-first
permute: its output IS the latent the vendor pipeline concatenates
(CogVideoXImageToVideoPipeline.prepare_latents: `image_latents = scaling_factor * latent_dist.sample()`).
The CogVideoX builder scaled it a second time, so CogVideoX-5b-I2V was conditioned on 0.49 x mode against the
vendor's 0.7 x sample: step-0 CFG output cos 0.92 against the vendor fed the same noise, frames 11 dB
(2026-10-04 ladder). Both engines.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import neurobrix.core.runtime.resolution.i2v_conditioning as compiled_i2v
import neurobrix.triton.i2v_conditioning as triton_i2v

LATENT_FF = (1, 1, 16, 4, 6)     # the encoder's output: frames-first [B, F=1, C, lh, lw]
STATE = (1, 13, 16, 4, 6)        # the denoiser's frames-first state [B, T, C, lh, lw]


class _Resolver:
    def __init__(self):
        self.resolved = {}

    def get(self, key):
        return self.resolved[key]


class _Ctx:
    def __init__(self, cache_path):
        self.pkg = type("Pkg", (), {"cache_path": cache_path, "manifest": {"model_name": "cogvideox-i2v-test"}})()
        self.variable_resolver = _Resolver()


def _ctx(tmp_path, invert):
    d = tmp_path / "components" / "vae"
    d.mkdir(parents=True, exist_ok=True)
    (d / "profile.json").write_text(json.dumps({"config": {
        "latent_channels": 16, "scaling_factor": 0.7, "invert_scale_latents": invert}}))
    return _Ctx(tmp_path)


def _latent():
    return np.random.default_rng(5).standard_normal(LATENT_FF).astype(np.float32)


def _expected(lat):
    out = np.zeros(STATE, np.float32)
    out[:, 0] = lat[:, 0]
    return out


@pytest.mark.parametrize("invert", [False, True])
def test_the_compiled_condition_is_the_encoders_output_padded(tmp_path, invert):
    ctx = _ctx(tmp_path, invert)
    lat = _latent()
    ctx.variable_resolver.resolved["vae_encoder.output_0"] = torch.from_numpy(lat)
    ctx.variable_resolver.resolved["global.latents"] = torch.zeros(STATE)
    got = compiled_i2v.build_condition(ctx, {"style": "cogvideox", "condition_component": "vae_encoder"}, 49)
    assert tuple(got.shape) == STATE
    np.testing.assert_array_equal(got.numpy(), _expected(lat))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NBXTensor is device-backed")
@pytest.mark.parametrize("invert", [False, True])
def test_the_triton_condition_is_the_encoders_output_padded(tmp_path, invert):
    from neurobrix.kernels.nbx_tensor import NBXTensor
    ctx = _ctx(tmp_path, invert)
    lat = _latent()
    ctx.variable_resolver.resolved["vae_encoder.output_0"] = NBXTensor.from_numpy(lat).to("cuda")
    ctx.variable_resolver.resolved["global.latents"] = NBXTensor.from_numpy(np.zeros(STATE, np.float32)).to("cuda")
    got = triton_i2v.build_condition(ctx, {"style": "cogvideox", "condition_component": "vae_encoder"}, 49)
    assert tuple(got.shape) == STATE
    np.testing.assert_array_equal(got.numpy(), _expected(lat))


def _installed(model):
    from neurobrix.core.paths import cache_dir
    root = Path(cache_dir()) / model
    return root if (root / "components" / "vae_encoder" / "graph.json").exists() else None


@pytest.mark.parametrize("model", ["CogVideoX-5b-I2V"])
def test_the_encoder_graph_already_carries_the_scale(model):
    """The convention read from the data: the installed container's vae_encoder graph ends with a mul by the
    vae profile's scaling_factor, then the frames-first permute — so the builder must not scale again."""
    root = _installed(model)
    if root is None:
        pytest.skip(f"{model} is not installed here")
    g = json.loads((root / "components" / "vae_encoder" / "graph.json").read_text())
    ops = {o["op_uid"]: o for o in (g["ops"].values() if isinstance(g["ops"], dict) else g["ops"])}
    by_out = {t: o for o in ops.values() for t in o["output_tensor_ids"]}
    last = by_out[g["output_tensor_ids"][0]]
    assert last["op_type"] == "aten::permute" and last["attributes"]["dims"] == [0, 2, 1, 3, 4]
    mul = by_out[last["input_tensor_ids"][0]]
    prof = json.loads((root / "components" / "vae" / "profile.json").read_text())
    cfg = prof.get("config") if isinstance(prof.get("config"), dict) else prof
    sf = float(cfg["scaling_factor"])
    want = 1.0 / sf if cfg.get("invert_scale_latents") else sf
    assert mul["op_type"] == "aten::mul"
    assert [a["value"] for a in mul["attributes"]["args"] if a["type"] == "scalar"] == [pytest.approx(want)]
