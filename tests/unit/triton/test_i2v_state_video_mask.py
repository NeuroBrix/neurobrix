"""Unit test: triton state_video_mask i2v conditioning mirror (R30).

Semantic-parity gate for the Allegro-TI2V `layout: state_video_mask` style:
the triton NBXTensor brick (`neurobrix.triton.i2v_conditioning`) must produce
the SAME condition tensor as the compiled brick
(`neurobrix.core.runtime.resolution.i2v_conditioning`) —
[B, C + ratio, T, lh, lw], channels [latent(C) then folded mask(ratio)], mask
1 everywhere except the folded plane(s) covering the conditioned pixel-frame
indices (default [0]) which are 0.

Structure:
  - CPU tests (always run): spec normalization parity, host folded-mask
    bit-parity vs the compiled torch build, fold-constraint raise parity.
  - GPU test (skips gracefully when CUDA is unavailable — NBXTensor is
    CUDA-backed): full triton build_condition on NBXTensor, downloaded and
    diffed bit-exactly against the compiled CPU reference.

Runnable two ways:
  - pytest:  PYTHONPATH=src python3 -m pytest tests/unit/triton/test_i2v_state_video_mask.py -v
  - script:  PYTHONPATH=src python3 tests/unit/triton/test_i2v_state_video_mask.py
"""
from __future__ import annotations

import json

import numpy as np
import torch

import neurobrix.core.runtime.resolution.i2v_conditioning as compiled_i2v
import neurobrix.triton.i2v_conditioning as triton_i2v

# ---------------------------------------------------------------------------
# Fixture scaffolding: fake ctx + vae profile, mirroring the runtime contract
# (pkg.manifest / pkg.cache_path / variable_resolver.resolved).
# ---------------------------------------------------------------------------

# vae_encoder graph output layout: frames-first [B, F, C, lh, lw] (the traced
# Allegro encode permutes to frames-first) — both bricks must detect the
# channel axis and arrange to channels-first [B, C, F, lh, lw].
_LATENT_SHAPE_FF = (1, 22, 4, 6, 8)   # B=1, F(latent)=22, C=4, lh=6, lw=8
_NUM_FRAMES = 88                      # pixel frames; ratio 4 -> latent_T 22
_RATIO = 4


class _Resolver:
    def __init__(self):
        self.resolved = {}

    def get(self, key):
        return self.resolved[key]


class _Pkg:
    def __init__(self, cache_path):
        self.cache_path = cache_path
        self.manifest = {"model_name": "allegro-ti2v-test"}


class _Ctx:
    def __init__(self, cache_path):
        self.pkg = _Pkg(cache_path)
        self.variable_resolver = _Resolver()


def _make_ctx(tmp_path) -> _Ctx:
    vae_dir = tmp_path / "components" / "vae"
    vae_dir.mkdir(parents=True, exist_ok=True)
    (vae_dir / "profile.json").write_text(json.dumps({
        "config": {
            "temporal_compression_ratio": _RATIO,
            "latent_channels": 4,
            "scaling_factor": 0.13,
        }
    }))
    return _Ctx(tmp_path)


def _latent_np() -> np.ndarray:
    rng = np.random.default_rng(23)
    return rng.standard_normal(_LATENT_SHAPE_FF).astype(np.float32)


def _spec(**overrides) -> dict:
    spec = {"style": "state_video_mask", "condition_component": "vae_encoder",
            "channel_dim": 1, "cond_frame_indices": [0]}
    spec.update(overrides)
    return spec


def _compiled_reference(tmp_path, spec: dict, num_frames: int) -> np.ndarray:
    """Compiled-brick condition on CPU torch — the parity oracle."""
    ctx = _make_ctx(tmp_path)
    ctx.variable_resolver.resolved["vae_encoder.output_0"] = (
        torch.from_numpy(_latent_np()))
    cond = compiled_i2v.build_condition(ctx, dict(spec), num_frames)
    assert cond is not None
    return cond.numpy()


# ---------------------------------------------------------------------------
# CPU: spec normalization parity (layout alias, defaults)
# ---------------------------------------------------------------------------

def test_spec_normalization_parity(tmp_path, monkeypatch):
    flag = {"layout": "state_video_mask"}
    monkeypatch.setattr(compiled_i2v, "get_component_flag",
                        lambda *a, **k: dict(flag))
    monkeypatch.setattr(triton_i2v, "get_component_flag",
                        lambda *a, **k: dict(flag))
    ctx = _make_ctx(tmp_path)
    spec_c = compiled_i2v.conditioning_spec(ctx, "transformer")
    spec_t = triton_i2v.conditioning_spec(ctx, "transformer")
    assert spec_c == spec_t, f"spec divergence: compiled={spec_c} triton={spec_t}"
    assert spec_t["style"] == "state_video_mask"
    assert spec_t["channel_dim"] == 1
    assert spec_t["cond_frame_indices"] == [0]
    assert spec_t["condition_component"] == "vae_encoder"
    assert (compiled_i2v.condition_channel_dim(ctx, "transformer")
            == triton_i2v.condition_channel_dim(ctx, "transformer") == 1)


# ---------------------------------------------------------------------------
# CPU: folded pixel mask bit-parity (host numpy vs compiled torch build)
# ---------------------------------------------------------------------------

def test_folded_mask_bit_parity_default_index(tmp_path):
    ref = _compiled_reference(tmp_path, _spec(), _NUM_FRAMES)
    assert ref.shape == (1, 4 + _RATIO, 22, 6, 8)
    # channels [latent(4) then folded mask(ratio=4)]
    ref_latent, ref_mask = ref[:, :4], ref[:, 4:]
    # latent channels pass through untouched (channels-first arrangement only)
    lat_cf = np.transpose(_latent_np(), (0, 2, 1, 3, 4))
    assert np.array_equal(ref_latent, lat_cf)
    mask = triton_i2v._state_video_mask_np(1, _NUM_FRAMES, _RATIO, 6, 8, [0])
    assert np.array_equal(mask, ref_mask), "triton host mask != compiled mask"
    # semantics: 1 everywhere except folded (channel 0, latent frame 0) which
    # covers pixel frame 0 (block split: channel k, frame t -> pixel k*T + t)
    assert mask[0, 0, 0].min() == mask[0, 0, 0].max() == 0.0
    assert int((mask == 0.0).sum()) == 6 * 8
    assert mask.sum() == mask.size - 6 * 8


def test_folded_mask_bit_parity_multi_index(tmp_path):
    spec = _spec(cond_frame_indices=[0, 3])
    ref = _compiled_reference(tmp_path, spec, _NUM_FRAMES)
    mask = triton_i2v._state_video_mask_np(1, _NUM_FRAMES, _RATIO, 6, 8, [0, 3])
    assert np.array_equal(mask, ref[:, 4:])
    # pixel frame 3 folds to (channel 0, latent frame 3): block split, k*22+t
    assert mask[0, 0, 3].max() == 0.0 and mask[0, 1, 3].min() == 1.0


def test_fold_constraint_raise_parity():
    # num_frames=85 odd: latent_T=(85-1)//4+1=22, 4*22=88 != 85 -> both raise
    try:
        triton_i2v._state_video_mask_np(1, 85, _RATIO, 6, 8, [0])
        raised = False
    except ValueError:
        raised = True
    assert raised, "triton fold-constraint guard did not raise"


# ---------------------------------------------------------------------------
# GPU: full triton build_condition vs compiled reference (bit-exact)
# ---------------------------------------------------------------------------

def _cuda_available() -> bool:
    return torch.cuda.is_available()


def test_triton_build_condition_bit_parity(tmp_path):
    if not _cuda_available():
        import pytest
        pytest.skip("NBXTensor is CUDA-backed and no GPU is available; "
                    "structural parity is covered by the CPU tests above "
                    "(spec + host mask + latent pass-through, bit-exact)")
    from neurobrix.kernels.nbx_tensor import NBXTensor

    ref = _compiled_reference(tmp_path, _spec(), _NUM_FRAMES)

    ctx = _make_ctx(tmp_path)
    ctx.variable_resolver.resolved["vae_encoder.output_0"] = (
        NBXTensor.from_numpy(_latent_np()))
    cond = triton_i2v.build_condition(ctx, _spec(), _NUM_FRAMES)
    assert cond is not None
    got = cond.numpy()
    assert got.shape == ref.shape == (1, 8, 22, 6, 8)
    assert np.array_equal(got, ref), (
        f"triton condition != compiled condition: max|diff|="
        f"{np.abs(got - ref).max()}")


def test_triton_latent_t_mismatch_raises(tmp_path):
    if not _cuda_available():
        import pytest
        pytest.skip("NBXTensor is CUDA-backed and no GPU is available")
    from neurobrix.kernels.nbx_tensor import NBXTensor

    ctx = _make_ctx(tmp_path)
    ctx.variable_resolver.resolved["vae_encoder.output_0"] = (
        NBXTensor.from_numpy(_latent_np()))  # latent_T=22
    # num_frames=44 -> folded latent_T=11 != encoded 22: compiled raises via
    # torch.cat dim validation; the mirror enforces the contract explicitly.
    try:
        triton_i2v.build_condition(ctx, _spec(), 44)
        raised = False
    except ValueError:
        raised = True
    assert raised, "triton latent_T-agreement guard did not raise"


if __name__ == "__main__":  # script-mode under a pytest-less GPU runtime venv
    import sys
    import tempfile
    from pathlib import Path

    failures = 0
    for name, fn in sorted(globals().items()):
        if not (name.startswith("test_") and callable(fn)):
            continue
        with tempfile.TemporaryDirectory() as td:
            kwargs = {}
            code = fn.__code__
            if "tmp_path" in code.co_varnames[:code.co_argcount]:
                kwargs["tmp_path"] = Path(td)
            if "monkeypatch" in code.co_varnames[:code.co_argcount]:
                class _MP:
                    def setattr(self, obj, attr, val):
                        setattr(obj, attr, val)
                kwargs["monkeypatch"] = _MP()
            try:
                fn(**kwargs)
                print(f"PASS {name}")
            except BaseException as exc:  # noqa: BLE001 — report and continue
                if type(exc).__name__ == "Skipped":
                    print(f"SKIP {name}: {exc}")
                else:
                    failures += 1
                    print(f"FAIL {name}: {exc}")
    sys.exit(1 if failures else 0)
