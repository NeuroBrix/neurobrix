"""The temporal latent extent is derived even when the spatial one cannot be.

`_inject_dynamic_latent_dimensions` derives two independent things: the spatial
latent size (from height, width and the VAE scale factor) and the temporal one,
`latent_frames = (num_frames - 1) // temporal_compression_ratio + 1`, which needs
neither height, nor width, nor a VAE scale factor.

They shared one early-return path, so the temporal derivation was hostage to the
spatial one's preconditions. `Open-Sora-v2` declares `num_frames: 51` and
`temporal_compression_ratio: 4` — everything the temporal derivation needs — but
no height, no width and no `vae_scale_factor` (it carries
`spatial_compression_ratio` instead). The method returned at the height/width
check, `latent_frames` was never computed, and the request died in the Triton
denoise loop resolving `global.latents`:

    RuntimeError: Key 'latent_frames' not found in runtime/defaults.json

The container held the answer — 13 — the whole time.

Run: PYTHONPATH=src python -m pytest tests/unit/runtime/test_latent_frames_is_not_hostage_to_spatial.py
"""
from __future__ import annotations

import types


def _executor():
    """The method under test reads only `self.pkg.topology`, so a stand-in
    carrying that is enough — building a real executor needs a container."""
    from neurobrix.core.runtime.executor import RuntimeExecutor

    ex = object.__new__(RuntimeExecutor)
    ex.pkg = types.SimpleNamespace(
        topology={"flow": {"type": "iterative_process"}}, manifest={})
    return ex


# Open-Sora-v2's runtime/defaults.json keys, verbatim (2026-09-10).
OPEN_SORA = {
    "batch_size": 1, "dtype": "bfloat16", "fps": 24, "guidance_scale": 7.5,
    "num_frames": 51, "num_inference_steps": 50, "prompt": "",
    "spatial_compression_ratio": 8, "temporal_compression_ratio": 4,
}


def test_latent_frames_is_derived_without_height_or_width():
    ex = _executor()
    out = ex._inject_dynamic_latent_dimensions(dict(OPEN_SORA), comp_configs={})
    assert "latent_frames" in out, (
        "no height/width, so the spatial derivation cannot run — but the "
        "temporal one needs neither")
    assert out["latent_frames"] == (51 - 1) // 4 + 1 == 13


def test_latent_frames_is_still_derived_when_the_spatial_side_can_run():
    """The reorder must not lose the case that already worked."""
    ex = _executor()
    d = dict(OPEN_SORA, height=256, width=256)
    out = ex._inject_dynamic_latent_dimensions(d, comp_configs={})
    assert out["latent_frames"] == 13


def test_a_model_without_frames_gains_nothing():
    """An image model has no temporal extent and must not acquire one."""
    ex = _executor()
    out = ex._inject_dynamic_latent_dimensions(
        {"height": 512, "width": 512}, comp_configs={})
    assert "latent_frames" not in out


def test_an_autoregressive_model_is_still_skipped_entirely():
    from neurobrix.core.runtime.executor import RuntimeExecutor

    ex = object.__new__(RuntimeExecutor)
    ex.pkg = types.SimpleNamespace(
        topology={"flow": {"type": "autoregressive_generation"}}, manifest={})
    out = ex._inject_dynamic_latent_dimensions(dict(OPEN_SORA), comp_configs={})
    assert "latent_frames" not in out
