"""The VACE control signal is built as the vendor builds it — inactive, reactive and the folded mask.

2026-09-27: the engine's brick encoded one clip and used it for BOTH halves of the control
(``cat([enc, enc, ones])``). The vendor WanVACEPipeline encodes ``V * (1 - M)`` (inactive) and
``V * M`` (reactive) from a control clip V and a mask M, and folds M into 64 mask channels. With an
input image the engine's reactive half was the image's latent where the vendor's is the latent of
zeros. The references below are verbatim ports of diffusers 0.37.0 pipeline_wan_vace.py
(preprocess_conditions, prepare_video_latents, prepare_masks). On the old brick the pair and the
split do not exist and the control's reactive half equals its inactive half: these fail.
"""
import types

import numpy as np
import pytest
import torch

from neurobrix.core.module.vision.image_dsp import vace_control_pair_np, vace_fold_mask_np


def _vendor_fold(mask, vae_s=8, vae_t=4, patch=2):
    """diffusers prepare_masks, one video, no reference image."""
    mask_ = mask[0]
    _c, t, h, w = mask_.shape
    new_t = (t + vae_t - 1) // vae_t
    new_h = h // (vae_s * patch) * patch
    new_w = w // (vae_s * patch) * patch
    mask_ = mask_[0].view(t, new_h, vae_s, new_w, vae_s).permute(2, 4, 0, 1, 3).flatten(0, 1)
    return torch.nn.functional.interpolate(mask_.unsqueeze(0), size=(new_t, new_h, new_w),
                                           mode="nearest-exact")


@pytest.mark.parametrize("t,h,w,keep", [(33, 256, 448, 1), (81, 352, 832, 1), (33, 256, 448, 9),
                                        (33, 256, 448, 0), (1, 64, 64, 1)])
def test_the_folded_mask_is_the_vendors(t, h, w, keep):
    clip = np.random.default_rng(0).standard_normal((1, 3, t, h, w)).astype(np.float32)
    _pair, m = vace_control_pair_np(clip if keep else None, keep, t, h, w)
    ours = vace_fold_mask_np(m, (t + 3) // 4, h // 8, w // 8, 64)
    ref = _vendor_fold(torch.from_numpy(m)).numpy()
    assert ours.shape == ref.shape
    assert np.array_equal(ours, ref)


def test_the_pair_is_inactive_and_reactive():
    rng = np.random.default_rng(1)
    clip = np.zeros((1, 3, 33, 32, 32), np.float32)
    clip[:, :, 0] = rng.standard_normal((1, 3, 32, 32))            # the image, padded with zeros
    pair, m = vace_control_pair_np(clip, 1, 33, 32, 32)
    v, mt = torch.from_numpy(clip), torch.from_numpy(m)
    mt = torch.where(mt > 0.5, 1.0, 0.0)                            # vendor prepare_video_latents
    assert np.array_equal(pair[0:1], (v * (1 - mt)).numpy())         # inactive: the image kept
    assert np.array_equal(pair[1:2], (v * mt).numpy())               # reactive: zeros here
    assert not pair[1].any() and pair[0, :, 0].any()
    zeros_pair, ones_m = vace_control_pair_np(None, 0, 33, 32, 32)   # vendor: V = 0, M = 1
    assert zeros_pair.shape == (2, 3, 33, 32, 32) and not zeros_pair.any() and ones_m.all()


def test_the_brick_splits_the_pair_into_the_two_halves(monkeypatch):
    from neurobrix.core.runtime.resolution import vace_control_conditioning as B
    lat = torch.stack([torch.full((16, 9, 4, 4), 1.0), torch.full((16, 9, 4, 4), 2.0)])  # [2,16,9,4,4]
    _pair, m = vace_control_pair_np(np.zeros((1, 3, 33, 32, 32), np.float32), 1, 33, 32, 32)
    resolved = {"vae_encoder.output_0": lat, "global.vace_pixel_mask": torch.from_numpy(m)}
    ctx = types.SimpleNamespace(variable_resolver=types.SimpleNamespace(resolved=resolved, get=resolved.get))
    monkeypatch.setattr(B, "_vae_latent_stats", lambda _ctx: (None, None, 16))
    ctrl = B.build_control(ctx, {"condition_component": "vae_encoder", "mask_channels": 64})
    assert tuple(ctrl.shape) == (1, 96, 9, 4, 4)
    assert torch.equal(ctrl[:, :16], lat[0:1]) and torch.equal(ctrl[:, 16:32], lat[1:2])
    assert torch.equal(ctrl[:, 32:], torch.from_numpy(vace_fold_mask_np(m, 9, 4, 4, 64)))
