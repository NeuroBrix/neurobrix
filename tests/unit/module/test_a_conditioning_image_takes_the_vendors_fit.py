"""An I2V conditioning image takes the fit its container declares (`condition_image_fit` on the vae_encoder): Allegro-
TI2V's own preprocessing (rhymes-ai Allegro single_inference_ti2v.py L20-26: ToTensorVideo, CenterCropResizeVideo,
2x - 1; data_process.py L47-68: the window at the request's aspect, centred, then `F.interpolate(..., mode="bilinear",
align_corners=True, antialias=True)`) cuts the image to the request's aspect and resamples it; the engine stretched it
(the 2.3c finding of 2026-11-04: a square apple stretched to 16:9 as frame 0 of a 720x1280 video).

The oracle is torch's own `F.interpolate` on the vendor's crop (the vendor oracle environment; skipped by name
elsewhere). What this test would do if the code were wrong: the image stretched instead of cropped, the crop off
centre or rounded otherwise, the resample without its antialias at a downscale or with the other corner convention,
the [-1, 1] normalisation moved — red at the comparison; a declaration the engine does not reproduce accepted
silently — red at the refusal."""
from __future__ import annotations

import numpy as np
import pytest

from neurobrix.core.module.vision import image_dsp


def _png(tmp_path, h, w, seed):
    from PIL import Image
    a = np.random.default_rng(seed).integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    p = tmp_path / f"img_{h}x{w}.png"
    Image.fromarray(a).save(p)
    return p, a


FIT = {"crop": "center", "resample": "linear", "antialias": True, "align_corners": True}


@pytest.mark.parametrize("src,dst", [((448, 448), (720, 1280)), ((300, 500), (90, 160)), ((720, 1280), (720, 1280)),
                                     ((333, 211), (64, 48))])
def test_the_fit_is_the_vendors(tmp_path, src, dst):
    torch = pytest.importorskip("torch")
    F = torch.nn.functional
    path, a = _png(tmp_path, *src, seed=src[0])
    got = image_dsp.i2v_vae_condition_np(str(path), height=dst[0], width=dst[1], fit=FIT)
    # the vendor: [1, C, H, W] in [0, 1], the centred window at the request's aspect, the resample, then 2x - 1
    x = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).double() / 255.0
    h, w = src
    tr = dst[0] / dst[1]
    nh, nw = (int(w * tr), w) if h / w > tr else (h, int(h / tr))
    i, j = int(round((h - nh) / 2.0)), int(round((w - nw) / 2.0))
    ref = F.interpolate(x[..., i:i + nh, j:j + nw], size=dst, mode="bilinear", align_corners=True, antialias=True)
    ref = (2.0 * ref - 1.0).numpy()[:, :, None]                        # [1, 3, 1, H, W]
    assert got.shape == ref.shape and got.dtype == np.float32
    assert float(np.abs(got - ref).max()) < 2e-6


def test_without_a_declaration_the_image_is_stretched(tmp_path):
    from PIL import Image
    path, a = _png(tmp_path, 40, 40, seed=3)
    got = image_dsp.i2v_vae_condition_np(str(path), height=20, width=60)
    want = np.asarray(Image.fromarray(a).resize((60, 20), Image.LANCZOS), dtype=np.float32) / 127.5 - 1.0
    np.testing.assert_array_equal(got[0, :, 0], want.transpose(2, 0, 1))


@pytest.mark.parametrize("bad", [{**FIT, "crop": "top"}, {**FIT, "resample": "bicubic"}, {**FIT, "antialias": False},
                                 {k: v for k, v in FIT.items() if k != "align_corners"}, {**FIT, "scale": 2}])
def test_a_fit_the_engine_does_not_reproduce_is_refused(tmp_path, bad):
    path, _ = _png(tmp_path, 40, 40, seed=4)
    with pytest.raises(RuntimeError, match="ZERO FALLBACK: condition_image_fit"):
        image_dsp.i2v_vae_condition_np(str(path), height=20, width=60, fit=bad)


def test_the_container_flag_is_carried():
    from neurobrix.nbx import component_flags
    assert "condition_image_fit" in component_flags.RUNTIME_COMPONENT_FLAGS
