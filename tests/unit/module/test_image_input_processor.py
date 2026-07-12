"""Value-equivalence tests — ImageInputProcessor vs the former run.py
inline --input-image block (P-IMAGE-INPUT-FLOW migration, OMNI Stage 1a).

The reference implementations below are verbatim ports of the inline CLI
code this processor replaces; the processor must produce bit-identical
float32 values for both contracts (i2v_vae_condition ± temporal pad, and
clip_centercrop). The full-zoo video battery is the land gate (maintainer
amendment 2026-07-11); these tests are the cheap first line.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from neurobrix.core.module.vision.input_processor import ImageInputProcessor


@pytest.fixture(scope="module")
def synthetic_png(tmp_path_factory):
    from PIL import Image
    rng = np.random.default_rng(23)
    arr = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
    p = tmp_path_factory.mktemp("img") / "synthetic.png"
    Image.fromarray(arr).save(p)
    return str(p)


def _reference_i2v(image_path, height, width, pad_nf):
    """Verbatim port of the former inline block (global.image)."""
    from PIL import Image as _PILImage
    _img = _PILImage.open(image_path).convert("RGB")
    _h = int(height) if height else _img.height
    _w = int(width) if width else _img.width
    if (_img.width, _img.height) != (_w, _h):
        _img = _img.resize((_w, _h), _PILImage.LANCZOS)
    _arr = np.asarray(_img, dtype="float32") / 127.5 - 1.0
    out = (torch.from_numpy(_arr).permute(2, 0, 1).unsqueeze(0).unsqueeze(2))
    if pad_nf > 1 and out.shape[2] == 1:
        _zeros = torch.zeros(out.shape[0], out.shape[1], pad_nf - 1,
                             out.shape[3], out.shape[4], dtype=out.dtype)
        out = torch.cat([out, _zeros], dim=2)
    return out


def _reference_clip(image_path, pc):
    """Verbatim port of the former inline block (global.pixel_values)."""
    from PIL import Image as _PILImage

    def _dim(d, k, default):
        v = d.get(k)
        if isinstance(v, dict):
            return int(v.get("height", v.get("shortest_edge", default)))
        return int(v) if v is not None else default

    _rs = _dim(pc, "size", 224)
    _cs = _dim(pc, "crop_size", _rs)
    _mean = np.asarray(pc.get("image_mean",
                              [0.48145466, 0.4578275, 0.40821073]), dtype="float32")
    _std = np.asarray(pc.get("image_std",
                             [0.26862954, 0.26130258, 0.27577711]), dtype="float32")
    _ci = _PILImage.open(image_path).convert("RGB")
    if pc.get("do_resize", True):
        _scale = _rs / min(_ci.width, _ci.height)
        _ci = _ci.resize((max(1, round(_ci.width * _scale)),
                          max(1, round(_ci.height * _scale))), _PILImage.BICUBIC)
    _l = (_ci.width - _cs) // 2
    _t = (_ci.height - _cs) // 2
    _ci = _ci.crop((_l, _t, _l + _cs, _t + _cs))
    _ca = np.asarray(_ci, dtype="float32") / 255.0
    if pc.get("do_normalize", True):
        _ca = (_ca - _mean) / _std
    return torch.from_numpy(_ca).permute(2, 0, 1).unsqueeze(0).contiguous()


@pytest.mark.parametrize("h,w", [(None, None), (256, 384), (480, 640)])
def test_i2v_single_frame_bit_identical(synthetic_png, h, w):
    got = ImageInputProcessor.process(
        "i2v_vae_condition", synthetic_png, height=h, width=w)
    ref = _reference_i2v(synthetic_png, h, w, 0)
    assert got.shape == ref.shape and got.dtype == ref.dtype == torch.float32
    assert torch.equal(got, ref.contiguous())


def test_i2v_temporal_pad_bit_identical(synthetic_png):
    got = ImageInputProcessor.process(
        "i2v_vae_condition", synthetic_png, height=128, width=128,
        pad_to_num_frames=13)
    ref = _reference_i2v(synthetic_png, 128, 128, 13)
    assert got.shape == ref.shape == (1, 3, 13, 128, 128)
    assert torch.equal(got, ref.contiguous())
    assert torch.count_nonzero(got[:, :, 1:]) == 0  # padded frames zero


@pytest.mark.parametrize("pc", [
    {},  # all defaults
    {"size": 336, "crop_size": 336},
    {"size": {"shortest_edge": 256}, "crop_size": {"height": 224, "width": 224},
     "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5]},
    {"do_resize": False, "crop_size": 128},
    {"do_normalize": False},
])
def test_clip_centercrop_bit_identical(synthetic_png, pc):
    got = ImageInputProcessor.process(
        "clip_centercrop", synthetic_png, preprocessor_config=dict(pc))
    ref = _reference_clip(synthetic_png, dict(pc))
    assert got.shape == ref.shape and got.dtype == torch.float32
    assert torch.equal(got, ref)


def test_unknown_type_zero_fallback(synthetic_png):
    with pytest.raises(RuntimeError, match="ZERO FALLBACK"):
        ImageInputProcessor.process("native_patch_grid", synthetic_png)


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        ImageInputProcessor.process("i2v_vae_condition", "/nonexistent.png")


def test_clip_requires_config(synthetic_png):
    with pytest.raises(RuntimeError, match="preprocessor_config"):
        ImageInputProcessor.process("clip_centercrop", synthetic_png)
