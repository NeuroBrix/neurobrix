"""Vendor-free numpy image DSP core (R34) — shared by compiled AND triton.

Every function takes a file path plus data-driven parameters, and returns
a contiguous float32 numpy array in the model's expected layout. NO torch,
NO transformers image processors, NO torchvision — the engine boundary
(torch.Tensor for compiled, NBXTensor for triton) is applied by the caller.

PIL usage note (R34): PIL decodes the file (boundary I/O) AND performs the
resize/crop resampling. Keeping the resampling in PIL is deliberate — the
vendor preprocessors and the former inline CLI block both use PIL
LANCZOS/BICUBIC, and the bit-identical vendor-parity contract (unit-tested
via torch.equal against verbatim reference ports) forces the same kernels;
a numpy reimplementation of the resamplers would break that parity. All
arithmetic beyond resampling (scale, normalize, transpose, pad) is numpy.

Preprocessing types implemented here:
  i2v_vae_condition — the video I2V conditioning contract (vendor
      VideoProcessor.preprocess semantics): resize to the run
      height/width, normalize to [-1, 1], single conditioning frame
      [1, 3, 1, H, W]; optionally zero-padded to [1, 3, T, H, W] for
      temporal-VAE denoisers (Wan-I2V class).
  clip_centercrop — the CLIP preprocessor contract, data-driven from the
      embedded preprocessor_config.json (resize shortest side, center
      crop, scale to [0, 1], normalize by mean/std) → [1, 3, cs, cs].

Planned types (land WITH their consumer model, never speculatively —
zero-fallback until then): native_patch_grid (Qwen3-VL / GLM-4.1V dynamic
resolution), anyres_slice (MiniCPM-o LLaVA-UHD slicing).
"""

from typing import Optional

import numpy as np

# CLIP defaults — used ONLY when the embedded preprocessor_config.json
# omits the keys (mirrors the config's own documented defaults).
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _load_rgb(image_path: str):
    """Boundary file I/O (R34-allowed): decode the image file to RGB."""
    from PIL import Image
    return Image.open(image_path).convert("RGB")


def i2v_vae_condition_np(
    image_path: str,
    height: Optional[int] = None,
    width: Optional[int] = None,
    pad_to_num_frames: int = 0,
) -> np.ndarray:
    """I2V VAE-conditioning clip. [1,3,1,H,W] float32 in [-1,1]; when
    `pad_to_num_frames` > 1, zero-padded on the temporal axis to
    [1,3,T,H,W] (frame0 = image, rest = zeros — Wan-I2V temporal-VAE
    contract, data-driven via the caller's registry flag)."""
    from PIL import Image
    img = _load_rgb(image_path)
    h = int(height) if height else img.height
    w = int(width) if width else img.width
    if (img.width, img.height) != (w, h):
        img = img.resize((w, h), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0        # HWC
    clip = np.ascontiguousarray(
        arr.transpose(2, 0, 1))[np.newaxis, :, np.newaxis]       # 1,3,1,H,W
    t = int(pad_to_num_frames or 0)
    if t > 1:
        out = np.zeros((1, 3, t, h, w), dtype=np.float32)
        out[:, :, 0] = clip[:, :, 0]
        return out
    return np.ascontiguousarray(clip)


def clip_centercrop_np(image_path: str, preprocessor_config: dict) -> np.ndarray:
    """CLIP-preprocessed view [1,3,cs,cs] float32, data-driven from the
    embedded modules/image_processor/preprocessor_config.json."""
    from PIL import Image
    pc = preprocessor_config

    def _dim(d, k, default):
        v = d.get(k)
        if isinstance(v, dict):
            return int(v.get("height", v.get("shortest_edge", default)))
        return int(v) if v is not None else default

    rs = _dim(pc, "size", 224)
    cs = _dim(pc, "crop_size", rs)
    mean = np.asarray(pc.get("image_mean", list(_CLIP_MEAN)), dtype=np.float32)
    std = np.asarray(pc.get("image_std", list(_CLIP_STD)), dtype=np.float32)

    img = _load_rgb(image_path)
    if pc.get("do_resize", True):
        scale = rs / min(img.width, img.height)
        img = img.resize(
            (max(1, round(img.width * scale)), max(1, round(img.height * scale))),
            Image.BICUBIC)
    left = (img.width - cs) // 2
    top = (img.height - cs) // 2
    img = img.crop((left, top, left + cs, top + cs))
    a = np.asarray(img, dtype=np.float32) / 255.0
    if pc.get("do_normalize", True):
        a = (a - mean) / std
    return np.ascontiguousarray(a.transpose(2, 0, 1))[np.newaxis]  # 1,3,cs,cs
