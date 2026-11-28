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
      VideoProcessor.preprocess semantics): fitted to the run
      height/width the caller resolved, as the container declares
      (`condition_image_fit`: crop none|center, then resample
      lanczos|bicubic in PIL or an antialiased linear in numpy),
      normalize to [-1, 1], single conditioning frame
      [1, 3, 1, H, W]; optionally zero-padded to [1, 3, T, H, W] for
      temporal-VAE denoisers (Wan-I2V class).
  clip_centercrop — the CLIP preprocessor contract, data-driven from the
      embedded preprocessor_config.json (resize shortest side, center
      crop, scale to [0, 1], normalize by mean/std) → [1, 3, cs, cs].
  native_patch_grid — dynamic-resolution flattened patch grid + grid_thw
      (GLM-4.1V / Qwen-VL lineage; landed with GLM-4.1V, its first
      consumer).
  minicpm_adaptive_slice — LLaVA-UHD adaptive-slice NaViT contract
      (MiniCPM-o lineage; landed with MiniCPM-o-4_5, its first
      consumer). v1 = single image, max_slice_nums 1 (no slicing):
      smart-resize to the scale_resolution budget, normalize by the
      topology's mean/std, pack to the patch-major [1, N, C*p*p]
      layout the built vpm graph consumes.
"""

from typing import Optional

import numpy as np

from neurobrix.core.components.runtime_deps import ensure_python_dep, resolve_ffmpeg_exe

# CLIP defaults — used ONLY when the embedded preprocessor_config.json
# omits the keys (mirrors the config's own documented defaults).
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _load_rgb(image_path: str):
    """Boundary file I/O (R34-allowed): decode the image file to RGB."""
    from PIL import Image
    return Image.open(image_path).convert("RGB")


def _center_crop_box(h: int, w: int, th: int, tw: int):
    """The window of an h x w image at the target's aspect th / tw, centred (rhymes-ai Allegro
    `data_process.center_crop_th_tw`: the long side cut, the offsets rounded as Python rounds)."""
    tr = th / tw
    nh, nw = (int(w * tr), w) if h / w > tr else (h, int(h / tr))
    i, j = int(round((h - nh) / 2.0)), int(round((w - nw) / 2.0))
    return i, j, nh, nw


def _linear_weights(n_in: int, n_out: int, align_corners: bool) -> np.ndarray:
    """[n_out, n_in] weights of a separable linear resample with antialiasing — a triangle filter
    widened by the downscale factor, each row normalised (the antialiased bilinear of
    `torch.nn.functional.interpolate(..., antialias=True)`; `align_corners` scales by
    (n_in - 1) / (n_out - 1))."""
    scale = (n_in - 1) / (n_out - 1) if align_corners and n_out > 1 else n_in / n_out
    support, inv = (scale, 1.0 / scale) if scale >= 1.0 else (1.0, 1.0)
    out = np.zeros((n_out, n_in), dtype=np.float64)
    for i in range(n_out):
        c = scale * (i + 0.5)
        lo, hi = max(int(c - support + 0.5), 0), min(int(c + support + 0.5), n_in)
        wgt = np.maximum(0.0, 1.0 - np.abs((np.arange(lo, hi) - c + 0.5) * inv))
        out[i, lo:hi] = wgt / wgt.sum()
    return out


#: what a `condition_image_fit` declaration may say (the vae_encoder's container flag): the crop and the resample
#: always, the two linear options with the linear resample only
_FIT_KEYS = {"crop": ("none", "center"), "resample": ("lanczos", "bicubic", "linear"), "antialias": (True,),
             "align_corners": (True, False)}
_LINEAR_OPTIONS = ("antialias", "align_corners")
#: the resamples PIL performs (the vendor preprocessors' own kernels, VaeImageProcessor's PIL resize)
_PIL_FILTERS = {"lanczos": "LANCZOS", "bicubic": "BICUBIC"}


def _check_fit(fit) -> None:
    """Refuse, by name, a `condition_image_fit` this DSP does not reproduce: an unknown key or value, a missing
    crop/resample (or linear option), a linear option on a PIL resample, a resample before a centre crop."""
    if not isinstance(fit, dict):
        raise RuntimeError("ZERO FALLBACK: condition_image_fit is required: the conditioning image's crop and "
                           "resample are its container's (vae_encoder), never this DSP's choice")
    for k, v in fit.items():
        if k != "source" and (k not in _FIT_KEYS or v not in _FIT_KEYS[k]):
            raise RuntimeError(f"ZERO FALLBACK: condition_image_fit.{k} = {v!r} is not a fit this engine "
                               f"reproduces ({_FIT_KEYS})")
    linear = fit.get("resample") == "linear"
    missing = sorted(k for k in ("crop", "resample") + (_LINEAR_OPTIONS if linear else ()) if k not in fit)
    if missing:
        raise RuntimeError(f"ZERO FALLBACK: condition_image_fit declares no {missing}")
    stray = [k for k in _LINEAR_OPTIONS if k in fit and not linear]
    if stray:
        raise RuntimeError(f"ZERO FALLBACK: condition_image_fit.{stray} are options of the linear resample, not of "
                           f"{fit['resample']!r} (a PIL filter)")
    if fit["crop"] != "none" and [k for k in fit if k in ("crop", "resample")][0] == "resample":
        raise RuntimeError("ZERO FALLBACK: condition_image_fit resamples before its centre crop (the key order is "
                           "the order of operations): no vendor this engine reproduces does; it crops, then resamples")


def _fit_condition(img, h: int, w: int, fit: dict) -> np.ndarray:
    """The conditioning image fitted as the vendor's own preprocessing declares (`fit`, the container's
    `condition_image_fit`), in its order: a centre crop to the target's aspect first when declared (Allegro-TI2V),
    then the declared resample to h x w — PIL's LANCZOS / BICUBIC (VaeImageProcessor's resize, CogVideoX / Wan), or
    the antialiased linear in numpy (Allegro's `F.interpolate`). HWC float32 in [-1, 1]."""
    _check_fit(fit)
    if fit["crop"] == "center":
        i, j, nh, nw = _center_crop_box(img.height, img.width, h, w)
        img = img.crop((j, i, j + nw, i + nh))
    if fit["resample"] == "linear":
        a = np.asarray(img, dtype=np.float64)
        ac = bool(fit["align_corners"])
        a = np.einsum("oh,hwc->owc", _linear_weights(a.shape[0], h, ac),
                      np.einsum("pw,hwc->hpc", _linear_weights(a.shape[1], w, ac), a))
        return (a / 127.5 - 1.0).astype(np.float32)
    from PIL import Image
    if (img.width, img.height) != (w, h):
        img = img.resize((w, h), getattr(Image, _PIL_FILTERS[fit["resample"]]))
    return np.asarray(img, dtype=np.float32) / 127.5 - 1.0


def i2v_vae_condition_np(
    image_path: str,
    height: Optional[int] = None,
    width: Optional[int] = None,
    pad_to_num_frames: int = 0,
    fit: Optional[dict] = None,
) -> np.ndarray:
    """I2V VAE-conditioning clip. [1,3,1,H,W] float32 in [-1,1]; when
    `pad_to_num_frames` > 1, zero-padded on the temporal axis to
    [1,3,T,H,W] (frame0 = image, rest = zeros — Wan-I2V temporal-VAE
    contract, data-driven via the caller's registry flag). The image is
    fitted to the caller's height x width (the run's size, which the
    caller resolved: never the image's own) as the container declares
    (`fit`, the vae_encoder's `condition_image_fit`: crop none|center,
    resample lanczos|bicubic|linear, in the vendor's order)."""
    if not height or not width:
        raise RuntimeError(f"ZERO FALLBACK: i2v_vae_condition needs the run's height and width (given {height!r} x "
                           f"{width!r}): the caller resolves them from the container's size policy; the image's own "
                           f"size is not the run's")
    h, w = int(height), int(width)
    arr = _fit_condition(_load_rgb(image_path), h, w, fit)                  # HWC
    clip = np.ascontiguousarray(
        arr.transpose(2, 0, 1))[np.newaxis, :, np.newaxis]       # 1,3,1,H,W
    t = int(pad_to_num_frames or 0)
    if t > 1:
        out = np.zeros((1, 3, t, h, w), dtype=np.float32)
        out[:, :, 0] = clip[:, :, 0]
        return out
    return np.ascontiguousarray(clip)


def _smart_resize(num_frames: int, height: int, width: int,
                  temporal_factor: int, factor: int,
                  min_pixels: int, max_pixels: int):
    """Verbatim port of the vendor Glm4vImageProcessor.smart_resize
    (dynamic-resolution pixel-budget fit, all dims snapped to `factor`)."""
    import math
    if num_frames < temporal_factor:
        raise RuntimeError(
            f"t:{num_frames} must be larger than temporal_factor:{temporal_factor}")
    if height < factor or width < factor:
        raise RuntimeError(
            f"height:{height} or width:{width} must be larger than factor:{factor}")
    if max(height, width) / min(height, width) > 200:
        raise RuntimeError(
            f"absolute aspect ratio must be smaller than 200, "
            f"got {max(height, width) / min(height, width)}")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    t_bar = round(num_frames / temporal_factor) * temporal_factor
    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((num_frames * height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (num_frames * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _require_values(cfg, keys, what: str) -> None:
    """Refuse, by name, every preprocessing value the container does not carry (no family default)."""
    missing = [k for k in keys if not isinstance(cfg, dict) or cfg.get(k) is None]
    if missing:
        raise RuntimeError(
            f"ZERO FALLBACK: {what} preprocessing is missing {missing} in the vision "
            "component's preprocessing block (topology.flow.vlm.preprocessing). The "
            "container must carry these values, taken from the model's "
            "preprocessor_config.json: rebuild the container so the block holds them.")


def _pixel_budget(cfg: dict):
    """The pixel budget the container declares: (flat_form, min_pixels, max_pixels).

    The two vendor processor-config forms carry two different budget arithmetics:
    flat ``min_pixels``/``max_pixels`` -> per-frame 2-D budget h_bar*w_bar;
    ``size.{shortest_edge,longest_edge}`` -> budget with the temporal factor t_bar*h_bar*w_bar.
    The flat form wins when the block carries it; a form in part, or none, is refused by name.
    """
    if cfg.get("min_pixels") is not None or cfg.get("max_pixels") is not None:
        _require_values(cfg, ("min_pixels", "max_pixels"), "native_patch_grid pixel budget")
        return True, int(cfg["min_pixels"]), int(cfg["max_pixels"])
    _require_values(cfg.get("size"), ("shortest_edge", "longest_edge"),
                    "native_patch_grid pixel budget (min_pixels and max_pixels, or size)")
    return False, int(cfg["size"]["shortest_edge"]), int(cfg["size"]["longest_edge"])


def native_patch_grid_np(image_path: str, preprocessing_cfg: dict):
    """Dynamic-resolution flattened patch grid (Qwen2-VL / GLM-4.1V lineage),
    vendor Glm4vImageProcessor contract: smart-resize to the pixel budget
    (dims snapped to patch_size*merge_size), BICUBIC, rescale 1/255,
    normalize mean/std, temporal-duplicate the single image to
    temporal_patch_size frames, 9-dim patchify.

    Returns (flatten_patches float32 [t*h*w, C*Tp*P*P],
             grid_thw int64 [1, 3])."""
    from PIL import Image
    cfg = preprocessing_cfg
    _require_values(cfg, ("patch_size", "temporal_patch_size", "merge_size",
                          "image_mean", "image_std"), "native_patch_grid")
    p, tp, merge = int(cfg["patch_size"]), int(cfg["temporal_patch_size"]), int(cfg["merge_size"])
    mean = np.asarray(cfg["image_mean"], dtype=np.float32)
    std = np.asarray(cfg["image_std"], dtype=np.float32)
    img = _load_rgb(image_path)
    flat_budget, min_px, max_px = _pixel_budget(cfg)
    if flat_budget:
        h_bar, w_bar = _smart_resize_2d(img.height, img.width,
                                        factor=p * merge,
                                        min_pixels=min_px, max_pixels=max_px)
    else:
        h_bar, w_bar = _smart_resize(tp, img.height, img.width,
                                     temporal_factor=tp, factor=p * merge,
                                     min_pixels=min_px, max_pixels=max_px)
    if (img.width, img.height) != (w_bar, h_bar):
        img = img.resize((w_bar, h_bar), Image.BICUBIC)
    # Vendor rescale semantics, bit-exact: float64 multiply by the scale
    # factor, downcast to float32; then float32 normalize in HWC.
    a = (np.asarray(img, dtype=np.uint8).astype(np.float64)
         * (1.0 / 255.0)).astype(np.float32)                  # HWC
    a = (a - mean) / std
    a = a.transpose(2, 0, 1)                                  # CHW
    frames = np.stack([a] * tp)                               # Tp,C,H,W (vendor
    # pads a lone image by repeating it to temporal_patch_size frames)
    grid_t = frames.shape[0] // tp
    grid_h, grid_w = h_bar // p, w_bar // p
    patches = frames.reshape(grid_t, tp, 3, grid_h // merge, merge, p,
                             grid_w // merge, merge, p)
    patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flat = patches.reshape(grid_t * grid_h * grid_w, 3 * tp * p * p)
    return (np.ascontiguousarray(flat, dtype=np.float32),
            np.array([[grid_t, grid_h, grid_w]], dtype=np.int64))


def _smart_resize_2d(height: int, width: int, factor: int,
                     min_pixels: int, max_pixels: int):
    """Verbatim port of the vendor Qwen2-VL image/video smart_resize:
    per-frame 2-D pixel-budget fit (h*w vs the budget — NO temporal term,
    unlike the GLM variant above), dims snapped to `factor`."""
    import math
    if height < factor or width < factor:
        raise RuntimeError(
            f"height:{height} or width:{width} must be larger than factor:{factor}")
    if max(height, width) / min(height, width) > 200:
        raise RuntimeError(
            f"absolute aspect ratio must be smaller than 200, "
            f"got {max(height, width) / min(height, width)}")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _open_video_reader(imageio, video_path: str):
    """Open an imageio video reader, resolving ffmpeg runtime deps if missing."""
    import os
    try:
        return imageio.get_reader(str(video_path))
    except Exception as exc:
        msg = str(exc).lower()
        if not any(k in msg for k in ("ffmpeg", "imageio", "backend", "plugin")):
            raise
        os.environ.setdefault("IMAGEIO_FFMPEG_EXE", resolve_ffmpeg_exe(allow_prompt=True))
        return _retry_video_reader_with_ffmpeg_dep(imageio, video_path)


def _retry_video_reader_with_ffmpeg_dep(imageio, video_path: str):
    """Retry opening the video reader, ensuring imageio_ffmpeg if imageio requires it."""
    try:
        return imageio.get_reader(str(video_path))
    except Exception as exc:
        msg = str(exc).lower()
        if "imageio_ffmpeg" not in msg and "imageio-ffmpeg" not in msg and "ffmpeg" not in msg:
            raise
        ensure_python_dep("imageio_ffmpeg", "imageio-ffmpeg>=0.4.0", "reading input video frames")
        return imageio.get_reader(str(video_path))


def native_patch_grid_video_np(video_path: str, preprocessing_cfg: dict,
                               fps: Optional[float] = None):
    """Video variant of the flattened patch grid — vendor
    Qwen2VLVideoProcessor contract (the AutoVideoProcessor resolution for
    the Qwen3-Omni lineage): uniform fps-based frame sampling
    (indices = arange(0, total, total/n) in float32, exactly the vendor
    arithmetic), per-frame 2-D smart-resize to the pixel budget, BICUBIC,
    rescale 1/255, normalize mean/std, temporal grouping by
    temporal_patch_size, 9-dim patchify.

    Defaults below mirror the vendor processor class's own documented
    defaults (same policy as native_patch_grid_np) and apply ONLY when
    the embedded preprocessing block omits the keys: fps 1.0,
    min_frames 4, max_frames 768.

    Returns (flatten_patches float32 [t*h*w, C*Tp*P*P],
             grid_thw int64 [1, 3],
             second_per_grid float — temporal_patch_size / fps, the
             M-RoPE temporal scale carried per video)."""
    import math
    from PIL import Image
    cfg = preprocessing_cfg
    p = int(cfg.get("patch_size", 14))
    tp = int(cfg.get("temporal_patch_size", 2))
    merge = int(cfg.get("merge_size", 2))
    # The VIDEO pixel budget is NOT the image budget: the vendor
    # processors override it per lineage, and both known lineages
    # resolve to the same formula — 128·factor² / 768·factor²
    # (omni Qwen3OmniMoeProcessor videos_kwargs _defaults 128·32²/
    # 768·32²; Qwen2VLVideoProcessor class defaults 128·28²/768·28²).
    # Explicit video_min_pixels / video_max_pixels keys win.
    factor = p * merge
    min_px = int(cfg.get("video_min_pixels", 128 * factor * factor))
    max_px = int(cfg.get("video_max_pixels", 768 * factor * factor))
    mean = np.asarray(cfg.get("image_mean", list(_CLIP_MEAN)), dtype=np.float32)
    std = np.asarray(cfg.get("image_std", list(_CLIP_STD)), dtype=np.float32)
    tgt_fps = float(fps if fps is not None else cfg.get("video_fps", 1.0))
    min_frames = int(cfg.get("video_min_frames", 4))
    max_frames = int(cfg.get("video_max_frames", 768))

    # Boundary file I/O (R34-allowed): decode the clip + container fps.
    imageio = ensure_python_dep("imageio", "imageio>=2.30.0", "reading input video frames")
    reader = _open_video_reader(imageio, video_path)
    meta = reader.get_meta_data()
    video_fps = float(meta.get("fps") or 0.0)
    if video_fps <= 0:
        raise RuntimeError(
            f"ZERO FALLBACK: no fps metadata in {video_path} — fps-based "
            "frame sampling needs the container frame rate.")
    frames_raw = [np.asarray(f) for f in reader]
    reader.close()
    total = len(frames_raw)
    if total < tp:
        raise RuntimeError(
            f"ZERO FALLBACK: video has {total} frames, fewer than "
            f"temporal_patch_size {tp}.")

    # Vendor sample_frames arithmetic (fps mode), verbatim:
    max_f = math.floor(min(max_frames, total) / tp) * tp
    n = total / video_fps * tgt_fps
    n = min(min(max(n, min_frames), max_f), total)
    n = math.floor(n / tp) * tp
    if n <= 0 or n > total:
        raise RuntimeError(
            f"ZERO FALLBACK: video can't be sampled — inferred "
            f"num_frames={n} from total={total} fps={video_fps} "
            f"target_fps={tgt_fps}.")
    # torch.arange(0, total, step).int() with a python-float step runs in
    # float32 — replicate that exact dtype so indices match the vendor.
    indices = np.arange(0, total, total / n, dtype=np.float32).astype(np.int32)

    # All frames of one container share a shape (the vendor groups by
    # shape; one video = one group) — resolve the budget fit once.
    first = Image.fromarray(frames_raw[int(indices[0])]).convert("RGB")
    h_bar, w_bar = _smart_resize_2d(
        first.height, first.width, factor=factor,
        min_pixels=min_px, max_pixels=max_px)
    proc = []
    for idx in indices:
        img = Image.fromarray(frames_raw[int(idx)]).convert("RGB")
        if (img.width, img.height) != (w_bar, h_bar):
            img = img.resize((w_bar, h_bar), Image.BICUBIC)
        a = (np.asarray(img, dtype=np.uint8).astype(np.float64)
             * (1.0 / 255.0)).astype(np.float32)               # HWC
        a = (a - mean) / std
        proc.append(a.transpose(2, 0, 1))                      # CHW
    frames = np.stack(proc)                                    # N,C,H,W

    grid_t = frames.shape[0] // tp
    grid_h, grid_w = h_bar // p, w_bar // p
    patches = frames.reshape(grid_t, tp, 3, grid_h // merge, merge, p,
                             grid_w // merge, merge, p)
    patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flat = patches.reshape(grid_t * grid_h * grid_w, 3 * tp * p * p)
    return (np.ascontiguousarray(flat, dtype=np.float32),
            np.array([[grid_t, grid_h, grid_w]], dtype=np.int64),
            float(tp) / tgt_fps)


def _clip_sizes(pc: dict):
    """(resize side, crop side) of the CLIP view `clip_centercrop_np` produces."""
    def _dim(d, k, default):
        v = d.get(k)
        if isinstance(v, dict):
            return int(v.get("height", v.get("shortest_edge", default)))
        return int(v) if v is not None else default

    rs = _dim(pc, "size", 224)
    return rs, _dim(pc, "crop_size", rs)


def clip_view_shape(preprocessor_config: dict) -> tuple:
    """The shape of the CLIP view `clip_centercrop_np` produces, [1, 3, cs, cs] — the plan binds the
    image encoder's input to it (`pytorch/prism/flow_bindings.FlowBindings.pixel_view`)."""
    cs = _clip_sizes(preprocessor_config)[1]
    return (1, 3, cs, cs)


def clip_centercrop_np(image_path: str, preprocessor_config: dict) -> np.ndarray:
    """CLIP-preprocessed view [1,3,cs,cs] float32, data-driven from the
    embedded modules/image_processor/preprocessor_config.json."""
    from PIL import Image
    pc = preprocessor_config
    rs, cs = _clip_sizes(pc)
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


def minicpm_adaptive_slice_np(image_path: str, preprocessing_cfg: dict) -> dict:
    """LLaVA-UHD adaptive-slice NaViT preprocessing (MiniCPM-o lineage),
    v1 single-image single-slice contract (max_slice_nums == 1).

    Vendor source: MiniCPMVImageProcessor (processing_minicpmo.py). With
    max_slice_nums == 1, get_sliced_grid returns None (multiple =
    min(ceil(w*h/sr**2), 1) <= 1) and slice_image takes the no-slice
    branch:

        best_size = self.find_best_resize(original_size, scale_resolution,
                                          patch_size, allow_upscale=True)
        source_image = image.resize(best_size, resample=BICUBIC)

    find_best_resize (vendor, verbatim arithmetic below):

        if (width * height > scale_resolution * scale_resolution) or allow_upscale:
            r = width / height
            height = int(scale_resolution / math.sqrt(r))
            width = int(height * r)
        best_width = ensure_divide(width, patch_size)    # max(round(w/p)*p, p)
        best_height = ensure_divide(height, patch_size)

    Then per slice: to_numpy .astype(float32)/255, normalize mean/std
    (HWC), channel-first, reshape_by_patch:

        patches = F.unfold(image, (p, p), stride=(p, p))     # [C*p*p, L]
        patches = patches.reshape(C, p, p, -1)
        patches = patches.permute(0, 1, 3, 2).reshape(C, p, -1)   # [C, p, L*p]

    and the model wrapper (get_vision_embedding) packs
    `i.flatten(end_dim=1).permute(1, 0)` -> [L*p, C*p]. The built vpm
    graph consumes the PATCH-MAJOR form [1, N, C*p*p] whose definitional
    inverse (the build wrapper's repack) is
    reshape(B, N, C, p, p).permute(0, 2, 3, 1, 4).reshape(B, C, p, N*p)
    — i.e. row n = patch (row-major over the gh x gw grid), inner order
    (c, i, j). The equivalent direct numpy packing from the CHW array:
    reshape(C, gh, p, gw, p) -> transpose(1, 3, 0, 2, 4) -> [gh*gw, C*p*p].

    Returns the vpm graph's input dict:
        all_pixel_values     float32 [1, N, C*p*p]
        patch_attention_mask bool    [1, 1, N] (all-True, B=1 contract)
        tgt_sizes            int32   [[gh, gw]] (values feed in-graph
                                     position arithmetic — must be exact)
    """
    import math
    from PIL import Image
    cfg = preprocessing_cfg
    missing = [k for k in ("patch_size", "scale_resolution", "max_slice_nums",
                           "image_mean", "image_std") if cfg.get(k) is None]
    if missing:
        raise RuntimeError(
            "ZERO FALLBACK: minicpm_adaptive_slice preprocessing block is "
            f"missing {missing} — the build's topology.flow.vlm.preprocessing "
            "must carry them (registry-emitted).")
    p = int(cfg["patch_size"])
    scale_resolution = int(cfg["scale_resolution"])
    max_slice_nums = int(cfg["max_slice_nums"])
    if max_slice_nums != 1:
        raise RuntimeError(
            "ZERO FALLBACK: minicpm_adaptive_slice v1 implements the "
            f"single-slice contract only (max_slice_nums == 1, got "
            f"{max_slice_nums}); multi-slice lands with a multi-slice vpm "
            "container (B > 1 slice batch).")
    mean = np.asarray(cfg["image_mean"], dtype=np.float32)
    std = np.asarray(cfg["image_std"], dtype=np.float32)

    img = _load_rgb(image_path)
    width, height = img.size
    # find_best_resize(original_size, scale_resolution, patch_size,
    # allow_upscale=True) — vendor arithmetic verbatim (allow_upscale
    # short-circuits the budget check to the resize branch).
    r = width / height
    height = int(scale_resolution / math.sqrt(r))
    width = int(height * r)
    best_width = max(round(width / p) * p, p)     # ensure_divide
    best_height = max(round(height / p) * p, p)   # ensure_divide
    img = img.resize((best_width, best_height), Image.Resampling.BICUBIC)

    a = np.asarray(img).astype(np.float32) / 255.0            # HWC, vendor /255
    a = (a - mean) / std                                      # vendor normalize
    a = a.transpose(2, 0, 1)                                  # CHW
    gh, gw = best_height // p, best_width // p
    # Patch-major packing (see docstring): [C,H,W] -> [gh*gw, C*p*p].
    packed = (a.reshape(a.shape[0], gh, p, gw, p)
              .transpose(1, 3, 0, 2, 4)
              .reshape(gh * gw, a.shape[0] * p * p))
    return {
        "all_pixel_values":
            np.ascontiguousarray(packed, dtype=np.float32)[np.newaxis],
        "patch_attention_mask": np.ones((1, 1, gh * gw), dtype=bool),
        "tgt_sizes": np.array([[gh, gw]], dtype=np.int32),
    }


#: The clips the VACE control encoder encodes in one pass, stacked on its batch axis in this order
#: (vendor WanVACEPipeline.prepare_video_latents): the part kept, the part to generate. The plan
#: sizes the encoder at this batch (pytorch/prism/flow_bindings).
VACE_CONTROL_CLIPS = ("inactive", "reactive")


def vace_control_pair_np(clip: Optional[np.ndarray], keep_frames: int, num_frames: int,
                         height: int, width: int):
    """The VACE control encoder's input and its pixel mask (vendor WanVACEPipeline).

    The vendor encodes TWO clips from a control video V and a mask M (1 = generate, 0 = keep):
    ``inactive = V * (1 - M)`` and ``reactive = V * M`` (prepare_video_latents), and folds M
    into the mask channels (prepare_masks). With no control video V is zeros and M is ones
    (preprocess_conditions: the all-generate path); a still image is the clip's first
    ``keep_frames`` frames, the rest padding, and those frames are the kept ones.

    Returns ``(pair [2, 3, T, H, W], mask [1, 1, T, H, W])`` float32: the two clips stacked on
    the batch axis so ONE encoder pass makes both latents (batch 0 = inactive, 1 = reactive).
    """
    if clip is None:
        v = np.zeros((1, 3, num_frames, height, width), dtype=np.float32)
        keep_frames = 0
    else:
        v = np.asarray(clip, dtype=np.float32)
        if v.ndim != 5 or v.shape[0] != 1:
            raise ValueError(f"VACE control: the clip must be [1, 3, T, H, W], got {tuple(v.shape)}")
    t = v.shape[2]
    if not 0 <= keep_frames <= t:
        raise ValueError(f"VACE control: {keep_frames} kept frames in a {t}-frame clip")
    m = np.ones((1, 1) + tuple(v.shape[2:]), dtype=np.float32)
    m[:, :, :keep_frames] = 0.0
    clips = {"inactive": v * (1.0 - m), "reactive": v * m}
    pair = np.concatenate([clips[name] for name in VACE_CONTROL_CLIPS], axis=0)
    return np.ascontiguousarray(pair), m


def vace_fold_mask_np(mask: np.ndarray, latent_t: int, latent_h: int, latent_w: int,
                      mask_channels: int) -> np.ndarray:
    """Fold a pixel mask [1, 1, T, H, W] into the VACE mask channels [1, C, T', H', W'].

    Vendor WanVACEPipeline.prepare_masks: the spatial VAE factor (s x s pixels per latent cell)
    moves into the channel axis — view(T, H', s, W', s).permute(2, 4, 0, 1, 3) — and the time
    axis is resampled T -> T' with ``nearest-exact`` (source index floor((i + 0.5) * T / T')).
    The spatial factor is read from the shapes (H / H'), never assumed; its square must be the
    container's declared mask width.
    """
    m = np.asarray(mask, dtype=np.float32)
    _, _, t, h, w = m.shape
    sh, sw = h // latent_h, w // latent_w
    if sh * latent_h != h or sw * latent_w != w or sh != sw or sh * sw != mask_channels:
        raise ValueError(
            f"VACE mask fold: pixels {h}x{w} over latent {latent_h}x{latent_w} is not a square "
            f"factor whose square is the declared {mask_channels} mask channels")
    f = m[0, 0].reshape(t, latent_h, sh, latent_w, sw).transpose(2, 4, 0, 1, 3)
    f = f.reshape(sh * sw, t, latent_h, latent_w)
    src = np.minimum(np.floor((np.arange(latent_t) + 0.5) * (t / latent_t)).astype(np.int64), t - 1)
    return np.ascontiguousarray(f[:, src][None])
