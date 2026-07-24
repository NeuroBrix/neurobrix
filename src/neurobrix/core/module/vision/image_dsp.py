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
  native_patch_grid — dynamic-resolution flattened patch grid + grid_thw
      (GLM-4.1V / Qwen-VL lineage; landed with GLM-4.1V, its first
      consumer).
  minicpm_adaptive_slice — LLaVA-UHD adaptive-slice NaViT contract
      (MiniCPM-o lineage; landed with MiniCPM-o-4_5, its first
      consumer). v1 = single image, max_slice_nums 1 (no slicing):
      smart-resize to the scale_resolution budget, normalize by the
      topology's mean/std, pack to the patch-major [1, N, C*p*p]
      layout the traced vpm graph consumes.
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
    # Defaults below are used ONLY when the embedded preprocessor_config.json
    # omits the keys — they mirror the vendor processor class's own
    # documented defaults (same policy as _CLIP_MEAN/_CLIP_STD).
    p = int(cfg.get("patch_size", 14))
    tp = int(cfg.get("temporal_patch_size", 2))
    merge = int(cfg.get("merge_size", 2))
    mean = np.asarray(cfg.get("image_mean", list(_CLIP_MEAN)), dtype=np.float32)
    std = np.asarray(cfg.get("image_std", list(_CLIP_STD)), dtype=np.float32)

    img = _load_rgb(image_path)
    # Pixel-budget form, read from the emitted block — the two vendor
    # processor-config shapes carry two DIFFERENT budget arithmetics and
    # the config shape is the only data-driven signal for which one:
    #   flat `min_pixels`/`max_pixels` (Qwen2-VL / Qwen2.5-VL / bailingmm
    #   processors)  -> per-frame 2-D budget h_bar*w_bar, no temporal term
    #   `size.{shortest_edge,longest_edge}` (Glm4vImageProcessor)
    #   -> budget includes the temporal factor t_bar*h_bar*w_bar
    # The video variant below already uses the 2-D form unconditionally.
    # Clean end state: an explicit emitted flag instead of this key-shape
    # proxy (registry/topology change, not a runtime one).
    if cfg.get("min_pixels") is not None or cfg.get("max_pixels") is not None:
        min_px = int(cfg.get("min_pixels", 56 * 56))
        max_px = int(cfg.get("max_pixels", 28 * 28 * 1280))
        h_bar, w_bar = _smart_resize_2d(img.height, img.width,
                                        factor=p * merge,
                                        min_pixels=min_px, max_pixels=max_px)
    else:
        size = cfg.get("size", {}) or {}
        min_px = int(size.get("shortest_edge", 112 * 112))
        max_px = int(size.get("longest_edge", 14 * 14 * 2 * 2 * 2 * 6144))
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
    import imageio
    reader = imageio.get_reader(str(video_path))
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
    `i.flatten(end_dim=1).permute(1, 0)` -> [L*p, C*p]. The traced vpm
    graph consumes the PATCH-MAJOR form [1, N, C*p*p] whose definitional
    inverse (the trace wrapper's repack) is
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
            "trace (B > 1 slice batch).")
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
