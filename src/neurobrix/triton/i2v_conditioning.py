"""Triton-branch I2V latent channel-concat conditioning (NBXTensor, R33-pure).

This is the TRITON-side brick. It mirrors the SEMANTICS of the compiled brick
`core/runtime/resolution/i2v_conditioning.py` but lives entirely in the triton
branch and computes end-to-end with NBXTensor — no torch, no `F.*`, no reach
into `core/` compute. The two branches stay separate (Hocine's architecture
invariant); this file is the R30 mirror of the core conditioning.

Wan-I2V is the motivating case: hidden_states = cat([noise(16), mask(4),
vae_encoded_image(16)]) = 36ch. The triton flow builds the 20ch condition once
from the vae_encoder NBXTensor output plus a deterministic frame mask, and the
triton flow / triton CFG engine channel-concat it onto the (possibly batched)
NBXTensor state before the transformer. Allegro-TI2V (`layout:
state_video_mask`) is the third style: hidden_states = cat([latents(4),
masked_video(4), folded_pixel_mask(4)]) = 12ch, with the pixel-space frame
mask temporally folded into vae_temporal_ratio channels.

Allowed imports here: NBXTensor + triton device glue (R33), numpy for the
host-side deterministic mask / per-channel stats (R34 CPU glue), and the shared
registry-flag / profile readers (data-driven config, not compute — the triton
flow already reads registry flags the same way).
"""

import json
from typing import Any, Optional

import numpy as np

from neurobrix.kernels.nbx_tensor import NBXTensor, DeviceAllocator
from neurobrix.triton.device_transfer import transfer_tensor
from neurobrix.core.runtime.registry_flags import get_component_flag

CONDITION_VAR = "global.i2v_condition"


def conditioning_spec(ctx: Any, loop_comp: str) -> Optional[dict]:
    """Return the i2v_latent_conditioning flag dict for the denoiser, or None.

    Identical data-driven read as the compiled brick (registry flag = config,
    not compute). Inert (None) for every model without the flag (R23).
    """
    model_name = ctx.pkg.manifest.get("model_name")
    flag = get_component_flag(model_name, loop_comp, "i2v_latent_conditioning",
                              default=None)
    if not flag:
        return None
    spec = dict(flag) if isinstance(flag, dict) else {}
    # `layout` is the registry alias for `style` (Allegro-TI2V declares
    # `layout: state_video_mask`); both feed the same style switch.
    style = spec.setdefault("style", spec.get("layout") or "wan")
    spec.setdefault("condition_component", "vae_encoder")
    if style == "wan":
        spec.setdefault("mask_channels", 4)
        spec.setdefault("channel_dim", 1)
    elif style == "cogvideox":
        spec.setdefault("channel_dim", 2)
    elif style == "state_video_mask":
        # Allegro-TI2V: transformer input = cat([latents, masked_video, mask],
        # dim=1); state is channels-first [B, C, T, H, W] -> concat dim 1.
        spec.setdefault("channel_dim", 1)
        # Frame indices carrying a conditioning image (vendor
        # conditional_images_indices); a single first-frame image by default.
        spec.setdefault("cond_frame_indices", [0])
    return spec


def condition_channel_dim(ctx: Any, loop_comp: str) -> int:
    """Channel axis for apply()'s concat, data-driven from the spec (R30 mirror
    of the compiled brick). 1 (channels-first default) when there is no spec."""
    spec = conditioning_spec(ctx, loop_comp)
    return int(spec.get("channel_dim", 1)) if spec else 1


def _vae_scaling(ctx: Any) -> tuple:
    """Read (scaling_factor, invert_scale_latents) from the vae profile.

    Scalar latent scaling for VAEs without per-channel mean/std (CogVideoX).
    Config read (data-driven), not compute.
    """
    prof_path = ctx.pkg.cache_path / "components" / "vae" / "profile.json"
    prof = json.loads(prof_path.read_text())
    cfg = prof.get("config") if isinstance(prof.get("config"), dict) else prof
    sf = cfg.get("scaling_factor") or prof.get("scaling_factor")
    invert = bool(cfg.get("invert_scale_latents") or prof.get("invert_scale_latents"))
    return (float(sf) if sf else None), invert


def _vae_temporal_ratio(ctx: Any) -> Optional[int]:
    """Read the VAE temporal compression ratio (pixel frames per latent frame).

    From the vae profile config: `temporal_compression_ratio` (diffusers-style
    configs, e.g. AutoencoderKLAllegro) or `vae_scale_factor[0]` (vendor
    pipelines expose a (t, h, w) tuple). Data-driven, never hardcoded — the
    same fallback chain as the compiled brick (config read, not compute).
    """
    prof_path = ctx.pkg.cache_path / "components" / "vae" / "profile.json"
    prof = json.loads(prof_path.read_text())
    cfg = prof.get("config") if isinstance(prof.get("config"), dict) else prof
    ratio = (cfg.get("temporal_compression_ratio")
             or prof.get("temporal_compression_ratio"))
    if not ratio:
        vsf = cfg.get("vae_scale_factor") or prof.get("vae_scale_factor")
        if isinstance(vsf, (list, tuple)) and vsf:
            ratio = vsf[0]
    return int(ratio) if ratio else None


def _target_latent_frames(ctx: Any) -> Optional[int]:
    """The denoiser's latent frame count (frames-first state dim 1), read from
    the initialized noise latent NBXTensor."""
    for var in ("global.latents", "global.hidden_states"):
        st = ctx.variable_resolver.resolved.get(var)
        if st is None:
            try:
                st = ctx.variable_resolver.get(var)
            except Exception:
                st = None
        if isinstance(st, NBXTensor) and st.dim() == 5:
            return int(st.shape[1])
    return None


def _vae_latent_stats(ctx: Any) -> tuple:
    """Read (latents_mean, latents_std, latent_channels) from the vae profile."""
    prof_path = ctx.pkg.cache_path / "components" / "vae" / "profile.json"
    prof = json.loads(prof_path.read_text())
    cfg = prof.get("config") if isinstance(prof.get("config"), dict) else prof
    mean = cfg.get("latents_mean") or prof.get("latents_mean")
    std = cfg.get("latents_std") or prof.get("latents_std")
    ch = (cfg.get("latent_channels") or prof.get("latent_channels")
          or (len(mean) if mean else 16))
    return mean, std, int(ch)


def _to_channels_first(latent: NBXTensor, latent_channels: int) -> NBXTensor:
    """Arrange a 5D NBXTensor latent to [B, C, T, H, W] (channels at dim 1)."""
    if latent.dim() != 5:
        return latent
    if latent.shape[1] == latent_channels:
        return latent
    if latent.shape[2] == latent_channels:
        return latent.permute(0, 2, 1, 3, 4).contiguous()
    return latent


def _frame_mask_np(num_frames: int, latent_t: int, lh: int, lw: int,
                   mask_channels: int) -> np.ndarray:
    """Deterministic Wan-I2V frame mask -> np[1, mask_channels, latent_t, lh, lw].

    Bit-mirror of the compiled `_build_frame_mask` (torch) in numpy: only the
    first frame is conditioned, repeat-interleaved by the temporal compression.
    """
    vst = mask_channels
    mask = np.ones((1, 1, num_frames, lh, lw), dtype=np.float32)
    if num_frames > 1:
        mask[:, :, 1:] = 0.0
    first = mask[:, :, 0:1]
    first = np.repeat(first, vst, axis=2)
    mask = np.concatenate([first, mask[:, :, 1:, :]], axis=2)
    mask = mask.reshape(1, -1, vst, lh, lw)
    mask = np.transpose(mask, (0, 2, 1, 3, 4))
    return np.ascontiguousarray(mask)


def _state_video_mask_np(b: int, num_frames: int, ratio: int, lh: int, lw: int,
                         cond_indices: list) -> np.ndarray:
    """Allegro-TI2V folded pixel mask -> np[b, ratio, latent_T, lh, lw].

    Host-side deterministic mask (same numpy CPU-glue allowance as
    `_frame_mask_np`), bit-mirror of the compiled brick's torch build:

    - PIXEL-space frame mask with INVERTED semantics (1 = NOT conditioned,
      0 = conditioned; vendor pipeline_allegro_ti2v.py:901-902), 1 channel,
      built directly at the LATENT spatial grid: the vendor
      F.interpolate(bilinear) resizes per-frame CONSTANT fields (each pixel
      frame is all-0 or all-1), and bilinear resampling of a constant field
      is the identity — value-identical to building at (lh, lw). (Documented
      shortcut shared with the compiled brick.)
    - Temporally FOLDED into `ratio` channels via the flat view
      (b, ratio, latent_T, lh, lw) (vendor line 916): folded channel k at
      latent frame t covers PIXEL frame k * latent_T + t — a block split of
      the temporal axis, NOT an interleaved grouping of consecutive frames.

    Raises on the same constraints the compiled brick (and the vendor's flat
    view) enforces: num_frames must equal ratio * latent_T.
    """
    # Latent temporal extent from PIXEL num_frames (vendor lines 910-914).
    if num_frames % 2 == 1:
        latent_size_t = (num_frames - 1) // ratio + 1
    else:
        latent_size_t = num_frames // ratio
    if num_frames != ratio * latent_size_t:
        # The vendor's flat view (line 916) enforces the same equality — it
        # would raise an opaque view error on the identical constraint.
        raise ValueError(
            f"state_video_mask mask fold needs num_frames == "
            f"vae_temporal_ratio * latent_T ({num_frames} != "
            f"{ratio} * {latent_size_t})")
    mask = np.ones((b, 1, num_frames, lh, lw), dtype=np.float32)
    mask[:, :, cond_indices] = 0.0
    # Fold the pixel-temporal axis into `ratio` channels (flat reshape ==
    # torch .view on the contiguous mask).
    return np.ascontiguousarray(mask.reshape(b, ratio, latent_size_t, lh, lw))


def _nbx_on(arr: np.ndarray, dev_idx: int) -> NBXTensor:
    """from_numpy onto a specific device (from_numpy uses the current device)."""
    prev = DeviceAllocator.get_device()
    DeviceAllocator.set_device(dev_idx)
    try:
        return NBXTensor.from_numpy(arr)
    finally:
        DeviceAllocator.set_device(prev)


def build_condition(ctx: Any, spec: dict, num_frames: int) -> Optional[NBXTensor]:
    """Build the per-step-invariant condition for the active style (R30 mirror
    of the compiled brick). "wan" (frame-mask + mean/std, channels-first),
    "cogvideox" (scalar-scaled, temporally-padded image latent, no mask,
    frames-first) or "state_video_mask" (Allegro-TI2V: VAE-encoded masked
    video + temporally-folded pixel mask, channels-first). Returns None if the
    vae_encoder output is not yet resolved.
    """
    if spec.get("style") == "cogvideox":
        return _build_condition_cogvideox(ctx, spec)
    if spec.get("style") == "state_video_mask":
        return _build_condition_state_video_mask(ctx, spec, num_frames)
    return _build_condition_wan(ctx, spec, num_frames)


def _build_condition_cogvideox(ctx: Any, spec: dict) -> Optional[NBXTensor]:
    """CogVideoX-I2V conditioning (NBXTensor): scalar-scaled VAE image latent,
    temporally padded (frame 0 = image, rest zeros) to the denoiser's latent
    frame count, NO mask, frames-first [B, T, C, H, W]. R33-pure mirror of the
    compiled _build_condition_cogvideox.
    """
    cond_comp = spec["condition_component"]
    res = ctx.variable_resolver.resolved
    img = res.get(f"{cond_comp}.output_0")
    if img is None:
        img = res.get(f"{cond_comp}.output")
    if not isinstance(img, NBXTensor):
        return None
    _, _, latent_channels = _vae_latent_stats(ctx)
    img = _to_channels_first(img, latent_channels).float()  # [B, C, F, H, W]
    dev_idx = img._device_idx
    b, c, f, h, w = img.shape
    sf, invert = _vae_scaling(ctx)
    if sf:
        factor = (1.0 / sf) if invert else sf
        scale_full = np.full((b, c, f, h, w), np.float32(factor), dtype=np.float32)
        img = img * _nbx_on(scale_full, dev_idx)
    t_target = _target_latent_frames(ctx)
    if t_target and t_target > f:
        pad = _nbx_on(np.zeros((b, c, t_target - f, h, w), dtype=np.float32), dev_idx)
        img = NBXTensor.cat([img, pad], dim=2)  # [B, C, T, H, W], frame 0 = image
    # frames-first to match the CogVideoX state layout [B, T, C, H, W]
    return img.permute(0, 2, 1, 3, 4).contiguous()


def _build_condition_state_video_mask(ctx: Any, spec: dict,
                                      num_frames: int) -> Optional[NBXTensor]:
    """Allegro-TI2V conditioning (NBXTensor): cat([masked_video_latent(C),
    folded_mask(R)], dim=1) -> [B, C + ratio, T, lh, lw]. R33-pure mirror of
    the compiled _build_condition_state_video_mask.

    Mirrors AllegroTI2VPipeline.prepare_mask_masked_video (vendor
    pipeline_allegro_ti2v.py:895-921) and the denoise-loop concat
    cat([latents, masked_video, mask], dim=1) (vendor line 787):

    - masked_video: the conditioning image(s) at their frame indices, zeros
      elsewhere, VAE-encoded and latent-scaled. In this runtime the
      vae_encoder COMPONENT output IS that latent — its traced graph
      internalizes encoder -> quant_conv -> moments -> mode * scaling_factor
      -> permute to frames-first [B, F, C, H, W]. The CLI synthesizes its
      pixel input (image at frame 0, zeros padded to num_frames) via the
      vae_encoder `pad_image_to_num_frames` flag.
    - mask: host-built folded pixel mask (`_state_video_mask_np`, numpy CPU
      glue like the wan `_frame_mask_np`), uploaded once and cast to the
      latent's dtype (0/1 values are exact in every float dtype).

    apply() appends the condition after the state, so the transformer input
    is [state, masked_video, mask] — the vendor's exact order (hence the
    layout name). Returns None if the vae_encoder output is not yet resolved.
    """
    cond_comp = spec["condition_component"]
    res = ctx.variable_resolver.resolved
    latent = res.get(f"{cond_comp}.output_0")
    if latent is None:
        latent = res.get(f"{cond_comp}.output")
    if not isinstance(latent, NBXTensor):
        return None

    _, _, latent_channels = _vae_latent_stats(ctx)
    latent = _to_channels_first(latent, latent_channels)  # [B, C, T, lh, lw]
    dev_idx = latent._device_idx
    b, _c, latent_t, lh, lw = latent.shape

    # Pixel frames folded per latent frame; profile-driven, with the flag's
    # mask_channels as fallback (the fold width IS the temporal ratio).
    ratio = _vae_temporal_ratio(ctx) or int(spec.get("mask_channels") or 0)
    if not ratio:
        raise ValueError(
            "state_video_mask conditioning needs the vae temporal compression "
            "ratio (vae profile temporal_compression_ratio / "
            "vae_scale_factor[0], or the flag's mask_channels)")

    cond_indices = list(spec.get("cond_frame_indices") or [0])
    mask_np = _state_video_mask_np(b, num_frames, ratio, lh, lw, cond_indices)

    # Latent_T agreement between the encoded latent and the folded mask: the
    # compiled brick gets this from torch.cat (which validates non-cat dims);
    # NBXTensor.cat sizes the output from tensors[0] without validating, so
    # the mirror enforces the same contract explicitly.
    if mask_np.shape[2] != latent_t:
        raise ValueError(
            f"state_video_mask latent_T mismatch: encoded latent has "
            f"{latent_t} latent frames but the folded mask has "
            f"{mask_np.shape[2]} (num_frames={num_frames}, ratio={ratio})")

    mask_t = _nbx_on(mask_np, dev_idx)
    if mask_t._dtype != latent._dtype:
        mask_t = mask_t.to(latent._dtype)
    return NBXTensor.cat([latent, mask_t], dim=1)  # [B, C + ratio, T, lh, lw]


def _build_condition_wan(ctx: Any, spec: dict, num_frames: int) -> Optional[NBXTensor]:
    """Build the 20ch per-step-invariant condition (channels at dim 1) as NBXTensor.

    condition = cat([frame_mask(mask_channels), normalized_vae_latent(C)], dim=1)
    Returns None if the vae_encoder output is not yet resolved.
    """
    cond_comp = spec["condition_component"]
    res = ctx.variable_resolver.resolved
    latent = res.get(f"{cond_comp}.output_0")
    if latent is None:
        latent = res.get(f"{cond_comp}.output")
    if not isinstance(latent, NBXTensor):
        return None

    mean, std, latent_channels = _vae_latent_stats(ctx)
    latent = _to_channels_first(latent, latent_channels).float()
    dev_idx = latent._device_idx
    b, _c, latent_t, lh, lw = latent.shape

    # Normalize exactly as the vendor pipeline: (x - mean) * (1/std). Pre-
    # broadcast the per-channel stats to the full latent shape in numpy so the
    # NBXTensor elementwise kernels run same-shape (no broadcast dependency).
    if mean is not None and std is not None:
        shape = (b, latent_channels, latent_t, lh, lw)
        mean_full = np.ascontiguousarray(
            np.broadcast_to(np.asarray(mean, np.float32).reshape(1, -1, 1, 1, 1), shape))
        inv_std_full = np.ascontiguousarray(
            np.broadcast_to((1.0 / np.asarray(std, np.float32)).reshape(1, -1, 1, 1, 1), shape))
        mean_t = _nbx_on(mean_full, dev_idx)
        inv_std_t = _nbx_on(inv_std_full, dev_idx)
        latent = (latent - mean_t) * inv_std_t

    mask_t = _nbx_on(_frame_mask_np(num_frames, latent_t, lh, lw,
                                    spec["mask_channels"]), dev_idx)
    return NBXTensor.cat([mask_t, latent], dim=1)  # [B, mask_ch + C, T, H, W]


def apply(state: NBXTensor, condition: NBXTensor,
          channel_dim: int = 1) -> NBXTensor:
    """Channel-concat the condition onto the state — batch, device, dtype aware.

    The condition is shared across CFG cond/uncond, so it is repeated to the
    state's batch (CFG-batched state [2, ...] vs condition [1, ...]). Moved to
    the state's device (D2D) and cast to the state's dtype before the concat.
    `channel_dim` is the concat axis: 1 for Wan channels-first [B, C, T, H, W],
    2 for CogVideoX frames-first [B, T, C, H, W]. Default 1 keeps Wan unchanged.
    """
    if condition.shape[0] != state.shape[0] and state.shape[0] % condition.shape[0] == 0:
        condition = condition.repeat(state.shape[0] // condition.shape[0], 1, 1, 1, 1)
    if condition._device_idx != state._device_idx:
        condition = transfer_tensor(condition, state._device_idx)
    if condition._dtype != state._dtype:
        condition = condition.to(state._dtype)
    # Extent guard: the condition must span every non-batch, non-channel dim of
    # the state so the channel-concat is well-formed. CogVideoX-I2V builds a
    # single-frame image condition (frame 0 = image latent); it must be zero-
    # padded along the latent frame axis to the state's frame count (frame 0 =
    # image, frames 1..T-1 = zeros). The compiled mirror does this at build time
    # via _target_latent_frames, but that lookup returns None in the triton
    # resolution order (the latent is not yet under global.latents when
    # build_condition runs before the loop), leaving a 1-frame condition. Pad
    # here against the LIVE state — timing-independent, semantically identical to
    # the compiled temporal pad, and a no-op for Wan (its condition is already
    # full-extent). Without it, the downstream cat silently mis-lays the batch
    # and only branch 0 receives the condition → CFG per-branch divergence.
    for d in range(state.ndim):
        if d == 0 or d == channel_dim:
            continue
        if condition.shape[d] < state.shape[d]:
            pad_shape = list(condition.shape)
            pad_shape[d] = state.shape[d] - condition.shape[d]
            pad = NBXTensor.zeros(tuple(pad_shape), dtype=condition._dtype,
                                  device=f"cuda:{condition._device_idx}")
            condition = NBXTensor.cat([condition, pad], dim=d)
    return NBXTensor.cat([state, condition], dim=channel_dim)
