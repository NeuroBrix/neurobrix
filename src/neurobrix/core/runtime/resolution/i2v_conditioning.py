"""I2V latent channel-concat conditioning (data-driven, R23-gated).

Some image-to-video denoisers condition by channel-concatenating a per-step-
invariant signal onto the noise latents BEFORE the transformer: the denoiser's
`in_channels` exceeds the latent channel count. The signal is built once from
the VAE-encoded first frame plus a deterministic frame mask.

Wan-I2V is the motivating case: hidden_states = cat([noise(16), mask(4),
vae_encoded_image(16)]) = 36ch. Unlike CogVideoX — whose transformer.forward
concatenates a separate `image_latents` input internally, so the runtime only
feeds it via a connection — the Wan transformer takes the pre-concatenated
tensor, so the runtime builds the condition channels here. Allegro-TI2V
(`layout: state_video_mask`) is the third style: hidden_states =
cat([latents(4), masked_video(4), folded_pixel_mask(4)]) = 12ch, with the
pixel-space frame mask temporally folded into vae_temporal_ratio channels.

Driven entirely by the `i2v_latent_conditioning` registry flag on the denoiser
component. Returns None / no-ops for every model without the flag (R23 inert).

The built 20ch condition is stored in `variable_resolver` under
`global.i2v_condition` so both the flow (non-CFG path) and the CFG engine
(batched / sequential) can channel-concat it onto the (possibly batched) state.
"""

import json
from typing import Any, Optional

import torch

from neurobrix.core.runtime.registry_flags import get_component_flag

CONDITION_VAR = "global.i2v_condition"


def conditioning_spec(ctx: Any, loop_comp: str) -> Optional[dict]:
    """Return the i2v_latent_conditioning flag dict for the denoiser, or None.

    The flag may be `true` (defaults) or a dict with overrides. Returns a
    normalized dict only when conditioning is enabled, else None (inert).
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
        # Wan-I2V: frame-mask + per-channel mean/std norm. State is channels-
        # first [B, C, T, H, W], so the channel-concat is dim 1.
        spec.setdefault("mask_channels", 4)
        spec.setdefault("channel_dim", 1)
    elif style == "cogvideox":
        # CogVideoX-I2V: scalar-scaled, temporally-padded image latent, no mask.
        # State is frames-first [B, T, C, H, W], so the channel-concat is dim 2.
        spec.setdefault("channel_dim", 2)
    elif style == "state_video_mask":
        # Allegro-TI2V: transformer input = cat([latents, masked_video, mask],
        # dim=1) (vendor pipeline_allegro_ti2v.py:787). State is channels-first
        # [B, C, T, H, W] (topology hidden_states = [2, 4, 1, 90, 160]), so the
        # channel-concat is dim 1.
        spec.setdefault("channel_dim", 1)
        # Frame indices carrying a conditioning image (vendor
        # conditional_images_indices); a single first-frame image by default.
        spec.setdefault("cond_frame_indices", [0])
    return spec


def condition_channel_dim(ctx: Any, loop_comp: str) -> int:
    """Channel axis for apply()'s concat, data-driven from the spec.

    Returns 1 (Wan channels-first default) when there is no spec, so callers
    that channel-concat the condition stay correct for the default style.
    """
    spec = conditioning_spec(ctx, loop_comp)
    return int(spec.get("channel_dim", 1)) if spec else 1


def _vae_scaling(ctx: Any) -> tuple:
    """Read (scaling_factor, invert_scale_latents) from the vae profile.

    Scalar latent scaling for VAEs without per-channel mean/std (CogVideoX):
    `latent = scaling_factor * latent` (or `1/scaling_factor` when inverted),
    mirroring the vendor pipeline. Data-driven, never hardcoded.
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
    pipelines expose a (t, h, w) tuple). Data-driven, never hardcoded.
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
    """The denoiser's latent frame count (frames-first state dim 1).

    Read from the initialized noise latent so the conditioning's temporal
    extent matches the state exactly (no num_frames recompute).
    """
    for var in ("global.latents", "global.hidden_states"):
        st = ctx.variable_resolver.resolved.get(var)
        if st is None:
            try:
                st = ctx.variable_resolver.get(var)
            except Exception:
                st = None
        if isinstance(st, torch.Tensor) and st.dim() == 5:
            return int(st.shape[1])
    return None


def _vae_latent_stats(ctx: Any) -> tuple:
    """Read (latents_mean, latents_std, latent_channels) from the vae profile.

    Data-driven, never hardcoded — the same values the vendor pipeline applies
    after vae.encode: `latent = (latent - mean) * (1/std)`.
    """
    prof_path = ctx.pkg.cache_path / "components" / "vae" / "profile.json"
    prof = json.loads(prof_path.read_text())
    # profile.json nests the vendor config under "config"; fall back to
    # top-level for older layouts. (Reading top-level only silently returned
    # None -> normalization skipped -> the unnormalized condition poisoned the
    # denoiser into a uniform/checkerboard output.)
    cfg = prof.get("config") if isinstance(prof.get("config"), dict) else prof
    mean = cfg.get("latents_mean") or prof.get("latents_mean")
    std = cfg.get("latents_std") or prof.get("latents_std")
    ch = (cfg.get("latent_channels") or prof.get("latent_channels")
          or (len(mean) if mean else 16))
    return mean, std, int(ch)


def _to_channels_first(latent: torch.Tensor, latent_channels: int) -> torch.Tensor:
    """Arrange a 5D latent to [B, C, T, H, W] (channels at dim 1).

    The vae_encoder graph output can carry channels at dim 1 or dim 2 depending
    on the traced layout; detect the channel axis by matching latent_channels.
    """
    if latent.dim() != 5:
        return latent
    if latent.shape[1] == latent_channels:
        return latent
    if latent.shape[2] == latent_channels:
        return latent.permute(0, 2, 1, 3, 4).contiguous()
    return latent


def build_condition(ctx: Any, spec: dict, num_frames: int) -> Optional[torch.Tensor]:
    """Build the per-step-invariant conditioning tensor for the active style.

    Multi-style, data-driven (spec["style"]): "wan" (frame-mask + per-channel
    mean/std norm, channels-first), "cogvideox" (scalar-scaled, temporally-
    padded image latent, no mask, frames-first) or "state_video_mask"
    (Allegro-TI2V: VAE-encoded masked video + temporally-folded pixel mask,
    channels-first). Returns None if the vae_encoder output is not yet
    resolved.
    """
    if spec.get("style") == "cogvideox":
        return _build_condition_cogvideox(ctx, spec)
    if spec.get("style") == "state_video_mask":
        return _build_condition_state_video_mask(ctx, spec, num_frames)
    return _build_condition_wan(ctx, spec, num_frames)


def _build_condition_cogvideox(ctx: Any, spec: dict) -> Optional[torch.Tensor]:
    """CogVideoX-I2V conditioning: scalar-scaled VAE image latent, temporally
    padded (frame 0 = image, rest zeros) to the denoiser's latent frame count,
    NO mask, frames-first [B, T, C, H, W]. Mirrors
    CogVideoXImageToVideoPipeline.prepare_latents (the patch_size_t branch is a
    no-op for models with patch_size_t=None, e.g. CogVideoX-5b-I2V).
    """
    cond_comp = spec["condition_component"]
    img = ctx.variable_resolver.resolved.get(f"{cond_comp}.output_0")
    if img is None:
        img = ctx.variable_resolver.get(f"{cond_comp}.output_0")
    if not isinstance(img, torch.Tensor):
        return None
    _, _, latent_channels = _vae_latent_stats(ctx)
    img = _to_channels_first(img, latent_channels)  # [B, C, F, H, W], F=1
    sf, invert = _vae_scaling(ctx)
    if sf:
        img = (img / sf) if invert else (img * sf)
    b, c, f, h, w = img.shape
    t_target = _target_latent_frames(ctx)
    if t_target and t_target > f:
        pad = torch.zeros(b, c, t_target - f, h, w, device=img.device, dtype=img.dtype)
        img = torch.cat([img, pad], dim=2)  # [B, C, T, H, W], frame 0 = image
    import os as _os
    if _os.environ.get("NBX_DIAG_I2V") == "1":
        _f = img.float()
        print(f"   [NBX-DIAG-I2V] cogvideox condition shape={list(img.shape)} "
              f"scaling={sf} invert={invert} mean={_f.mean():.3f} std={_f.std():.3f}")
    # frames-first to match the CogVideoX state layout [B, T, C, H, W]
    return img.permute(0, 2, 1, 3, 4).contiguous()


def _build_condition_state_video_mask(ctx: Any, spec: dict,
                                      num_frames: int) -> Optional[torch.Tensor]:
    """Allegro-TI2V conditioning: cat([masked_video_latent(C), folded_mask(R)], dim=1).

    Mirrors AllegroTI2VPipeline.prepare_mask_masked_video (vendor
    pipeline_allegro_ti2v.py:895-921) and the denoise-loop concat
    cat([latents, masked_video, mask], dim=1) (vendor line 787):

    - masked_video: the conditioning image(s) at their frame indices, zeros
      elsewhere, VAE-encoded and latent-scaled (vendor lines 897-904,
      `input_video * (mask < 0.5)` then vae.encode * scale_factor). In this
      runtime the vae_encoder COMPONENT output IS that latent — its traced
      graph internalizes encoder -> quant_conv -> moments -> mode *
      scaling_factor -> permute to frames-first [B, F, C, H, W] (mode instead
      of the vendor's generator-driven .sample(): deterministic, baked into
      the graph). The CLI synthesizes its pixel input (image at frame 0,
      zeros padded to num_frames — exactly the vendor's masked video for
      cond index 0) via the vae_encoder `pad_image_to_num_frames` flag.
    - mask: PIXEL-space frame mask with INVERTED semantics (1 = NOT
      conditioned, 0 = conditioned; vendor lines 901-902), 1 channel,
      spatially resized to the latent grid, then temporally FOLDED into
      vae_temporal_ratio channels via view(B, ratio, latent_T, lh, lw)
      (vendor lines 905-916). The fold is a flat view: folded channel k at
      latent frame t covers PIXEL frame k * latent_T + t — a block split of
      the temporal axis, NOT an interleaved grouping of consecutive frames.

    apply() appends the condition after the state, so the transformer input
    is [state, masked_video, mask] — the vendor's exact order (hence the
    layout name). Returns None if the vae_encoder output is not yet resolved.
    """
    cond_comp = spec["condition_component"]
    latent = ctx.variable_resolver.resolved.get(f"{cond_comp}.output_0")
    if latent is None:
        latent = ctx.variable_resolver.get(f"{cond_comp}.output_0")
    if not isinstance(latent, torch.Tensor):
        return None

    _, _, latent_channels = _vae_latent_stats(ctx)
    latent = _to_channels_first(latent, latent_channels)  # [B, C, T, lh, lw]
    b, _c, _latent_t, lh, lw = latent.shape
    device, dtype = latent.device, latent.dtype

    # Pixel frames folded per latent frame; profile-driven, with the flag's
    # mask_channels as fallback (the fold width IS the temporal ratio).
    ratio = _vae_temporal_ratio(ctx) or int(spec.get("mask_channels") or 0)
    if not ratio:
        raise ValueError(
            "state_video_mask conditioning needs the vae temporal compression "
            "ratio (vae profile temporal_compression_ratio / "
            "vae_scale_factor[0], or the flag's mask_channels)")

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

    cond_indices = list(spec.get("cond_frame_indices") or [0])

    # Pixel-space inverted frame mask, built directly at the LATENT spatial
    # grid: the vendor F.interpolate(bilinear) (line 915) resizes per-frame
    # CONSTANT fields (each pixel frame is all-0 or all-1), and bilinear
    # resampling of a constant field is the identity — value-identical to
    # building at (lh, lw).
    mask = torch.ones(b, 1, num_frames, lh, lw, device=device, dtype=dtype)
    mask[:, :, cond_indices] = 0
    # Fold the pixel-temporal axis into `ratio` channels (vendor line 916).
    mask = mask.view(b, ratio, latent_size_t, lh, lw)

    # torch.cat enforces latent_T agreement between the encoded latent and
    # the folded mask (dim 2), exactly as the vendor concat does.
    condition = torch.cat([latent, mask], dim=1)  # [B, C + ratio, T, lh, lw]
    import os as _os
    if _os.environ.get("NBX_DIAG_I2V") == "1":
        _lf = latent.float()
        print(f"   [NBX-DIAG-I2V] state_video_mask condition "
              f"shape={list(condition.shape)} video mean={_lf.mean():.3f} "
              f"std={_lf.std():.3f} | mask zeros={int((mask == 0).sum())} "
              f"cond_indices={cond_indices} ratio={ratio}")
    return condition


def _build_condition_wan(ctx: Any, spec: dict, num_frames: int) -> Optional[torch.Tensor]:
    """Build the per-step-invariant conditioning tensor (channels dim 1).

    condition = cat([frame_mask(mask_channels), normalized_vae_latent(C)], dim=1)
    Returns None if the vae_encoder output is not yet resolved.
    """
    cond_comp = spec["condition_component"]
    latent = ctx.variable_resolver.resolved.get(f"{cond_comp}.output_0")
    if latent is None:
        latent = ctx.variable_resolver.get(f"{cond_comp}.output_0")
    if not isinstance(latent, torch.Tensor):
        return None

    mean, std, latent_channels = _vae_latent_stats(ctx)
    latent = _to_channels_first(latent, latent_channels)  # [B, C, T, H, W]
    device, dtype = latent.device, latent.dtype

    import os as _os
    _diag = _os.environ.get("NBX_DIAG_I2V") == "1"
    if _diag:
        _lf = latent.float()
        print(f"   [NBX-DIAG-I2V] RAW vae_latent shape={list(latent.shape)} "
              f"mean={_lf.mean():.3f} std={_lf.std():.3f} "
              f"min={_lf.min():.3f} max={_lf.max():.3f}")

    # Normalize exactly as the vendor pipeline: (x - mean) * (1/std).
    if mean is not None and std is not None:
        mean_t = torch.tensor(mean, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
        std_t = (1.0 / torch.tensor(std, device=device, dtype=dtype)).view(1, -1, 1, 1, 1)
        latent = (latent - mean_t) * std_t

    b, _c, latent_t, lh, lw = latent.shape
    mask = _build_frame_mask(num_frames, latent_t, lh, lw,
                             spec["mask_channels"], device, dtype)
    condition = torch.cat([mask, latent], dim=1)  # [B, mask_ch + C, T, H, W]
    if _diag:
        _nf = latent.float()
        print(f"   [NBX-DIAG-I2V] NORMALIZED latent mean={_nf.mean():.3f} "
              f"std={_nf.std():.3f} min={_nf.min():.3f} max={_nf.max():.3f} | "
              f"condition shape={list(condition.shape)} num_frames={num_frames} | "
              f"latents_mean[0:3]={mean[:3] if mean else None} latents_std[0:3]={std[:3] if std else None}")
    return condition


def _build_frame_mask(num_frames: int, latent_t: int, lh: int, lw: int,
                      mask_channels: int, device, dtype) -> torch.Tensor:
    """Deterministic Wan-I2V frame mask -> [1, mask_channels, latent_t, lh, lw].

    Mirrors WanImageToVideoPipeline.prepare_latents: only the first frame is
    conditioned (mask=1), the rest are 0; the first frame is repeat-interleaved
    by the temporal compression so the latent-temporal layout matches.
    """
    vst = mask_channels  # vae_scale_factor_temporal == mask_channels for Wan
    mask = torch.ones(1, 1, num_frames, lh, lw, device=device, dtype=dtype)
    if num_frames > 1:
        mask[:, :, 1:] = 0
    first = mask[:, :, 0:1]
    first = torch.repeat_interleave(first, dim=2, repeats=vst)
    mask = torch.cat([first, mask[:, :, 1:, :]], dim=2)
    mask = mask.view(1, -1, vst, lh, lw)
    mask = mask.transpose(1, 2).contiguous()  # [1, vst, latent_t, lh, lw]
    return mask


def apply(state: torch.Tensor, condition: torch.Tensor,
          channel_dim: int = 1) -> torch.Tensor:
    """Channel-concat the condition onto the state, batch-aware.

    The condition is shared across CFG cond/uncond, so it is repeated to the
    state's batch (e.g. CFG-batched state [2, ...] vs condition [1, ...]).
    `channel_dim` is the concat axis: 1 for Wan channels-first [B, C, T, H, W],
    2 for CogVideoX frames-first [B, T, C, H, W]. Default 1 keeps Wan unchanged.
    """
    if condition.shape[0] != state.shape[0] and state.shape[0] % condition.shape[0] == 0:
        condition = condition.repeat(state.shape[0] // condition.shape[0], 1, 1, 1, 1)
    if condition.dtype != state.dtype:
        condition = condition.to(state.dtype)
    if condition.device != state.device:
        condition = condition.to(state.device)
    return torch.cat([state, condition], dim=channel_dim)
