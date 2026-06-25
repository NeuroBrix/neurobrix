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
NBXTensor state before the transformer.

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
    style = spec.setdefault("style", "wan")
    spec.setdefault("condition_component", "vae_encoder")
    if style == "wan":
        spec.setdefault("mask_channels", 4)
        spec.setdefault("channel_dim", 1)
    elif style == "cogvideox":
        spec.setdefault("channel_dim", 2)
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
    of the compiled brick). "wan" (frame-mask + mean/std, channels-first) or
    "cogvideox" (scalar-scaled, temporally-padded image latent, no mask,
    frames-first). Returns None if the vae_encoder output is not yet resolved.
    """
    if spec.get("style") == "cogvideox":
        return _build_condition_cogvideox(ctx, spec)
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
    return NBXTensor.cat([state, condition], dim=channel_dim)
