"""I2V latent channel-concat conditioning (data-driven, R23-gated).

Some image-to-video denoisers condition by channel-concatenating a per-step-
invariant signal onto the noise latents BEFORE the transformer: the denoiser's
`in_channels` exceeds the latent channel count. The signal is built once from
the VAE-encoded first frame plus a deterministic frame mask.

Wan-I2V is the motivating case: hidden_states = cat([noise(16), mask(4),
vae_encoded_image(16)]) = 36ch. Unlike CogVideoX — whose transformer.forward
concatenates a separate `image_latents` input internally, so the runtime only
feeds it via a connection — the Wan transformer takes the pre-concatenated
tensor, so the runtime builds the condition channels here.

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
    spec.setdefault("condition_component", "vae_encoder")
    spec.setdefault("mask_channels", 4)
    return spec


def _vae_latent_stats(ctx: Any) -> tuple:
    """Read (latents_mean, latents_std, latent_channels) from the vae profile.

    Data-driven, never hardcoded — the same values the vendor pipeline applies
    after vae.encode: `latent = (latent - mean) * (1/std)`.
    """
    prof_path = ctx.pkg.cache_path / "components" / "vae" / "profile.json"
    prof = json.loads(prof_path.read_text())
    mean = prof.get("latents_mean")
    std = prof.get("latents_std")
    ch = prof.get("latent_channels") or (len(mean) if mean else 16)
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

    # Normalize exactly as the vendor pipeline: (x - mean) * (1/std).
    if mean is not None and std is not None:
        mean_t = torch.tensor(mean, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
        std_t = (1.0 / torch.tensor(std, device=device, dtype=dtype)).view(1, -1, 1, 1, 1)
        latent = (latent - mean_t) * std_t

    b, _c, latent_t, lh, lw = latent.shape
    mask = _build_frame_mask(num_frames, latent_t, lh, lw,
                             spec["mask_channels"], device, dtype)
    condition = torch.cat([mask, latent], dim=1)  # [B, mask_ch + C, T, H, W]
    import os as _os
    if _os.environ.get("NBX_DIAG_I2V") == "1":
        print(f"   [NBX-DIAG-I2V] vae_latent(ch-first)={list(latent.shape)} "
              f"mask={list(mask.shape)} condition={list(condition.shape)} "
              f"num_frames={num_frames}")
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


def apply(state: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
    """Channel-concat the condition onto the state, batch-aware.

    The condition is shared across CFG cond/uncond, so it is repeated to the
    state's batch (e.g. CFG-batched state [2, ...] vs condition [1, ...]).
    """
    if condition.shape[0] != state.shape[0] and state.shape[0] % condition.shape[0] == 0:
        condition = condition.repeat(state.shape[0] // condition.shape[0], 1, 1, 1, 1)
    if condition.dtype != state.dtype:
        condition = condition.to(state.dtype)
    if condition.device != state.device:
        condition = condition.to(state.device)
    return torch.cat([state, condition], dim=1)
