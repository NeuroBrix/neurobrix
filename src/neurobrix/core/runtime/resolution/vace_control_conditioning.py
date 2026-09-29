"""VACE control conditioning (data-driven, R23-gated).

WanVACE denoisers take TWO extra inputs each diffusion step that the standard
Wan transformer does not:

  * ``control_hidden_states``  — a 96-channel latent control signal, built (once,
    step-invariant) as ``cat([inactive_latent(16), reactive_latent(16),
    reshaped_mask(64)], dim=1)``. The 32 latent channels come from VAE-encoding
    the control video split into an *inactive* part (kept, ``video·(1-mask)``)
    and a *reactive* part (generated, ``video·mask``); the 64 mask channels come
    from folding the 8×8 spatial VAE-downsample factor of the binary mask into
    the channel dim (WanVACEPipeline.prepare_masks).
  * ``control_hidden_states_scale`` — a per-VACE-layer scale vector,
    ``ones(len(vace_layers))``, injected at the layers listed in the transformer
    config's ``vace_layers``.

Unlike Wan-I2V (which channel-concats its condition ONTO the noise latents — see
``i2v_conditioning.py``), VACE feeds these as *separate* transformer inputs, so
the runtime stores them as ``global.control_hidden_states`` /
``global.control_hidden_states_scale`` and the InputResolver binds them by the
zero-semantic ``global.<name>`` fallback.

The CLI builds the vendor's pair — ``V * (1 - M)`` and ``V * M`` for a control
clip V and a mask M (1 = generate) — on the encoder's batch axis, so one
``vae_encoder`` pass makes both halves: all-generate (no image) is V = 0, M = 1;
an input image is its padded clip with its frame kept (M = 0 there).

Driven entirely by the ``vace_control_conditioning`` registry flag on the
denoiser component. No flag → ``conditioning_spec`` returns None and every
function below is inert (R23 zero blast radius on other models).
"""

import json
from typing import Any, Optional

import torch

from neurobrix.core.runtime.registry_flags import get_component_flag

CONTROL_VAR = "global.control_hidden_states"
SCALE_VAR = "global.control_hidden_states_scale"
MASK_VAR = "global.vace_pixel_mask"


#: What a VACE conditioning spec must declare (the container's flag, from its registry entry).
SPEC_KEYS = ("condition_component", "mask_channels", "vace_layers", "z_dim")


def conditioning_spec(ctx: Any, loop_comp: str) -> Optional[dict]:
    """Return the normalized vace_control_conditioning spec, or None (inert)."""
    model_name = ctx.pkg.manifest.get("model_name")
    flag = get_component_flag(model_name, loop_comp, "vace_control_conditioning",
                              default=None)
    if not flag:
        return None
    spec = dict(flag) if isinstance(flag, dict) else {}
    # The container declares the whole spec (the registry's flag, carried at build). The four
    # values stood here as constants (15 layers, 64 mask channels, z_dim 16, "vae_encoder") —
    # one model's numbers, served to any container whose flag was a bare switch: refused now.
    missing = [k for k in SPEC_KEYS if spec.get(k) is None]
    if missing:
        raise RuntimeError(
            f"ZERO FALLBACK: {loop_comp}.vace_control_conditioning lacks {', '.join(missing)} — "
            f"the container must declare its VACE spec, never a model's constants.")
    return spec


def _vae_latent_stats(ctx: Any) -> tuple:
    """Read (latents_mean, latents_std, latent_channels) from the vae profile.

    Same data-driven source and normalization as i2v_conditioning: the values
    the vendor applies after vae.encode — ``latent = (latent - mean) * (1/std)``.
    """
    prof_path = ctx.pkg.cache_path / "components" / "vae" / "profile.json"
    prof = json.loads(prof_path.read_text())
    cfg = prof.get("config") if isinstance(prof.get("config"), dict) else prof
    mean = cfg.get("latents_mean") or prof.get("latents_mean")
    std = cfg.get("latents_std") or prof.get("latents_std")
    ch = (cfg.get("latent_channels") or prof.get("latent_channels") or cfg.get("z_dim")
          or (len(mean) if mean else 16))
    return mean, std, int(ch)


def _to_channels_first(latent: torch.Tensor, latent_channels: int) -> torch.Tensor:
    """Arrange a 5D latent to [B, C, T, H, W] (channels at dim 1)."""
    if latent.dim() != 5:
        return latent
    if latent.shape[1] == latent_channels:
        return latent
    if latent.shape[2] == latent_channels:
        return latent.permute(0, 2, 1, 3, 4).contiguous()
    return latent


def build_control(ctx: Any, spec: dict) -> Optional[torch.Tensor]:
    """Build control_hidden_states = cat([inactive, reactive, mask], dim=1) as the vendor does.

    The vae_encoder encoded the pair the CLI built (batch 0 = ``V * (1 - M)``, batch 1 =
    ``V * M``, image_dsp.vace_control_pair_np); both are normalized as the vendor normalizes
    (``(x - mean) / std``) and the pixel mask ``global.vace_pixel_mask`` is folded into the
    mask channels on the latent grid (image_dsp.vace_fold_mask_np). Returns None if the
    vae_encoder output is not yet resolved.
    """
    from neurobrix.core.module.vision.image_dsp import vace_fold_mask_np

    cond_comp = spec["condition_component"]
    latent = ctx.variable_resolver.resolved.get(f"{cond_comp}.output_0")
    if latent is None:
        latent = ctx.variable_resolver.get(f"{cond_comp}.output_0")
    if not isinstance(latent, torch.Tensor):
        return None
    pixel_mask = ctx.variable_resolver.resolved.get(MASK_VAR)
    if pixel_mask is None:
        raise RuntimeError(f"ZERO FALLBACK: VACE control needs {MASK_VAR}, the pixel mask the CLI builds")

    mean, std, latent_channels = _vae_latent_stats(ctx)
    latent = _to_channels_first(latent, latent_channels)  # [2, C, T, H, W]
    if latent.shape[0] != 2:
        raise RuntimeError(
            f"ZERO FALLBACK: the VACE control encoder must see the (inactive, reactive) pair on its "
            f"batch axis, got batch {latent.shape[0]}")
    device, dtype = latent.device, latent.dtype

    # Normalize exactly as the vendor pipeline: (x - mean) * (1/std).
    if mean is not None and std is not None:
        mean_t = torch.tensor(mean, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
        std_t = (1.0 / torch.tensor(std, device=device, dtype=dtype)).view(1, -1, 1, 1, 1)
        latent = (latent - mean_t) * std_t

    video_latents = torch.cat([latent[0:1], latent[1:2]], dim=1)  # [1, 2*z_dim, T, H, W]
    _b, _c, lt, lh, lw = video_latents.shape
    pm = pixel_mask.detach().cpu().numpy() if isinstance(pixel_mask, torch.Tensor) else pixel_mask
    mask = torch.from_numpy(vace_fold_mask_np(pm, lt, lh, lw, int(spec["mask_channels"]))).to(
        device=device, dtype=dtype)
    control = torch.cat([video_latents, mask], dim=1)  # [1, 2*z_dim + mask_ch, ...]

    import os as _os
    if _os.environ.get("NBX_DIAG_VACE") == "1":
        _lf = latent.float()
        print(f"   [NBX-DIAG-VACE] vae_latent shape={list(latent.shape)} "
              f"norm mean={_lf.mean():.3f} std={_lf.std():.3f} | "
              f"control shape={list(control.shape)} mask mean={float(mask.float().mean()):.3f} "
              f"(expect C={2 * latent_channels + int(spec['mask_channels'])})")
    return control


def build_scale(spec: dict, device, dtype) -> torch.Tensor:
    """control_hidden_states_scale = ones(len(vace_layers))."""
    return torch.ones(int(spec["vace_layers"]), device=device, dtype=dtype)


def batch_for_cfg(control: torch.Tensor) -> torch.Tensor:
    """Repeat the batch-1 control to batch=2 for the batched CFG forward.

    The control signal is shared across CFG cond/uncond (the vendor passes the
    SAME tensor to both forwards), so in batched mode it is repeated [c, c] to
    match the batch=2 hidden_states. Mirrors i2v_conditioning.apply's repeat.
    """
    if control.shape[0] == 1:
        return torch.cat([control, control], dim=0)
    return control
