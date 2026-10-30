"""The image-to-video conditioning facts both branches read (core/runtime/resolution/i2v_conditioning for the
PyTorch branch, triton/i2v_conditioning for Triton): the registry flag and the VAE profile, no tensor. Torch-free."""
import json
from typing import Any, Optional

from neurobrix.core.runtime.registry_flags import get_component_flag


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
        # CogVideoX-I2V: the encoded (already scaled) image latent, temporally padded, no mask.
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
