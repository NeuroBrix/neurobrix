"""The VACE control-conditioning facts both branches read (core/runtime/resolution/vace_control_conditioning and
triton/vace_control_conditioning): the registry flag and the VAE profile, no tensor. Torch-free."""
import json
from typing import Any, Optional

from neurobrix.core.runtime.registry_flags import get_component_flag

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
