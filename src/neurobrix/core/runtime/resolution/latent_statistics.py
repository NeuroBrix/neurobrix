"""The latent's own statistics — the affine a decoder expects its input in.

Some video VAEs (Wan 2.1 / 2.2) are trained on a NORMALISED latent: the vendor pipeline
maps the denoised latent back before decoding, per channel,

    latents = latents * latents_std + latents_mean          (diffusers pipeline_wan.py:641-644,
                                                             written there as `/ (1/std) + mean`)

with both vectors declared on the VAE's config — and the decode graph, traced from the
VAE alone, cannot contain that step. Neither flow applied it (2026-09-21): Wan T2V decoded
the raw flow latent and rendered an orange field, a degenerate artefact under R29. Read
from the container's own declaration, never a constant: a VAE whose profile carries no
statistics gets None and its latent passes untouched.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple


def decoder_profile(pkg: Any, decoder_name: str) -> Dict[str, Any]:
    """The decoder's profile.json from the package's cache path — the same file the
    I2V and VACE conditioning read their statistics from; {} when the component has none."""
    path = pkg.cache_path / "components" / decoder_name / "profile.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def latent_affine(vae_profile: Dict[str, Any]) -> Optional[Tuple[List[float], List[float]]]:
    """(std, mean) per latent channel from the VAE profile's config, or None."""
    cfg = (vae_profile or {}).get("config") or vae_profile or {}
    mean = cfg.get("latents_mean")
    std = cfg.get("latents_std")
    if not (isinstance(mean, (list, tuple)) and isinstance(std, (list, tuple))):
        return None
    if len(mean) != len(std) or not mean:
        raise RuntimeError(
            f"ZERO FALLBACK: the VAE declares latents_mean of {len(mean)} channels and "
            f"latents_std of {len(std)}: the container's declaration is inconsistent.")
    return [float(v) for v in std], [float(v) for v in mean]
