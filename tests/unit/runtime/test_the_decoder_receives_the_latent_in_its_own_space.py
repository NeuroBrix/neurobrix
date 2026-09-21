"""A VAE trained on a normalised latent declares latents_mean / latents_std; the vendor
maps the latent back per channel before decoding (`latents * std + mean`, diffusers
pipeline_wan.py:641-644). Neither flow applied it and Wan T2V decoded the raw flow latent
into an orange field (R29, 2026-09-21). What this test would do if the code were wrong:
answer None for a declaring profile (the field), or accept mismatched declarations."""
from __future__ import annotations

import pytest

from neurobrix.core.runtime.resolution.latent_statistics import latent_affine


def test_a_declaring_profile_gives_std_then_mean_per_channel():
    prof = {"config": {"latents_mean": [-0.75, 0.1], "latents_std": [2.8, 1.4], "latent_channels": 2}}
    assert latent_affine(prof) == ([2.8, 1.4], [-0.75, 0.1])


def test_a_profile_without_statistics_leaves_the_latent_alone():
    assert latent_affine({"config": {"latent_channels": 16}}) is None
    assert latent_affine({}) is None


def test_inconsistent_declarations_refuse():
    with pytest.raises(RuntimeError, match="ZERO FALLBACK"):
        latent_affine({"config": {"latents_mean": [0.0, 0.0], "latents_std": [1.0]}})
