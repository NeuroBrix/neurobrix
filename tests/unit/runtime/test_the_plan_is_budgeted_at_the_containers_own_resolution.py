"""One authority for the container's own output size — the executor renders at it, and
the plan is budgeted at it (2026-09-21, Wan2.1-T2V-1.3B: the flow decoded 81 frames at
480x832 while Prism, handed height=None, bound the VAE's spatial symbols to nothing and
estimated 1.74 GiB for a decode whose first conv input is 24.84 GB).

Shapes below are the T2V container's: the backbone's traced latent [1, 16, 21, 60, 104]
with spatial_compression_ratio 8 → 480x832; the VAE traced smaller ([1, 16, 9, 14, 22]) and
consulted only when the backbone cannot answer. What this test would do if the code were
wrong: a VAE-first walk answers 112x176; a missing scale answers None; a family constant
would answer 512."""
from __future__ import annotations

from neurobrix.core.runtime.resolution.container_size import container_output_size, vae_scale_factor


def _components(backbone_shape, vae_shape):
    return {"transformer": {"shapes": {"hidden_states": backbone_shape}},
            "vae": {"shapes": {"z": vae_shape}}}


def test_the_backbones_traced_latent_times_the_scale_is_the_output_size():
    comps = _components([1, 16, 21, 60, 104], [1, 16, 9, 14, 22])
    assert vae_scale_factor({}, {"spatial_compression_ratio": 8}, comps) == 8
    assert container_output_size({}, {"spatial_compression_ratio": 8}, comps) == (480, 832)


def test_the_vae_answers_only_when_the_backbone_cannot():
    comps = {"transformer": {"shapes": {"img": [1, 60, 64]}},          # a flattened latent (Open-Sora)
             "vae": {"shapes": {"z": [1, 16, 9, 14, 22]}}}
    assert container_output_size({}, {"vae_scale_factor": 8}, comps) == (112, 176)


def test_without_a_scale_or_a_spatial_latent_the_answer_is_none_never_a_constant():
    comps = _components([1, 16, 21, 60, 104], [1, 16, 9, 14, 22])
    assert container_output_size({}, {}, comps) is None                # no scale declared, no trace attrs
    assert container_output_size({}, {"vae_scale_factor": 8}, {"model": {"shapes": {"input_ids": [1, 23]}}}) is None
