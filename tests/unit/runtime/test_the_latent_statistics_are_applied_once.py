"""The decoder's latent statistics (latents_mean / latents_std) are applied ONCE, by the
flow (`resolution.latent_statistics`), never again by the VAE handler.

Both used to apply them: every container declaring the statistics decoded
`(x*std + mean)*std + mean` (measured 2026-10-04 — mochi-1-preview's first decode tile
matched the vendor's conv_in on the twice-mapped latent, 534.777 vs 534.765, once-mapped
539.601; Wan2.1-T2V's handler received a latent already equal to final*std + mean, to
0.0, and mapped it again, std 2.64 -> 6.38). What this test would do if the code were
wrong: the handler's output would differ from its input for a declaring profile and
the first two tests fail (seen failing on injection of the old block).
"""
import pytest
import torch

from neurobrix.core.components.base import ComponentConfig
from neurobrix.core.components.handlers.vae_handler import VAEComponentHandler

MEAN = [-0.75, 0.1, 0.3]
STD = [2.8, 1.4, 0.9]


def _handler(scaling_factor=None):
    cfg = ComponentConfig(class_name="AutoencoderKLMochi", component_type="vae",
                          scaling_factor=scaling_factor,
                          raw_profile={"config": {"latents_mean": MEAN, "latents_std": STD,
                                                  "scaling_factor": scaling_factor}})
    return VAEComponentHandler(cfg, cache_path="")


def _latent():
    return torch.randn(1, 3, 2, 4, 5, generator=torch.Generator().manual_seed(0))


@pytest.mark.parametrize("sf", [None, 1.0])
def test_a_declaring_vae_receives_the_flow_latent_untouched(sf):
    x = _latent()
    out = _handler(sf).transform_inputs({"z": x.clone()}, "post_loop")["z"]
    assert torch.equal(out, x)


def test_the_scaling_factor_is_still_the_handlers():
    x = _latent()
    out = _handler(2.0).transform_inputs({"z": x.clone()}, "post_loop")["z"]
    assert torch.allclose(out, x / 2.0)


def test_outside_the_post_loop_nothing_changes():
    x = _latent()
    assert torch.equal(_handler(2.0).transform_inputs({"z": x.clone()}, "loop")["z"], x)
