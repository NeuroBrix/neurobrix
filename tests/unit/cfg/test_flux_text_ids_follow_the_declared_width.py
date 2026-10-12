"""The FLUX-family text position ids follow the WIDTH the container declares, never a
convention guessed for every model.

The vendor's conventions, read from its own source: with three axes (t, h, w) — FLUX.1,
Flex.1-alpha, Open-Sora 2.0 (Flux-initialised) — every text token's ids are zeros
(diffusers `pipeline_flux.py`, `encode_prompt`: ``text_ids = torch.zeros(prompt_embeds.shape[1], 3)``);
with a fourth axis the last channel is the text sequence coordinate L (diffusers
`pipeline_flux2_klein.py`, `_prepare_text_ids`: ``cartesian_prod(t, h, w, arange(L))``).

Measured 2026-10-12 on Flex.1-alpha (nbx/campaigns/2026_10_12_mq25_gate/judge/oracle/verdicts.txt):
both CFG engines wrote L into the third channel of its 3-axis ids, so every text token carried an
image-grid rotary (sin table l2 165.9 against the vendor's 131.2) and, on the vendor's own draw at
512x384 with the vendor's guidance, the native engine rendered red petals where the vendor renders
the apple; the weights, the embedders and the image ids all matched the vendor's.

What this test does if the code is wrong: the three-axis cell reads the text ids back and fails
on any non-zero element (seen red on both engines before the fix, 2026-10-12); the four-axis cell
fails if the sequence coordinate is no longer written into the last channel.

Run: python -m pytest tests/unit/cfg/test_flux_text_ids_follow_the_declared_width.py
"""
import numpy as np
import pytest


class _Resolver:
    def __init__(self, defaults):
        self.defaults = dict(defaults)
        self.values = {}

    def set(self, name, val):
        self.values[name] = val


class _Ctx:
    def __init__(self, defaults):
        self.variable_resolver = _Resolver(defaults)


def _as_numpy(v):
    if hasattr(v, "numpy"):
        return np.asarray(v.numpy())
    return np.asarray(v)


def _engine(engine):
    if engine == "triton":
        from neurobrix.triton.cfg.engine import _synthesize_position_ids
    else:
        from neurobrix.core.cfg.engine import _synthesize_position_ids
    return _synthesize_position_ids


@pytest.mark.parametrize("engine", ["triton", "core"])
@pytest.mark.parametrize("shape", [[512, 3], [1, 31, 3]])
def test_three_axis_text_ids_are_zeros(engine, shape):
    ctx = _Ctx({"latent_height": 48, "latent_width": 64})
    _engine(engine)(ctx, ["hidden_states", "txt_ids", "img_ids"],
                    {"hidden_states": [1, 768, 64], "txt_ids": shape, "img_ids": [768, 3]})
    ids = _as_numpy(ctx.variable_resolver.values["global.txt_ids"])
    assert list(ids.shape) == shape
    assert int(np.abs(ids).max()) == 0, "a three-axis text id carries a position"


@pytest.mark.parametrize("engine", ["triton", "core"])
@pytest.mark.parametrize("shape", [[512, 4], [1, 31, 4]])
def test_four_axis_text_ids_carry_the_sequence_coordinate_last(engine, shape):
    ctx = _Ctx({"latent_height": 48, "latent_width": 64})
    _engine(engine)(ctx, ["hidden_states", "txt_ids", "img_ids"],
                    {"hidden_states": [1, 768, 64], "txt_ids": shape, "img_ids": [768, 4]})
    ids = _as_numpy(ctx.variable_resolver.values["global.txt_ids"]).reshape(-1, 4)
    n = ids.shape[0]
    assert np.array_equal(ids[:, -1], np.arange(n)), "the fourth axis is the text sequence coordinate"
    assert int(np.abs(ids[:, :3]).max()) == 0, "t, h, w of a text token are zero"
