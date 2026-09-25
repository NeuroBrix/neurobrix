"""The FLUX-family image position ids follow the REQUEST's latent grid, never a
square guess from the token count.

Measured 2026-09-25 on the re-traced Flex.1-alpha at 512x1536 (latent 64x192,
patch grid 32x96 = 3072 tokens): both CFG engines computed the count from the
runtime latent dims and then rebuilt the grid as int(sqrt(3072)) = 55 rows by 55
columns = 3025 ids — the RoPE multiply failed: ``Cannot broadcast (1, 24, 3584, 128)
and (1, 1, 3537, 128)`` (validation_outputs/signature_retrace_2026_09_25/Flex.1-alpha/512x1536.log).
A square request hid it (64x64 = sqrt(4096)).

What this test does if the code is wrong: the non-square cell reads the ids back
and fails on their count and their row/column extents (seen red on the sqrt form,
2026-09-25, both engines); the refusal cell fails if a missing latent size is
guessed instead of refused (ZERO FALLBACK).

Run: python -m pytest tests/unit/cfg/test_flux_image_ids_follow_the_request_grid.py
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


COMP_INPUTS = ["hidden_states", "img_ids"]
COMP_SHAPES = {"hidden_states": [1, 4096, 64], "img_ids": [4096, 3]}


def _as_numpy(v):
    if hasattr(v, "numpy"):
        return np.asarray(v.numpy())
    return np.asarray(v)


@pytest.mark.parametrize("engine", ["triton", "core"])
def test_a_non_square_request_gets_its_own_rows_and_columns(engine):
    if engine == "triton":
        from neurobrix.triton.cfg.engine import _synthesize_position_ids
    else:
        from neurobrix.core.cfg.engine import _synthesize_position_ids
    ctx = _Ctx({"latent_height": 64, "latent_width": 192})   # 512x1536 / 8
    _synthesize_position_ids(ctx, COMP_INPUTS, COMP_SHAPES)
    ids = _as_numpy(ctx.variable_resolver.values["global.img_ids"]).reshape(-1, 3)
    assert ids.shape[0] == 32 * 96, ids.shape
    assert int(ids[:, 1].max()) == 31 and int(ids[:, 2].max()) == 95, (ids[:, 1].max(), ids[:, 2].max())


@pytest.mark.parametrize("engine", ["triton", "core"])
def test_a_request_without_latent_sizes_is_refused_not_guessed(engine):
    if engine == "triton":
        from neurobrix.triton.cfg.engine import _synthesize_position_ids
    else:
        from neurobrix.core.cfg.engine import _synthesize_position_ids
    ctx = _Ctx({})
    with pytest.raises(RuntimeError, match="ZERO FALLBACK"):
        _synthesize_position_ids(ctx, COMP_INPUTS, COMP_SHAPES)
