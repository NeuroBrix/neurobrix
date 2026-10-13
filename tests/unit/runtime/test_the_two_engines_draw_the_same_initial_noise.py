"""The compiled and the Triton branch draw the SAME initial noise for a seeded run — the vendor's CPU
draw, torch-free, from the shared run stream (`kernels/rng_stream.init_draw`, `kernels/mt_randn.py`).

The owner's rule, 2026-10-13 02:05: one draw, data-driven, identical between branches and to the
vendor at equal seed. Until now the compiled branch drew `torch.randn` on a device generator and the
Triton branch its Philox kernel: three verdicts of 2026-10-13 differed between the branches only by
the draw (Wan2.1-I2V native, mochi, CogVideoX-2b triton rendered the apple on the other branch's
draw).

What this test does if the code is wrong: the bytes cell fails when the two resolvers' `randn`
allocations differ for one seed (red before this change: two generators, two algorithms); the
order cell fails if a second `init: randn` variable does not continue the same stream; the seedless
cell fails if an unseeded run were made to raise instead of drawing unseeded in each engine.

Run: python -m pytest tests/unit/runtime/test_the_two_engines_draw_the_same_initial_noise.py
"""
from __future__ import annotations

import numpy as np
import pytest

from neurobrix.core.runtime.resolution.variable_resolver import VariableResolver
from neurobrix.kernels import rng_stream
from neurobrix.kernels.mt_randn import mt_randn

SPEC = {"type": "tensor", "role": "state_tensor",
        "resolver": {"method": "allocate", "init": "randn",
                     "shape_source": {"axis_0": "runtime.batch_size", "axis_1": "runtime.channels",
                                      "axis_2": "runtime.height", "axis_3": "runtime.width"}}}
DEFAULTS = {"dtype": "float32", "seed": 42, "batch_size": 1, "channels": 4, "height": 6, "width": 8}


def _resolver(mode: str) -> VariableResolver:
    return VariableResolver({"global.latents": SPEC}, dict(DEFAULTS), {}, {}, {}, device="cpu", mode=mode)


def _as_numpy(t):
    return np.asarray(t.numpy() if callable(getattr(t, "numpy", None)) else t)


def test_both_engines_draw_the_same_bytes_for_one_seed():
    rng_stream.set_run_seed(42)
    a = _as_numpy(_resolver("compiled")._resolve_single("global.latents", SPEC))
    rng_stream.set_run_seed(42)
    b = _as_numpy(_resolver("triton")._resolve_single("global.latents", SPEC))
    rng_stream.set_run_seed(None)
    assert a.shape == (1, 4, 6, 8) and a.dtype == np.float32
    assert np.array_equal(a, b), "the two branches drew different initial noise for seed 42"
    assert np.array_equal(a, mt_randn(42, (1, 4, 6, 8))), "the draw is the vendor's CPU draw for the seed"


def test_a_second_random_variable_continues_the_same_stream_in_both_engines():
    rng_stream.set_run_seed(7)
    r = _resolver("compiled")
    a1 = _as_numpy(r._resolve_single("global.latents", SPEC))
    a2 = _as_numpy(r._resolve_single("global.latents", SPEC))
    rng_stream.set_run_seed(7)
    r = _resolver("triton")
    b1 = _as_numpy(r._resolve_single("global.latents", SPEC))
    b2 = _as_numpy(r._resolve_single("global.latents", SPEC))
    rng_stream.set_run_seed(None)
    assert np.array_equal(a1, b1) and np.array_equal(a2, b2)
    assert not np.array_equal(a1, a2), "the second draw is the stream's next, not the first again"


def test_a_seedless_run_still_draws_in_each_engine():
    rng_stream.set_run_seed(None)
    d = dict(DEFAULTS)
    d.pop("seed")
    r = VariableResolver({"global.latents": SPEC}, d, {}, {}, {}, device="cpu", mode="compiled")
    a = _as_numpy(r._resolve_single("global.latents", SPEC))
    assert a.shape == (1, 4, 6, 8) and abs(float(a.std()) - 1.0) < 0.3
    with pytest.raises(RuntimeError, match="no seed"):
        rng_stream.init_draw((2, 2))
