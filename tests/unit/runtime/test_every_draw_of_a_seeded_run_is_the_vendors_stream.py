"""Every draw of a seeded run is the vendor's CPU stream, continued, in both engines (nbxtensor/random/rng_stream).

The vendor threads one generator through every draw of a run: the image encoder's posterior sample, the initial
noise, each ancestral step's noise. Measured 2026-10-14: Allegro-TI2V at its gate request rendered the same clip in
both engines at one step (mean |a - b| under 1 of 255 per frame) and different content at ten — the Euler Ancestral
step noise was the only difference (torch's device generator vs Philox); CogVideoX-5b-I2V's first op departing
between the kernel oracle and the Triton branch was the posterior sample `vae_encoder/aten.randn_like::0`.

What this test does if the code is wrong: the stream cell fails if next_normal / next_uniform do not continue the
generator the initial draw started (torch's CPU randn / rand for the seed); the compiled graph-op cell fails if
`run_generator.draw` does not take the stream (red before this change: the run's torch generator); the compiled
ancestral cell fails if the step's noise is not the stream's next draw (red before: torch.randn on the generator);
the seedless cell fails if an unseeded run stops drawing; the Triton cells (a card) fail if the Triton randn_like /
the Triton ancestral step do not take the same values as the compiled branch (red before: Philox).

Run: python -m pytest tests/unit/runtime/test_every_draw_of_a_seeded_run_is_the_vendors_stream.py
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from nbxtensor.random import rng_stream
from nbxtensor.random.mt_randn import MT19937

CFG = {"num_train_timesteps": 1000, "beta_start": 0.0001, "beta_end": 0.02, "beta_schedule": "linear",
       "prediction_type": "epsilon", "timestep_spacing": "linspace", "steps_offset": 0}
SHAPE = (1, 4, 3, 8, 8)
card = pytest.mark.skipif(os.environ.get("CUDA_VISIBLE_DEVICES", None) == "" or not torch.cuda.is_available(),
                          reason="needs a card")


@pytest.fixture(autouse=True)
def _disarm():
    yield
    rng_stream.set_run_seed(None)


def test_the_stream_continues_the_initial_draws_generator():
    rng_stream.set_run_seed(42)
    a = rng_stream.init_draw((2, 16))
    b = rng_stream.next_normal((3, 16))
    u = rng_stream.next_uniform((32,))
    m = MT19937(42)
    assert np.array_equal(a.ravel(), m.randn(32)) and np.array_equal(b.ravel(), m.randn(48))
    assert np.array_equal(u, m.uniform_f32(32))


def test_a_compiled_graph_random_op_draws_from_the_stream():
    from neurobrix.core.runtime.graph import run_generator
    rng_stream.set_run_seed(7)
    run_generator.arm(lambda: torch.Generator("cpu").manual_seed(999))
    try:
        x = run_generator.draw("randn_like", (2, 16), torch.float32, "cpu")
        r = run_generator.draw("rand", (16,), torch.float32, "cpu")
    finally:
        run_generator.arm(None)
    m = MT19937(7)
    assert np.array_equal(x.numpy().ravel(), m.randn(32)), "the graph's randn is the stream's next normals"
    assert np.array_equal(r.numpy(), m.uniform_f32(16))


def _compiled_step(seed, sample, eps):
    from neurobrix.core.module.scheduler.diffusion.euler import EulerAncestralDiscreteScheduler
    s = EulerAncestralDiscreteScheduler(dict(CFG))
    s.set_timesteps(10, device="cpu")
    rng_stream.set_run_seed(seed)
    out = s.step(torch.from_numpy(eps), s.timesteps[0], torch.from_numpy(sample),
                 generator=torch.Generator("cpu").manual_seed(999))
    return (out["prev_sample"] if isinstance(out, dict) else out).numpy(), s


def test_the_compiled_ancestral_noise_is_the_streams_next_draw():
    rs = np.random.RandomState(0)
    sample, eps = rs.randn(*SHAPE).astype(np.float32), rs.randn(*SHAPE).astype(np.float32)
    got, s = _compiled_step(42, sample, eps)
    sig, sig_n = float(s.sigmas[0]), float(s.sigmas[1])
    up = (sig_n ** 2 * (sig ** 2 - sig_n ** 2) / sig ** 2) ** 0.5
    down = (sig_n ** 2 - up ** 2) ** 0.5
    denoised = sample - sig * eps
    want = sample + (sample - denoised) / sig * (down - sig) + up * MT19937(42).randn(sample.size).reshape(SHAPE)
    assert np.allclose(got, want, atol=1e-5), float(np.abs(got - want).max())


def test_a_seedless_run_still_draws():
    from neurobrix.core.runtime.graph import run_generator
    rng_stream.set_run_seed(None)
    run_generator.arm(lambda: torch.Generator("cpu").manual_seed(3))
    try:
        x = run_generator.draw("randn", (4, 16), torch.float32, "cpu")
    finally:
        run_generator.arm(None)
    assert x.shape == (4, 16) and np.array_equal(x.numpy(), torch.randn(4, 16, generator=torch.Generator("cpu").manual_seed(3)).numpy())


@card
def test_the_triton_randn_like_takes_the_compiled_branchs_values():
    from nbxtensor.tensor import NBXDtype, NBXTensor
    from nbxtensor.ops.wrappers import randn_like_wrapper
    from neurobrix.core.runtime.graph import run_generator
    rng_stream.set_run_seed(11)
    t = randn_like_wrapper(NBXTensor.from_numpy(np.zeros((2, 16), np.float32))).numpy()
    rng_stream.set_run_seed(11)
    run_generator.arm(lambda: None)
    try:
        c = run_generator.draw("randn_like", (2, 16), torch.float32, "cpu").numpy()
    finally:
        run_generator.arm(None)
    assert np.array_equal(t, c)


@card
def test_the_two_engines_ancestral_steps_agree():
    from nbxtensor.tensor import NBXTensor
    from neurobrix.triton.scheduler.euler import TritonEulerAncestralDiscreteScheduler
    rs = np.random.RandomState(1)
    sample, eps = rs.randn(*SHAPE).astype(np.float32), rs.randn(*SHAPE).astype(np.float32)
    want, s = _compiled_step(5, sample, eps)
    ts = TritonEulerAncestralDiscreteScheduler(dict(CFG))
    ts.set_timesteps(10)
    rng_stream.set_run_seed(5)
    out = ts.step(NBXTensor.from_numpy(eps), ts.timesteps[0] if hasattr(ts, "timesteps") else s.timesteps[0],
                  NBXTensor.from_numpy(sample))
    got = (out["prev_sample"] if isinstance(out, dict) else out).numpy()
    assert np.allclose(got, want, atol=1e-4), float(np.abs(got - want).max())
