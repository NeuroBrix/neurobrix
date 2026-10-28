"""`NBX_FORCE_RAND_SEED` (nbxtensor/random/rng_pin) is a diagnostic contract of BOTH engines: under it the Euler ancestral
schedulers of the compiled and the Triton engine add the SAME noise at a step, so a pinned cross-engine drift past the
first stochastic step compares kernels, not two noise streams. The Mac, 2026-10-25 02:43: the pinned 2-step drift of
Allegro-TI2V parted exactly at the transformer's second-step input (-413 700 vs -455 092) — the Triton scheduler drew
from the pin, the compiled one from the run stream.

What this test would do if the code were wrong: with the compiled scheduler ignoring the pin (the behaviour before),
its noise differs from the pinned draw and the first assertion fails. Injection seen red on 2026-10-25, restored green.
"""
import numpy as np
import pytest

from nbxtensor.random import rng_pin

CFG = {"num_train_timesteps": 1000, "beta_start": 0.00085, "beta_end": 0.012, "beta_schedule": "scaled_linear",
       "prediction_type": "epsilon", "timestep_spacing": "linspace"}


def _pin(monkeypatch, seed):
    monkeypatch.setenv("NBX_FORCE_RAND_SEED", str(seed))
    rng_pin._STATE = None
    rng_pin._SEED_USED = "<unset>"


def _step(torch, cls, eps, t, sample):
    sch = cls(CFG); sch.set_timesteps(4)
    out = sch.step(eps, t, sample, return_dict=False)
    return out[0] if isinstance(out, (tuple, list)) else out


def test_the_compiled_scheduler_adds_the_pinned_noise(monkeypatch):
    torch = pytest.importorskip("torch")
    from neurobrix.core.module.scheduler.diffusion.euler import EulerAncestralDiscreteScheduler as E
    rng = np.random.default_rng(0)
    sample = torch.from_numpy(rng.standard_normal((1, 2, 4, 4)).astype(np.float32))
    eps = torch.from_numpy(rng.standard_normal((1, 2, 4, 4)).astype(np.float32))
    t = E(CFG).__class__(CFG); t.set_timesteps(4); t = t.timesteps[0]
    _pin(monkeypatch, 7); p7 = _step(torch, E, eps, t, sample)
    _pin(monkeypatch, 7); again = _step(torch, E, eps, t, sample)
    assert torch.equal(p7, again)                                        # pinned: reproducible across instances
    _pin(monkeypatch, 7); n7 = rng_pin.pinned_normal((1, 2, 4, 4))
    _pin(monkeypatch, 11); n11 = rng_pin.pinned_normal((1, 2, 4, 4))
    _pin(monkeypatch, 11); p11 = _step(torch, E, eps, t, sample)
    # the two pinned runs differ by sigma_up * (n7 - n11) and nothing else: one amplitude over every element
    ratio = (p7 - p11).numpy().astype(np.float64) / (n7 - n11).astype(np.float64)
    assert np.allclose(ratio, ratio.flat[0], rtol=1e-3) and ratio.flat[0] > 0, ratio


def test_both_engines_add_the_same_pinned_noise(monkeypatch):
    torch = pytest.importorskip("torch")
    from neurobrix.core.module.scheduler.diffusion.euler import EulerAncestralDiscreteScheduler
    from neurobrix.triton.scheduler.euler import TritonEulerAncestralDiscreteScheduler
    from neurobrix.kernels.nbx_tensor import NBXTensor
    rng = np.random.default_rng(1)
    sample = rng.standard_normal((1, 2, 4, 4)).astype(np.float32); eps = rng.standard_normal((1, 2, 4, 4)).astype(np.float32)
    c = EulerAncestralDiscreteScheduler(CFG); c.set_timesteps(4)
    _pin(monkeypatch, 5)
    a = c.step(torch.from_numpy(eps), c.timesteps[0], torch.from_numpy(sample), return_dict=False)
    a = (a[0] if isinstance(a, (tuple, list)) else a).numpy()
    tr = TritonEulerAncestralDiscreteScheduler(CFG); tr.set_timesteps(4)
    _pin(monkeypatch, 5)
    try:
        b = tr.step(NBXTensor.from_numpy(eps), int(c.timesteps[0]), NBXTensor.from_numpy(sample), return_dict=False)
    except Exception as exc:                                             # the Triton scheduler needs a device for its ops
        pytest.skip(f"the Triton scheduler does not step on the host here: {type(exc).__name__}: {exc}")
    b = (b[0] if isinstance(b, (tuple, list)) else b).numpy()
    assert np.allclose(a, b, atol=1e-4), np.abs(a - b).max()
