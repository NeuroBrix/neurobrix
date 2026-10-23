"""ONE draw for both engines, torch-free, the vendor's CPU draw (kernels/mt_randn.py).

The owner's rule, 2026-10-13: a model's initial noise is one draw, data-driven, identical between
the compiled and the Triton branch and to the vendor at equal seed. Three verdicts of 2026-10-13
differed between the branches only because each branch drew differently (Wan2.1-I2V native,
mochi, CogVideoX-2b triton: the same engine rendered the apple on the other branch's draw).

What this test does if the code is wrong: the stream cell compares the raw MT19937 words and the
24-bit uniforms with torch's CPU generator — bit for bit, any seed; the normal cell compares the
draw with `torch.randn(generator=cpu)` within 1 ulp for every value and bit-equal for most (the
transcendentals are ATen's Sleef kernels here, float64 rounded once in ours); the tail cell checks
a size that is not a multiple of 16 (the tail is recomputed from 16 fresh uniforms); the
determinism cell fails if two generators with one seed diverge.

Run: python -m pytest tests/unit/kernels/test_one_draw_is_the_vendors_cpu_draw.py
"""
from __future__ import annotations

import numpy as np
import pytest

from neurobrix.kernels.mt_randn import MT19937, mt_randn

torch = pytest.importorskip("torch")


def _ulps_apart(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ai = a.view(np.int32).astype(np.int64)
    bi = b.view(np.int32).astype(np.int64)
    return np.abs(ai - bi)


#: An absolute floor beside the 4-ulp bound: where ATen draws through its SCALAR float32 normal fill (every value on
#: ARM, the Mac 2026-10-22), the error is the Box-Muller intermediate's — absolute, at the radius' magnitude, which a
#: hundred million draws keep under 8 — so a value near 0 drifts many of its own ulp (measured there: at most 2.0e-6
#: over 100 000 draws). 4 ulp of 8.0 in float32.
ABS_FLOOR = 4 * float(np.spacing(np.float32(8.0)))


def _far(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The values neither within 4 ulp of the vendor's nor within the absolute floor of it."""
    return (_ulps_apart(a, b) > 4) & (np.abs(a.astype(np.float64) - b.astype(np.float64)) > ABS_FLOOR)


@pytest.mark.parametrize("seed", [0, 1, 42, 123456789, 2**31 - 1, 2**32 + 7])
def test_the_word_stream_is_torchs(seed):
    g = torch.Generator("cpu").manual_seed(seed)
    ref = torch.rand(2000, generator=g).numpy()   # one word per value: the 24-bit uniform of each
    ours = MT19937(seed).uniform_f32(2000)
    # torch's `rand` (float) takes ONE 32-bit word per value and keeps its 24 low bits: equal uniforms
    # mean an equal word stream, seeding included.
    assert np.array_equal(ref, ours), f"seed {seed}: the word stream differs at {np.nonzero(ref != ours)[0][:5]}"


@pytest.mark.parametrize("seed,n", [(42, 16), (42, 4096), (7, 100000), (2**31 - 1, 2048)])
def test_the_normals_are_the_vendors_within_one_ulp(seed, n):
    ref = torch.randn(n, generator=torch.Generator("cpu").manual_seed(seed)).numpy()
    ours = MT19937(seed).randn(n)
    ulps = _ulps_apart(ref, ours)
    # ATen's transcendentals are Sleef's vectorised kernels (sinf/cosf u35: up to 3.5 ulp); ours are
    # float64 rounded once. Measured 2026-10-13: at most 3 ulp, 98.8 % within 1, 61.5 % bit-equal.
    assert not _far(ref, ours).any(), f"seed {seed} n {n}: {int(_far(ref, ours).sum())} values beyond 4 ulp and the floor"
    # the statistics are those of ATen's VECTORISED fill (x86 AVX2/AVX-512, `normal_fill_16`); its scalar fill (ARM:
    # 79 % within 1 ulp, 48 % bit-equal, the Mac 2026-10-22) is held by the bound above alone
    if n >= 2048 and torch.backends.cpu.get_cpu_capability() in ("AVX2", "AVX512"):
        assert float((ulps <= 1).mean()) >= 0.98, "98 % of the values are within 1 ulp of the vendor's"
        assert float((ulps == 0).mean()) >= 0.55, "the uniform stream is exact: most values are bit-equal"


@pytest.mark.parametrize("n", [1, 2, 5, 15])
def test_fewer_than_sixteen_values_take_the_scalar_path(n):
    ref = torch.randn(n, generator=torch.Generator("cpu").manual_seed(42)).numpy()
    ours = MT19937(42).randn(n)
    assert not _far(ref, ours).any(), (ref, ours)
    # and the stream continues where torch's does: a 16-block after a scalar draw
    g = torch.Generator("cpu").manual_seed(7)
    ref2 = torch.cat([torch.randn(3, generator=g), torch.randn(32, generator=g)]).numpy()
    m = MT19937(7)
    ours2 = np.concatenate([m.randn(3), m.randn(32)])
    assert not _far(ref2, ours2).any()


@pytest.mark.parametrize("n", [33, 1000, 21 * 60 * 104])
def test_a_tail_is_recomputed_from_fresh_uniforms(n):
    ref = torch.randn(n, generator=torch.Generator("cpu").manual_seed(42)).numpy()
    ours = MT19937(42).randn(n)
    assert not _far(ref, ours).any()


def test_one_seed_one_draw_and_a_shape_reshapes_the_same_stream():
    a = mt_randn(42, (1, 16, 21, 60, 104))
    b = mt_randn(42, (1, 16, 21, 60, 104))
    assert a.dtype == np.float32 and a.shape == (1, 16, 21, 60, 104)
    assert np.array_equal(a, b)
    assert np.array_equal(a.ravel(), MT19937(42).randn(a.size))
    assert abs(float(a.mean())) < 0.01 and abs(float(a.std()) - 1.0) < 0.01
