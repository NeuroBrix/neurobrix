"""A 1-D transposed convolution (groups 1, dilation 1) runs as a GEMM and an overlap-add (`conv_wrappers.
_conv_transpose1d_gemm`): the vocoders' upsamplers ran 20-307x slower through the scatter kernel (768 -> 384 channels,
K 16, stride 8 on a V100: 103 ms against 0.34 ms; torch eager 0.21 ms — the kernel-level benchmark, 2026-11-03).
Judged against an fp64 oracle with the route asserted taken; a grouped one keeps the scatter kernel; the census derives
the GEMM's keys from the same rule. What this test would do if the code were wrong: the scatter kernel still serving
it (route not taken), a wrong tap or a wrong input position in the overlap-add (parity), the census missing the
GEMM's key — RED. Needs a device."""
from __future__ import annotations

import numpy as np
import pytest

from nbxtensor.memory.allocator import DeviceAllocator
from nbxtensor.tensor import NBXTensor
from nbxtensor.ops import conv_wrappers as CW


def _has_gpu():
    try:
        DeviceAllocator.set_device(0)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_gpu(), reason="needs a device")


@pytest.fixture(autouse=True, scope="module")
def _profile():
    from neurobrix.kernels import autotune_certify as Z
    Z._bind_hardware_profile()


def _ref(x, w, b, s, p):
    N, Cin, L = x.shape
    _, Cout, K = w.shape
    Lo = (L - 1) * s - 2 * p + K
    full = np.zeros((N, Cout, (L - 1) * s + K))
    for i in range(L):
        full[:, :, i * s:i * s + K] += np.einsum("nc,cok->nok", x[:, :, i].astype(np.float64), w.astype(np.float64))
    out = full[:, :, p:p + Lo]
    return out + (0 if b is None else b.astype(np.float64)[None, :, None])


@pytest.mark.parametrize("N,Cin,L,Cout,K,s,p,bias", [
    (1, 96, 37, 48, 16, 8, 4, False),       # the upsampler's ratio, reduced
    (2, 64, 23, 32, 10, 5, 3, True),        # a batch, a bias, K not a multiple of the stride
    (1, 32, 50, 16, 8, 4, 2, False),
    (1, 16, 9, 8, 3, 1, 0, True),           # stride 1
])
def test_parity_and_the_route(N, Cin, L, Cout, K, s, p, bias, monkeypatch):
    calls = []
    real = CW._conv_transpose1d_gemm
    monkeypatch.setattr(CW, "_conv_transpose1d_gemm", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    rng = np.random.default_rng(Cin + L)
    x = (rng.standard_normal((N, Cin, L)) * 0.5).astype(np.float16)
    w = (rng.standard_normal((Cin, Cout, K)) / np.sqrt(Cin)).astype(np.float16)
    b = rng.standard_normal(Cout).astype(np.float16) if bias else None
    to = lambda a: NBXTensor.from_numpy(np.ascontiguousarray(a)).to("cuda:0")   # noqa: E731
    got = CW.conv_transpose_wrapper(to(x), to(w), None if b is None else to(b), stride=[s], padding=[p]).numpy()
    assert calls, "the GEMM route was not taken"
    r = _ref(x, w, b, s, p)
    assert got.shape == r.shape
    assert np.abs(got.astype(np.float64) - r).max() / np.abs(r).max() < 2e-3


def test_a_grouped_transposed_conv_keeps_the_scatter_kernel(monkeypatch):
    calls = []
    monkeypatch.setattr(CW, "_conv_transpose1d_gemm", lambda *a, **k: calls.append(1))
    rng = np.random.default_rng(1)
    x = rng.standard_normal((1, 8, 12)).astype(np.float16)
    w = rng.standard_normal((8, 1, 4)).astype(np.float16)        # depthwise (groups 8): Kokoro's F0/N pool
    to = lambda a: NBXTensor.from_numpy(np.ascontiguousarray(a)).to("cuda:0")   # noqa: E731
    CW.conv_transpose_wrapper(to(x), to(w), None, stride=[2], padding=[1], groups=8)
    assert not calls


def test_the_census_derives_the_gemm_s_key():
    from nbxtensor.ops import launch_keys as LK
    from nbxtensor.dtype.nbx_dtype import NBXDtype
    h = NBXDtype.float16
    got = LK.conv_launches([1, 32, 50], [32, 1, 3], [4], [2], [1], True, 1, h, h, None, 1 << 32, native_bf16=False)
    assert got == LK.mm_launches(3, 32, 50, h, h, False)          # M = C_out * K = 3: below the unit, keyed
    assert LK.conv_launches([1, 32, 50], [32, 1, 3], [4], [2], [1], True, 32, h, h, None, 1 << 32, native_bf16=False) == []
