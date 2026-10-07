"""Unit test — `conv2d` on the m8n8k4 matrix unit holds the contract of `conv2d_forward_kernel`.

Where the hardware profile declares `matrix_unit` with `mm` tiles (volta.yml), a convolution whose input and weight
are both the unit's operand dtype in memory runs the implicit GEMM `kernels/ops/conv2d_m8n8k4.py`. Each clause is
proven against an fp64 oracle computed in-test with the route asserted TAKEN: padding, stride, dilation, groups,
every M/N/K tail, a strided (non-contiguous) input, the fused bias, the output dtype. An fp32 operand must NOT take
the route, and the derived census lists no key where it runs. Skipped where the profile declares no matrix unit.

  PYTHONPATH=src python3 -m pytest tests/unit/kernels/test_conv2d_m8n8k4.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from neurobrix.kernels import wrappers as W
from neurobrix.kernels.nbx_tensor import DeviceAllocator, NBXTensor


def _unit_declared() -> bool:
    try:
        DeviceAllocator.set_device(0)
        return bool((W._matrix_unit() or {}).get("mm"))
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _unit_declared(), reason="the hardware profile declares no matrix_unit.mm")


@pytest.fixture(autouse=True, scope="module")
def _profile():
    from neurobrix.kernels import autotune_certify as Z
    Z._bind_hardware_profile()


@pytest.fixture
def taken(monkeypatch):
    calls = []
    real = W._conv2d_m8n8k4
    monkeypatch.setattr(W, "_conv2d_m8n8k4", lambda *a, **kw: (calls.append(a[0].shape), real(*a, **kw))[1])
    return calls


def _nb(x):
    return NBXTensor.from_numpy(np.ascontiguousarray(x)).to("cuda:0")


def _ref(x, w, b, s, p, d, g):
    """fp64 direct convolution (NCHW, OIHW)."""
    x, w = x.astype(np.float64), w.astype(np.float64)
    N, C, H, Wd = x.shape
    O, Cg, KH, KW = w.shape
    xp = np.pad(x, ((0, 0), (0, 0), (p[0], p[0]), (p[1], p[1])))
    OH = (H + 2 * p[0] - d[0] * (KH - 1) - 1) // s[0] + 1
    OW = (Wd + 2 * p[1] - d[1] * (KW - 1) - 1) // s[1] + 1
    out = np.zeros((N, O, OH, OW))
    Og = O // g
    for gi in range(g):
        for r in range(KH):
            for c in range(KW):
                patch = xp[:, gi * Cg:(gi + 1) * Cg, r * d[0]: r * d[0] + s[0] * (OH - 1) + 1: s[0],
                           c * d[1]: c * d[1] + s[1] * (OW - 1) + 1: s[1]]
                out[:, gi * Og:(gi + 1) * Og] += np.einsum("nchw,oc->nohw", patch, w[gi * Og:(gi + 1) * Og, :, r, c])
    if b is not None:
        out += b.astype(np.float64)[None, :, None, None]
    return out


CASES = [
    # N, C, H, W, O, KH, KW, stride, pad, dil, groups
    (2, 64, 32, 32, 128, 3, 3, (1, 1), (1, 1), (1, 1), 1),       # the VAE/UNet 3x3
    (1, 37, 19, 23, 50, 3, 3, (1, 1), (1, 1), (1, 1), 1),        # every tail
    (2, 32, 33, 31, 64, 3, 3, (2, 2), (1, 1), (1, 1), 1),        # downsample
    (1, 48, 20, 20, 96, 1, 1, (1, 1), (0, 0), (1, 1), 1),        # 1x1 projection
    (1, 16, 24, 24, 32, 3, 3, (1, 1), (2, 2), (2, 2), 1),        # dilation
    (2, 64, 16, 16, 64, 3, 3, (1, 1), (1, 1), (1, 1), 4),        # groups
    (1, 3, 64, 64, 96, 4, 4, (4, 4), (0, 0), (1, 1), 1),         # patch embed, K = 48
    (1, 24, 9, 40, 40, 1, 7, (1, 2), (0, 3), (1, 1), 1),         # rectangular kernel, mixed stride
]


@pytest.mark.parametrize("N,C,H,Wd,O,KH,KW,s,p,d,g", CASES)
@pytest.mark.parametrize("bias_dtype", [None, np.float16, np.float32])
def test_conv2d_matches_fp64(N, C, H, Wd, O, KH, KW, s, p, d, g, bias_dtype, taken):
    rng = np.random.default_rng(N * C + H + O + KH)
    x = (rng.standard_normal((N, C, H, Wd)) * 0.5).astype(np.float16)
    w = (rng.standard_normal((O, C // g, KH, KW)) * 0.3).astype(np.float16)
    b = None if bias_dtype is None else rng.standard_normal(O).astype(bias_dtype)
    out = W.conv2d_wrapper(_nb(x), _nb(w), None if b is None else _nb(b), stride=list(s), padding=list(p),
                           dilation=list(d), groups=g).numpy()
    assert taken, "the m8n8k4 route was not taken"
    ref = _ref(x, w, b, s, p, d, g)
    assert out.shape == ref.shape
    rel = np.abs(out.astype(np.float64) - ref).max() / np.abs(ref).max()
    assert rel < max(float(np.finfo(out.dtype).eps), 1e-5), f"rel {rel:.2e} dtype {out.dtype}"


def test_a_strided_input_is_read_through_its_strides(taken):
    rng = np.random.default_rng(11)
    x = (rng.standard_normal((1, 40, 18, 18)) * 0.5).astype(np.float16)
    w = (rng.standard_normal((24, 20, 3, 3)) * 0.3).astype(np.float16)
    xs = _nb(x).narrow(1, 10, 20)                                    # a channel slice: non-contiguous view
    out = W.conv2d_wrapper(xs, _nb(w), None, stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1).numpy()
    assert taken
    ref = _ref(x[:, 10:30], w, None, (1, 1), (1, 1), (1, 1), 1)
    rel = np.abs(out.astype(np.float64) - ref).max() / np.abs(ref).max()
    assert rel < max(float(np.finfo(out.dtype).eps), 1e-5), f"rel {rel:.2e}"


def test_an_fp32_operand_keeps_the_exact_path(taken):
    rng = np.random.default_rng(3)
    x = rng.standard_normal((1, 8, 12, 12)).astype(np.float32) * 70000.0     # beyond fp16
    w = rng.standard_normal((8, 8, 3, 3)).astype(np.float32)
    got = W.conv2d_wrapper(_nb(x), _nb(w), None, stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1).numpy()
    assert not taken, "an fp32 operand took the fp16 matrix unit"
    assert np.isfinite(got).all()


def test_the_census_derives_no_key_where_the_unit_runs():
    from neurobrix.kernels import launch_keys as LK
    from neurobrix.kernels.nbx_tensor import NBXDtype
    h, f = NBXDtype.float16, NBXDtype.float32
    args = (1, 64, 32, 32, 128, 3, 3, 1, 1, 1, 1, 1, 1, 1)
    assert LK.conv2d_launches(*args, h, h, None, 1 << 32) == []
    assert LK.conv2d_launches(*args, f, f, None, 1 << 32) != []
