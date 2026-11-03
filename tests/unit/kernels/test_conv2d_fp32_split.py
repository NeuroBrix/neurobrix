"""Unit test — an fp32 convolution on the m8n8k4 matrix unit through the hi + lo split loses nothing against the fp32 path.

Where the hardware profile declares `matrix_unit.fp32_split` (volta.yml), the convolution's implicit GEMM
(ops/conv2d_m8n8k4.py) carries an fp32 operand as the GEMM does (ops/matmul_m8n8k4.py): hi + lo of the unit's operand
dtype under a power-of-two scale per row of A (an output position) and per column of B (an output channel) of each K
tile, three MMAs, fp32 accumulation. The fp32 convolutions are the video VAEs' (CogVideoX, Allegro, Wan, Sana 4K: the
census's 2 920 fp32 conv2d rows on a V100-16GB), measured at 63 % (CogVideoX-2b) and 51 % (Allegro) of the GPU time
of a gate request on the FMA path (nsys, 2026-11-03). The split was MEASURED faster than the FMA path only for a 1x1
window whose tile fills (x1.26-1.95; a 3x3 is slower at every tile: volta.yml `fp32_split.conv`), so only such a
convolution takes it (`launch_keys.matrix_unit_conv_operands`); a 3x3 or a narrow 1x1 keeps the FMA path. Each case is judged against an fp64 oracle computed in-test AND
against the FMA path the operands took before (the unit withheld): the split's error, per output position against
that position's own magnitude, must not exceed the FMA path's beyond four fp32 epsilons. What this test would do if
the code were wrong: a 1x1 fp32 convolution left on the FMA path, or a 3x3 sent to the slower split — RED; a split
without its lo term or its scales (fp16-level error, overflow beyond 65504) — RED. Skipped where the profile declares no split.

  PYTHONPATH=src python3 -m pytest tests/unit/kernels/test_conv2d_fp32_split.py -v
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

from nbxtensor.ops import wrappers as W
from nbxtensor.ops import conv_wrappers as CW
from nbxtensor.memory.allocator import DeviceAllocator
from nbxtensor.tensor import NBXTensor
from tests.unit.kernels.test_conv2d_m8n8k4 import _ref


def _no_gpu_reason():
    """Why these cells cannot run here, or None: only the absence of a device or of the profile's split rows skips."""
    from nbxtensor.backend.runtime import gpu_runtime
    try:
        gpu_runtime()
    except (OSError, RuntimeError) as exc:
        return f"no GPU runtime: {exc}"
    if DeviceAllocator.device_count() < 1:
        return "the process sees no device"
    DeviceAllocator.set_device(0)
    if not ((W._matrix_unit() or {}).get("fp32_split") or {}).get("mm"):
        return "the hardware profile declares no matrix_unit.fp32_split"
    return None


_SKIP = _no_gpu_reason()
pytestmark = pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "")

_SLACK = 4 * float(np.finfo(np.float32).eps)


@pytest.fixture(autouse=True, scope="module")
def _profile():
    from neurobrix.kernels import autotune_certify as Z
    Z._bind_hardware_profile()


def _nb(x):
    return NBXTensor.from_numpy(np.ascontiguousarray(x)).to("cuda:0")


def _per_position(got, ref):
    """max over output positions of |got - ref| / max|ref| over that position's channels."""
    g = np.moveaxis(got, 1, -1).reshape(-1, got.shape[1])
    r = np.moveaxis(ref, 1, -1).reshape(-1, ref.shape[1])
    return float((np.abs(g - r).max(axis=1) / np.maximum(np.abs(r).max(axis=1), 1e-300)).max())


def _both(run, monkeypatch):
    """(split result, the split flags it ran with), FMA-path result — the second with the matrix unit withheld."""
    calls = []
    real = CW._conv2d_m8n8k4
    monkeypatch.setattr(CW, "_conv2d_m8n8k4", lambda *a, **kw: (calls.append(kw.get("split")), real(*a, **kw))[1])
    split = run().numpy().astype(np.float64)
    assert calls, "the m8n8k4 route was not taken"
    flags = calls[0]
    from nbxtensor.ops import _configs as C
    monkeypatch.setattr(C, "matrix_unit", lambda: {})
    for _m in (W, *(importlib.import_module(f"nbxtensor.ops.{f}") for f in W._FAMILY_MODULES)):
        monkeypatch.setattr(_m, "_matrix_unit", lambda: {})
    calls.clear()
    fma = run().numpy().astype(np.float64)
    assert not calls
    return split, flags, fma


def _judge(split, fma, ref):
    es, ef = _per_position(split, ref), _per_position(fma, ref)
    assert np.isfinite(split).all()
    assert es <= ef + _SLACK, f"split {es:.2e} vs fma {ef:.2e} (per output position, relative to fp64)"


CASES = [
    # N, C, H, W, O, KH, KW, stride, pad, dil, groups, weight dtype — the census's fp32 1x1 classes, reduced in extent
    (1, 384, 30, 52, 1152, 1, 1, (1, 1), (0, 0), (1, 1), 1, np.float32),   # 384 -> 1152 (x1.95 measured)
    (1, 384, 30, 52, 384, 1, 1, (1, 1), (0, 0), (1, 1), 1, np.float32),    # 384 -> 384 (x1.58)
    (2, 192, 40, 52, 384, 1, 1, (1, 1), (0, 0), (1, 1), 1, np.float32),    # 192 -> 384 (even)
    (1, 512, 40, 40, 256, 1, 1, (1, 1), (0, 0), (1, 1), 1, np.float32),    # 512 -> 256 (x1.53)
    (1, 256, 37, 41, 256, 1, 1, (1, 1), (0, 0), (1, 1), 1, np.float32),    # every M tail
]

FMA_CASES = [
    (1, 128, 48, 48, 128, 3, 3, (1, 1), (1, 1), (1, 1), 1),     # a VAE 3x3: the split is slower, no row
    (1, 512, 20, 24, 512, 3, 3, (1, 1), (1, 1), (1, 1), 1),     # the mid block's 3x3
    (1, 16, 60, 104, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1),      # a 1x1 whose 16 channels fill a quarter of the tile (x0.36)
]


@pytest.mark.parametrize("N,C,H,Wd,O,KH,KW,s,p,d,g,wdt", CASES)
def test_census_shapes(N, C, H, Wd, O, KH, KW, s, p, d, g, wdt, monkeypatch):
    rng = np.random.default_rng(N * C + H + O)
    x = rng.standard_normal((N, C, H, Wd)).astype(np.float32)
    w = (rng.standard_normal((O, C // g, KH, KW)) / np.sqrt(C * KH * KW / g)).astype(wdt)
    b = rng.standard_normal(O).astype(np.float32)
    run = lambda: CW.conv2d_wrapper(_nb(x), _nb(w), _nb(b), stride=list(s), padding=list(p), dilation=list(d), groups=g)
    from nbxtensor.ops import launch_keys as LK
    from nbxtensor.dtype.nbx_dtype import NBXDtype
    xd, wd = LK.conv_dtypes(NBXDtype.float32, NBXDtype.float16 if wdt == np.float16 else NBXDtype.float32)
    ref = _ref(x, w, b, s, p, d, g)
    reps = LK.matrix_unit_operands(xd, wd, ref.shape[0] * ref.shape[2] * ref.shape[3])   # the launcher's own operand rule
    split, flags, fma = _both(run, monkeypatch)
    assert flags == (reps[0] == "split", reps[1] == "split")
    _judge(split, fma, ref)


_EDGES = {
    "beyond fp16 (1e6)": lambda r, s: r.standard_normal(s) * 1e6,
    "below fp16 normals (1e-8)": lambda r, s: r.standard_normal(s) * 1e-8,
    "positions 1e6 next to positions 1e-6": lambda r, s: r.standard_normal(s) * np.where(np.arange(s[3]) % 2, 1e6, 1e-6),
    "elements over 24 decades": lambda r, s: r.standard_normal(s) * 10.0 ** r.uniform(-12, 12, s),
}


@pytest.mark.parametrize("edge", list(_EDGES))
def test_range_edges(edge, monkeypatch):
    rng = np.random.default_rng(len(edge))
    x = _EDGES[edge](rng, (1, 128, 24, 24)).astype(np.float32)
    w = rng.standard_normal((128, 128, 1, 1)).astype(np.float32)
    run = lambda: CW.conv2d_wrapper(_nb(x), _nb(w), None, stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1)
    split, flags, fma = _both(run, monkeypatch)
    assert flags == (True, True)
    _judge(split, fma, _ref(x, w, None, (1, 1), (0, 0), (1, 1), 1))


@pytest.mark.parametrize("N,C,H,Wd,O,KH,KW,s,p,d,g", FMA_CASES)
def test_where_the_split_was_not_measured_faster_the_fma_path_stays(N, C, H, Wd, O, KH, KW, s, p, d, g, monkeypatch):
    calls = []
    real = CW._conv2d_m8n8k4
    monkeypatch.setattr(CW, "_conv2d_m8n8k4", lambda *a, **kw: (calls.append(1), real(*a, **kw))[1])
    rng = np.random.default_rng(C + O)
    x = rng.standard_normal((N, C, H, Wd)).astype(np.float32)
    w = (rng.standard_normal((O, C, KH, KW)) / np.sqrt(C * KH * KW)).astype(np.float32)
    out = CW.conv2d_wrapper(_nb(x), _nb(w), None, stride=list(s), padding=list(p), dilation=list(d), groups=g).numpy()
    assert not calls, "an fp32 convolution the split does not speed up took the matrix unit"
    ref = _ref(x, w, None, s, p, d, g)
    assert np.abs(out - ref).max() / np.abs(ref).max() < 1e-5


def test_the_census_derives_no_key_where_the_split_runs():
    """The derived census forms the launcher's own decision (`launch_keys.conv2d_launches`): no autotune key where the
    unit takes an fp32 convolution through the split; a key where the profile declares no split."""
    from nbxtensor.ops import launch_keys as LK
    from nbxtensor.dtype.nbx_dtype import NBXDtype
    f = NBXDtype.float32
    one = (1, 128, 32, 32, 128, 1, 1, 1, 1, 0, 0, 1, 1, 1)
    three = (1, 128, 32, 32, 128, 3, 3, 1, 1, 1, 1, 1, 1, 1)
    assert LK.conv2d_launches(*one, f, f, None, 1 << 32) == []
    assert LK.conv2d_launches(*three, f, f, None, 1 << 32) != []
    mu = {k: v for k, v in W._matrix_unit().items() if k != "fp32_split"}
    assert LK.conv2d_launches(*one, f, f, None, 1 << 32, unit=mu) != []
