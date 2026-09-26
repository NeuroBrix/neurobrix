"""Narrowing a weight to bfloat16 rounds to nearest, ties to even — it does not truncate.

Until 2026-09-26 the Triton weight loaders kept the top sixteen bits of each fp32 (and of each
fp16 widened to fp32): a truncation toward zero, half a bf16 ulp of magnitude lost on average,
the same sign every time. The ATen branch, the vendors' `torch.Tensor.to(torch.bfloat16)` and
this engine's own device cast all round to nearest even. Measured on PixArt-XL-2-1024-MS on
Apple: the Triton copy of `adaln.emb.aspect_ratio_embedder.proj_1.weight` equalled the
truncation of the container's fp32 on 100 % of its elements and the oracle's copy on 49.9 %.

The reference here is computed WITHOUT the bit trick the fix uses: for each value, the two
bf16 neighbours are enumerated and the nearer one taken in float64, the even one on a tie —
so the test cannot share a defect with the code it judges.
"""
from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from neurobrix.kernels.nbx_tensor import DeviceAllocator, NBXDtype, float32_to_bf16_bits


def _bits_value(b: int) -> float:
    """The float64 value of a bf16 pattern, read as IEEE does while rounding: exponent 255 with
    a zero mantissa is 2**128 (the value that rounds to infinity), not a special case."""
    sign = -1.0 if b & 0x8000 else 1.0
    e, m = (b >> 7) & 0xFF, b & 0x7F
    return sign * (m / 128.0) * 2.0 ** -126 if e == 0 else sign * (1 + m / 128.0) * 2.0 ** (e - 127)


def _reference_bits(x: np.ndarray) -> np.ndarray:
    """Nearest bf16 by enumeration in float64; ties to the even pattern; NaN quiet; infinities kept;
    a value past the midpoint above the bf16 maximum rounds to infinity (IEEE)."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    out = np.empty(x.size, dtype=np.uint16)
    for i, v in enumerate(x.tolist()):
        sign = 0x8000 if np.signbit(np.float32(v)) else 0
        if v != v:
            out[i] = sign | 0x7FC0
            continue
        if v in (float("inf"), float("-inf")):
            out[i] = sign | 0x7F80
            continue
        lo = int(np.array([v], dtype=np.float32).view(np.uint32)[0] >> 16)        # toward zero
        hi = lo + 1                                                               # away from zero
        dlo, dhi = abs(v - _bits_value(lo)), abs(v - _bits_value(hi))
        out[i] = lo if dlo < dhi or (dlo == dhi and lo % 2 == 0) else hi
    return out


def _cases() -> np.ndarray:
    rng = np.random.default_rng(20260926)
    random = (rng.standard_normal(4096) * 0.05).astype(np.float32)
    ties = ((rng.integers(1, 1 << 15, 256).astype(np.uint32) << 16) | 0x8000).view(np.float32)   # exact halves
    specials = np.array([0.0, -0.0, 1.0, -1.0, 12345.0, 3.3895313892515355e38, 3.4028234663852886e38,
                         -3.4028234663852886e38, np.inf, -np.inf, np.nan, 1e-40, -1e-40], dtype=np.float32)
    return np.concatenate([random, ties, specials])


def test_the_helper_rounds_to_nearest_even():
    x = _cases()
    np.testing.assert_array_equal(float32_to_bf16_bits(x).reshape(-1), _reference_bits(x))


def test_the_helper_is_not_a_truncation():
    """The planted defect: the top sixteen bits. It must differ from the helper on about half of
    random values — if it did not, this test file could not tell the two apart."""
    x = _cases()
    truncated = (x.view(np.uint32) >> 16).astype(np.uint16)
    assert np.mean(truncated != _reference_bits(x)) > 0.3


def _has_gpu() -> bool:
    try:
        return DeviceAllocator.device_count() > 0
    except Exception:                                  # pragma: no cover
        return False


def _device_bits(t, n) -> np.ndarray:
    import ctypes
    return np.ctypeslib.as_array(ctypes.cast(t.data_ptr(), ctypes.POINTER(ctypes.c_uint16)), shape=(n,)).copy()


@pytest.mark.skipif(not _has_gpu(), reason="pinned host allocation needs a device runtime")
@pytest.mark.parametrize("source", ["float32", "float16"])
def test_staging_to_pinned_host_rounds_to_nearest_even(source):
    from neurobrix.triton.weight_loader import _load_to_pinned_cpu
    x = _cases()[:4096 + 256]
    src = x.astype(np.float16) if source == "float16" else x
    staged = _load_to_pinned_cpu(src.tobytes(), (src.size,),
                                 NBXDtype.float16 if source == "float16" else NBXDtype.float32, NBXDtype.bfloat16)
    np.testing.assert_array_equal(_device_bits(staged, src.size), _reference_bits(src.astype(np.float32)))


@pytest.mark.skipif(not _has_gpu(), reason="the arena loader needs a device")
def test_the_arena_loader_rounds_to_nearest_even(tmp_path):
    """The path most weights take: a shard read into the component's device arena."""
    from neurobrix.triton.weight_loader import load_component_weights
    x = _cases()[:4096 + 256]
    wdir = tmp_path / "components" / "c" / "weights"; wdir.mkdir(parents=True)
    header = {"w": {"dtype": "F32", "shape": [x.size], "data_offsets": [0, x.nbytes]}}
    hb = json.dumps(header).encode(); hb += b" " * (-len(hb) % 8)
    (wdir / "shard_000.safetensors").write_bytes(struct.pack("<Q", len(hb)) + hb + x.tobytes())
    weights = load_component_weights(str(tmp_path), "c", 0, compute_dtype=NBXDtype.bfloat16)
    t = weights["w"]
    assert t.nbx_dtype == NBXDtype.bfloat16
    got = np.ascontiguousarray(t.numpy()).view(np.uint16).reshape(-1)
    np.testing.assert_array_equal(got, _reference_bits(x))
