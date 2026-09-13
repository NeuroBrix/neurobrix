"""A bf16 bias must still be bf16 in the key the autotuner computes.

Measured 2026-09-11: every `addmm_kernel` shape whose bias is bf16 failed
certification with

    the wrapper computed key (..., 'fp32', 'fp32', 'fp32', 'fp32') for inputs
    synthesized from (..., 'fp32', 'fp32', 'bf16', 'fp32'): the census and the
    kernel disagree — nothing certified for this key

Four shapes, and it explains by itself why the Apple certified directory holds
thirty entries and not one in bf16: the oracle never runs on them, the census
and the kernel never agree, and nothing is ever written.

It is a defect in its OWN RIGHT, independent of the oracle's raw bf16 read
that is under audit beside it. If that read turns out to be sound, this
survives; if the read is repaired, this survives too. Two defects reached
through the same dtype are still two defects -- the audit already had to
separate "the oracle decided" from "the oracle never ran", and this is the
second half.

What the wrapper says it does: a bias narrower than an fp32 accumulator is
WIDENED ON LOAD inside the kernel (`PROMOTE_BIAS`, exact) and keeps its own
dtype; only a bias that would have to be NARROWED is converted beforehand. So
a bf16 bias against an fp32 activation must reach the kernel as bf16, and the
key must say so.

Runnable: PYTHONPATH=src python3 -m pytest \
    tests/unit/kernels/test_a_bf16_bias_survives_to_the_key.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from check_measurement_environment import owned_cache_env       # noqa: E402

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin", reason="the Apple certification path")

M, N, K = 15, 768, 128


@pytest.fixture(autouse=True)
def _owns_its_cache(tmp_path, monkeypatch):
    for var, value in owned_cache_env(tmp_path).items():
        monkeypatch.setenv(var, value)


def _tensors():
    from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype
    from neurobrix.kernels.autotune_certify import f32_to_bf16_bits

    rng = np.random.default_rng(20260912)
    a = NBXTensor.from_numpy(rng.standard_normal((M, K), dtype=np.float32))
    b = NBXTensor.from_numpy(rng.standard_normal((K, N), dtype=np.float32))
    bias_f32 = rng.standard_normal(N, dtype=np.float32)
    bias = NBXTensor.from_numpy(f32_to_bf16_bits(bias_f32),
                                dtype=NBXDtype.bfloat16)
    return a, b, bias


def _key_seen(bias, a, b):
    """The key the autotuner would form for this call, captured at its seam."""
    from neurobrix.kernels import wrappers as W
    from neurobrix.triton import autotune_cache as atc
    from neurobrix.kernels.ops.matmul import addmm_kernel

    tuner = addmm_kernel
    seen = {}
    saved = tuner.run

    def spy(*args, **kwargs):
        tuner.nargs = dict(zip(tuner.arg_names, args))
        seen.setdefault("key", atc.key_of(tuner, args, kwargs))
        return saved(*args, **kwargs)

    tuner.run = spy
    try:
        W.addmm(bias, a, b)
    finally:
        tuner.run = saved
    return seen.get("key")


def test_the_bias_reaches_the_kernel_as_bf16():
    from neurobrix.kernels.nbx_tensor import NBXDtype

    a, b, bias = _tensors()
    assert bias.nbx_dtype == NBXDtype.bfloat16, "the fixture itself must be bf16"
    key = _key_seen(bias, a, b)
    assert key is not None, "the kernel was never launched; this measured nothing"
    dtypes = [k for k in key if isinstance(k, str) and
              k.replace("torch.", "") in ("float32", "bfloat16", "float16",
                                          "fp32", "bf16", "fp16")]
    assert dtypes, f"no dtype in the key {key!r}"
    assert any("bf" in d or "bfloat" in d for d in dtypes), (
        f"the key carries no bf16 anywhere: {dtypes}. A bf16 bias against an "
        f"fp32 activation must reach the kernel as bf16 -- PROMOTE_BIAS widens "
        f"it ON LOAD and is exact. If it is converted beforehand, the census "
        f"and the kernel can never agree on a key and nothing bf16 is ever "
        f"certified.")


def test_the_key_matches_what_the_census_would_record():
    """The certification's own equality, stated directly.

    `certify` refuses when `key_of` on the synthesized call differs from the
    key the census recorded. This asserts the third dtype -- the bias, by the
    kernel's argument order `(a_ptr, b_ptr, bias_ptr, c_ptr)` -- is the one
    that was handed in.
    """
    a, b, bias = _tensors()
    key = _key_seen(bias, a, b)
    assert key is not None
    dtypes = [k for k in key if isinstance(k, str)]
    assert len(dtypes) >= 3, f"fewer than three dtypes in {key!r}"
    assert "bf" in dtypes[2], (
        f"the third dtype is the BIAS by the kernel's argument order and it "
        f"reads {dtypes[2]!r}; the census recorded bf16 there. This is the "
        f"disagreement that refuses every bf16 shape.")
