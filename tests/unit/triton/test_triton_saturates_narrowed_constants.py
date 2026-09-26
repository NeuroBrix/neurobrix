"""A finite constant narrowed into a half dtype saturates to that dtype's extreme; it does not become infinite.

A graph traced in fp32 carries finfo(fp32).min as a literal mask sentinel. The vendor, running in the half
dtype, would have used finfo(half).min, which is finite. Measured 2026-09-26 on PixArt-XL-2-1024-MS's T5:
the Triton branch's `aten.where::0` wrote -inf into 13 200 bf16 positions where the ATen branch kept the
finite fp32 value. The ATen dtype engine already clamps fill and creation scalars to the half range; this is
the Triton mirror (R30), extended to the one-element operand `where` narrows.
"""
from __future__ import annotations

import numpy as np
import pytest

from neurobrix.kernels.nbx_tensor import NBXDtype, NBXTensor, DeviceAllocator
from neurobrix.triton.dtype import TritonDtypeEngine, saturate_scalar

F32_MIN = float(np.finfo(np.float32).min)          # -3.4028234663852886e38
BF16_MAX = 3.3895313892515355e38


def test_saturate_scalar_clamps_only_what_would_overflow():
    assert saturate_scalar(F32_MIN, NBXDtype.bfloat16) == -BF16_MAX
    assert saturate_scalar(-3.3895e38, NBXDtype.float16) == -65504.0
    assert saturate_scalar(1.5, NBXDtype.bfloat16) == 1.5
    assert saturate_scalar(float("-inf"), NBXDtype.bfloat16) == float("-inf")     # an infinity written on purpose
    assert saturate_scalar(F32_MIN, NBXDtype.float32) == F32_MIN                  # not a half dtype
    assert saturate_scalar(True, NBXDtype.float16) is True


def _gpu():
    try:
        return DeviceAllocator.device_count() > 0
    except Exception:                                     # pragma: no cover
        return False


def _host(t):
    h = t.numpy()
    if h.dtype.kind == "V":
        h = (np.ascontiguousarray(h).view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    return np.asarray(h, dtype=np.float64)


@pytest.mark.skipif(not _gpu(), reason="needs a device")
def test_where_writes_the_half_extreme_not_minus_infinity():
    from neurobrix.kernels import wrappers as w
    eng = TritonDtypeEngine(NBXDtype.bfloat16)
    where = eng.wrap_op("where", w.where_wrapper)
    cond = NBXTensor.from_numpy(np.array([[True, False], [False, True]]))
    x = NBXTensor.from_numpy(np.zeros((), dtype=np.uint16), dtype=NBXDtype.bfloat16)   # 0.0 in bf16, one element
    y = NBXTensor.from_numpy(np.array(F32_MIN, dtype=np.float32))                       # the fp32 sentinel
    out = _host(where(cond, x, y))
    assert np.isfinite(out).all(), f"where wrote {out.tolist()}: the sentinel became infinite in bf16"
    assert out[0, 1] == -BF16_MAX and out[0, 0] == 0.0


@pytest.mark.skipif(not _gpu(), reason="needs a device")
def test_masked_fill_and_full_saturate_in_half_dtypes():
    from neurobrix.kernels import wrappers as w
    from neurobrix.kernels.dispatch import dispatch
    eng = TritonDtypeEngine(NBXDtype.float16)
    x = NBXTensor.from_numpy(np.ones((2, 2), dtype=np.float16))
    mask = NBXTensor.from_numpy(np.array([[True, False], [False, False]]))
    filled = _host(eng.wrap_op("masked_fill", w.masked_fill)(x, mask, F32_MIN))
    assert filled[0, 0] == -65504.0 and filled[1, 1] == 1.0
    full = eng.wrap_op("full", dispatch("aten::full"))
    made = _host(full([2, 2], -3.3895e38, dtype=NBXDtype.float16))
    assert (made == -65504.0).all(), f"full wrote {made.tolist()}"
