"""The DtypeEngine counts the copies its wrappers make and drop inside an op on its `copy_meter` — the operand casts
(lower precision, fp32 upcast, promote) and a wider result stored narrower — when a compiled sequence on a device whose
allocator frees only once the device has run the work sets the meter (MPS, `CompiledSequence`); nothing is counted with
no meter. The Mac's RUN 33 (2026-11-04 20:27): those copies were the uncounted frees that kept mochi's VAE out of memory.

What this test would do if the code were wrong: a wrapper's cast not counted — red at its figure; the wide result of a
narrowed store not counted — red; anything counted with no meter — red at the last test. Card-less (CPU tensors)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from neurobrix.core.dtype.engine import DtypeEngine  # noqa: E402


def _engine(meter=True):
    e = DtypeEngine(torch.float16, graph_dtype=torch.float16)
    e.copy_meter = [0] if meter else None
    return e


def test_a_lower_precision_cast_is_counted():
    e = _engine()
    f = e._make_lower_precision_wrapper(lambda a, b: a + b)
    f(torch.ones(100, dtype=torch.float32), torch.ones(100, dtype=torch.float16))
    assert e.copy_meter == [200]                      # the fp32 operand's fp16 copy: 100 x 2 bytes


def test_an_fp32_upcast_is_counted():
    e = _engine()
    f = e._make_fp32_wrapper(lambda a: a * 2)
    f(torch.ones(100, dtype=torch.float16))
    assert e.copy_meter == [400]                      # the fp16 operand's fp32 copy


def test_a_wide_result_stored_narrower_is_counted():
    e = _engine()
    f = e._make_store_in_compute_dtype(lambda a: a.float())
    out = f(torch.ones(100, dtype=torch.float16))
    assert out.dtype == torch.float16 and e.copy_meter == [400]   # the fp32 result, dropped


def test_no_meter_counts_nothing():
    e = _engine(meter=False)
    e._make_lower_precision_wrapper(lambda a: a)(torch.ones(10, dtype=torch.float32))
    e._make_fp32_wrapper(lambda a: a)(torch.ones(10, dtype=torch.float16))
    assert e.copy_meter is None
