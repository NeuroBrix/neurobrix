"""A row-windowed matmul oracle reads only the rows it windows.

`screen_oracle._mm(named, rows=(r0, r1))` is the launch oracle of the certifier and the
runtime screen: three row windows of a product "are a few megabytes" (its own docstring).
Measured 2026-09-28 on the Mac (results/certifier_price/phases, matmul 262144x1024x512 bf16):
the certifier's footprint spiked by 3 684 MB in the oracle phase and fell back — the WHOLE
operand `a` read to the host and cast whole to float64 (`_to_f64(a)` then `a[r0:r1]`), once
per window, so the window saved nothing on the host and the certifier's price for the matmul
family was 16 bytes per element where the draws and copies account for 8.

What this test does if the code is wrong: the host peak of one windowed call on a
1 048 576 x 256 fp32 operand is the whole operand in float64 (2 GiB) plus its fp32 copy, far
above the bound; seen red before the fix. The value is checked against the whole product's
rows so the cut cannot move the numbers.
"""
from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from neurobrix.kernels import screen_oracle as SO
from neurobrix.kernels.nbx_tensor import NBXTensor, DeviceAllocator


@pytest.fixture
def operands():
    try:
        DeviceAllocator.get_device()
    except Exception as exc:                            # noqa: BLE001
        pytest.skip(f"no device for a live operand: {exc}")
    rng = np.random.default_rng(7)
    a_np = rng.standard_normal((1 << 20, 256), dtype=np.float32)
    b_np = rng.standard_normal((256, 64), dtype=np.float32)
    a, b = NBXTensor.from_numpy(a_np), NBXTensor.from_numpy(b_np)
    yield a, b, a_np, b_np
    del a, b


def test_a_row_window_reads_the_rows_not_the_operand(operands):
    a, b, a_np, b_np = operands
    r0, r1 = 4096, 4096 + 64
    tracemalloc.start()
    tracemalloc.reset_peak()
    ref = SO._mm({"a_ptr": a, "b_ptr": b}, rows=(r0, r1))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert ref is not None and ref.shape == (r1 - r0, 64)
    rows_bytes = (r1 - r0) * 256 * (4 + 8) + 256 * 64 * (4 + 8)      # the window's rows and b, fp32 read + float64
    assert peak <= rows_bytes + 16 * 2**20, (
        f"the windowed oracle held {peak / 2**20:.0f} MiB on the host for {(r1 - r0)} rows "
        f"(bound {(rows_bytes + 16 * 2**20) / 2**20:.0f} MiB): the whole operand crossed")
    expect = a_np[r0:r1].astype(np.float64) @ b_np.astype(np.float64)
    assert np.array_equal(ref, expect), "the window's rows are not the product's rows"
