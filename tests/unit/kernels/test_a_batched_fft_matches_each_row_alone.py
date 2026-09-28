"""The radix-2 FFT runs every row of a batch in one launch per step, and a row computed in a batch
is bit-identical to the same row computed alone.

The rows were launched one at a time from the host. An STFT of a spoken sentence is thousands of
frames, so thousands of launches per butterfly stage: chatterbox's census walk spent every py-spy
sample inside that loop (2026-09-28, three samples of three), and every live Triton run through
`aten::stft` / `aten::istft` (chatterbox, MiniCPM-o-4_5) paid the same.

What each test would do if the code were wrong: a row offset missing from the batched kernels makes
every row after the first read the first row's data — the bit-identity assertion fails on row 1; a
wrong twiddle sign or bit reversal fails the numpy comparison; the inverse without its 1/N scale
fails the round trip. (Seen red: the row offset removed from `fft_stage_rows_kernel`.)
The byte identity against the one-row kernels this replaces is `tools/bare_return_bitgate.py`
(`capture` on main vs on this branch, `compare`), recorded in the commit.
"""
from __future__ import annotations

import numpy as np
import pytest

from neurobrix.kernels.nbx_tensor import DeviceAllocator

pytestmark = pytest.mark.skipif(DeviceAllocator.device_count() == 0, reason="needs a device to run the kernels")


def _nbx(a):
    from neurobrix.kernels.nbx_tensor import NBXTensor
    return NBXTensor.from_numpy(np.ascontiguousarray(a, dtype=np.float32))


def _np(t):
    return np.asarray(t.numpy(), dtype=np.float32)


@pytest.mark.parametrize("rows,n", [(1, 8), (37, 256), (129, 1024)])
def test_a_row_in_a_batch_is_the_row_alone_and_is_the_fft(rows, n):
    from neurobrix.kernels.wrappers import _triton_fft_forward
    rng = np.random.default_rng(7)
    xr = rng.standard_normal((rows, n)).astype(np.float32)
    xi = rng.standard_normal((rows, n)).astype(np.float32)
    br, bi = _triton_fft_forward(_nbx(xr), _nbx(xi))
    br, bi = _np(br), _np(bi)
    for r in (0, rows // 2, rows - 1):
        ar, ai = _triton_fft_forward(_nbx(xr[r]), _nbx(xi[r]))
        assert np.array_equal(br[r], _np(ar)) and np.array_equal(bi[r], _np(ai)), f"row {r} differs from itself alone"
    ref = np.fft.fft(xr.astype(np.float64) + 1j * xi.astype(np.float64), axis=-1)
    scale = np.abs(ref).max()
    assert np.abs(br - ref.real).max() / scale < 1e-5 and np.abs(bi - ref.imag).max() / scale < 1e-5


def test_the_inverse_undoes_the_forward_on_every_row():
    from neurobrix.kernels.wrappers import _triton_fft_forward, _triton_ifft
    rng = np.random.default_rng(11)
    xr = rng.standard_normal((53, 512)).astype(np.float32)
    xi = rng.standard_normal((53, 512)).astype(np.float32)
    fr, fi = _triton_fft_forward(_nbx(xr), _nbx(xi))
    yr, yi = _triton_ifft(fr, fi)
    assert np.abs(_np(yr) - xr).max() < 1e-4 and np.abs(_np(yi) - xi).max() < 1e-4
