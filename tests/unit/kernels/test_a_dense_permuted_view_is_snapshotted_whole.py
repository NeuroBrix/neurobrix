"""A permuted view of a dense buffer is snapshotted over its own bytes; a view with gaps is refused.

Since 9c2ab78e (2026-10-08) every 3x3 conv weight on Apple is a KRSC buffer read as a (K, C, R, S)
view, and the certifier builds the same view. `_writable_buffers` refused any non-contiguous
argument, so from that commit no 3x3 conv key could be certified on Apple: the re-measurement of
2026-10-09 00:41 (nbx-atelier/campagnes/2026_09_22_apple/results/certifier_price/remeasure_2026_10_09,
k4_conv_256_770x2048, p_conv2d_0_5M, p_conv2d_1_344M) failed with "a strided view among the
arguments — not certifiable", and the launcher's screen skipped every such conv. A permutation of
a dense buffer has no gaps: its span from `data_ptr()` is exactly its `numel * itemsize` bytes, so
the snapshot covers the tensor and nothing else. A view with gaps (a slice, an expand) still is
refused, for the reason the docstring of `_writable_buffers` records.
"""
from __future__ import annotations

import numpy as np

from neurobrix.kernels import launcher as L
from neurobrix.kernels.nbx_tensor import NBXTensor


def _krsc_view():
    w = np.arange(4 * 3 * 3 * 5, dtype=np.float32).reshape(4, 3, 3, 5)      # K, R, S, C in memory
    return NBXTensor.from_numpy(np.ascontiguousarray(w)).permute(0, 3, 1, 2)  # read as K, C, R, S


def test_a_krsc_conv_weight_is_snapshotted_over_its_own_bytes():
    t = _krsc_view()
    assert not t.is_contiguous()
    assert L._writable_buffers([t]) == [(t.data_ptr(), 4 * 5 * 3 * 3 * 4, "fp32")]


def test_a_view_with_gaps_is_still_refused():
    base = NBXTensor.from_numpy(np.zeros((4, 6), dtype=np.float32))
    assert L._writable_buffers([base[:, :3]]) is None                       # every row skips 3
    assert L._writable_buffers([base.permute(1, 0)[:2]]) is None             # permuted AND gapped
    assert L._writable_buffers([NBXTensor.from_numpy(np.zeros((1, 6), dtype=np.float32)).expand(4, 6)]) is None
