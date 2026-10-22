"""A stride-1 conv3d runs its taps in place, ONE conv2d per batch sample over that sample's frames [T_out, Cin, H, W]
(`wrappers.conv3d_wrapper`): its key's batch is T_out, whatever the batch. The census computes the launch with
`launch_keys.conv_launches`; keyed at B*T_out, SANA-Video's guidance-batch patch embed (B = 2, 11 frames) formed
(11, 128, 21, 42, 2240, ...) where the census held (22, ...) and the certified-only W1+Apple gate missed it (2026-10-22).
Other temporal strides stack every sample's frames: B*T_out. And `aten::_safe_softmax` runs the softmax wrapper
(`kernels/dispatch.py`), so the census derives its key (Ming-Lite's vision softmax missed the same gate).

Injection (seen red, then restored green, 2026-10-22): the 5-D batch back to B*T_out -> the first cell red; `_safe_softmax`
out of the census's softmax kinds -> the last cell red."""
from __future__ import annotations

import sys
from pathlib import Path

from neurobrix.kernels import launch_keys as LK
from neurobrix.kernels.nbx_tensor import NBXDtype

F32 = NBXDtype.float32


def _batch(launches):
    assert launches, "no launch"
    return {int(k[0]) for _kern, k in launches}


def test_a_stride_one_conv3d_keys_one_sample_s_frames():
    L = LK.conv_launches([2, 128, 11, 21, 42], [2240, 128, 1, 1, 1], [1, 1, 1], [0, 0, 0], [1, 1, 1], False, 1,
                         F32, F32, F32, 1 << 40)
    assert _batch(L) == {11}, L


def test_a_strided_conv3d_stacks_every_sample_s_frames():
    L = LK.conv_launches([2, 128, 11, 21, 42], [2240, 128, 1, 1, 1], [2, 1, 1], [0, 0, 0], [1, 1, 1], False, 1,
                         F32, F32, F32, 1 << 40)
    assert _batch(L) == {2 * 6}, L


def test_the_census_derives_a_safe_softmax():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
    import derived_census as D
    assert "aten::_safe_softmax" in D._KEYED_KINDS
