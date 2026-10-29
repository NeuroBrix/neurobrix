"""A conv3d decomposed into temporal taps (`_conv3d_via_conv2d`) reads each tap's frames as a strided view and writes
its result through a view of the [B, C, T, H, W] output — no permute copy either way — and still computes the
definition: against a float64 conv3d built from the per-tap fp64 conv2d reference, a batch of two, temporal and
spatial padding, a dilated and a strided case."""
import numpy as np
import pytest

from nbxtensor.ops import wrappers as Wr
from nbxtensor.tensor import DeviceAllocator, NBXTensor
from nbxtensor.ops.oracles.conv2d_fp64 import conv2d_reference


def _gpu() -> bool:
    try:
        DeviceAllocator.set_device(0)
        return True
    except Exception:
        return False


def _conv3d_ref(x, w, pt, ph, sh, dh):
    B, C, T, H, W = x.shape
    O, Cg, kt, kh, kw = w.shape
    xp = np.pad(x.astype(np.float64), ((0, 0), (0, 0), (pt, pt), (0, 0), (0, 0)))
    To = xp.shape[2] - (kt - 1)
    outs = []
    for b in range(B):
        frames = []
        for t in range(To):
            acc = None
            for k in range(kt):
                r = conv2d_reference(xp[b:b + 1, :, t + k], w[:, :, k].astype(np.float64),
                                     stride=(sh, sh), padding=(ph, ph), dilation=(dh, dh), groups=1)
                acc = r if acc is None else acc + r
            frames.append(acc[0])
        outs.append(np.stack(frames, axis=1))
    return np.stack(outs)


@pytest.mark.parametrize("B,C,T,H,W,O,pt,ph,sh,dh", [
    (1, 8, 5, 12, 14, 16, 1, 1, 1, 1),
    (2, 16, 4, 9, 11, 8, 1, 0, 1, 1),     # a batch: one launch per sequence, each in place
    (1, 8, 6, 13, 13, 8, 0, 2, 1, 2),     # dilated, no temporal pad
    (2, 8, 5, 16, 16, 16, 1, 1, 2, 1),    # spatially strided
])
def test_the_taps_compute_the_definition(B, C, T, H, W, O, pt, ph, sh, dh):
    if not _gpu():
        pytest.skip("no device")
    rng = np.random.default_rng(3)
    x = rng.standard_normal((B, C, T, H, W)).astype(np.float32)
    w = rng.standard_normal((O, C, 3, 3, 3)).astype(np.float32)
    ref = _conv3d_ref(x, w, pt, ph, sh, dh)
    got = Wr._conv3d_via_conv2d(NBXTensor.from_numpy(x), NBXTensor.from_numpy(w), None,
                                (1, sh, sh), (pt, ph, ph), (1, dh, dh), 1).numpy().astype(np.float64)
    assert got.shape == ref.shape
    err = np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-12)
    assert err < 1e-2, f"relative error {err:.3g}"
