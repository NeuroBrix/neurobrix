"""Unit test for conv3d temporal chunk-streaming in the Triton wrapper.

Guard for the `_conv3d_via_conv2d_chunked` path (kernels/wrappers.py):
the one-shot temporal decomposition of conv3d needs ~4-5 FULL folded
transients simultaneously (pad copy, x2 fold, conv2d output, permuted
copy, accumulator), which OOMs native-resolution video-VAE encodes in
triton mode while the same component fits in compiled mode (cuDNN needs
input + output + bounded workspace). Reference failure: Allegro-TI2V
vae_encoder at 720x1280x12f fp32 — 30.4 GB live at aten.convolution::1
on a 32 GB V100.

The chunked path streams the folded batch axis (B*T_out frames) with a
bounded per-transient size. This test proves the chunked math is
equivalent to both torch.nn.functional.conv3d and the one-shot
`_conv3d_via_conv2d` path across kernel/stride/pad/dilation/groups/bias
configs and chunk sizes (tc = 1 / 2 / 3 forced via `frame_bytes`).

Runnable two ways:
  - pytest:  PYTHONPATH=src /usr/bin/python3 -m pytest tests/unit/kernels/test_conv3d_chunk_stream.py -v
  - script:  PYTHONPATH=src <gpu-venv>/bin/python tests/unit/kernels/test_conv3d_chunk_stream.py
"""
from __future__ import annotations

try:
    import pytest
except ModuleNotFoundError:  # script-mode under the pytest-less GPU runtime venv
    class _NoPytest:
        class mark:
            @staticmethod
            def parametrize(*a, **k):
                return lambda fn: fn

        @staticmethod
        def skip(*a, **k):
            raise SystemExit(0)

    pytest = _NoPytest()  # type: ignore

import numpy as np
import torch

from neurobrix.kernels import wrappers
from neurobrix.kernels.nbx_tensor import (
    DeviceAllocator, NBXTensor, nbx_to_torch,
)


def _nbx(arr):
    return NBXTensor.from_numpy(np.ascontiguousarray(arr))


def _triple(v):
    return (v, v, v) if isinstance(v, int) else tuple(v)


# (config-id, x.shape, w.shape, stride, padding, dilation, groups, bias)
_CONFIGS = [
    # Allegro temp_conv class: temporal-only kernel at 1x1 spatial.
    ("temporal_k311", (1, 6, 9, 10, 12), (8, 6, 3, 1, 1),
     1, (1, 0, 0), 1, 1, True),
    # Full 3x3x3 VAE-style conv.
    ("full_k333", (2, 5, 7, 9, 11), (7, 5, 3, 3, 3),
     1, (1, 1, 1), 1, 1, True),
    # Strided (index_select temporal path) without bias.
    ("strided_222", (1, 4, 9, 12, 14), (6, 4, 3, 3, 3),
     (2, 2, 2), (1, 1, 1), 1, 1, False),
    # Patch-embed-ish: kt=1, spatial stride.
    ("patch_kt1", (2, 4, 6, 12, 12), (10, 4, 1, 3, 3),
     (1, 2, 2), (0, 1, 1), 1, 1, True),
    # Grouped conv.
    ("groups2", (1, 6, 8, 9, 10), (8, 3, 3, 3, 3),
     1, (1, 1, 1), 1, 2, True),
    # Temporal dilation.
    ("dilated_t2", (1, 4, 11, 8, 9), (5, 4, 3, 3, 3),
     1, (2, 1, 1), (2, 1, 1), 1, True),
]


def _run_case(xs, ws, stride, padding, dilation, groups, with_bias, tc_forced):
    rng = np.random.default_rng(7)
    x_np = rng.standard_normal(xs).astype(np.float32)
    w_np = (rng.standard_normal(ws) * 0.2).astype(np.float32)
    b_np = rng.standard_normal(ws[0]).astype(np.float32) if with_bias else None

    st, sh, sw = _triple(stride)
    pt, ph, pw = _triple(padding)
    dt, dh, dw = _triple(dilation)

    # Reference 1: torch conv3d.
    ref = torch.nn.functional.conv3d(
        torch.from_numpy(x_np).cuda(), torch.from_numpy(w_np).cuda(),
        torch.from_numpy(b_np).cuda() if b_np is not None else None,
        stride=(st, sh, sw), padding=(pt, ph, pw),
        dilation=(dt, dh, dw), groups=groups).float().cpu()

    # Reference 2: one-shot triton decomposition (gate never fires on these
    # tiny folds — deterministic floor is 1 GiB).
    oneshot = nbx_to_torch(wrappers._conv3d_via_conv2d(
        _nbx(x_np), _nbx(w_np),
        _nbx(b_np) if b_np is not None else None,
        (st, sh, sw), (pt, ph, pw), (dt, dh, dw), groups)).float().cpu()

    # Chunked path, forced chunk size: tc = _NBX_CONV3D_CHUNK_BYTES // frame_bytes.
    frame_bytes = max(1, wrappers._NBX_CONV3D_CHUNK_BYTES // tc_forced)
    got = nbx_to_torch(wrappers._conv3d_via_conv2d_chunked(
        _nbx(x_np), _nbx(w_np),
        _nbx(b_np) if b_np is not None else None,
        st, sh, sw, pt, ph, pw, dt, dh, dw, groups,
        frame_bytes)).float().cpu()

    return got, oneshot, ref


@pytest.mark.parametrize("tc", [1, 2, 3])
@pytest.mark.parametrize(
    "cid,xs,ws,stride,padding,dilation,groups,with_bias", _CONFIGS)
def test_chunked_matches_torch_and_oneshot(
        cid, xs, ws, stride, padding, dilation, groups, with_bias, tc):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    got, oneshot, ref = _run_case(
        xs, ws, stride, padding, dilation, groups, with_bias, tc)

    assert got.shape == ref.shape, f"{cid}/tc={tc}: shape {got.shape} != {ref.shape}"
    # vs torch: same envelope as the documented one-shot parity (~3e-5).
    assert torch.allclose(got, ref, rtol=1e-4, atol=1e-4), (
        f"{cid}/tc={tc} vs torch: max|d|={(got - ref).abs().max().item()}")
    # vs one-shot: identical math, launch shapes differ (autotune configs
    # may vary) — tight-but-not-bitwise tolerance.
    assert torch.allclose(got, oneshot, rtol=1e-4, atol=1e-4), (
        f"{cid}/tc={tc} vs one-shot: max|d|={(got - oneshot).abs().max().item()}")


def test_device_free_bytes_probe():
    """The gate's feasibility probe returns a plausible positive value on a
    CUDA host (and -1, never an exception, when the query is unavailable)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    free = DeviceAllocator.device_free_bytes()
    assert isinstance(free, int)
    assert free > 0


if __name__ == "__main__":  # script mode
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    for tc in (1, 2, 3):
        for cfg in _CONFIGS:
            cid = cfg[0]
            got, oneshot, ref = _run_case(*cfg[1:], tc)
            dt_ = (got - ref).abs().max().item()
            do_ = (got - oneshot).abs().max().item()
            status = "OK" if (dt_ < 1e-4 and do_ < 1e-4) else "FAIL"
            print(f"[{status}] {cid} tc={tc} max|d| torch={dt_:.2e} oneshot={do_:.2e}")
    print("device_free_bytes:", DeviceAllocator.device_free_bytes())
