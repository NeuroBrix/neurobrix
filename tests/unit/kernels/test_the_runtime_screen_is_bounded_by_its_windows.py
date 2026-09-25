"""The runtime screen of an uncertified key is bounded by its windows — for every kernel family.

The Mac, 2026-09-25 (its report, "engine defect"): PixArt at 2048x1024, the VAE's last
convolution (128 -> 3 channels, bf16) is not certified at that size, so the launcher screens
its candidates at runtime. The screen's budget test counted the buffers' DEVICE bytes, and the
row windows (`NBX_SCREEN_WINDOWS`) existed only for the matrix kernels, so a convolution was
screened whole or not at all: its 537 MB bf16 input became 2.1 GB of float64 on the host before
the first tap — a 27.5 GB footprint on a 24 GB machine and a kill. On this rack the same key was
seated UNSCREENED ("the output could not be row-windowed"), which is the other face of the same
absence: nothing verified it.

Three things hold now, and this file holds each:

* the convolution family's reference is computed on FLAT OUTPUT ROWS of `[N*Cout*Ho, Wo]`,
  reading only the rows' receptive field through a slab cut on the device — equal, row for
  row, to the whole reference (CPU cells, every geometry class: stride, padding, dilation,
  groups, depthwise);
* the full screen is refused when its float64 footprint — not the device bytes — exceeds the
  budget, and the windowed screen takes it (CPU cell);
* the Mac's key itself, on a card: screened by the fp64 oracle on named windows, host peak
  under a bound an order of magnitude below the whole input in float64, and the kernel's rows
  on the last window equal to the reference. Seen RED against main f29981a8 (the key seated
  unscreened, no window record), GREEN on the branch.

    CUDA_VISIBLE_DEVICES=3 NEUROBRIX_REPLAY_CACHE=<scratch>/replay_cache_x PYTHONPATH=src \
      pytest tests/unit/kernels/test_the_runtime_screen_is_bounded_by_its_windows.py -p no:cacheprovider
"""
from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from neurobrix.kernels.oracles import conv2d_fp64 as C

GEOMETRIES = [
    # n, cin, h, w, cout, k, stride, pad, dilation, groups
    (2, 4, 9, 11, 6, 3, 1, 1, 1, 1),
    (1, 6, 10, 7, 4, 3, 2, 1, 1, 2),
    (2, 3, 8, 8, 5, 2, 1, 0, 2, 1),
    (1, 5, 12, 9, 5, 3, 2, 2, 1, 5),      # depthwise through the grouped path
    (3, 4, 6, 6, 4, 1, 1, 0, 1, 1),
]


def _named(n, cin, h, w, cout, oh, ow, k, s, p, d, g):
    return dict(batch_dim=n, in_feat_dim=cin, in_height=h, in_width=w, out_feat_dim=cout,
                out_height=oh, out_width=ow, kernel_height=k, kernel_width=k, stride_height=s,
                stride_width=s, padding_height=p, padding_width=p, dilation_height=d,
                dilation_width=d, groups=g)


@pytest.mark.parametrize("geom", GEOMETRIES)
def test_the_flat_row_reference_equals_the_whole_reference_on_every_window(geom):
    n, cin, h, w, cout, k, s, p, d, g = geom
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n, cin, h, w))
    wt = rng.standard_normal((cout, cin // g, k, k))
    full = C.conv2d_reference(x, wt, stride=(s, s), padding=(p, p), dilation=(d, d), groups=g)
    oh, ow = full.shape[2:]
    flat = full.reshape(n * cout * oh, ow)
    M = flat.shape[0]
    named = _named(n, cin, h, w, cout, oh, ow, k, s, p, d, g)
    reads = []

    def slab(n0, n1, u0, u1, v0, v1):
        reads.append((u1 - u0) * (v1 - v0))
        return x[n0:n1, :, u0:u1, v0:v1]

    halo = d * (k - 1)
    for r0, r1 in [(0, 1), (M - 1, M), (M // 2, min(M, M // 2 + 3)), (oh - 1, min(M, oh + 2)),
                   (cout * oh - 2, min(M, cout * oh + 2)), (0, M)]:
        reads.clear()
        got = C.conv2d_rows(named, slab, wt.reshape(-1), (r0, r1))
        assert got is not None and got.shape == (r1 - r0, ow), (r0, r1)
        np.testing.assert_allclose(got, flat[r0:r1], rtol=0, atol=1e-12)
        # EVERY window — the ones straddling a channel or a batch element included — reads a
        # slab bounded by its own rows and their halo, never the whole input: a window
        # that crosses a channel used to be widened to every row of the element (the
        # guardian, 2026-09-26), which for the Mac's key is the whole 2.1 GB input.
        bound = ((r1 - r0) * s + halo) * w
        assert max(reads) <= bound, f"window {(r0, r1)} read {max(reads)} elements, bound {bound}"


def test_a_window_straddling_a_channel_reads_only_its_rows():
    """The window the Mac's key would place at rows 2046..2051 crosses from channel 0 to
    channel 1; each side is its own segment with its own slab."""
    n, cin, h, w, cout, k = 1, 4, 8, 6, 3, 3
    rng = np.random.default_rng(3)
    x = rng.standard_normal((n, cin, h, w)); wt = rng.standard_normal((cout, cin, k, k))
    full = C.conv2d_reference(x, wt, stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1)
    oh, ow = full.shape[2:]
    named = _named(n, cin, h, w, cout, oh, ow, k, 1, 1, 1, 1)
    segs = list(C._row_segments((oh - 2, oh + 3), cout, oh))
    assert segs == [(0, 0, oh - 2, oh, oh - 2, oh), (0, 1, 0, 3, oh, oh + 3)], segs
    reads = []
    got = C.conv2d_rows(named, lambda n0, n1, u0, u1, v0, v1: (reads.append(u1 - u0), x[n0:n1, :, u0:u1, v0:v1])[1],
                        wt.reshape(-1), (oh - 2, oh + 3))
    np.testing.assert_allclose(got, full.reshape(-1, ow)[oh - 2:oh + 3], atol=1e-12)
    assert max(reads) <= 3 + (k - 1) and len(reads) == 2, reads


def test_a_declaration_that_contradicts_the_arrays_is_a_refusal_with_its_reason():
    named = _named(1, 4, 8, 6, 3, 8, 6, 3, 1, 1, 1, 1)
    named["out_height"] = 7                            # not the formula's 8
    with pytest.raises(ValueError, match="formula"):
        C.conv2d_rows(named, lambda *a: np.zeros((1, 4, 1, 6)), np.zeros(3 * 4 * 9), (0, 1))
    with pytest.raises(ValueError, match="row cost"):
        C.oracle_row_bytes("conv2d_forward_kernel", {"batch_dim": 1})
    assert C.oracle_row_bytes("matmul_kernel", {}) is None


def test_the_depthwise_rows_equal_the_whole_reference():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 5, 9, 8))
    wt = rng.standard_normal((5, 3, 3))
    full = C.depthwise_reference(x, wt, stride=(2, 2), padding=(1, 1))
    oh, ow = full.shape[2:]
    flat = full.reshape(-1, ow)
    named = dict(N=2, C=5, H_in=9, W_in=8, H_out=oh, W_out=ow, kh=3, kw=3, stride_h=2,
                 stride_w=2, pad_h=1, pad_w=1)
    for r0, r1 in [(0, 2), (flat.shape[0] - 2, flat.shape[0]), (oh - 1, oh + 3), (0, flat.shape[0])]:
        got = C.depthwise_rows(named, lambda n0, n1, u0, u1, v0, v1: x[n0:n1, :, u0:u1, v0:v1],
                               wt.reshape(-1), (r0, r1))
        np.testing.assert_allclose(got, flat[r0:r1], rtol=0, atol=1e-12)


def test_the_full_screen_is_measured_by_its_float64_footprint_not_the_device_bytes():
    """The Mac's key: 12 MB of output, 537 MB of input on the device, 2.2 GB of float64 for the
    reference. The device bytes were what the budget saw. This drives the launcher's own
    decision: a buffer set UNDER the budget in device bytes and OVER it in float64 must be
    windowed — delete the footprint term and this cell is red."""
    from neurobrix.kernels import launcher as L
    from neurobrix.kernels import screen_oracle as S
    budget = 32 * 1024 * 1024
    macs = [(0, 128 * 2048 * 1024 * 2, "bfloat16"), (1, 3 * 128 * 9 * 2, "bfloat16"),
            (2, 3 * 2048 * 1024 * 2, "bfloat16")]
    assert S.fp64_footprint(macs) == 4 * sum(b for _a, b, _d in macs)
    assert "over the screening budget" in L._needs_windowing(macs, budget)
    small_device_big_reference = [(0, budget // 2, "int8"), (1, 4096, "float16")]   # 16 MiB on the device, 128 MiB in float64
    why = L._needs_windowing(small_device_big_reference, budget)
    assert why is not None and "float64 footprint" in why, why
    assert L._needs_windowing([(0, 4096, "float32"), (1, 4096, "float32")], budget) is None
    with pytest.raises(ValueError, match="not an NBXDtype"):
        S.fp64_footprint([(0, 8, "nonsense")])
    assert "conv2d_forward_kernel" in S.ROW_WINDOWABLE and "depthwise_conv2d_kernel" in S.ROW_WINDOWABLE


# ───────────────────────────── the Mac's key, on a card ─────────────────────────────

N_, CIN, H_, W_, COUT, K_ = 1, 128, 2048, 1024, 3, 3
#: What the whole input costs in float64 — the footprint the windows exist to avoid.
WHOLE_INPUT_F64 = N_ * CIN * H_ * W_ * 8
#: The bound the windowed screen must stay under on the host: an order of magnitude below.
HOST_PEAK_BOUND = WHOLE_INPUT_F64 // 8


def _cuda_free_bytes():
    try:
        import ctypes
        cuda = ctypes.CDLL("libcudart.so")
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        return free.value if cuda.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)) == 0 else 0
    except OSError:
        return 0


def test_the_macs_key_is_screened_on_windows_with_a_bounded_host_footprint(monkeypatch, tmp_path):
    triton = pytest.importorskip("triton")            # noqa: F841
    # A fresh replay cache: a sweep persisted by an earlier run of this key would serve it
    # without a screen, and this cell would then measure nothing (seen 2026-09-26).
    monkeypatch.setenv("NEUROBRIX_REPLAY_CACHE", str(tmp_path / "replay_cache"))
    from neurobrix.kernels import launcher as L, wrappers as W
    from neurobrix.kernels import screen_oracle as S
    from neurobrix.kernels.nbx_tensor import DeviceAllocator, NBXTensor
    try:
        DeviceAllocator.empty_cache_pool()
    except Exception:                                   # noqa: BLE001
        pass
    need = N_ * CIN * H_ * W_ * 2 * 3 + (1 << 30)       # input, a candidate's traffic, headroom
    if _cuda_free_bytes() < need:
        pytest.skip(f"needs {need / 2 ** 30:.1f} GB free on one card")
    monkeypatch.setenv("NBX_SCREEN_WINDOWS", "3")
    L.clear_screened()
    L.SCREEN_WINDOWS.clear()
    rng = np.random.default_rng(7)
    x = NBXTensor.from_numpy(rng.standard_normal((N_, CIN, H_, W_)).astype(np.float32)).to("cuda").to("bfloat16")
    w = NBXTensor.from_numpy((rng.standard_normal((COUT, CIN, K_, K_)) * 0.05).astype(np.float32)).to("cuda").to("bfloat16")

    tracemalloc.start()
    y = W.conv2d_wrapper(x, w, padding=1)
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert tuple(y.shape) == (N_, COUT, H_, W_)

    unscreened_conv = [u for u in L.unscreened() if u.kernel == "conv2d_forward_kernel"]
    assert not unscreened_conv, f"the key was seated without a screen: {unscreened_conv[0].reason}"
    records = [r for r in L.SCREEN_WINDOWS.values() if f"of {N_ * COUT * H_}" in r["windows"]]
    assert records, f"no window record for the key's {N_ * COUT * H_} flat rows: {L.SCREEN_WINDOWS}"
    rec = records[-1]
    assert rec["adjudicated_by"] == "fp64 oracle", rec
    assert rec["kept"] >= 1, rec
    assert peak < HOST_PEAK_BOUND, (f"host peak {peak / 2 ** 20:.0f} MiB during the screen; the bound is "
                                    f"{HOST_PEAK_BOUND / 2 ** 20:.0f} MiB (the whole input in float64 is "
                                    f"{WHOLE_INPUT_F64 / 2 ** 20:.0f} MiB)")

    # the kernel's own rows on the LAST window equal the reference — the window anchored at the
    # final row, where an index that wraps shows or nowhere
    M = N_ * COUT * H_
    r0, r1 = M - 5, M
    named = _named(N_, CIN, H_, W_, COUT, H_, W_, K_, 1, 1, 1, 1)
    ref = C.conv2d_rows(named, S._slab_reader(x, (N_, CIN, H_, W_)), S._to_f64(w), (r0, r1))
    got = S._to_f64(y.contiguous().view(M, W_)[r0:r1, :].contiguous())
    np.testing.assert_allclose(got, ref, rtol=5e-2, atol=5e-2)
