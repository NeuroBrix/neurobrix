"""Flash attention on an arch whose dot lowers to scalar FMA (Triton >= 3.3 below sm_80,
triton-lang/triton#5066) contracts QK^T in head-dim chunks with a 64x32 tile at 8 warps — stated by
the arch profile, read by the wrapper; every other arch launches exactly as before.

2026-10-03: Allegro's self-attention (B=2, H=24, T=14 080, D=96) took 19.95 s in our kernel and
0.131 s in the vendor's; the kernel ran at 0.18 TFLOP/s at every D and T on a V100 (PTX: 0 mma.sync).

What would this file do if the code were wrong? The Volta rows without qk_chunk -> the first test RED
and the chunked path never taken (the speed test RED); a qk_chunk row added to Ampere or Hopper ->
the first test RED; the launch meta read from a row without qk_chunk -> the second RED; a chunk
loop that drops or misreads a chunk -> the oracle test RED.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]
VENDORS = REPO / "src" / "neurobrix" / "config" / "vendors" / "nvidia"


def _prefill_rows(arch):
    rows = yaml.safe_load((VENDORS / f"{arch}.yml").read_text()).get("sdpa_thresholds") or []
    return [r for r in rows if "seqlen_q_le" not in r]


def test_volta_states_its_fma_tile_and_matrix_archs_do_not():
    below_256 = [r for r in _prefill_rows("volta") if r.get("head_dim_ge", 0) < 256]
    assert below_256 and all(r.get("qk_chunk") == 32 and r.get("num_warps") == 8
                             and (r["block_m"], r["block_n"]) == (64, 32) for r in below_256)
    for arch in ("ampere", "hopper"):
        assert not any("qk_chunk" in r for r in _prefill_rows(arch)), arch


def test_the_launch_meta_comes_only_from_a_chunk_row(monkeypatch):
    from neurobrix.kernels.ops import _configs as K
    rows = [{"seqlen_q_le": 16, "block_m": 16, "block_n": 64, "num_warps": 4},
            {"head_dim_ge": 256, "block_m": 32, "block_n": 32, "num_warps": 4},
            {"head_dim_lt": 256, "block_m": 64, "block_n": 32, "num_warps": 8, "qk_chunk": 32}]
    monkeypatch.setattr(K, "active_vendor_profile", lambda: {"sdpa_thresholds": rows})
    assert K.sdpa_launch_meta(1, 128) == {}                          # decode row: no chunk stated
    assert K.sdpa_launch_meta(4096, 256) == {}
    assert K.sdpa_launch_meta(4096, 96) == {"num_warps": 8, "qk_chunk": 32}
    monkeypatch.setattr(K, "active_vendor_profile", lambda: {})
    assert K.sdpa_launch_meta(4096, 96) == {}


def _cuda_cc():
    try:
        import ctypes
        cuda = ctypes.CDLL("libcudart.so")
        n = ctypes.c_int()
        if cuda.cudaGetDeviceCount(ctypes.byref(n)) != 0 or n.value == 0:
            return None
        major, minor = ctypes.c_int(), ctypes.c_int()
        cuda.cudaDeviceGetAttribute(ctypes.byref(major), 75, 0)
        cuda.cudaDeviceGetAttribute(ctypes.byref(minor), 76, 0)
        return major.value * 10 + minor.value
    except OSError:
        return None


volta = pytest.mark.skipif(_cuda_cc() != 70, reason="needs a Volta card (the FMA-path row in force)")


def _attn(q, k, v):
    from neurobrix.kernels import wrappers as W
    real = W._lk.sdpa_route
    W._lk.sdpa_route = lambda *a, **kw: ("flash", 0)
    try:
        return W.scaled_dot_product_attention_wrapper(q, k, v, k_pre_transposed=False)
    finally:
        W._lk.sdpa_route = real


@volta
@pytest.mark.parametrize("D", [64, 96, 128])
def test_the_chunked_kernel_agrees_with_float64(D):
    from neurobrix.kernels.nbx_tensor import NBXTensor
    rng = np.random.default_rng(D)
    a = [(rng.standard_normal((1, 2, 333, D)) * 0.5).astype(np.float16) for _ in range(3)]
    o = _attn(*[NBXTensor.from_numpy(x) for x in a]).numpy().astype(np.float64)
    q, k, v = [x.astype(np.float64) for x in a]
    s = q @ k.transpose(0, 1, 3, 2) / np.sqrt(D)
    s -= s.max(-1, keepdims=True); p = np.exp(s); p /= p.sum(-1, keepdims=True)
    assert np.abs(o - p @ v).max() < 5e-3


@volta
def test_the_chunked_launch_is_several_times_the_held_one_on_this_card(monkeypatch):
    """Same process, same inputs: the profile's chunk row against the same call with the row's
    launch meta withheld (the held form). Measured 14x at T 4096, D 96 (2026-10-03)."""
    from neurobrix.kernels import wrappers as W
    from neurobrix.kernels.nbx_tensor import NBXTensor, DeviceAllocator
    rng = np.random.default_rng(0)
    q, k, v = [NBXTensor.from_numpy((rng.standard_normal((1, 8, 2048, 96)) * 0.5).astype(np.float16)) for _ in range(3)]

    def timed():
        _attn(q, k, v); DeviceAllocator.sync_device()
        t0 = time.perf_counter(); _attn(q, k, v); DeviceAllocator.sync_device()
        return time.perf_counter() - t0
    chunked = timed()
    monkeypatch.setattr(W, "_sdpa_launch_meta", lambda *a: {})
    held = timed()
    assert held / chunked > 4, (held, chunked)
