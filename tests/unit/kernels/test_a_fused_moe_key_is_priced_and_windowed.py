"""A fused MoE key's certification is priced before any draw and its fp64 oracle is windowed like a matmul's.

The Mac, 2026-11-04 14:58 (RUN 24): with the working set bounded at the profile's 18 186 MiB, a `fused_moe_kernel` key
took ~25 GB of the unified pool 30 s after its start — `_key_operands` knew no MoE key, so `price_key` answered None and
the key ran unpriced, and its oracle held every pair's activations and result in float64 at once (327 680 pairs).

What this test would do if the code were wrong: a MoE key unpriced again — red at the first test; an oracle computed
whole above the cap, or a price that does not bound what the oracle really allocates (numpy's allocations, measured by
tracemalloc) — red at the second; a window that is not the whole computation's rows, or a routed weight dropped — red
at the third; the last test certifies a windowed key on a card."""
from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from neurobrix.kernels import autotune_certify as AC

QUAL = "nbxtensor.ops.fused_moe.fused_moe_kernel"


def _key(N, K, nv, E, mul=False):
    t = "fp16"
    return (N, K, nv, E, mul, not mul, t, "int64", t, t, "int64", "int64", "int64")


def _tuner():
    from nbxtensor.ops.fused_moe import fused_moe_kernel
    return fused_moe_kernel


def test_the_macs_key_is_priced_and_its_oracle_windowed():
    pr = AC.price_key(QUAL, _tuner(), _key(768, 2048, 327680, 128))
    assert pr is not None, "a fused MoE key must be priced before its draw"
    whole_f64 = 327680 * (2048 + 768) * 8
    assert pr["oracle"] < whole_f64 / 4, pr                    # windowed: far below the whole float64 product
    assert pr["draws"] >= 327680 * 2048 * 2 + 128 * 768 * 2048 * 2    # the activations and the experts, as drawn


@pytest.mark.parametrize("cap", [AC.ORACLE_MAX_MACS, 3 * 64 * 96 * 128])
def test_the_price_bounds_what_the_oracle_allocates(monkeypatch, cap):
    monkeypatch.setattr(AC, "ORACLE_MAX_MACS", cap)
    N, K, nv, E = 96, 128, 4096, 8
    rng = np.random.default_rng(0)
    x = AC._arr(rng, (nv, K), "fp16")
    ws = [AC._arr(rng, (N, K), "fp16") for _ in range(E)]
    score = AC._arr(rng, (nv,), "fp16")
    route = rng.integers(0, E, size=nv).astype(np.int64)
    build = AC._moe_oracle_fn(x, ws, score, route, True)
    tracemalloc.start()
    tracemalloc.reset_peak()
    ref = build()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    pr = AC.price_key(QUAL, _tuner(), _key(N, K, nv, E, mul=True))
    assert peak <= pr["oracle"], (peak, pr["oracle"])
    assert (cap < N * K * nv) == hasattr(ref, "slices")


def test_a_window_is_the_whole_computations_rows(monkeypatch):
    N, K, nv, E = 32, 48, 600, 5
    rng = np.random.default_rng(1)
    x = AC._arr(rng, (nv, K), "fp16")
    ws = [AC._arr(rng, (N, K), "fp16") for _ in range(E)]
    score = AC._arr(rng, (nv,), "fp16")
    route = rng.integers(0, E, size=nv).astype(np.int64)
    whole = AC._moe_oracle_fn(x, ws, score, route, True)()
    xs = AC._f64(x)
    for p in (0, 299, 599):
        want = xs[p] @ AC._f64(ws[route[p]]).T * float(AC._f64(score)[p])
        np.testing.assert_allclose(whole[p], want, rtol=1e-12)
    monkeypatch.setattr(AC, "ORACLE_MAX_MACS", 3 * 50 * N * K)
    windowed = AC._moe_oracle_fn(x, ws, score, route, True)()
    for (r0, r1), block in windowed.blocks:
        np.testing.assert_array_equal(block, whole[r0:r1])


def test_a_windowed_key_certifies_on_the_device(monkeypatch):
    """On a card: the certification of a key above the cap measures every candidate on the oracle's row windows."""
    from nbxtensor.memory.allocator import DeviceAllocator
    if DeviceAllocator.device_count() < 1:
        pytest.skip("needs a device")
    AC._bind_hardware_profile()
    monkeypatch.setattr(AC, "ORACLE_MAX_MACS", 3 * 64 * 96 * 128)
    key = _key(96, 128, 2048, 8, mul=True)
    entry = AC.certify_key(QUAL, _tuner(), key, 1e-2, np.random.default_rng(3), bench=lambda fn: (fn(), 1.0)[1])
    assert entry["excluded"] == [] and entry["proof"]["accepted"] >= 1, entry
    assert "row window" in entry["proof"]["oracle"], entry["proof"]["oracle"]
