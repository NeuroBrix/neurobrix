"""P-NONDET-LONG-ROW pinning tests — deterministic long-context prefill.

The measured cause chain (2026-08-23): the long row's prefill (T=4,164,
D=128 pow2) has an fp32 scores tensor 3% OVER the Volta math budget →
routed to the flash kernel → the exact D>=128 pad-by-one detour lands
it in the masked-load specialisation → the DOCUMENTED Volta race:
kernel-level 3 distinct outputs in 5 identical calls at that shape;
end-to-end the first replayed-step logits differed on EVERY pair of
runs, and one greedy near-tie (~token 10) made the row's output
bimodal (the stable sha pair). Forcing math: 4/4 runs byte-identical
at every step. Fix: chunked math for over-budget pow2 shapes, bounded
by `memory.sdpa_math_max_chunks` (vendor yml).

Three pins:
  1. DETERMINISM at the exact trigger shape through the chunked route
     — three calls, identical bytes. FAILS on the pre-fix flash route
     (measured 3 distinct in 5).
  2. CORRECTNESS: chunked output vs the independent float64 reference.
  3. ROUTING + ACTIVATION: over-budget pow2 routes to
     _math_attention_chunked when the ceiling allows; keeps flash when
     the chunk count exceeds it (the registered video-class residual);
     and the route is proven by a call counter, not assumed.

The budget helpers are patched in-test (a bare test process has no
hardware profile — the vacuous-equivalence lesson: pin the route).

Runnable two ways:
  - pytest:  PYTHONPATH=src python3 -m pytest tests/unit/kernels/test_prefill_determinism.py -v
  - script:  PYTHONPATH=src python3 tests/unit/kernels/test_prefill_determinism.py
"""
from __future__ import annotations

import ctypes

try:
    import pytest
except ModuleNotFoundError:  # script-mode under the pytest-less GPU venv
    class _NoPytest:
        @staticmethod
        def skip(*a, **k):
            raise SystemExit(0)

    pytest = _NoPytest()  # type: ignore

import numpy as np

from neurobrix.kernels.nbx_tensor import NBXTensor, DeviceAllocator
from neurobrix.kernels import wrappers as W

BOUND = 2e-03  # same derived bound as the decode oracles


def _has_gpu() -> bool:
    try:
        DeviceAllocator.set_device(0)
        return True
    except Exception:
        return False


def _d2h(t) -> bytes:
    buf = (ctypes.c_char * t._nbytes)()
    DeviceAllocator.memcpy(ctypes.addressof(buf), t.data_ptr(),
                           t._nbytes, 2)
    return bytes(buf)


class _pinned_route:
    """Patch the two budget helpers so the chunked route is decided by
    THIS test, not by the machine's profile."""

    def __init__(self, budget, max_chunks):
        self.budget, self.max_chunks = budget, max_chunks

    def __enter__(self):
        self._b = W._sdpa_math_scores_budget_bytes
        self._f = getattr(W, "_sdpa_math_scores_device_fraction", None)
        self._c = W._sdpa_math_max_chunks
        W._sdpa_math_scores_budget_bytes = lambda: self.budget
        if self._f is not None:
            W._sdpa_math_scores_device_fraction = lambda: 0.0
        W._sdpa_math_max_chunks = lambda: self.max_chunks
        return self

    def __exit__(self, *a):
        W._sdpa_math_scores_budget_bytes = self._b
        if self._f is not None:
            W._sdpa_math_scores_device_fraction = self._f
        W._sdpa_math_max_chunks = self._c


def _mk(B, H, H_kv, T, D, seed=7):
    rng = np.random.default_rng(seed)
    q = NBXTensor.from_numpy(
        (rng.standard_normal((B, H, T, D)) * 0.1).astype(np.float16))
    k = NBXTensor.from_numpy(
        (rng.standard_normal((B, H_kv, T, D)) * 0.1).astype(np.float16))
    v = NBXTensor.from_numpy(
        (rng.standard_normal((B, H_kv, T, D)) * 0.1).astype(np.float16))
    return q, k, v


def test_long_prefill_is_deterministic_through_the_chunked_route() -> None:
    """The exact trigger: pow2 D, causal, scores just over budget.
    Three identical calls, identical bytes. The pre-fix flash route
    measured 3 distinct outputs in 5 at this shape class."""
    if not _has_gpu():
        pytest.skip("no GPU")
    B, H, H_kv, T, D = 1, 32, 8, 4164, 128
    q, k, v = _mk(B, H, H_kv, T, D)
    calls = {"n": 0}
    orig = W._math_attention_chunked

    def counting(*a, **kw):
        calls["n"] += 1
        return orig(*a, **kw)

    W._math_attention_chunked = counting
    try:
        with _pinned_route(budget=2147483648, max_chunks=16):
            outs = [_d2h(W.scaled_dot_product_attention_wrapper(
                q, k, v, attn_mask=None, is_causal=True,
                scale=1.0 / np.sqrt(D))) for _ in range(3)]
    finally:
        W._math_attention_chunked = orig
    assert calls["n"] == 3, (
        f"chunked route fired {calls['n']} times, wanted 3 — the "
        f"trigger shape did not route as pinned (vacuous determinism)")
    assert outs[0] == outs[1] == outs[2], (
        "long-prefill outputs differ across identical calls — the "
        "P-NONDET-LONG-ROW race is back")


def test_chunked_matches_float64() -> None:
    """Chunked math vs the independent float64 reference (causal)."""
    if not _has_gpu():
        pytest.skip("no GPU")
    B, H, H_kv, T, D = 1, 8, 2, 512, 64
    rng = np.random.default_rng(3)
    qn = (rng.standard_normal((B, H, T, D)) * 0.1).astype(np.float16)
    kn = (rng.standard_normal((B, H_kv, T, D)) * 0.1).astype(np.float16)
    vn = (rng.standard_normal((B, H_kv, T, D)) * 0.1).astype(np.float16)
    q, k, v = (NBXTensor.from_numpy(x) for x in (qn, kn, vn))
    out = W._math_attention_chunked(q, k, v, None, True,
                                    1.0 / np.sqrt(D), chunk_rows=128)
    got = np.frombuffer(_d2h(out), dtype=np.float16).astype(
        np.float64).reshape(B, H, T, D)

    g = H // H_kv
    K = np.repeat(kn.astype(np.float64), g, axis=1)
    V = np.repeat(vn.astype(np.float64), g, axis=1)
    s = np.einsum("bhqd,bhkd->bhqk", qn.astype(np.float64), K) / np.sqrt(D)
    tri = np.triu(np.ones((T, T)), k=1).astype(bool)
    s[:, :, tri] = -np.inf
    m = s.max(-1, keepdims=True)
    e = np.exp(s - m)
    ref = np.einsum("bhqk,bhkd->bhqd", e / e.sum(-1, keepdims=True), V)
    err = float(np.abs(got - ref).max() / np.abs(ref).max())
    assert err <= BOUND, f"chunked math rel err {err:.3e} > {BOUND:.0e}"


def test_video_scale_keeps_its_path() -> None:
    """A shape whose chunk count exceeds the ceiling must NOT take the
    chunked route (the registered residual class keeps its behavior)."""
    if not _has_gpu():
        pytest.skip("no GPU")
    calls = {"n": 0}
    orig = W._math_attention_chunked

    def counting(*a, **kw):
        calls["n"] += 1
        return orig(*a, **kw)

    W._math_attention_chunked = counting
    try:
        # tiny budget => chunk_rows 128, T=4096 => 32 chunks > ceiling 16
        with _pinned_route(budget=128 * 1 * 4 * 4096 * 4, max_chunks=16):
            B, H, H_kv, T, D = 1, 4, 4, 4096, 64
            q, k, v = _mk(B, H, H_kv, T, D, seed=9)
            W.scaled_dot_product_attention_wrapper(
                q, k, v, attn_mask=None, is_causal=True,
                scale=1.0 / np.sqrt(D))
    finally:
        W._math_attention_chunked = orig
    assert calls["n"] == 0, (
        "over-ceiling shape took the chunked route — the video-class "
        "cost bound is not being honored")


if __name__ == "__main__":
    if not _has_gpu():
        raise SystemExit("no GPU")
    test_long_prefill_is_deterministic_through_the_chunked_route()
    print("PASS: long-prefill deterministic x3 through the chunked route")
    test_chunked_matches_float64()
    print("PASS: chunked math matches float64")
    test_video_scale_keeps_its_path()
    print("PASS: over-ceiling shapes keep their path")


# ---------------------------------------------------------------------------
# Per-DEVICE budget composition (P-PREFILL-TRITON internal bar, 2026-08-31).
# Discovery: the rig is heterogeneous (2x V100-16G + 2x V100-32G); the
# per-ARCH volta.yml budget (2 GiB) let a 1.90 GiB scores tensor route to
# UN-chunked math on a 16 GB card whose free memory was 1.87 GiB -> OOM at
# xlong prefill (block_scatter, model on cuda:0). The budget must compose
# with the EXECUTING device's capacity, data-driven from the profile
# (no driver query): min(yml bytes, yml fraction x device memory_mb).
# fraction 0.066 keeps 32 GB cards at the 2 GiB base via min() (zero
# route change there) and caps 16 GB cards at ~1.06 GiB (chunked arms
# instead).
# ---------------------------------------------------------------------------

class _FakeDev:
    def __init__(self, index, memory_mb):
        self.index, self.memory_mb = index, memory_mb
        self.architecture, self.brand = "volta", "nvidia"


class _FakeProfile:
    def __init__(self, devs):
        self.devices = devs


def test_budget_composes_with_executing_device_memory() -> None:
    import neurobrix.kernels.wrappers as W
    # The values the auto profile ACTUALLY records (config/hardware/
    # default.yml, from nvidia-smi memory.total — verified 2026-09-01):
    # 16384 / 32768. Expectations below are computed from these same
    # constants, never asserted as free-standing round byte counts —
    # the 2026-08-31 gardien review caught an "exactly 2 GiB" identity
    # baked into an earlier draft of this test.
    devs = [_FakeDev(0, 16384), _FakeDev(2, 32768)]
    orig_prof, orig_base = W.get_hardware_profile, W._sdpa_math_scores_budget_bytes
    orig_frac = W._sdpa_math_scores_device_fraction
    W.get_hardware_profile = lambda: _FakeProfile(devs)
    W._sdpa_math_scores_budget_bytes = lambda: 2 << 30
    W._sdpa_math_scores_device_fraction = lambda: 0.066
    try:
        # 16G card: capped at 0.066 x 16384 MiB ~= 1.06 GiB
        assert W._sdpa_math_scores_budget_bytes_for(0) == int(
            16384 * 1024 * 1024 * 0.066)
        # 32G card: 0.066 x 32768 MiB > 2 GiB -> min() keeps the base
        assert W._sdpa_math_scores_budget_bytes_for(2) == 2 << 30
        # unknown index / None (absence, not error): plain base budget
        assert W._sdpa_math_scores_budget_bytes_for(None) == 2 << 30
        assert W._sdpa_math_scores_budget_bytes_for(7) == 2 << 30
    finally:
        W.get_hardware_profile = orig_prof
        W._sdpa_math_scores_budget_bytes = orig_base
        W._sdpa_math_scores_device_fraction = orig_frac


def test_nbxtensor_device_index_field_reaches_the_cap() -> None:
    """The call site consumes NBXTensor._device_idx — NOT ._device
    (bare type string 'cuda') and NOT .device (returns SELF, engraved
    trap). Guard the field contract the routing depends on."""
    from neurobrix.kernels.nbx_tensor import NBXTensor
    # Instance-level field contract: every construction path must stamp
    # a real integer index (bare "cuda" defaults to device 0).
    t = NBXTensor.__new__(NBXTensor)
    assert "_device_idx" in NBXTensor.__slots__, \
        "_device_idx slot is the routing contract"
    del t
    import inspect
    import neurobrix.kernels.wrappers as W
    src = inspect.getsource(W.scaled_dot_product_attention_wrapper)
    assert "_device_idx" in src, "route decision must read q._device_idx"
    assert 'getattr(q, "_device",' not in src.replace(
        "_device_idx", ""), "bare _device (type string) must not be used"


def test_budget_unchanged_without_fraction_key() -> None:
    import neurobrix.kernels.wrappers as W
    devs = [_FakeDev(0, 16384)]
    orig_prof, orig_base = W.get_hardware_profile, W._sdpa_math_scores_budget_bytes
    orig_frac = W._sdpa_math_scores_device_fraction
    W.get_hardware_profile = lambda: _FakeProfile(devs)
    W._sdpa_math_scores_budget_bytes = lambda: 2 << 30
    W._sdpa_math_scores_device_fraction = lambda: 0.0
    try:
        assert W._sdpa_math_scores_budget_bytes_for(0) == 2 << 30
    finally:
        W.get_hardware_profile = orig_prof
        W._sdpa_math_scores_budget_bytes = orig_base
        W._sdpa_math_scores_device_fraction = orig_frac


# ---------------------------------------------------------------------------
# ACTIVATION legs (gardien review 2026-08-31, LAND-BLOCKING conditions):
# route decisions proven on REAL NBXTensors on REAL devices of each memory
# class — not on strings the call site can never produce. Spy on the two
# math entry points; assert the decided route per (device class, shape
# window). Skipped cleanly off-GPU / on homogeneous rigs.
# ---------------------------------------------------------------------------

def _gpu_classes():
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout
        devs = [(int(l.split(",")[0]), int(l.split(",")[1]))
                for l in out.strip().splitlines() if l.strip()]
        small = next((i for i, m in devs if m < 20000), None)
        big = next((i for i, m in devs if m >= 20000), None)
        return small, big
    except Exception:
        return None, None


def _stash_real_profile():
    """A bare test process has no stashed profile -> budget 0 -> both
    activation tests would pass VACUOUSLY on the 2 GiB fallback (the
    exact vacuous-equivalence class). Load the machine's real profile
    and stash it; return a zero-arg restore callable that puts BOTH
    stash-touched globals (profile + native-bf16 flag) back, so later
    tests in the same process see the bare-process state again."""
    import neurobrix.kernels.wrappers as W
    prev_prof = W.get_hardware_profile()
    prev_flag = W.has_native_bf16()
    if prev_prof is None:
        from neurobrix.core.prism.autodetect import load_default_profile
        W.set_hardware_profile(load_default_profile())

    def _restore():
        W._NBX_HW_PROFILE = prev_prof
        W._NBX_HAS_NATIVE_BF16 = prev_flag

    return _restore


def _route_spy(q_shape, kv_shape, device_idx):
    """Run the wrapper with spies on both math entries; return which
    route fired ('chunked' | 'math' | 'flash')."""
    import numpy as np
    import neurobrix.kernels.wrappers as W
    from neurobrix.kernels.nbx_tensor import NBXTensor

    def mk(shape):
        t = NBXTensor.empty(shape, dtype="float16",
                            device=f"cuda:{device_idx}")
        z = np.zeros(shape, dtype=np.float16)
        from neurobrix.kernels.nbx_tensor import DeviceAllocator
        DeviceAllocator.memcpy(t.data_ptr(),
                               z.ctypes.data, z.nbytes, kind=1)  # H2D
        return t

    q, k, v = mk(q_shape), mk(kv_shape), mk(kv_shape)
    fired = {}
    orig_c, orig_m = W._math_attention_chunked, W._math_attention

    def spy_c(*a, **kw):
        fired["route"] = "chunked"
        raise _SpyHit()

    def spy_m(*a, **kw):
        fired["route"] = "math"
        raise _SpyHit()

    W._math_attention_chunked, W._math_attention = spy_c, spy_m
    try:
        try:
            W.scaled_dot_product_attention_wrapper(q, k, v, is_causal=True)
            fired.setdefault("route", "flash")
        except _SpyHit:
            pass
    finally:
        W._math_attention_chunked, W._math_attention = orig_c, orig_m
    return fired.get("route")


class _SpyHit(Exception):
    pass


def test_16g_nonpow2_window_routes_chunked_on_device() -> None:
    """16G card, hd=112 (Sana class), scores in (device cap, 2 GiB]:
    must CHUNK — the pre-fix behaviour on this window was un-chunked
    math (OOM class); the un-capped non-pow2 fallthrough was silent
    flash (band-risk class). Both are wrong answers here."""
    import pytest
    small, _ = _gpu_classes()
    if small is None:
        pytest.skip("no <20GB GPU on this host")
    restore = _stash_real_profile()
    try:
        # B=1 H=20 T=4096 hd=112: scores 20*4096^2*4 = 1.34 GiB — inside
        # (0.066 x 16384 MiB ~= 1.06 GiB, 2 GiB]. row=20*4096*4=320KiB,
        # chunk_rows 3456 -> 2 chunks <= ceiling 16.
        route = _route_spy((1, 20, 4096, 112), (1, 20, 4096, 112), small)
    finally:
        restore()
    assert route == "chunked", f"expected chunked on 16G window, got {route}"


def test_32g_pow2_window_keeps_prefix_route_on_device() -> None:
    """32G card, hd=128 pow2, scores ~1.5 GiB (inside the 16G cap
    window but under the 2 GiB base): route must stay plain math —
    byte-identical to the pre-fix decision. THE 32G no-change proof by
    activation, not arithmetic."""
    import pytest
    _, big = _gpu_classes()
    if big is None:
        pytest.skip("no >=20GB GPU on this host")
    restore = _stash_real_profile()
    try:
        # B=1 H=24 T=4096 hd=128: scores 24*4096^2*4 = 1.61 GiB (the Wan
        # block-0 class the gardien named) — under 2 GiB base, over the
        # 16G cap. On 32G the cap is min(2GiB, 0.066x32768MiB)=2GiB.
        route = _route_spy((1, 24, 4096, 128), (1, 24, 4096, 128), big)
    finally:
        restore()
    assert route == "math", f"expected un-chunked math on 32G, got {route}"
