"""`nbxtensor.backend.device_measure` and `nbxtensor.io.read_rate.evict` / `cold_read_gbps`: a device's GEMM, GEMV
and copy speeds and a file's cold read, measured by the library. They lived in the engine (NeuroBrix's
`core/prism/facts_probe.py`) until 2026-11-12.

What this test would do if the code were wrong: a GEMM rate not 2*n^3 over the timed milliseconds, a GEMV rate over
anything but the weight's bytes, a device-to-device copy counted once instead of read plus write, a copy rate taken
from the wrong direction's time, an empty file refused by the eviction, a cold read that does not evict — RED."""
import os

import pytest

from nbxtensor.backend import device_measure as M
from nbxtensor.io import read_rate as R


def _accelerator():
    from nbxtensor.memory.allocator import DeviceAllocator
    if DeviceAllocator.device_count() == 0:          # only a machine without an accelerator skips; any other error fails
        pytest.skip("no accelerator on this machine")


def test_a_gemm_rate_is_two_n_cubed_over_the_timed_milliseconds(monkeypatch):
    _accelerator()
    from nbxtensor.backend import launcher as L
    assert M.matmul_tflops("float16", sizes=(32, 64), warmup=1, rep=2)["best"] > 0   # timed for real (tuned once)
    ms = {32: 0.5, 64: 2.0}
    sizes = iter((32, 64))
    monkeypatch.setattr(L, "do_bench", lambda fn, warmup, rep: (fn(), ms[next(sizes)])[1])
    got = M.matmul_tflops("float16", sizes=(32, 64), warmup=1, rep=1)
    want = {n: 2.0 * n ** 3 / (ms[n] * 1e-3) / 1e12 for n in (32, 64)}
    assert got["per_size"] == pytest.approx(want) and got["best"] == pytest.approx(max(want.values()))


def test_a_gemv_rate_is_the_weight_bytes_over_the_time(monkeypatch):
    _accelerator()
    from nbxtensor.backend import launcher as L
    assert M.gemv_gbps("float16", k=256, n=512, reps=2) > 0                          # timed for real (tuned once)
    monkeypatch.setattr(L, "do_bench", lambda fn, warmup, rep: (fn(), 0.001)[1])
    assert M.gemv_gbps("float16", k=256, n=512, reps=1) == round(256 * 512 * 2 / 1e-6 / 1e9, 1)


def test_each_copy_rate_is_its_bytes_over_its_own_direction_s_time(monkeypatch):
    _accelerator()
    seconds = {3: 1.0, 1: 2.0, 2: 4.0}
    seen = []

    def timed(dst, src, nbytes, kind, reps):
        seen.append((kind, reps))
        return seconds[kind] if (kind, reps) != (1, 1) else 8.0     # the pageable upload: kind 1, half the reps
    monkeypatch.setattr(M, "_timed_copies", timed)
    n = 1 << 20
    got = M.copies(n, reps=2)
    assert got == {"memory_gbps": 2 * n / 1.0 / 1e9, "h2d_gbps": n / 2.0 / 1e9, "d2h_gbps": n / 4.0 / 1e9,
                   "h2d_pageable_gbps": n / 8.0 / 1e9}
    assert seen == [(3, 4), (1, 2), (2, 2), (1, 1)]


def test_the_copies_are_timed_on_the_device():
    _accelerator()
    got = M.copies(1 << 20, reps=2)
    assert set(got) == {"memory_gbps", "h2d_gbps", "d2h_gbps", "h2d_pageable_gbps"}
    assert all(v > 0 for v in got.values()), got


def test_a_cold_read_evicts_the_file_then_reads_all_of_it(tmp_path, monkeypatch):
    empty, full = tmp_path / "empty", tmp_path / "full"
    empty.write_bytes(b"")
    full.write_bytes(os.urandom(3 << 20))
    R.evict([empty, full])                                   # an empty file has nothing cached: no refusal
    evicted = []
    real = R._evict
    monkeypatch.setattr(R, "_evict", lambda p, size: evicted.append((p, size)) or real(p, size))
    assert R.cold_read_gbps(full) > 0
    assert evicted == [(str(full), 3 << 20)]
