"""`backend.launcher.do_bench` spends about `rep` milliseconds on its timed repetitions, whatever each repetition's
flush and timing cost beside the kernel: the count is sized from one whole repetition.

Before (trial M of 0.6.13, 2026-11-12): sized from the kernel's own time, a 0.01 ms kernel got 10 000 repetitions,
each behind a 256 MB clear; a new user's first long request on Metal (kernel configurations measured once) ran past
10 minutes.

What this test does if the code is wrong: with a fast kernel and a slow flush, thousands of repetitions run instead
of a few — red.

Run: python -m pytest nbxtensor/tests/test_a_benchmark_spends_its_time_budget_not_more.py
"""
import time


def test_a_fast_kernel_behind_a_slow_flush_runs_within_the_budget(monkeypatch):
    from nbxtensor.backend import launcher as L
    from nbxtensor.memory.allocator import DeviceAllocator

    class Flush:
        _nbytes = 4

        def data_ptr(self):
            return 0
    monkeypatch.setattr(L, "_time_ms", lambda fn, n: 0.01)                     # a 0.01 ms kernel
    monkeypatch.setattr(DeviceAllocator, "stream_synchronize", staticmethod(lambda *a: None))
    clears = []
    monkeypatch.setattr(DeviceAllocator, "memset_cuda",
                        staticmethod(lambda *a: (clears.append(1), time.sleep(0.002))))   # a 2 ms clear
    import nbxtensor.tensor as T
    monkeypatch.setattr(T.NBXTensor, "empty", staticmethod(lambda *a, **k: Flush()))
    began = time.perf_counter()
    L.do_bench(lambda: None, warmup=1, rep=100, budget_ms=1e9)
    spent = time.perf_counter() - began
    assert len(clears) <= 60, f"{len(clears)} repetitions: sized from the kernel alone, not from a whole repetition"
    assert spent < 1.0, f"{spent:.2f} s for a 100 ms budget"


def test_a_certification_keeps_the_count_it_has_always_used(monkeypatch):
    """Certification sizes its repetitions from the kernel's time (`size_repetitions_by="kernel"`, passed by name by
    `ops.autotune_harness`): its tables stay comparable with every one certified before (the Orchestrateur's decision,
    2026-11-12)."""
    import inspect
    from nbxtensor.backend import launcher as L
    from nbxtensor.memory.allocator import DeviceAllocator
    from nbxtensor.ops import autotune_harness as H

    class Flush:
        _nbytes = 4

        def data_ptr(self):
            return 0
    clears = []
    monkeypatch.setattr(L, "_time_ms", lambda fn, n: 0.5)                      # a 0.5 ms kernel
    monkeypatch.setattr(DeviceAllocator, "stream_synchronize", staticmethod(lambda *a: None))
    monkeypatch.setattr(DeviceAllocator, "memset_cuda", staticmethod(lambda *a: (clears.append(1), time.sleep(0.002))))
    import nbxtensor.tensor as T
    monkeypatch.setattr(T.NBXTensor, "empty", staticmethod(lambda *a, **k: Flush()))
    L.do_bench(lambda: None, warmup=1, rep=100, budget_ms=1e9, size_repetitions_by="kernel")
    assert len(clears) == 200                                                   # rep / kernel time: 100 / 0.5
    src = inspect.getsource(H)
    assert src.count('size_repetitions_by="kernel"') == src.count("L.do_bench("), "a certification timer lost its count"
