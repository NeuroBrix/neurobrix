"""Accelerate before macOS 15 has no BLASSetThreading (macOS 15.0+; PyMC #8573, libKriging #416). nbxtensor 0.0.8 then
found no BLAS it knew and raised at the CPU pool's setup, so every Triton run died at start on macOS 14 (measured by the
Mac, 2026-11-09 16:35, simulating that Accelerate through the entry point). The kind is now named, its threads are not
set, and the pool runs.

What this test would do if the code were wrong: blas() raising on an Accelerate without the symbol (red), a pool thread's
initializer or parallel_for raising with it (red), doctor not naming it (red), an Accelerate WITH the symbol no longer
controlled per thread (red)."""
from __future__ import annotations

import pytest

from nbxtensor.backend.cpu import blas as B
from nbxtensor.backend.cpu import pool as P


class _Accelerate14:
    """libBLAS of macOS 14: the BLAS, no thread API."""


class _Accelerate15:
    def __init__(self):
        self.calls = []
        outer = self

        class _Fn:
            argtypes = restype = None

            def __init__(self, name):
                self.name = name

            def __call__(self, *a):
                outer.calls.append((self.name, a))
                return 0 if self.name == "set" else 1
        self.BLASSetThreading = _Fn("set")
        self.BLASGetThreading = _Fn("get")


@pytest.fixture
def darwin(monkeypatch):
    def use(lib):
        monkeypatch.setattr(B.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(B.ctypes, "CDLL", lambda path: lib)
        B.blas.cache_clear()
    yield use
    monkeypatch.undo()                  # the real platform back BEFORE anything asks blas() again
    B.blas.cache_clear()
    P.set_threads(None)                 # the pool rebuilt on this process's real BLAS


def test_accelerate_without_thread_api_is_named_and_the_pool_runs(darwin):
    darwin(_Accelerate14())
    assert B.blas()[0] == B.ACCELERATE_UNCONTROLLED
    B.set_blas_threads(4)                                   # nothing to set, nothing raised
    assert B.blas_threads_of_this_thread() is None
    P.set_threads(4)
    P.set_split_threshold("blas14", 1)
    assert P.parallel_for(8, lambda a, b: b - a, "blas14") == [2, 2, 2, 2]


def test_doctor_names_it(darwin, monkeypatch, capsys):
    import nbxtensor.__main__ as main_mod
    from nbxtensor.backend import metal_backend
    darwin(_Accelerate14())
    monkeypatch.setattr(main_mod.sys, "platform", "darwin")
    monkeypatch.setattr(metal_backend, "is_apple_silicon", lambda: False)
    main_mod.doctor()
    assert "BLAS: Accelerate without per-thread control" in capsys.readouterr().out


def test_accelerate_with_the_thread_api_is_still_set_per_thread(darwin):
    lib = _Accelerate15()
    darwin(lib)
    assert B.blas()[0] == "accelerate"
    B.set_blas_threads(1)
    assert ("set", (1,)) in lib.calls
