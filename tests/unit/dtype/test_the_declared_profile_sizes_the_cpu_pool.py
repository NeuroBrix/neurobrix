"""The library's CPU pool is sized by the hardware profile the engine declares (Hocine 2026-10-29 18:11, rule 1: the
profile is the machine's identity card — its compute threads, never a count read off the machine).

What would this file do if the engine did not hand the profile's count to the pool? The pool keeps the library's own
measured default: RED (7 expected)."""
from types import SimpleNamespace

from nbxtensor.backend.cpu import pool


def test_the_declared_profile_sets_the_pool_to_its_compute_threads(monkeypatch):
    from neurobrix.triton import dtype as TD
    monkeypatch.setattr(TD, "profile_triton_has_fp64", lambda profile: False)
    before = pool._STATE.get("threads")
    cpu = SimpleNamespace(compute_threads=7)
    prof = SimpleNamespace(devices=[], cpu=cpu, has_native_bf16=False)
    try:
        TD.declare_hardware_profile(prof)
        assert pool.threads() == 7
    finally:
        pool.set_threads(before)


def test_a_profile_without_its_cpu_is_refused_by_name(monkeypatch):
    import pytest
    from neurobrix.triton import dtype as TD
    monkeypatch.setattr(TD, "profile_triton_has_fp64", lambda profile: False)
    with pytest.raises(ValueError, match="cpu"):
        TD.declare_hardware_profile(SimpleNamespace(devices=[], cpu=None, has_native_bf16=False))
