"""The certifier's MoE synthesis builds its expert-address tables with the runtime's own builder
(`pointer_tables.build_ptr_tables`), never from `data_ptr()` by hand: on Metal a shader dereferences captured addresses
under a pinned scope, and a hand-written table made every fused MoE key of the Apple census fail its oracle with a
relative error of exactly 1 — the experts never read (the Mac, 2026-11-04 11:34).

What this test would do if the code were wrong: the hand-written table — the builder is never called (red). One card."""
import numpy as np
import pytest

pytestmark = pytest.mark.usefixtures("needs_device")


@pytest.mark.parametrize("qual,key", [
    ("nbxtensor.ops.fused_moe.fused_moe_kernel",
     (512, 256, 64, 8, False, True, "fp16", "int64", "fp16", "fp16", "int64", "int64", "int64")),
])
def test_a_moe_key_is_synthesised_through_the_runtimes_table_builder(qual, key, monkeypatch):
    from neurobrix.kernels import autotune_certify as A
    from nbxtensor.memory import pointer_tables as PT
    import importlib
    A._bind_hardware_profile()
    tuner = getattr(importlib.import_module(qual.rsplit(".", 1)[0]), qual.rsplit(".", 1)[1])
    built = []
    real = PT.build_ptr_tables
    monkeypatch.setattr(PT, "build_ptr_tables", lambda g, u, d: (built.append(len(g)), real(g, u, d))[1])
    run, oracle, _out = A.synthesize(qual, tuner, key, np.random.default_rng(0))
    got = run().numpy().astype(np.float64)
    want = oracle()
    assert built, "the synthesis wrote its own address table"
    assert np.abs(got - want).max() / np.abs(want).max() < 2e-2
