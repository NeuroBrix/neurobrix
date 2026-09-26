"""The certifier's synthesis is bounded: a key too large for the card is refused before a value
is drawn, and an operand is drawn in its own precision, not in float64.

On 2026-09-26 two certifiers sat an hour in `synthesize` holding 65 GB and 52 GB of host memory
(30 GB left on the host) for keys whose operands asked 37 GiB of a 31.7 GiB card: the operands
were drawn in float64 (plus a scaled float64 copy) before the device allocation failed and the
key was reported TOO LARGE. The Mac reported the same class on a 24 GB unified-memory machine.

What these would do on the old synthesis: the refusal test fails (the draw happens and then no
refusal is raised), and the peak test fails (the float64 draw and its scaled copy peak near five
times an fp32 operand's bytes).
"""
from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from neurobrix.kernels import autotune_certify as AC


class _NoDraw:
    def __getattr__(self, name):
        raise AssertionError(f"a value was drawn ({name}) for a key refused as too large")


class _Tuner:
    pass


@pytest.fixture
def fp16_out(monkeypatch):
    monkeypatch.setattr(AC.C, "output_dtype", lambda tuner, key: "fp16")


def test_a_key_larger_than_the_card_is_refused_before_any_draw(fp16_out):
    qual = "neurobrix.kernels.ops.matmul.matmul_kernel"
    key = (1 << 20, 4096, 4096, True, False, "fp16", "fp16", "fp16")     # 8 GiB + 32 MiB + 8 GiB
    with pytest.raises(AC.KeyTooLargeForClass) as e:
        AC.synthesize(qual, _Tuner(), key, _NoDraw(), card_bytes=16 * 2**30)
    assert e.value.asked > 16 * 2**30 and e.value.card == 16 * 2**30
    assert AC.oversize_for_class(e.value) == (e.value.asked, 16 * 2**30)


def test_a_key_that_fits_is_drawn(fp16_out):
    qual = "neurobrix.kernels.ops.matmul.matmul_kernel"
    made = AC.synthesize(qual, _Tuner(), (64, 32, 16, True, False, "fp16", "fp16", "fp16"),
                         np.random.default_rng(0), card_bytes=16 * 2**30)
    assert made is not None


def test_bf16_operands_are_exactly_representable_and_rounded_to_nearest_even():
    a = AC._arr(np.random.default_rng(1), (257, 129), "bf16")
    assert a._nbx_dtype == "bf16" and a.dtype == np.float32
    u = np.asarray(a).view(np.uint32)
    assert not np.any(u & np.uint32(0xFFFF))
    ref = np.random.default_rng(1).standard_normal((257, 129), dtype=np.float32) * np.float32(0.1)
    assert np.array_equal(AC.bf16_bits_to_f32(AC.f32_to_bf16_bits(ref)), np.asarray(a))


@pytest.mark.parametrize("dt,ceiling", [("fp32", 1.2), ("fp16", 3.2), ("bf16", 1.2)])
def test_the_host_peak_is_a_small_multiple_of_the_operand(dt, ceiling):
    shape = (2048, 2048)
    tracemalloc.start()
    a = AC._arr(np.random.default_rng(2), shape, dt)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak <= ceiling * a.nbytes, f"{dt}: peak {peak} for {a.nbytes} operand bytes"
