"""The certifier holds a key's operands at the kernel's own width, and on unified memory it is
bounded by the key's phase price, never by a constant per element.

The Mac's certify on merge-queue-19 (2026-10-05 12:44, `logs/certify_post_write_mq19_2026_10_05`)
refused eight keys as "operands larger than the card" that the planner places whole on the same
profile (the rack, 13:40: default-9f169c79, 96 plans, rc 0): Real-ESRGAN x4plus / x8 and
swin2SR-realworld convolutions at 4224-5120 px, Wan2.1-VACE's VAE conv at batch 162, Wan2.1-T2V's
fp32 addmm at M=163840. Their refused byte counts were exactly `_unified_bytes_per_element`'s
10 bytes per half element (a float32 host draw + the device copy + a float32 readback), a
constant the phase price (`price_key`, measured 2026-09-28) had already shown 40 % over on a
large key. A bf16 operand was carried on the host as float32 values, a fp16 one drawn whole in
float32 before its cast.

What this file does on the old certifier: the eight keys are refused (red, seen on 71dc0710);
a bf16 operand is 4 bytes an element and a fp16 draw peaks at three times its operand; the
phase refusal names no phase; `values` does not exist. The operand's host peak at its own width
is `test_the_certifier_synthesis_is_bounded.test_the_host_peak_is_a_small_multiple_of_the_operand`.
"""
from __future__ import annotations

import numpy as np
import pytest

from neurobrix.kernels import autotune_certify as AC

CONV = "neurobrix.kernels.ops.conv2d.conv2d_forward_kernel"
ADDMM = "neurobrix.kernels.ops.matmul.addmm_kernel"
MM = "neurobrix.kernels.ops.matmul.matmul_kernel"
CARD = 19069403136                                   # default-9f169c79's 18 186 MiB, the refusals' card

REFUSED_ON_MQ19 = [
    (CONV, (1, 64, 4224, 4224, 64, 4224, 4224, 3, 3, 1, 1, 1, 1, 1, 1, 1, False, 'bf16', 'bf16', 'bf16')),
    (CONV, (1, 64, 4288, 4288, 64, 4288, 4288, 3, 3, 1, 1, 1, 1, 1, 1, 1, False, 'bf16', 'bf16', 'bf16')),
    (CONV, (1, 64, 4480, 4480, 64, 4480, 4480, 3, 3, 1, 1, 1, 1, 1, 1, 1, False, 'bf16', 'bf16', 'bf16')),
    (CONV, (1, 64, 5120, 5120, 64, 5120, 5120, 3, 3, 1, 1, 1, 1, 1, 1, 1, False, 'bf16', 'bf16', 'bf16')),
    (CONV, (162, 96, 162, 418, 96, 160, 416, 3, 3, 1, 1, 0, 0, 1, 1, 1, False, 'bf16', 'bf16', 'bf16')),
    (CONV, (162, 96, 266, 266, 96, 264, 264, 3, 3, 1, 1, 0, 0, 1, 1, 1, False, 'bf16', 'bf16', 'bf16')),
    (ADDMM, (163840, 1536, 8960, False, False, 'fp32', 'fp32', 'bf16', 'fp32')),
    (ADDMM, (163840, 8960, 1536, False, False, 'fp32', 'fp32', 'bf16', 'fp32')),
]


class _Tuner:
    pass


class _NoDraw:
    def __getattr__(self, name):
        raise AssertionError(f"a value was drawn ({name}) for a key refused before its draw")


class _Drawn(Exception):
    pass


@pytest.fixture
def out_dtype_of_key(monkeypatch):
    """The output dtype the key's last field names (what `output_dtype` reads off a real tuner)."""
    monkeypatch.setattr(AC.C, "output_dtype", lambda tuner, key: key[-1])


@pytest.mark.parametrize("qual,key", REFUSED_ON_MQ19, ids=[str(k[:4]) for _, k in REFUSED_ON_MQ19])
def test_a_key_the_plan_places_whole_reaches_its_draw_on_unified_memory(qual, key, out_dtype_of_key,
                                                                        monkeypatch):
    def drawn(*a, **k):
        raise _Drawn()
    monkeypatch.setattr(AC, "_arr", drawn)
    with pytest.raises(_Drawn):
        AC.synthesize(qual, _Tuner(), key, np.random.default_rng(0), card_bytes=CARD, unified=True)


def test_on_unified_memory_a_key_over_the_card_is_refused_by_its_phases(out_dtype_of_key):
    qual, key = REFUSED_ON_MQ19[3]
    peak = AC.price_key(qual, _Tuner(), key)["peak"]
    with pytest.raises(AC.KeyTooLargeForClass) as e:
        AC.synthesize(qual, _Tuner(), key, _NoDraw(), card_bytes=peak - 1, unified=True)
    assert e.value.asked == peak and e.value.card == peak - 1
    for word in ("draws", "device", "oracle", "unified"):
        assert word in str(e.value), f"the refusal does not name {word!r}: {e.value}"


def test_the_draws_are_priced_at_the_kernels_width(out_dtype_of_key):
    qual, key = REFUSED_ON_MQ19[0]
    x, w = 64 * 4224 * 4224, 64 * 64 * 9
    assert AC.price_key(qual, _Tuner(), key)["draws"] == 2 * (x + w)
    qual, key = REFUSED_ON_MQ19[6]
    a, b, bias = 163840 * 8960, 8960 * 1536, 1536
    assert AC.price_key(qual, _Tuner(), key)["draws"] == 4 * (a + b) + 2 * bias


def test_a_bf16_operand_is_its_rounded_bits_and_reads_back_as_its_values():
    a = AC._arr(np.random.default_rng(1), (257, 129), "bf16")
    assert a._nbx_dtype == "bf16" and a.dtype == np.uint16
    ref = np.random.default_rng(1).standard_normal((257, 129), dtype=np.float32) * np.float32(0.1)
    bits = AC.f32_to_bf16_bits(ref)
    assert np.array_equal(np.asarray(a), bits)
    v = AC.values(a)
    assert v.dtype == np.float32 and np.array_equal(v, AC.bf16_bits_to_f32(bits))
    assert np.array_equal(AC.values(a[3:9, 1:4]), AC.bf16_bits_to_f32(bits)[3:9, 1:4])


def test_the_same_draws_as_before_for_every_width():
    """Drawn in chunks, the values are the whole draw's: a certificate made before stays the
    proof of the same numbers."""
    for dt, np_dt in (("fp16", np.float16), ("fp32", np.float32)):
        a = AC._arr(np.random.default_rng(5), (3, 1_000_003), dt)
        ref = (np.random.default_rng(5).standard_normal((3, 1_000_003), dtype=np.float32)
               * np.float32(0.1)).astype(np_dt)
        assert np.array_equal(np.asarray(a), ref), dt


def test_a_bf16_carrier_refuses_a_float_cast():
    """Its bits cast to float are integers, not values: refused by name, never silently read."""
    a = AC._arr(np.random.default_rng(1), (4, 4), "bf16")
    with pytest.raises(TypeError, match="values"):
        a.astype(np.float64)
    assert np.asarray(a).astype(np.uint32).dtype == np.uint32       # the bits stay readable as bits


def test_the_oracles_read_bf16_values_not_bits():
    r = np.random.default_rng(3)
    a, b = AC._arr(r, (2, 24, 16), "bf16"), AC._arr(r, (2, 16, 8), "bf16")
    got = AC._matmul_oracle_fn(a, b)()
    assert np.array_equal(got, AC.values(a).astype(np.float64) @ AC.values(b).astype(np.float64))
    x, w = AC._arr(r, (1, 4, 9, 7), "bf16"), AC._arr(r, (6, 4, 3, 3), "bf16")
    from neurobrix.kernels.oracles.conv2d_fp64 import conv2d_reference_window
    ref = conv2d_reference_window(AC.values(x).astype(np.float64), AC.values(w).astype(np.float64),
                                  stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1)
    assert np.array_equal(AC._conv_oracle_fn(x, w, (1, 1), (1, 1), (1, 1), 1)(), ref)
