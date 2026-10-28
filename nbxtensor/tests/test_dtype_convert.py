"""nbxtensor.dtype.convert: the one place bfloat16 meets float32 and float16 on the host."""
import numpy as np
import pytest

from nbxtensor.dtype import convert as C


def _every_pattern():
    return np.arange(2 ** 16, dtype=np.uint32).astype(np.uint16)


def test_widening_is_exact_and_round_trips():
    bits = _every_pattern()
    f = C.bf16_bits_to_f32(bits)
    finite = np.isfinite(f)
    assert np.array_equal(C.f32_to_bf16_bits(f[finite]), bits[finite])


def test_rounding_is_nearest_even():
    # 1 + 2^-8 sits exactly between two bf16 values (1 and 1 + 2^-7): ties to the even one, 1.0
    half = np.array([1 + 2 ** -8, 1 + 3 * 2 ** -8], dtype=np.float32)
    got = C.bf16_bits_to_f32(C.f32_to_bf16_bits(half))
    assert got.tolist() == [1.0, 1 + 2 ** -6]


def test_every_nan_becomes_torchs_quiet_nan():
    """torch/headeronly/util/BFloat16.h round_to_nearest_even: `if (std::isnan(src)) return UINT16_C(0x7FC0);`
    — the sign of a NaN is not kept."""
    nans = np.array([0x7FC00000, 0xFFC00000, 0x7F800001, 0xFF800001], dtype=np.uint32).view(np.float32)
    assert C.f32_to_bf16_bits(nans).tolist() == [0x7FC0] * 4


def test_narrowing_to_half_is_widening_then_numpys_cast():
    bits = _every_pattern()
    want = C.bf16_bits_to_f32(bits).astype(np.float16)
    assert np.array_equal(C.bf16_bits_to_f16(bits).view(np.uint16), want.view(np.uint16))
    out = np.empty(bits.size, dtype=np.float16)
    C.bf16_bits_to_f16(bits[:100], out=out[:100])
    assert np.array_equal(out[:100].view(np.uint16), want[:100].view(np.uint16))


def test_the_void_carrier_is_read_and_anything_else_refused():
    bits = np.array([0x3F80, 0x4000], dtype=np.uint16)
    assert C.bf16_bits_to_f32(bits.view("V2")).tolist() == [1.0, 2.0]
    with pytest.raises(TypeError, match="bfloat16 host carrier"):
        C.bf16_bits_to_f32(np.zeros(2, dtype=np.float32))


def test_the_rounding_streams_beyond_one_step():
    a = np.random.default_rng(1).standard_normal(3 * (1 << 18) + 5).astype(np.float32)
    ref = ((a.view(np.uint32) + 0x7FFF + ((a.view(np.uint32) >> 16) & 1)) >> 16).astype(np.uint16)
    assert np.array_equal(C.f32_to_bf16_bits(a), ref)
