"""The convolution kernel computes 64-bit offsets only where an operand's reach needs them (`_conv_offsets_wide`):
32-bit offsets are exact below 2^31 and faster (17 % on Apple's ALUs once the Metal backend stopped narrowing the
64-bit ones, 2026-10-20); a reach within a factor of two of 2^31 keeps 64-bit, covering a masked lane's overshoot.
Bit identity of the two forms on V100: stable/conv32/probe.py (four shapes, fp16 and fp32, identical bytes).
What this test would do if the threshold were wrong: Sana 4K's 64 x 4099 x 8194 input narrowed -> red."""
from types import SimpleNamespace as NS

from nbxtensor.ops.conv_wrappers import _conv_offsets_wide


def _t(shape):
    st, acc = [], 1
    for n in reversed(shape):
        st.insert(0, acc)
        acc *= n
    return NS(shape=tuple(shape), stride=lambda st=tuple(st): st)


def test_a_small_convolution_runs_on_32_bit_offsets():
    assert _conv_offsets_wide(_t((1, 64, 64, 64)), _t((64, 64, 3, 3)), _t((1, 64, 64, 64))) is False


def test_an_operand_past_2_to_the_31_keeps_64_bit_offsets():
    assert _conv_offsets_wide(_t((1, 64, 4099, 8194)), _t((3, 64, 3, 3)), _t((1, 3, 4097, 8192))) is True


def test_the_margin_covers_an_overshoot_below_2_to_the_31():
    assert _conv_offsets_wide(_t((1, 70, 4096, 4096)), _t((3, 70, 3, 3)), _t((1, 3, 4096, 4096))) is True
