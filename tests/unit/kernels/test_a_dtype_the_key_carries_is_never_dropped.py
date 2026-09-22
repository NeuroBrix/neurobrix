"""A dtype name the certifier does not recognise SHIFTS every operand after it.

`key_dtypes` filtered a key's strings against a vocabulary and silently skipped anything
missing from it. `uint8` was missing. So the census key

    (64, 1024, 128, True, False, True, 'fp16', 'fp16', 'fp16', 'uint8')

returned THREE dtypes, not four; `synthesize`'s `bias_dt = dts[3] if len(dts) > 3 else dts[0]`
fell off the end and took `dts[0]`; the certifier built an **fp16** bias for a **uint8** key;
the wrapper then computed an fp16 key — correctly — and the certifier reported the census key
UNREACHABLE, "the engine cannot produce this key".

It could. MiniCPM-o forms that key on every run and misses on it. A real miss was read for a
day as an engine limitation, because a lookup table skipped a name instead of refusing it.

This is the SAME SHAPE as `_NP` in `autotune_certify.py`, which did not know any integer dtype
and turned a uint8 attention mask into float32 in silence — fixed on 2026-09-22 without
noticing that a second table had the same hole. The same bug written twice is a missing brick.

The fix has two halves and both are pinned here: the vocabulary knows the unsigned types, and
an unknown string is a REFUSAL rather than a skip, so the third table cannot be silent.
"""
from __future__ import annotations

import numpy as np
import pytest

from neurobrix.kernels.autotune_certified import _DTYPES, key_dtypes
from neurobrix.kernels.autotune_certify import _NP, _arr

#: The key MiniCPM-o forms, verbatim from its run record.
MINICPM_KEY = (64, 1024, 128, True, False, True, "fp16", "fp16", "fp16", "uint8")


def test_the_key_that_was_called_unreachable_yields_four_dtypes():
    assert key_dtypes(MINICPM_KEY) == ["fp16", "fp16", "fp16", "uint8"]


def test_the_bias_operand_is_the_one_the_key_names():
    """The exact expression `synthesize` uses. With uint8 dropped this read 'fp16'."""
    dts = key_dtypes(MINICPM_KEY)
    bias_dt = dts[3] if len(dts) > 3 else (dts[0] if dts else "fp16")
    assert bias_dt == "uint8"


def test_dropping_one_name_shifts_every_operand_after_it():
    """Why a skip is not a harmless omission — stated as an executable claim."""
    short = [d for d in MINICPM_KEY if isinstance(d, str) and d != "uint8"]
    assert len(short) == 3
    assert (short[3] if len(short) > 3 else short[0]) == "fp16"   # the wrong operand, silently


@pytest.mark.parametrize("name", ["uint8", "uint16", "uint32", "uint64",
                                  "int1", "int8", "int16", "int32", "int64",
                                  "fp16", "bf16", "fp32", "fp64", "bool"])
def test_every_name_the_vocabulary_claims_can_also_be_SYNTHESISED(name):
    """A vocabulary entry the synthesiser cannot build would only move the failure.

    Two tables have now had the same hole; this cell ties them together so a name can never
    again be known to one and unknown to the other.
    """
    assert name in _DTYPES
    assert name in _NP, f"{name} is a recognised key dtype but _NP cannot synthesise it"
    arr = _arr(np.random.default_rng(0), (2, 3), name)
    assert np.asarray(arr).dtype == _NP[name]


def test_an_unrecognised_dtype_string_is_REFUSED_not_skipped():
    with pytest.raises(RuntimeError) as e:
        key_dtypes((64, 1024, "float8_e5m2"))
    msg = str(e.value)
    assert "float8_e5m2" in msg          # it names the offender
    assert "shift" in msg.lower()        # and why skipping is not an option


def test_non_string_key_elements_are_still_ignored():
    """Bucket integers and constexpr booleans are not dtypes and must pass through."""
    assert key_dtypes((64, 1024, 128, True, False, True)) == []


def test_the_integral_synthesis_is_a_mask_not_noise():
    """An integral operand in these kernels is a mask or an index: 0/1, reproducible by the
    fp64 oracle exactly. Sampling a normal distribution into a uint8 would wrap."""
    a = np.asarray(_arr(np.random.default_rng(0), (64, 64), "uint8"))
    assert a.dtype == np.uint8
    assert set(np.unique(a)).issubset({0, 1})
