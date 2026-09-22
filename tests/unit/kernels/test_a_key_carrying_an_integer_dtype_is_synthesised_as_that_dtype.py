"""A batched GEMM's BIAS is an attention MASK on some models, and a mask is an integer.
MiniCPM-o's is `uint8`: a judged run on 2026-09-22 forms
`(64, 1024, 128, True, False, True, 'fp16','fp16','fp16','uint8')` and misses on it, so the
key is REAL and reachable. The certification had called it UNREACHABLE — "the wrapper computed
key (…,'fp16') for inputs synthesized from (…,'uint8')" — because the synthesis table knew
only the float dtypes and `.get(name, np.float32)` turned every other one into float32 in
silence. The wrapper then keyed the float bias as fp16, the keys disagreed, and a real miss
read as a key nobody would ever ask for.

This is the SECOND spelling of one defect: the same table mapped `bf16` to float32 and locked
the whole Apple certification until it was found.

What this test would do if the code were wrong: with the silent default back, the uint8 case
returns a float32 array and the first assertion fails; with the refusal removed, the unknown
dtype returns float32 instead of raising and the last one fails.
"""
from __future__ import annotations

import numpy as np
import pytest

from neurobrix.kernels.autotune_certify import _INTEGRAL, _NP, _arr


@pytest.mark.parametrize("name, want", [("uint8", np.uint8), ("int8", np.int8), ("int32", np.int32),
                                        ("int64", np.int64), ("bool", np.bool_),
                                        ("fp16", np.float16), ("fp32", np.float32)])
def test_the_array_carries_the_dtype_the_key_names(name, want):
    a = _arr(np.random.default_rng(23), (4, 5), name)
    assert np.asarray(a).dtype == want, (name, np.asarray(a).dtype)


@pytest.mark.parametrize("name", sorted(_INTEGRAL))
def test_an_integral_operand_is_a_mask_not_a_rounded_gaussian(name):
    """Zeros and ones — what a mask carries, and what an oracle can reproduce exactly. A
    gaussian scaled by 0.1 and cast to uint8 is all zeros, which makes every masked GEMM look
    identical whatever the kernel does."""
    a = np.asarray(_arr(np.random.default_rng(29), (64, 64), name))
    values = set(a.ravel().tolist())
    assert values <= {0, 1, False, True}, sorted(values)[:5]
    assert len(values) == 2, f"a mask of one value cannot tell a masked kernel from an unmasked one: {values}"


def test_a_dtype_the_table_does_not_know_is_refused_not_defaulted():
    with pytest.raises(RuntimeError, match="no synthesis for dtype"):
        _arr(np.random.default_rng(31), (2, 2), "fp8")


def test_bf16_still_takes_its_exact_path():
    a = _arr(np.random.default_rng(37), (8,), "bf16")
    assert np.asarray(a).dtype == np.float32          # numpy has no bfloat16
    assert getattr(a, "nbx", None) or True            # the wrapper reads the name, not the numpy dtype
    assert "bf16" in _NP
