"""`NBXDtype` is an `IntEnum`, and what `str()` renders for one CHANGED between interpreters:
Python 3.10 gives `NBXDtype.bool_`, Python 3.11 gives `9`. The census shadow decided what a
value-read answers by matching `"bool" in str(dtype)`, so on 3.11 a guard's
`all(isfinite(x))` read 0.0, every diffusion shadow's step-boundary NaN gate ended the run at
step 1, and no post-loop key was harvested — the Mac measured PixArt-XL going 16 -> 33 keys
once the name was read instead (2026-09-21). On this rack's 3.10.12 the old form matched,
which is precisely why nothing here ever reported it: a census that silently harvests fewer
keys looks exactly like a census.

What this test would do if the code were wrong: it hands the reader a dtype whose `str()` is
a bare number — the 3.11 rendering — and a real `NBXDtype`; matching on `str()` returns the
float answer for the numeric one and the case fails. It therefore fails on 3.10 too, where
the defect itself cannot be reproduced, which is the point of pinning the NAME.
"""
from __future__ import annotations

import enum

from neurobrix.kernels.census import _dtype_name


class _Renders_As_A_Number(enum.IntEnum):
    """An IntEnum rendered the way Python 3.11 renders one."""
    bool_ = 9
    int64 = 4
    float16 = 1

    def __str__(self):                      # 3.11's IntEnum.__str__
        return str(self.value)


def test_a_dtype_whose_str_is_a_bare_number_is_still_read_by_name():
    assert str(_Renders_As_A_Number.bool_) == "9"          # the rendering that broke it
    assert _dtype_name(_Renders_As_A_Number.bool_) == "bool_"
    assert "bool" in _dtype_name(_Renders_As_A_Number.bool_)
    assert "int" in _dtype_name(_Renders_As_A_Number.int64)
    assert "int" not in _dtype_name(_Renders_As_A_Number.float16)


def test_the_racks_own_dtypes_read_the_same_by_name():
    from neurobrix.kernels.nbx_tensor import NBXDtype
    assert "bool" in _dtype_name(NBXDtype.bool_)
    assert "int" in _dtype_name(NBXDtype.int64)
    for f in ("float16", "float32", "bfloat16"):
        assert "int" not in _dtype_name(getattr(NBXDtype, f)), f


def test_a_plain_object_without_a_name_still_answers_a_string():
    assert _dtype_name("float32") == "float32"
