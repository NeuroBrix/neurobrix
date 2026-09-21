"""The census shadow's benign value-read keys off the dtype NAME, not value.

In a census shadow (`kernels/census.py`) no VALUE means anything: `install()`
replaces `NBXTensor.item` so a guard's `all(isfinite(x))` reads healthy (True)
and an integer token id reads 0. The dtype was inspected as
`str(self._dtype)`, but NBXDtype is an IntEnum, so `str(<NBXDtype.bool_: 9>)`
is "9" — "bool" never matched. Two silent failures followed:

  - a bool guard answered 0.0 → `bool(...)` False → the always-on diffusion
    loop-state NaN gate raised at step 1 and the shadow harvested no
    post-loop keys (PixArt-XL / CogVideoX-2b, 2026-09-21; Sana before them);
  - an integer read answered 0.0 (float), not 0 (int), where a token index
    is expected.

Reading `dtype.name` fixes both. Model-free: the extracted helper
`_shadow_item_value` is driven with each NBXDtype directly (no shadow install,
which refuses when a device is visible).
"""
from __future__ import annotations

from neurobrix.kernels.census import _shadow_item_value
from neurobrix.kernels.nbx_tensor import NBXDtype


def test_a_bool_guard_reads_healthy():
    # the finite gate: bool(all(isfinite(x)).item()) must be truthy in shadow
    v = _shadow_item_value(NBXDtype.bool_)
    assert v is True, (
        f"a bool value-read answered {v!r} in shadow — the finite gate then "
        f"reads falsy and refuses the shadow at step 1; read the dtype by name")


def test_an_integer_read_answers_int_zero():
    for dt in (NBXDtype.int64, NBXDtype.int32):
        v = _shadow_item_value(dt)
        assert v == 0 and isinstance(v, int) and not isinstance(v, bool), (
            f"{dt.name} read answered {v!r} ({type(v).__name__}); a token index "
            f"must be int 0, not float 0.0")


def test_a_float_read_answers_float_zero():
    v = _shadow_item_value(NBXDtype.float32)
    assert v == 0.0 and isinstance(v, float)
