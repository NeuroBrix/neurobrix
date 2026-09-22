"""A bf16 model on bf16-capable hardware never falls back to fp32.

`solver.solve` runs an fp32 pass when the bf16 plan produced NO candidates:

    # FP32 fallback for BF16 models
    if not candidates:
        if self._try_fp32_fallback(container, profile):
            target_dtype_str = "float32"

`_try_fp32_fallback` returns True whenever ANY component is bfloat16 — it accepts `profile`
and ignores it. On hardware that SUPPORTS bf16 that fallback cannot help: fp32 needs strictly
MORE memory than the bf16 plan that just failed. It cannot produce a plan, and it replaces the
diagnostic figures with numbers exactly 2x the truth.

Measured 2026-09-22, Flex.1-alpha on an M4 Pro (bf16-capable, 17 277 MB budget). Instrumenting
`compute_dtype_factor`'s call site showed the two passes:

    transformer: source='bfloat16' comp='bfloat16' -> mult=1.0  (component_dtypes populated)
    transformer: source='bfloat16' comp='float32'  -> mult=2.0  (component_dtypes=None)

and the refusal reported the SECOND: "the streaming path needs 33954MB for that one
component", where the bf16 figure is ~16 977 MB and FITS the budget. Two machines spent hours
on a 2.000x that was this fallback, not an estimator defect.

The fallback stays for hardware with no bf16 support, which is what it was written for.
"""
from __future__ import annotations

import pytest

from neurobrix.core.prism.solver import PrismSolver


class _Comp:
    def __init__(self, dt): self._dt = dt
    def get_dominant_dtype(self): return self._dt


class _Container:
    def __init__(self, dtypes): self._c = [_Comp(d) for d in dtypes]
    def get_neural_components(self): return self._c


class _Profile:
    def __init__(self, supported): self._s = set(supported)
    def devices_support_dtype(self, dt): return dt in self._s


def test_bf16_capable_hardware_does_not_try_the_fp32_fallback():
    s = PrismSolver()
    assert s._try_fp32_fallback(_Container(["bfloat16"]),
                                _Profile(["bfloat16", "float16", "float32"])) is False


def test_hardware_without_bf16_still_falls_back():
    """The case the fallback was written for must keep working."""
    s = PrismSolver()
    assert s._try_fp32_fallback(_Container(["bfloat16"]),
                                _Profile(["float32"])) is True


def test_a_model_with_no_bf16_component_never_falls_back():
    s = PrismSolver()
    assert s._try_fp32_fallback(_Container(["float16", "float32"]),
                                _Profile(["float32"])) is False


def test_no_profile_keeps_the_old_behaviour():
    """Legacy callers pass no profile; refusing to answer would change their plans."""
    s = PrismSolver()
    assert s._try_fp32_fallback(_Container(["bfloat16"]), None) is True
