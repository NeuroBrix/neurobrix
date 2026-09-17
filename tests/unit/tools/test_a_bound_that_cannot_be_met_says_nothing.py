"""A verdict at a bound no hardware can meet is not a strict test, it is an empty one.

The Liger portability probe judged five fused Triton kernels against ATen at an ABSOLUTE
1e-7 in fp32. One fp32 ulp at magnitude ten is about 1e-6, so that bound asks two
implementations to be bit-identical and calls every difference in rounding order a failure.
Its first run returned FAIL on four of five kernels whose largest disagreement was two to
three ulp (2026-09-16). The JSON said FAIL; the kernels were fine.

The replacement is the standard mixed bound, |diff| <= atol + rtol * magnitude. This file
exists to stop the obvious over-correction: a bound loose enough to pass anything says as
little as one strict enough to fail everything, so the cases below pin BOTH ends.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import liger_probe as L  # noqa: E402

ATOL, RTOL = 1e-6, 1e-6
FP32_EPS = 2 ** -23


def test_rounding_order_passes():
    """What the old bound failed: the measured rms_norm and rope readings of 2026-09-16."""
    assert L.judge(1.9073486328125e-06, 13.074162483215332, ATOL, RTOL) == "PASS"  # 2.5 ulp
    assert L.judge(9.5367431640625e-07, 11.20805835723877, ATOL, RTOL) == "PASS"   # 3.1 ulp
    assert L.judge(4.76837158203125e-07, 4.443190097808838, ATOL, RTOL) == "PASS"  # rope q
    assert L.judge(0.0, 11.478546142578125, ATOL, RTOL) == "PASS"                  # geglu, bit-identical


def test_a_real_error_still_fails():
    """The half of the bound that earns the other half."""
    assert L.judge(0.13, 13.07, ATOL, RTOL) == "FAIL"        # 1 % of the magnitude
    assert L.judge(13.07, 13.07, ATOL, RTOL) == "FAIL"       # the kernel returned zeros
    assert L.judge(1e-3, 13.07, ATOL, RTOL) == "FAIL"        # fp16-grade error in an fp32 test


def test_the_bound_sits_between_them_where_it_was_placed():
    """One fp32 ulp of the magnitude passes; a hundred do not. Stated as a ratio, so the
    numbers move with the dtype's epsilon rather than with a remembered constant."""
    mag = 13.07
    assert L.judge(8 * FP32_EPS * mag, mag, ATOL, RTOL) == "PASS"
    assert L.judge(100 * FP32_EPS * mag, mag, ATOL, RTOL) == "FAIL"


def test_a_near_zero_tensor_is_not_judged_against_a_near_zero_bound():
    """Without the floor, a tensor of magnitude 1e-9 gets a bound of 1e-15 and every
    implementation fails it. With the floor, a real error at that scale still fails."""
    assert L.judge(5e-7, 1e-9, ATOL, RTOL) == "PASS"
    assert L.judge(5e-3, 1e-9, ATOL, RTOL) == "FAIL"


def test_the_bound_that_was_shipped_would_fail_the_agreeing_kernels():
    """The red this landed against, kept as a case so the old form cannot come back."""
    absolute_1e_7 = lambda d, mag: "PASS" if d <= 1e-7 else "FAIL"  # noqa: E731
    agreeing = [(1.9073486328125e-06, 13.074162483215332), (9.5367431640625e-07, 11.20805835723877),
                (4.76837158203125e-07, 4.443190097808838)]
    assert all(absolute_1e_7(d, m) == "FAIL" for d, m in agreeing)
    assert all(L.judge(d, m, ATOL, RTOL) == "PASS" for d, m in agreeing)
