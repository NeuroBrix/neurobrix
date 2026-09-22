"""The divergence refusal must not divide by a tolerance of zero.

`_tolerance` refuses a MISSING `autotune_screen_rtol` and accepts a declared one — including
0.0, which is a legitimate thing for a profile to want (exact match). The refusal message
added on 2026-09-22 printed `best / tolerance`, so a zero tolerance would raise
ZeroDivisionError inside the error path and replace the diagnosis with a traceback: the very
failure that message exists to prevent. Found by the rack on review, latent (no profile
declares 0 today), fixed before it could be anything else.
"""
import re


def _message(best: float, tolerance: float) -> str:
    """The message's arithmetic, lifted verbatim from autotune_certify."""
    ratio = f", {best / tolerance:.1f}x tolerance" if tolerance else " against an EXACT-match tolerance"
    return f"every config diverges from the fp64 oracle beyond {tolerance:g} (best {best:.3g}{ratio})"


def test_a_zero_tolerance_does_not_raise_inside_the_error_path():
    msg = _message(0.31, 0.0)
    assert "EXACT-match" in msg
    assert "best 0.31" in msg


def test_a_normal_tolerance_still_names_the_multiple():
    msg = _message(0.8, 0.04)
    assert "20.0x tolerance" in msg
    assert "best 0.8" in msg


def test_the_source_guards_the_division():
    """The shipped code, not just this file's copy of its arithmetic."""
    src = open("src/neurobrix/kernels/autotune_certify.py", encoding="utf-8").read()
    assert re.search(r"if tolerance else", src), \
        "the refusal divides by `tolerance` with no guard for zero"
