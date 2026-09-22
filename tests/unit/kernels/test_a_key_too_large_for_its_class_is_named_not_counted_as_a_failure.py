"""A census key whose tensors cannot FIT the memory class is not a certification failure: it
is a census defect arriving late. Wan2.1-T2V's video VAE projects 81 frames of 722x1282x96 to
RGB, and its input alone is 26.8 GiB — the 16 GB census recorded it because the shadow planned
the op at its graph shape, while a real 16 GB run never forms that key because Prism tiles the
decode long before it. Reported beside genuine failures it buries both: the run says
"failed: 1" and nobody learns that the one is impossible by arithmetic.

What this test would do if the code were wrong: a malloc failure asking LESS than the card
holds — the ordinary out-of-memory of a key that is merely tight — would be swallowed as
"too large for this class" and never counted as a failure; the second case pins that it is
not. A poisoned-context failure must still stop the run, which the third case pins.

Shapes: 28 789 986 816 bytes against a 16 151 MB card are the real figures the certifier
printed; 8 GiB against the same card is the contrasting case, an allocation that could have
succeeded on an emptier card and is therefore a real failure.
"""
from __future__ import annotations

from neurobrix.kernels.autotune_certify import after_key_failure, oversize_for_class

REAL = ("GPU malloc failed (error 2) for 28789986816 bytes [device cuda:0 live_tracked=0MB "
        "pool_cached=0MB (0 blocks) driver_free=15842MB / driver_total=16151MB]")
TIGHT = ("GPU malloc failed (error 2) for 8589934592 bytes [device cuda:0 live_tracked=9000MB "
         "pool_cached=0MB (0 blocks) driver_free=5842MB / driver_total=16151MB]")
POISONED = "GPU malloc failed (error 700) for 256 bytes [device cuda:0]"


def test_a_key_that_cannot_fit_the_card_is_named_with_its_arithmetic():
    got = oversize_for_class(RuntimeError(REAL))
    assert got is not None
    asked, card = got
    assert asked == 28789986816 and card == 16151 * 1024 * 1024


def test_a_merely_tight_allocation_is_not_called_too_large():
    assert oversize_for_class(RuntimeError(TIGHT)) is None


def test_the_run_counts_them_apart_and_keeps_going(capsys):
    said = []
    summary = {}
    assert after_key_failure(RuntimeError(REAL), summary, "conv::wan", said.append) is False
    assert summary.get("oversize") == 1 and "failed" not in summary, summary
    assert "TOO LARGE FOR THIS CLASS" in said[0] and "26.8" in said[0], said

    summary2 = {}
    assert after_key_failure(RuntimeError(TIGHT), summary2, "conv::tight", said.append) is False
    assert summary2["failed"] == 1 and "oversize" not in summary2, summary2


def test_a_poisoned_context_still_stops_the_run():
    summary = {}
    assert after_key_failure(RuntimeError(POISONED), summary, "mm::x", lambda _m: None) is True
    assert summary["failed"] == 1 and summary["aborted"]["key"] == "mm::x"
