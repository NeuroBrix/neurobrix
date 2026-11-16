"""The verdict rule of `tools/measure_the_gated_delta_rule_prefill.py`, written before any card run, answers each way on
synthetic numbers: a point is won, lost or tied by the quartiles of the two sides' timed runs; a point a side did not
run, or where a side is beyond its correctness bound, has no verdict; a length is won only when every point at it is
won (a tie is not a win); the word is LOSES as soon as one length is lost.

What this test does if the rule is wrong: a tie read as a win, a fast but wrong side allowed to win, a dry run that
claims a verdict, a single lost point hidden by the others at its length - red."""
import pytest

import measure_the_gated_delta_rule_prefill as M

BOUND = 4e-6


def side(times, dev_out=1e-7, dev_state=1e-7):
    return dict(ran=True, why=None, dev_out=dev_out, dev_state=dev_state, **M.spread(times))


FAST = [1.0, 1.1, 1.2, 1.3, 1.4]
SLOW = [2.0, 2.1, 2.2, 2.3, 2.4]
OVERLAP = [1.2, 1.3, 1.4, 1.5, 1.6]


@pytest.mark.parametrize("scan,vendor,kind", [
    (FAST, SLOW, "win"),
    (SLOW, FAST, "loss"),
    (FAST, OVERLAP, "tie"),
])
def test_a_point_is_won_lost_or_tied_by_the_quartiles(scan, vendor, kind):
    assert M.point_verdict(side(scan), side(vendor), "float32", BOUND, BOUND)[0] == kind


def test_a_side_that_did_not_run_gives_no_measurement():
    scan = {"ran": False, "why": "the library's scan launches on a device only"}
    kind, why = M.point_verdict(scan, side(SLOW), "float32", BOUND, BOUND)
    assert kind == "no measurement" and "device only" in why


@pytest.mark.parametrize("who", ["scan", "vendor"])
@pytest.mark.parametrize("field", ["dev_out", "dev_state"])
def test_a_side_beyond_its_bound_is_not_comparable_however_fast(who, field):
    sides = {"scan": side(FAST), "vendor": side(SLOW)}
    sides[who][field] = 1e-3
    kind, why = M.point_verdict(sides["scan"], sides["vendor"], "float32", BOUND, BOUND)
    assert kind == "not comparable" and who in why


def test_the_output_bound_of_a_half_precision_output_adds_its_rounding():
    s = side(FAST, dev_out=3e-4)                        # within 4e-6 + 2^-11, beyond 4e-6
    assert M.point_verdict(s, side(SLOW), "float16", BOUND, BOUND)[0] == "win"
    assert M.point_verdict(s, side(SLOW), "float32", BOUND, BOUND)[0] == "not comparable"


def pts(*rows):
    return [{"L": L, "kind": kind} for L, kind in rows]


def test_the_scan_wins_only_when_it_wins_every_point_of_every_length():
    v = M.overall_verdict(pts((512, "win"), (512, "win"), (4096, "win")))
    assert v["word"] == "WINS" and v["wins"] == [512, 4096] and v["line"].startswith("VERDICT: THE SCAN WINS")


def test_one_lost_or_tied_point_loses_its_length_and_the_word_is_loses():
    v = M.overall_verdict(pts((512, "win"), (512, "win"), (2048, "win"), (2048, "tie"), (4096, "loss"),
                              (4096, "win")))
    assert v["word"] == "LOSES" and v["wins"] == [512] and v["losses"] == [2048, 4096]
    assert "loses at L = [2048, 4096]" in v["line"]


def test_a_length_with_an_unmeasured_point_has_no_verdict():
    v = M.overall_verdict(pts((512, "win"), (2048, "no measurement"), (2048, "win")))
    assert v["word"] == "NO VERDICT" and v["none"] == [2048] and v["wins"] == [512]


def test_a_dry_run_claims_no_verdict():
    v = M.overall_verdict(pts((8, "no measurement"), (70, "no measurement")))
    assert v["word"] == "NO VERDICT" and v["wins"] == [] and v["losses"] == []


def test_a_loss_is_called_even_beside_a_length_without_verdict():
    v = M.overall_verdict(pts((512, "not comparable"), (4096, "loss")))
    assert v["word"] == "LOSES" and v["none"] == [512] and v["losses"] == [4096]
