"""Step 5 of the state engine (core/state/policy.py, docs/design/state-engine.md §4.4): when the whole state does not fit,
an exact policy that fits always wins; a policy that changes the output is taken only when no exact one fits and only with
its quality cost measured against the vendor, which the decision carries; otherwise a refusal by name with the figures.

What this test does if the code were wrong: an output-changing policy chosen while an exact one fits (silent degradation)
turns the first cell red; an unmeasured changing policy taken turns the third red; a refusal without its figures, or no
refusal, turns the fourth red."""
import pytest

from neurobrix.core.state.policy import PolicyCandidate, StatePolicyRefused, choose_policy

MB = 2**20
PAGES = PolicyCandidate("pages", 900 * MB, exact=True)
OFFLOAD = PolicyCandidate("offload", 300 * MB, exact=True, slowdown=1.8)
Q8 = PolicyCandidate("quantize_kv", 450 * MB, exact=False, quality_cost=0.002, cost_source="vendor diff, request X")
WIN = PolicyCandidate("window", 200 * MB, exact=False, quality_cost=0.03, cost_source="vendor diff, request X")


def test_an_exact_policy_that_fits_always_wins_over_a_cheaper_changing_one():
    d = choose_policy([Q8, WIN, OFFLOAD, PAGES], 1000 * MB)
    assert d.name == "pages" and d.quality_cost is None
    d = choose_policy([Q8, WIN, OFFLOAD, PAGES], 500 * MB)
    assert d.name == "offload", "pages does not fit; offload is exact and fits, so it beats quantize_kv"


def test_a_changing_policy_is_taken_only_when_no_exact_fits_and_carries_its_measured_cost():
    d = choose_policy([PAGES, PolicyCandidate("offload", 600 * MB, exact=True), Q8, WIN], 500 * MB)
    assert d.name == "quantize_kv" and d.quality_cost == 0.002 and "vendor" in d.reason


def test_a_changing_policy_without_a_measured_cost_is_never_taken():
    unmeasured = PolicyCandidate("window", 200 * MB, exact=False, quality_cost=None)
    with pytest.raises(StatePolicyRefused, match="no measured quality cost"):
        choose_policy([PAGES, unmeasured], 250 * MB)


def test_nothing_fits_is_refused_by_name_with_the_figures():
    with pytest.raises(StatePolicyRefused, match=r"ZERO FALLBACK: the state does not fit: 100 MB .*\(the smallest needs 200 MB\)"):
        choose_policy([PAGES, OFFLOAD, Q8, WIN], 100 * MB)
