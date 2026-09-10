"""The screen may not seat a kernel the fp64 oracle contradicts.

`screen_configs` decides by CONSENSUS among the candidate configurations: it
runs each once, clusters them by agreement, and keeps the largest cluster. That
was the right answer to the 2026-09-07 incident — anchoring on a nominated
reference inverts the moment that reference is the broken one, and `matmul`'s
first config was one of three that wrote half the output. But consensus is a
VOTE, and a vote has two failure modes nothing here covers:

  * the majority cluster is wrong in the same way — the minority is excluded
    and the wrong kernel is seated;
  * **every** candidate agrees and all are wrong — `len(clusters) == 1` returns
    the whole space, silently.

Neither is hypothetical on hardware nobody has looked at. Our certified
directory covers `nvidia/volta` and nothing else, so on any other card the
runtime SWEEPS and this screen is the only thing between the user and a wrong
kernel. And the space it votes over differs by target: the same flash tile needs
98 304 bytes on sm_70 and 164 352 on sm_86, so sm_86 explores configurations we
never explore, with a majority we have never seen.

The report that prompted this: on sm_86, greedy decode, the Triton branch
returned 98 tokens against 31 compiled and collapsed into repetition. At
temperature zero there is no sampling, so the divergence is in the forward and
not in the draw. **This is a framing, not a cause** — it is not reproduced here
and nothing below asserts it is that.

One neighbouring fact, recorded as framing and not as a cause: the Metal agent
has just found `argsort` returning values absent from its input as soon as an
alternating stage appears, `_compare_and_swap` duplicating the tile instead of
swapping it. Different chain, and at temperature zero it does not touch our
measurements. It is the third defect this week living on a path our gates do not
take.

So the rule this file pins: **when an oracle is available, a configuration the
oracle contradicts is never seated, whatever the vote says.** Exercised against
a simulated foreign space — a majority that is wrong — because that is the space
we cannot run.

Run: PYTHONPATH=src python -m pytest tests/unit/kernels/test_screen_is_checked_against_an_oracle.py
"""
from __future__ import annotations

import numpy as np
import pytest


def _f32(values):
    return np.asarray(values, dtype=np.float32).tobytes()


# One tile of a MoE decode row: the oracle's answer, and the two ways a kernel
# gets it wrong — half the output left unwritten (the 2026-09-07 shape), and a
# value that is merely far.
ORACLE = _f32([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
CORRECT = _f32([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
HALF_WRITTEN = _f32([1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0])
FAR = _f32([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 99.0])


def test_a_wrong_majority_does_not_carry_the_vote():
    """The foreign-space shape: three configs wrong the same way, two right.
    Consensus alone seats the three. The oracle must overrule it."""
    from neurobrix.kernels.launcher import configs_agreeing_with_oracle

    space = [("m0", HALF_WRITTEN), ("m1", HALF_WRITTEN), ("m2", HALF_WRITTEN),
             ("ok0", CORRECT), ("ok1", CORRECT)]
    kept = configs_agreeing_with_oracle(space, ORACLE, "float32")
    assert {n for n, _ in kept} == {"ok0", "ok1"}, (
        "the screen seated a configuration the oracle contradicts because more "
        "configurations agreed with each other than with the truth")


def test_a_unanimous_wrong_space_is_refused_entirely():
    """The silent case: every candidate agrees, and all are wrong. Consensus
    returns the whole space and says nothing."""
    from neurobrix.kernels.launcher import configs_agreeing_with_oracle

    space = [("a", HALF_WRITTEN), ("b", HALF_WRITTEN), ("c", HALF_WRITTEN)]
    kept = configs_agreeing_with_oracle(space, ORACLE, "float32")
    assert kept == [], (
        "a space in which every configuration is wrong must seat none of them, "
        "not all of them")


def test_the_correct_space_is_untouched():
    """The control. A screen that refuses everything is not a screen — this
    must not become a rule that empties a healthy space."""
    from neurobrix.kernels.launcher import configs_agreeing_with_oracle

    space = [("a", CORRECT), ("b", CORRECT), ("c", CORRECT)]
    assert len(configs_agreeing_with_oracle(space, ORACLE, "float32")) == 3


def test_a_merely_distant_value_is_judged_by_the_profile_tolerance():
    """Not every difference is a defect: the tolerance is the profile's, and it
    is the same one the screen already uses between candidates."""
    from neurobrix.kernels.launcher import configs_agreeing_with_oracle

    kept = configs_agreeing_with_oracle([("far", FAR), ("ok", CORRECT)],
                                        ORACLE, "float32")
    assert {n for n, _ in kept} == {"ok"}


def test_no_oracle_seats_nothing_new_and_says_so():
    """The honest degradation: with no oracle the function refuses to pretend.
    It returns None so the caller keeps the consensus it had — and knows that is
    what it has."""
    from neurobrix.kernels.launcher import configs_agreeing_with_oracle

    assert configs_agreeing_with_oracle([("a", CORRECT)], None, "float32") is None


def test_the_cell_is_not_vacuous():
    """Non-vacuity, stated as its own test rather than assumed: given a
    deliberately wrong kernel, the predicate must SAY it is wrong. If this ever
    passes trivially, every test above is decoration."""
    from neurobrix.kernels.launcher import configs_agreeing_with_oracle

    kept = configs_agreeing_with_oracle([("deliberately_wrong", HALF_WRITTEN)],
                                        ORACLE, "float32")
    assert kept == [], (
        "the predicate accepted a kernel that wrote half its output — it "
        "cannot detect anything")
