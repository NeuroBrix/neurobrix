"""A bound hardware profile that declares no autotune ladder is REFUSED, not keyed exact.

Answering "step 1" for a profile that simply forgot the key is a silent degradation dressed
as a default. It produced no error and no log line, and it is how the bucket ladder came to
exist on `nvidia/volta.yml` alone while all twenty-six other vendor profiles carried none —
found on 2026-09-22 by the schema gate, never by a run. Every one of those profiles would
have keyed EXACTLY: a key explosion, silently, on every architecture but Volta.

The distinction the door draws, and why it is not simply "always refuse":

* **bound but ladderless** — a real profile matched and it lacks the key. That is a defect in
  the profile and it is refused by name.
* **unbound** (`{}`) — no target matched at all: the census behind `CUDA_VISIBLE_DEVICES=`
  before `install()` binds it, or a machine with no profile. That is a different condition,
  answered by binding the shadow to its profile, and keying exact there is unchanged. A
  refusal here would break the census's own door.

Shapes: 240 is a bucket top on the reference ladder and 226 a raw extent under it — the exact
pair from the census incident where a shadow that did not know its profile recorded 226 where
the launcher keys 240. `step: 1` is written as a real one-row ladder to prove that "exact" is
legitimate when DECLARED, and only refused when absent.
"""
from __future__ import annotations

import pytest

from neurobrix.kernels.autotune_bucket import bucket_of, ladder_for

_LADDER = {"autotune": {"buckets": {"default": [{"up_to": 64, "step": 1},
                                                {"up_to": 256, "step": 16}]}}}
_EXACT = {"autotune": {"buckets": {"default": [{"up_to": None, "step": 1}]}}}


def test_a_bound_profile_with_no_ladder_is_refused_by_name():
    with pytest.raises(RuntimeError) as e:
        ladder_for("M", {"architecture": "ampere"})
    msg = str(e.value)
    assert "ZERO FALLBACK" in msg
    assert "ampere" in msg, msg            # it names WHICH profile, not just that one failed
    assert "autotune.buckets" in msg       # and the key to add
    assert "'M'" in msg                    # and the dimension that had no bucket


def test_the_refusal_does_not_depend_on_the_profile_carrying_a_name():
    with pytest.raises(RuntimeError, match="ZERO FALLBACK"):
        ladder_for("N", {"block_sizes": {"gemm": 64}})


@pytest.mark.parametrize("value, top", [(226, 240), (64, 64), (65, 80), (1, 1)])
def test_a_declared_ladder_still_buckets(value, top):
    assert bucket_of("M", value, _LADDER) == top


def test_an_unbound_profile_still_keys_exact_because_that_is_a_different_condition():
    # The census door: no driver answers, so no profile matched. Refusing here would refuse
    # the census itself.
    assert ladder_for("M", {}) == [(None, 1)]
    assert bucket_of("M", 226, {}) == 226


def test_exact_is_legitimate_when_it_is_WRITTEN_DOWN():
    # A one-row step-1 ladder is a decision. Its absence is not.
    assert bucket_of("M", 226, _EXACT) == 226
    assert ladder_for("M", _EXACT) == [(None, 1)]


def test_every_shipped_profile_answers_without_refusing():
    """The door's own reachability check: after 2026-09-22 no shipped profile can hit it."""
    import pathlib
    import yaml

    vendors = pathlib.Path(__file__).resolve().parents[3] / "src/neurobrix/config/vendors"
    profiles = sorted(vendors.glob("*/*.yml"))
    assert profiles, "no vendor profiles found — the check would be vacuous"
    for p in profiles:
        profile = yaml.safe_load(p.read_text()) or {}
        for dim in ("M", "N", "K", "W"):
            assert bucket_of(dim, 226, profile) > 0, f"{p} refused dimension {dim}"
