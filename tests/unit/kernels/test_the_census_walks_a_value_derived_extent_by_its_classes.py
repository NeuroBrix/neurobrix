"""Some extents are learned only from VALUES, and a shadow has none: the number of speech
tokens a vocoder receives is what survived a filter on the tokens sampled, so one census run
records the keys of ONE length and a directory certified from it misses every other speech
(chatterbox's vocoder: 1 716 tokens and 1 856 tokens gave two full sets of convolution keys,
2026-09-21). `census.walk_extent` runs the stage at every KEY CLASS of the extent instead:
the ends, then the midpoint of any pair whose recorded key sets differ, until the differing
pair is adjacent. Every kernel's key is a monotone step function of the extent, so a pair
with one key set brackets a range with that key set.

What this test would do if the code were wrong: a walk that only sampled the ends would meet
2 classes of the 191 and the assertion on the missed set fails; a walk that swallowed a
refusing extent silently would pass the "every extent refused" case instead of raising.

Shapes: the synthetic extent runs 1..2048 (chatterbox's speech-token bound) through three
derived dimensions — n+204 (a context with a fixed conditioning), 2n (a mel at two frames a
token) and 256n (a waveform at the vocoder's rate) — on the profile's own ladders, so the
class count is the real one, not a toy. 191 classes is what those three ladders produce
there; the test asserts the walk meets all of them, never a number it chose.
"""
from __future__ import annotations

import pytest

from neurobrix.kernels import census
from neurobrix.kernels.autotune_bucket import bucket_of

PROFILE = None


def _profile():
    global PROFILE
    if PROFILE is None:
        import yaml
        from pathlib import Path
        root = Path(__file__).resolve().parents[3]
        PROFILE = yaml.safe_load((root / "src/neurobrix/config/vendors/nvidia/volta.yml").read_text())
    return PROFILE


def _classes_of(n: int):
    p = _profile()
    return (bucket_of("M", n + 204, p), bucket_of("M", 2 * n, p), bucket_of("W", 256 * n, p))


@pytest.fixture
def shadow(monkeypatch):
    monkeypatch.setitem(census._ACTIVE, "census", True)
    monkeypatch.setenv("NBX_CENSUS_EXTENTS", "1")
    yield
    monkeypatch.setitem(census._ACTIVE, "census", False)


def test_the_walk_meets_every_class_of_the_extent(shadow):
    ran = []

    def run(n):
        ran.append(n)
        for obs in census._OBSERVERS:
            for i, v in enumerate(_classes_of(n)):
                obs.add(f"k{i}::({v},)")
        return n

    assert census.walks_extents()
    last = census.walk_extent(1, 2048, run, name="synthetic")
    truth = {_classes_of(n) for n in range(1, 2049)}
    met = {_classes_of(n) for n in ran}
    assert truth - met == set(), sorted(truth - met)[:5]
    assert len(ran) < 4 * len(truth), (len(ran), len(truth))   # bounded: a walk, never the whole range
    assert last == ran[-1]


def test_an_extent_the_graph_refuses_is_its_own_class(shadow):
    def run(n):
        if n < 4:
            raise RuntimeError("integer division or modulo by zero")
        for obs in census._OBSERVERS:
            obs.add(f"k::({bucket_of('M', n, _profile())},)")
        return n

    census.walk_extent(1, 256, run, name="partly refused")     # returns; the refusal is a class


def test_a_walk_that_refuses_everywhere_fails_instead_of_reading_as_censused(shadow):
    def run(n):
        raise RuntimeError("integer division or modulo by zero")

    with pytest.raises(RuntimeError, match="every extent refused"):
        census.walk_extent(1, 256, run, name="all refused")


def test_outside_the_shadow_the_door_is_shut(monkeypatch):
    monkeypatch.setitem(census._ACTIVE, "census", True)
    monkeypatch.delenv("NBX_CENSUS_EXTENTS", raising=False)
    assert census.walks_extents() is False
