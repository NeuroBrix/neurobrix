"""The two-phase battery sends a cell to the pinned phase only when its container fits one card.

What would this file do if the code were wrong? A model over the fit sent to the pinned phase
→ the first cell fails; an unreadable size sent to the pinned phase → the second; a round-robin
that skips or doubles a name → the third.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "two_phase", Path(__file__).resolve().parents[3] / "tests" / "regression" / "two_phase.py")
two_phase = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(two_phase)

GB = 1024 ** 3


def test_a_container_over_the_fit_goes_to_the_whole_rig():
    sizes = {"tiny": 2 * GB, "mid": 13 * GB, "big": 31 * GB, "huge": 66 * GB}
    single, whole = two_phase.partition(list(sizes), 14 * GB, sizes)
    assert single == ["tiny", "mid"] and whole == ["big", "huge"]


def test_an_unreadable_size_is_the_whole_rig_s():
    single, whole = two_phase.partition(["ghost"], 14 * GB, {})
    assert single == [] and whole == ["ghost"]


def test_round_robin_covers_every_name_once():
    names = [f"m{i}" for i in range(7)]
    plan = two_phase.round_robin(names, ["0", "1", "2", "3"])
    flat = sorted(n for v in plan.values() for n in v)
    assert flat == sorted(names) and plan["0"] == ["m0", "m4"] and plan["3"] == ["m3"]
