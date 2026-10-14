"""A class's census ladder ends at the profile's OWN rung: the card's capacity, when the commercial
ladder does not name it.

A run with the whole card free plans at the card's capacity. On a commercial card (16 384, 32 768 MB)
that is a ladder rung and the ladder covers it; on the Mac's M4 Pro (17.8 GB, 18 227 MB) the ladder
stops at 16 384 and the run's own rung was never derived — Prism splits CogVideoX-5b-I2V's guidance
at every rung up to 16 384 and not at 18 227, so the run asked `addmm (480, 3072, 4096)` and the
derived table held M=240 only (the Mac, 2026-10-14 06:18; reproduced on the CPU with its profile:
the row at 18 227 is the run's key).

What this test does if the code is wrong: the M4 cell fails when the capacity is not appended
(red before this change); the V100 cell fails if a commercial capacity were appended twice or the
ladder below it changed.

Run: PYTHONPATH=src python -m pytest tests/unit/census/test_a_class_ladder_ends_at_the_profiles_own_rung.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))
import certified_census as CC  # noqa: E402


def _profile(monkeypatch, memory_mb):
    import neurobrix.core.prism.loader as L
    monkeypatch.setattr(L, "load_profile", lambda hw: SimpleNamespace(devices=[SimpleNamespace(memory_mb=memory_mb)]))
    import neurobrix.core.prism.memory_budget as MB
    monkeypatch.setattr(MB, "memory_ladder_mb", lambda: [4096, 6144, 8192, 11264, 12288, 16384, 24576, 32768])


def test_a_card_above_the_ladders_rung_gets_its_own_rung_last(monkeypatch):
    _profile(monkeypatch, 18227)
    assert CC.rungs_for("auto-apple-m4-pro-18g") == [4096, 6144, 8192, 11264, 12288, 16384, 18227]


def test_a_commercial_card_keeps_the_ladder_unchanged(monkeypatch):
    _profile(monkeypatch, 16384)
    assert CC.rungs_for("v100-16g") == [4096, 6144, 8192, 11264, 12288, 16384]
    _profile(monkeypatch, 32768)
    assert CC.rungs_for("v100-32g") == [4096, 6144, 8192, 11264, 12288, 16384, 24576, 32768]
