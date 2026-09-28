"""A test's child process sees the cards its caller gave the suite, never an ordinal of its own.

2026-09-27: a unit gate told `CUDA_VISIBLE_DEVICES=3` put a TinyLlama child on physical card 2
(the lazy-bind boundary test set "2") and a launch-path child on card 0 (the launcher set "0").
Twice it met a PixArt retrace that owned card 2, and the launcher test went red on a card it had
not been given. Before this branch the helper does not exist and the scan names three sites (a
fourth, test_serve_warm, carries its ordinal in a dict the scan cannot read): these fail on main.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.unit import child_env as C

TESTS = Path(__file__).resolve().parents[2]

# A child env handed a literal ordinal, or a lookup whose default is one. `monkeypatch.setenv`
# changes this process's own view for a test about the view itself, and is not a child's door.
_LITERAL_CARD = re.compile(
    r"""CUDA_VISIBLE_DEVICES["']\s*(?:\]\s*=|:)\s*(?:["']\d|[^#\n]*\.get\([^)]*["']\d)""")


def test_no_door_keeps_the_tests_own_default(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert C.the_callers_door("2") == "2"


def test_a_door_is_kept(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    assert C.the_callers_door("2") == "3"


def test_an_empty_door_skips_instead_of_finding_a_card(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(pytest.skip.Exception):
        C.the_callers_door("0")


def test_no_test_hands_a_child_a_card_of_its_own():
    sites = []
    for f in sorted(TESTS.rglob("*.py")):
        for n, line in enumerate(f.read_text().splitlines(), 1):
            if "monkeypatch" not in line and _LITERAL_CARD.search(line):
                sites.append(f"{f.relative_to(TESTS)}:{n}: {line.strip()}")
    assert not sites, "a child is given its own card:\n" + "\n".join(sites)


def test_a_pinned_profile_travels_with_the_tests_own_card(monkeypatch):
    """A 32 GB profile pinned with card 2 must not follow the caller's door onto a 16 GB card."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert C.the_pinned_profile("v100-32g") == "v100-32g"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert C.the_pinned_profile("v100-32g") is None
