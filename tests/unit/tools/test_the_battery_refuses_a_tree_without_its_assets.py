"""A battery run from a tree missing its gitignored assets is refused before the first cell.

On 2026-09-05 a battery ran from a frozen worktree that did not carry `test_speech_ref.wav`
and `test_upscale_input.png` — gitignored by design, so a worktree created from a commit never
has them — and returned **30 red cells of 33**, every one a missing file rather than a defect.
The cost was not the reds. It was the hours of cards spent producing them.

The question is now asked ONCE, at collection, instead of thirty times after the cards are spent.
This file proves both halves: the predicate answers correctly, and the HOOK actually calls it
(register 17 — a helper whose every test passes can still have no seam).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "regression"))
import conftest as B  # noqa: E402


class _Config:
    def getoption(self, name):
        return False


class _Item:
    keywords = ()

    def add_marker(self, _m):
        pass


def test_a_tree_with_both_assets_is_not_refused(tmp_path):
    for a in B.REQUIRED_ASSETS:
        (tmp_path / a).write_bytes(b"\x00")
    assert B.refuse_a_tree_without_its_ignored_assets(tmp_path) == []


def test_the_missing_ones_are_named(tmp_path):
    (tmp_path / B.REQUIRED_ASSETS[0]).write_bytes(b"\x00")
    assert B.refuse_a_tree_without_its_ignored_assets(tmp_path) == [B.REQUIRED_ASSETS[1]]
    assert sorted(B.refuse_a_tree_without_its_ignored_assets(tmp_path / "nowhere")) == \
        sorted(B.REQUIRED_ASSETS)


def test_the_hook_refuses_and_says_what_to_copy(tmp_path, monkeypatch):
    """The wiring. Without this the predicate above can be perfect and never called."""
    monkeypatch.setattr(B, "REPO_ROOT", tmp_path)
    with pytest.raises(BaseException) as e:
        B.pytest_collection_modifyitems(_Config(), [_Item()])
    said = str(e.value)
    assert "missing asset" in said, said
    for a in B.REQUIRED_ASSETS:
        assert a in said, f"{a} must be named: {said}"
    assert "cp " in said, "the refusal names the command that satisfies it"
    assert "30 red cells" in said, "and what it cost the day it was learned"


def test_an_empty_collection_is_not_refused(tmp_path, monkeypatch):
    """`--collect-only` on a machine that will never run the battery must still work, and a
    run that selected no cell spends no card, so there is nothing to refuse."""
    monkeypatch.setattr(B, "REPO_ROOT", tmp_path)
    B.pytest_collection_modifyitems(_Config(), [])


def test_the_real_tree_carries_them(tmp_path):
    """The premise of every battery launched from here, checked rather than assumed."""
    assert B.refuse_a_tree_without_its_ignored_assets(B.REPO_ROOT) == []
