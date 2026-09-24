"""The report-only census of what a refusing `_reshape` would reject (2026-09-24).

Step one of the owner's two-step decision: before `_reshape` is made to refuse a target that
no longer matches its input, we need to know how many of the 59 containers fall and which.
`NBX_RESHAPE_REPORT=<file>` records the sites that would be refused.

What it must NOT count is the whole catalogue. A target carrying a `-1` fails the exact-match
test by construction — `static_elements` skips the -1 — so it reaches the same code as a real
invention. A refusing implementation would still infer it happily. Counting those would report
every container as falling and measure nothing, which is the failure mode this repository has
a register for.

So the instrument is checked both ways here: it fires on an invention, and it stays silent on
an inference and on an exact match.

Runnable: PYTHONPATH=src python3 -m pytest \
  tests/unit/kernels/test_the_reshape_report_counts_inventions_not_inferences.py -v
"""
from __future__ import annotations

import json

import pytest
import torch

from neurobrix.kernels.metadata_ops import _reshape


def _records(tmp_path, monkeypatch, in_shape, target):
    out = tmp_path / "report.jsonl"
    monkeypatch.setenv("NBX_RESHAPE_REPORT", str(out))
    monkeypatch.setenv("NBX_RESHAPE_REPORT_MODEL", "UnitTest")
    x = torch.zeros(in_shape)
    try:
        _reshape([x], {"shape": list(target), "op_uid": "aten.view::7"})
    except Exception:
        pass                      # the report is written before any repair is attempted
    if not out.exists():
        return []
    return [json.loads(l) for l in out.read_text().splitlines() if l.strip()]


def test_an_invented_shape_is_recorded(tmp_path, monkeypatch):
    """The PixArt case: a baked target four times too small, no -1 to absorb it."""
    got = _records(tmp_path, monkeypatch, (2, 16384, 8), (8192, 8))
    assert len(got) == 1, got
    r = got[0]
    assert r["in_numel"] == 2 * 16384 * 8
    assert r["target"] == [8192, 8]
    assert r["ratio"] == pytest.approx(4.0)
    assert r["op_uid"] == "aten.view::7" and r["model"] == "UnitTest"


def test_an_inferred_shape_is_not_recorded(tmp_path, monkeypatch):
    """A -1 target reaches the same code and is NOT an invention."""
    assert _records(tmp_path, monkeypatch, (2, 16384, 8), (-1, 8)) == []


def test_an_exact_match_is_not_recorded(tmp_path, monkeypatch):
    assert _records(tmp_path, monkeypatch, (2, 4096, 8), (8192, 8)) == []


def test_nothing_is_written_when_the_variable_is_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("NBX_RESHAPE_REPORT", raising=False)
    out = tmp_path / "report.jsonl"
    try:
        _reshape([torch.zeros((2, 16384, 8))], {"shape": [8192, 8]})
    except Exception:
        pass
    assert not out.exists()
