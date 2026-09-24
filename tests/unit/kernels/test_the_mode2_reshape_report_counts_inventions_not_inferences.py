"""The report-only census of what a refusing reshape would reject, on MODE 2's site.

`test_the_reshape_report_counts_inventions_not_inferences.py` pins the instrument in
`metadata_ops._reshape`, which is the COMPILED-mode op set. The census that was meant to run
it (`--modes triton,triton-sequential`) never enters that module: mode 2 dispatches
`aten::view` / `aten::reshape` / `aten::_unsafe_view` through `dispatch._resolve_view_shape`,
and mode 1 under a census shadow is a door that refuses. So the first instrument was green and
unreachable from the pass it existed for (measured 2026-09-24: a Kokoro-82M census with it
set never created the report file).

This is the same instrument at the site mode 2 actually reaches, with the same record and the
same two-way check: it fires on an invention (a target with no -1 whose element count does
not match its input) and stays silent on an inference, on an exact match, and when the
variable is unset. A 1-D target is NOT exempt here: `NBXTensor.view` does not validate numel,
so a wrong 1-D length is an invention too.

Runnable: PYTHONPATH=src python3 -m pytest \
  tests/unit/kernels/test_the_mode2_reshape_report_counts_inventions_not_inferences.py -v
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("triton")

from neurobrix.kernels.dispatch import _resolve_view_shape  # noqa: E402


def _records(tmp_path, monkeypatch, in_shape, target):
    out = tmp_path / "report.jsonl"
    monkeypatch.setenv("NBX_RESHAPE_REPORT", str(out))
    monkeypatch.setenv("NBX_RESHAPE_REPORT_MODEL", "UnitTest")
    _resolve_view_shape(SimpleNamespace(shape=tuple(in_shape)), list(target))
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
    assert r["target_numel"] == 8192 * 8
    assert r["ratio"] == pytest.approx(4.0)
    assert r["model"] == "UnitTest" and r["site"] == "dispatch._resolve_view_shape"


def test_a_one_dimensional_invention_is_recorded(tmp_path, monkeypatch):
    got = _records(tmp_path, monkeypatch, (4, 6), (12,))
    assert len(got) == 1 and got[0]["ratio"] == pytest.approx(2.0), got


def test_an_inferred_shape_is_not_recorded(tmp_path, monkeypatch):
    """A -1 target is NOT an invention; a refusing implementation would still infer it."""
    assert _records(tmp_path, monkeypatch, (2, 16384, 8), (-1, 8)) == []


def test_an_exact_match_is_not_recorded(tmp_path, monkeypatch):
    assert _records(tmp_path, monkeypatch, (2, 4096, 8), (8192, 8)) == []


def test_nothing_is_written_when_the_variable_is_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("NBX_RESHAPE_REPORT", raising=False)
    monkeypatch.chdir(tmp_path)
    _resolve_view_shape(SimpleNamespace(shape=(2, 16384, 8)), [8192, 8])
    assert list(tmp_path.iterdir()) == []


def test_the_resolution_itself_is_unchanged(tmp_path, monkeypatch):
    """Report-only: the shape handed back is the same with the variable set or unset."""
    x = SimpleNamespace(shape=(1, 640, 14))
    monkeypatch.delenv("NBX_RESHAPE_REPORT", raising=False)
    unset = _resolve_view_shape(x, [1, 640, 23])
    monkeypatch.setenv("NBX_RESHAPE_REPORT", str(tmp_path / "r.jsonl"))
    assert _resolve_view_shape(x, [1, 640, 23]) == unset == [1, 640, 14]
