"""`tools/scatter_scan.py` reads the engine's two branches (RED_LINES.md lines 16-22): a definition of
`triton/<brick>/<rel>.py` is the twin of the same name in `pytorch/<brick>/<rel>.py`, held to its original's row; inside
each side the scan still reports every scattered definition. Proven on an engine tree written here.

What would this file do if the scan were wrong?
  * a twin counted as a second row beside its original's, or a twin of a reviewed name (NOT_SEMANTIC) reported: RED;
  * a scattered definition only the twin carries, the same name in a triton file that twins nothing, or the same name
    in two pytorch files, no longer reported: RED;
  * the raw count (`twin_aware=False`) no longer counting twins separately: RED."""
from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("scatter_scan", REPO / "tools" / "scatter_scan.py")
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)


def _write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def _engine(tmp_path: Path) -> Path:
    e = tmp_path / "src" / "neurobrix"
    for side in ("pytorch", "triton"):
        _write(e, f"{side}/prism/solver.py", "def _arena_bytes():\n    return 1\n")
        _write(e, f"{side}/prism/chunked_region.py", "def region_carriers():\n    return 1\n")   # reviewed: NOT_SEMANTIC
    _write(e, "triton/prism/solver.py", "def _arena_bytes():\n    return 1\n\ndef _twin_own_stride():\n    return 2\n")
    _write(e, "triton/prism/extra.py", "def _arena_bytes():\n    return 3\n")       # same name, a file twinning nothing
    _write(e, "pytorch/prism/one.py", "def _row_strides():\n    return 4\n")
    _write(e, "pytorch/prism/two.py", "def _row_strides():\n    return 4\n")
    return e


def _scan(e: Path, twin_aware: bool = True):
    return {(r[0], r[1]) for r in S.scan(e, twin_aware=twin_aware)}


def test_a_twin_is_held_to_its_originals_row_and_review(tmp_path):
    got = _scan(_engine(tmp_path))
    assert ("pytorch/prism/solver.py", "_arena_bytes") in got
    assert ("triton/prism/solver.py", "_arena_bytes") not in got, sorted(got)
    assert not any(s == "region_carriers" for _, s in got), sorted(got)


def test_each_branch_keeps_its_teeth(tmp_path):
    got = _scan(_engine(tmp_path))
    expected = {("triton/prism/solver.py", "_twin_own_stride"),     # only the twin carries it
                ("triton/prism/extra.py", "_arena_bytes"),          # same name in a triton file twinning nothing
                ("pytorch/prism/one.py", "_row_strides"),           # same name in two pytorch files
                ("pytorch/prism/two.py", "_row_strides")}
    assert not expected - got, sorted(expected - got)


def test_the_raw_count_counts_every_twin_separately(tmp_path):
    e = _engine(tmp_path)
    raw, aware = _scan(e, twin_aware=False), _scan(e)
    assert raw - aware == {("triton/prism/solver.py", "_arena_bytes"), ("triton/prism/chunked_region.py", "region_carriers")}
    assert S.twin_original("triton/prism/solver.py", e) == "pytorch/prism/solver.py"
    assert S.twin_original("triton/prism/extra.py", e) is None
    assert S.twin_original("triton/arena.py", e) is None
