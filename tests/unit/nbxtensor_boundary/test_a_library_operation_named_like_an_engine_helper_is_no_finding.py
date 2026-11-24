"""A library OPERATION that shares a name with an engine helper is no "fact" finding (qa 1326-1).

The scan calls a fact every public function of the library's FACT_MODULES. nbxtensor.memory.guarded (`guarded`,
`place`) holds operations of the element-wise proof, not facts; the engine's own `place` / `guarded` helpers only share
the name. Proven on a library and an engine written here: `memory/layout.py` (a fact module) and `memory/guarded.py`
(an operation module), the engine defining a function of each name.

What would this file do if guarded.py were swept in with its package? The engine's `place` is reported as a "fact"
finding (RED); if the fact module were dropped instead, `aligned` stops being reported (RED)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("engine_slim_scan", REPO / "tools" / "engine_slim_scan.py")
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)

_BODY = ("    total = 0\n    for i, v in enumerate(values):\n        total += v * i + len(values)\n"
         "        if total > 99:\n            total -= v\n    return total, total * 2\n")


def _write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def test_a_library_operation_named_like_an_engine_helper_is_no_fact_finding(tmp_path):
    lib, repo = tmp_path / "lib" / "nbxtensor", tmp_path / "repo"
    _write(lib, "__init__.py", "__version__ = '0'\n")
    _write(lib, "memory/__init__.py", "")
    _write(lib, "memory/layout.py", "def aligned(n):\n    return n\n")
    _write(lib, "memory/guarded.py", "def place(end, n):\n    return end - n\n")
    _write(repo, "src/neurobrix/a.py",
           "def place(values, limit):\n" + _BODY + "\ndef aligned(values, limit):\n" + _BODY + "\n"
           "def entry():\n    return place([1], 2), aligned([1], 2)\n")
    facts = {s for k, _, s, _ in S.scan(repo, lib) if k == "fact"}
    assert "aligned" in facts, "the memory fact module is no longer read"
    assert "place" not in facts, "an operation of memory/guarded.py was read as a fact"

