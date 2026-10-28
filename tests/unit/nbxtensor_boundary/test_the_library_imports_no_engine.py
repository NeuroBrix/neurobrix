"""The nbxtensor library stands alone: it imports neither the engine nor torch.

A library that imports `neurobrix` cannot be published on its own, and one
that imports torch is not a replacement for torch. No exception, not even an
optional interop module (Hocine, 28/10/2026 20:33): exchange goes through
DLPack, imported by the other side. The torch comparison harness lives
outside the library, in benchmarks/nbxtensor_vs_torch/.

Parsed with `ast`, not grepped: a docstring mentioning torch is not an import.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LIB = REPO / "nbxtensor" / "src" / "nbxtensor"

TORCH_ALLOWED: frozenset = frozenset()   # empty, and it stays empty


def _imports(path: Path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield node.lineno, a.name
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.lineno, node.module


def _offenders(top: str, allowed=frozenset()):
    bad = []
    for p in sorted(LIB.rglob("*.py")):
        rel = str(p.relative_to(LIB))
        if rel in allowed:
            continue
        for line, name in _imports(p):
            if name == top or name.startswith(top + "."):
                bad.append(f"{rel}:{line} imports {name}")
    return bad


def test_the_library_exists():
    assert (LIB / "__init__.py").is_file()


def test_the_library_imports_no_engine():
    bad = _offenders("neurobrix")
    assert not bad, "nbxtensor must not import the engine: " + "; ".join(bad)


def test_the_library_imports_no_torch_at_all():
    bad = _offenders("torch", TORCH_ALLOWED)
    assert not bad, "nbxtensor must not import torch: " + "; ".join(bad)
