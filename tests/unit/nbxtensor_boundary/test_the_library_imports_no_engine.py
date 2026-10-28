"""The nbxtensor library stands alone: it imports neither the engine nor torch.

A library that imports `neurobrix` cannot be published on its own, and one
that imports torch is not a replacement for torch. The only torch allowed is
the optional interop module (`nbxtensor/interop/torch.py`) and the benchmark
harness (`nbxtensor/bench/`), which compares against torch by definition.

Parsed with `ast`, not grepped: a docstring mentioning torch is not an import.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LIB = REPO / "nbxtensor" / "src" / "nbxtensor"

TORCH_ALLOWED = {"interop/torch.py"}


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


def test_the_library_imports_no_torch_outside_interop():
    bad = _offenders("torch", TORCH_ALLOWED)
    assert not bad, "nbxtensor must not import torch: " + "; ".join(bad)
