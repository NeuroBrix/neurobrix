"""Prism asks nbxtensor whether the weights fit float16; no range constant or scan lives in the engine."""
import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[3] / "src" / "neurobrix"
SIDES = pytest.mark.parametrize("side", ["pytorch", "triton"])     # each branch's own Prism (RED_LINES line 16)


def _solver(side):
    return SRC / side / "prism" / "solver.py"


def _tree(side):
    return ast.parse(_solver(side).read_text())


@SIDES
def test_the_engine_keeps_no_fp16_range_constant(side):
    names = {n.id for t in ast.walk(_tree(side)) if isinstance(t, ast.Assign) for n in ast.walk(t)
             if isinstance(n, ast.Name)}
    assert "FP16_MAX" not in names
    assert "65504" not in _solver(side).read_text()


@SIDES
def test_the_engine_keeps_no_local_range_scan(side):
    src = _solver(side).read_text()
    assert "bf16_abs_maxima" not in src
    assert "finite_max" not in src


@SIDES
def test_prism_imports_the_library_range_scan(side):
    imported = [(n.module, a.name) for n in ast.walk(_tree(side)) if isinstance(n, ast.ImportFrom) for a in n.names]
    assert ("nbxtensor.io.safetensors", "fp16_range_scan") in imported
