"""Prism asks nbxtensor whether the weights fit float16; no range constant or scan lives in the engine."""
import ast
import pathlib

SOLVER = pathlib.Path(__file__).resolve().parents[3] / "src" / "neurobrix" / "core" / "prism" / "solver.py"


def _tree():
    return ast.parse(SOLVER.read_text())


def test_the_engine_keeps_no_fp16_range_constant():
    names = {n.id for t in ast.walk(_tree()) if isinstance(t, ast.Assign) for n in ast.walk(t) if isinstance(n, ast.Name)}
    assert "FP16_MAX" not in names
    assert "65504" not in SOLVER.read_text()


def test_the_engine_keeps_no_local_range_scan():
    src = SOLVER.read_text()
    assert "bf16_abs_maxima" not in src
    assert "finite_max" not in src


def test_prism_imports_the_library_range_scan():
    imported = [(n.module, a.name) for n in ast.walk(_tree()) if isinstance(n, ast.ImportFrom) for a in n.names]
    assert ("nbxtensor.io.safetensors", "fp16_range_scan") in imported
