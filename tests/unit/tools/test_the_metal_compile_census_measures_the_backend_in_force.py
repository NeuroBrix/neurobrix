"""tools/metal_compile_census.py is the compile censuses' one harness (the Intel SPIR-V census imports its kernels,
signatures and device-function screen) and the Apple census itself. It must measure the Metal backend IN FORCE — the
one nbxtensor's table names (`backend/metal_backend.METAL_BACKENDS`) — never the fork archived on 2026-09-17: until
2026-11-10 its own run imported `triton_msl` and could not start on any current environment (ModuleNotFoundError).
What this would do if the code were wrong: no Apple census at all, while the tool's name says there is one."""
import ast
from pathlib import Path

TOOL = Path(__file__).resolve().parents[3] / "tools" / "metal_compile_census.py"


def imported_modules(source: str):
    out = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            out.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module.split(".")[0])
    return out


def test_the_census_imports_no_archived_backend_and_no_torch():
    mods = imported_modules(TOOL.read_text())
    assert "triton_msl" not in mods and "torch" not in mods, sorted(mods)


def test_the_backend_class_and_the_target_come_from_nbxtensor_s_table():
    src = TOOL.read_text()
    assert "backend_compiler_class()" in src and "metal_target()" in src
