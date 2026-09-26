"""The re-trace tool takes Forge's interpreter through `NBX_FORGE_PYTHON`, as it takes the
engine's through `NBX_PYTHON`; the default is the toolchain's own venv.

Forge moves onto the engine environment behind the executed graph gate (2026-09-26): the gate
is this tool run with Forge's python switched, so the path is a door, not a literal.
Run: python -m pytest tests/unit/tools/test_the_retrace_tool_takes_forges_python_through_a_door.py
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[3] / "tools"


def _fresh(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("NBX_FORGE_PYTHON", raising=False)
    else:
        monkeypatch.setenv("NBX_FORGE_PYTHON", value)
    sys.path.insert(0, str(TOOLS))
    sys.modules.pop("retrace_zoo", None)
    return importlib.import_module("retrace_zoo")


def test_the_door_switches_forges_python_and_the_default_is_the_toolchains_venv(monkeypatch):
    assert _fresh(monkeypatch, "/home/mlops/venvs/nbx_t214/bin/python").PY == "/home/mlops/venvs/nbx_t214/bin/python"
    assert _fresh(monkeypatch, None).PY == "/home/mlops/ml/venv/bin/python"
