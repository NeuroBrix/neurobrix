"""A capability that is a property of the launcher the engine selected is DECLARED to nbx_tensor
by the engine — nbx_tensor asks the engine nothing.

Until 2026-09-26 `backend_loads_pointers_from_memory` imported `nbxtensor.backend.metal_backend`
inside nbx_tensor.py to ask which Metal backend was selected (9828c7f7), and the boundary ratchet
`test_the_boundary_does_not_widen` was red on main. The answer is the same — triton_ext proved its
pinned scope, any other Metal launcher answers False — but it travels the other way: the selection
itself (`selected_metal_backend`, passed by the launcher's driver seam, the autotune key
resolution and the refusal reader before a kernel is launched) declares it. Seen RED on
main dcbd4748: no `declare_backend_capability`, the ratchet cell red.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from nbxtensor import tensor as T
import nbxtensor.backend.capabilities as _nbx_capabilities
import nbxtensor.backend.runtime as _nbx_runtime

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture
def on_metal(monkeypatch):
    monkeypatch.setattr(_nbx_runtime, "gpu_backend", lambda: "metal")
    monkeypatch.setattr(_nbx_capabilities, "_DECLARED_CAPABILITIES", {})


def test_undeclared_metal_answers_the_tables_false(on_metal):
    assert _nbx_capabilities.backend_loads_pointers_from_memory() is False


def test_the_engines_declaration_is_read_above_the_table(on_metal):
    _nbx_capabilities.declare_backend_capability("metal", loads_pointers_from_memory=True)
    assert _nbx_capabilities.backend_loads_pointers_from_memory() is True
    _nbx_capabilities.declare_backend_capability("metal", loads_pointers_from_memory=False)
    assert _nbx_capabilities.backend_loads_pointers_from_memory() is False


def test_a_declaration_for_another_backend_does_not_speak_for_this_one(on_metal):
    _nbx_capabilities.declare_backend_capability("hip", loads_pointers_from_memory=True)
    assert _nbx_capabilities.backend_loads_pointers_from_memory() is False


def test_a_declaration_is_a_bool():
    with pytest.raises(TypeError):
        _nbx_capabilities.declare_backend_capability("metal", loads_pointers_from_memory="yes")


def test_the_selection_declares_what_it_selected():
    src = (REPO / "nbxtensor/src/nbxtensor/backend/metal_backend.py").read_text()
    body = src.split("def selected_metal_backend(", 1)[1].split("\ndef ", 1)[0]
    assert body.count("return _declare_selected(") == 2 and "return declared" not in body
    helper = src.split("def _declare_selected(", 1)[1].split("\ndef ", 1)[0]
    assert re.search(r'declare_backend_capability\("metal",\s*loads_pointers_from_memory=\(name == "triton_ext"\)\)', helper)


def test_nbx_tensor_no_longer_imports_the_metal_backend():
    src = (REPO / "nbxtensor/src/nbxtensor/tensor.py").read_text()
    assert "nbxtensor.backend.metal_backend" not in src
