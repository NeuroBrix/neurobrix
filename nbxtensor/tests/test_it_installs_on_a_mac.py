"""nbxtensor installs on a Mac. PyPI publishes no Triton for macOS (triton 3.8.0: two manylinux wheels, no sdist): the
Metal Triton is built from source (Triton with TRITON_EXT_ENABLED=1 and triton-ext's AppleGPU backend), so a plain
"triton" dependency made `pip install nbxtensor` — and `pip install "neurobrix[pytorch]"`, the PyTorch engine on MPS —
ResolutionImpossible on a clean macOS venv (the Mac, 2026-11-06 23:03). The engine's own pyproject already carries the
marker. Where Triton is absent the library refuses by name with what to build (backend/metal_backend).

What this test would do if the packaging were wrong: a triton dependency that macOS must resolve from PyPI (red)."""
from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:          # python 3.10
    import tomli as tomllib

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_triton_is_not_resolved_from_pypi_on_macos():
    deps = tomllib.loads(PYPROJECT.read_text())["project"]["dependencies"]
    triton = [d.replace(" ", "").replace('"', "'") for d in deps if d.split(";")[0].strip() == "triton"]
    assert triton == ["triton;platform_system!='Darwin'"], triton
