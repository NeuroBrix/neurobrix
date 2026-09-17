"""The backend we pin has to be a name TRITON has, and ours are not its.

`_pin_triton_backend` sets `TRITON_DEFAULT_BACKEND` so Triton selects a backend directly instead
of calling `is_active()` on every registered one — upstream's AMD probe imports torch inside its
own, so merely asking the question puts torch in the process and breaks R33.

This engine names its backends for the vendor RUNTIME it loads: `cuda`, `hip`, `metal`. Triton
names its own for the compiler target, and they are the directory names under `triton/backends/`:
**`nvidia`** and **`amd`**. `triton/runtime/driver.py::_create_driver` raises
`Unknown backend device '<name>'. Available backends: [...]` on anything else.

So pinning `cuda` was always wrong, and it never showed: this engine's launcher does every launch
itself and never asks Triton for a driver. It showed the moment a launch went down Triton's own
path — torch 2.14 ships in-tree Triton ops and the seam hands those back to Triton, which then
read the pin (2026-09-17, the second defect behind the warm compiled bmm failure).

A name Triton does not have is worse than no pin at all, so an unknown mapping pins nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from neurobrix.kernels import nbx_tensor as N  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("TRITON_DEFAULT_BACKEND", raising=False)


def _pin_with(monkeypatch, registry):
    """Pin with a fake triton.backends registry, so the test states which names exist."""
    mod = type(sys)("triton.backends")
    mod.backends = registry
    monkeypatch.setitem(sys.modules, "triton.backends", mod)


def test_cuda_pins_tritons_nvidia(monkeypatch):
    _pin_with(monkeypatch, {"nvidia": object(), "amd": object()})
    assert N._pin_triton_backend("cuda") == "cuda", "the engine's own name is what it returns"
    import os
    assert os.environ["TRITON_DEFAULT_BACKEND"] == "nvidia", "but Triton is told Triton's name"


def test_hip_pins_tritons_amd(monkeypatch):
    _pin_with(monkeypatch, {"nvidia": object(), "amd": object()})
    N._pin_triton_backend("hip")
    import os
    assert os.environ["TRITON_DEFAULT_BACKEND"] == "amd"


def test_a_name_triton_does_not_have_pins_nothing(monkeypatch):
    """Worse than no pin: `_create_driver` raises on an unknown name, so every launch down
    Triton's own path dies with `Unknown backend device`."""
    _pin_with(monkeypatch, {"nvidia": object(), "amd": object()})
    N._pin_triton_backend("metal")          # no Metal plugin in this registry
    import os
    assert "TRITON_DEFAULT_BACKEND" not in os.environ


def test_metal_pins_metal_when_the_plugin_is_registered(monkeypatch):
    _pin_with(monkeypatch, {"nvidia": object(), "metal": object()})
    N._pin_triton_backend("metal")
    import os
    assert os.environ["TRITON_DEFAULT_BACKEND"] == "metal"


def test_an_explicit_choice_outranks_ours(monkeypatch):
    monkeypatch.setenv("TRITON_DEFAULT_BACKEND", "amd")
    _pin_with(monkeypatch, {"nvidia": object(), "amd": object()})
    N._pin_triton_backend("cuda")
    import os
    assert os.environ["TRITON_DEFAULT_BACKEND"] == "amd", "setdefault, not set"


def test_no_triton_at_all_still_pins_the_mapped_name(monkeypatch):
    """A compiled-only install has no Triton to ask; the mapping is still the right answer."""
    monkeypatch.setitem(sys.modules, "triton.backends", None)
    N._pin_triton_backend("cuda")
    import os
    assert os.environ["TRITON_DEFAULT_BACKEND"] == "nvidia"


def test_the_real_registry_has_the_names_this_mapping_targets():
    """The premise, checked against the installed Triton rather than assumed."""
    triton_backends = pytest.importorskip("triton.backends")
    names = set(getattr(triton_backends, "backends", {}))
    if not names:
        pytest.skip("this Triton exposes no backend registry to read")
    assert "nvidia" in names, f"the mapping targets 'nvidia'; this Triton has {sorted(names)}"
