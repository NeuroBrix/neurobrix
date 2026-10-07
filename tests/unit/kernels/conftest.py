"""Fixtures shared by the kernels cells."""

# The shared rig door, so every measuring cell in this directory can ask for a
# free rig by naming `a_free_rig` — see `_rig.py` for why it is a fixture and not
# a module-level mark.
from ._rig import a_free_rig, rig_reason  # noqa: F401


import pytest


@pytest.fixture
def without_matrix_unit(monkeypatch):
    """The card as if its profile declared no `matrix_unit`: every GEMM, convolution and attention takes the
    tl.dot kernels and their autotune keys. For the cells that prove the certifier, the sweep or the FMA routes —
    machinery a GEMM on the unit (no key, the profile's tile) never reaches."""
    from neurobrix.kernels import wrappers as W
    from neurobrix.kernels.ops import _configs as C
    monkeypatch.setattr(C, "matrix_unit", lambda: {})
    monkeypatch.setattr(W, "_matrix_unit", lambda: {})
