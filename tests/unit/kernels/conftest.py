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


@pytest.fixture
def host_backend_fp64(monkeypatch):
    """nbx_tensor's float64 capability declared as a run declares it (`set_hardware_profile`): from
    this host's hardware profile (`precision.kernels_carry_fp64.triton`), restored after the cell."""
    from neurobrix.core.prism.autodetect import load_default_profile
    from neurobrix.kernels import nbx_tensor as T
    from neurobrix.triton.dtype import profile_triton_has_fp64
    monkeypatch.setattr(T, "_BACKEND_HAS_FP64", None)
    T.set_backend_has_fp64(profile_triton_has_fp64(load_default_profile()))
