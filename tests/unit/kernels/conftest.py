"""Fixtures shared by the kernels cells."""

# The shared rig door, so every measuring cell in this directory can ask for a
# free rig by naming `a_free_rig` — see `_rig.py` for why it is a fixture and not
# a module-level mark.
from ._rig import a_free_rig, rig_reason  # noqa: F401


import pytest
import nbxtensor.backend.capabilities as _nbx_capabilities  # the module the moved names live in (M2 cut of tensor.py)


@pytest.fixture
def without_matrix_unit(monkeypatch):
    """The card as if its profile declared no `matrix_unit`: every GEMM, convolution and attention takes the
    tl.dot kernels and their autotune keys. For the cells that prove the certifier, the sweep or the FMA routes —
    machinery a GEMM on the unit (no key, the profile's tile) never reaches."""
    from nbxtensor.ops import wrappers as W
    from nbxtensor.ops import _configs as C
    monkeypatch.setattr(C, "matrix_unit", lambda: {})
    import importlib
    for _m in (W, *(importlib.import_module(f"nbxtensor.ops.{f}") for f in W._FAMILY_MODULES)):  # the core and its families (M2 split) each bind it
        monkeypatch.setattr(_m, "_matrix_unit", lambda: {})


@pytest.fixture
def host_backend_fp64(monkeypatch):
    """nbx_tensor's float64 capability declared as a run declares it (`set_hardware_profile`): from
    this host's hardware profile (`precision.kernels_carry_fp64.triton`), restored after the cell."""
    from neurobrix.core.prism.autodetect import load_default_profile
    from nbxtensor import tensor as T
    import nbxtensor.backend.capabilities as _nbx_capabilities
    from neurobrix.triton.dtype import profile_triton_has_fp64
    monkeypatch.setattr(_nbx_capabilities, "_BACKEND_HAS_FP64", None)
    _nbx_capabilities.set_backend_has_fp64(profile_triton_has_fp64(load_default_profile()))


@pytest.fixture
def as_the_m4_pro(monkeypatch):
    """The vendor profile in force as the Apple M4 Pro's (`apple/apple_m4_pro.yml`): its bucket ladders, no matrix unit.
    For the cells that price or synthesise keys MEASURED on that card (the certifier's price points, the keys refused
    on its 18 GB table): a key is formed under its card's profile, never under the one where the cell happens to run
    — on a V100 the volta ladder and matrix unit refused them as census defects."""
    from nbxtensor.ops import _configs as C
    from neurobrix.core.config.loader import get_vendor_config
    cfg = dict(get_vendor_config("apple", "apple_m4_pro"))
    cfg.update({"_vendor": "apple", "_profile": "apple_m4_pro", "_exact": True})
    monkeypatch.setattr(C, "_ACTIVE_PROFILE", cfg)
    return cfg
