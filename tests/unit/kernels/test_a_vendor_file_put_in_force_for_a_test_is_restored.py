"""A vendor file a test puts in force (`_vendor_file.bind_vendor_file`) is gone when the test ends: the description
in force, the wrappers' hardware profile and the kernels' float64 flag are what they were before.

The cells that bound the Volta file wrote `monkeypatch.setattr(CF, "_ACTIVE_PROFILE", CF.vendor_profile_for_arch(
"7.0"))`: the call records the Volta description INTO the dict monkeypatch then saves, so the teardown put Volta back
in force for the process, and a Triton MoE store later compiled the NVIDIA m8n8k4 mma kernel on the Apple backend
(`test_the_triton_store_is_byte_for_byte_the_resident_moe[2]`, red in the in-order unit run, green alone, 2026-11-24;
bisected by scripts/bisect_order_leak.py to `test_the_launcher_names_its_row_breakpoints.py`).

What this test would do if the code were wrong: with `bind_vendor_file` back to that spelling, the description in
force after the scope is Volta's (its `matrix_unit` declared) — red."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.needs_gpu_runtime

from tests.unit.kernels._vendor_file import bind_vendor_file
from tests.unit.prism._pinned_machine import V100_16GB, profile


def test_the_volta_file_bound_for_a_scope_is_not_in_force_after_it():
    """Bind Volta inside a monkeypatch scope, as a cell's fixture does, and find the process's state as it was."""
    from nbxtensor.backend import capabilities as K
    from nbxtensor.ops import _configs as CF
    from nbxtensor.ops import wrappers as W
    CF.active_vendor_profile()                                   # this host's description resolved first
    before = (dict(CF._ACTIVE_PROFILE), W._NBX_HW_PROFILE, W._NBX_HAS_NATIVE_BF16, K._BACKEND_HAS_FP64)
    with pytest.MonkeyPatch.context() as mp:
        bound = bind_vendor_file(mp, "7.0", profile(V100_16GB))
        assert bound.get("matrix_unit"), "the Volta file declares its matrix unit (the premise of this test)"
    after = (dict(CF._ACTIVE_PROFILE), W._NBX_HW_PROFILE, W._NBX_HAS_NATIVE_BF16, K._BACKEND_HAS_FP64)
    assert after[0] == before[0], "the Volta description is still in force after the scope"
    assert after[1:] == before[1:], "the wrappers' hardware profile or the float64 flag was not restored"
