"""A vendor file put in force for one test, and every process state the binding touches restored after it.

`_configs.vendor_profile_for_arch(arch)` RECORDS the description it resolves in `_configs._ACTIVE_PROFILE`, in
place (`_remember_profile` clears and updates the dict). The cells that wrote
`monkeypatch.setattr(CF, "_ACTIVE_PROFILE", CF.vendor_profile_for_arch("7.0"))` saved the attribute AFTER the call had
already filled the very dict monkeypatch saved, so the "restore" put the Volta description back in force for the
process: its `matrix_unit` (m8n8k4) went on to every later cell, and a Triton MoE store on the Apple backend compiled
the NVIDIA mma kernel ("inline assembly has no MSL lowering",
`test_the_triton_store_is_byte_for_byte_the_resident_moe[2]`, red in the in-order unit run and green alone,
2026-11-24; bisected by scripts/bisect_order_leak.py). `wrappers.set_hardware_profile` was not restored either.
"""
from __future__ import annotations


def bind_vendor_file(monkeypatch, arch: str, machine=None) -> dict:
    """Put the vendor file of `arch` in force (and, when given, the Prism profile `machine` declared to the wrappers)
    for the test that owns `monkeypatch`; return the description in force. The dicts and flags are saved BEFORE the
    resolution writes them, so the test's teardown hands back exactly what it found."""
    from nbxtensor.backend import capabilities as K
    from nbxtensor.ops import _configs as CF
    from nbxtensor.ops import wrappers as W
    for mod, name in ((W, "_NBX_HW_PROFILE"), (W, "_NBX_HAS_NATIVE_BF16"), (K, "_BACKEND_HAS_FP64")):
        monkeypatch.setattr(mod, name, getattr(mod, name))
    monkeypatch.setattr(CF, "_IN_FORCE", dict(CF._IN_FORCE))
    monkeypatch.setattr(CF, "_UNDOCUMENTED", dict(CF._UNDOCUMENTED))
    monkeypatch.setattr(CF, "_ACTIVE_PROFILE", {})          # the resolution records into this one, not the process's
    if machine is not None:
        W.set_hardware_profile(machine)
    resolved = CF.vendor_profile_for_arch(arch)
    monkeypatch.setattr(CF, "_ACTIVE_PROFILE", resolved)
    return CF.active_vendor_profile()
