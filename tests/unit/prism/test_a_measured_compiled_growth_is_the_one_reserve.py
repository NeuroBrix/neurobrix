"""The compiled engine's growth during a run, once a fact on the device, is ONE reserve and the rung is whole.

2026-11-08 (the Mac, RUN 78, mochi-1-preview native on the M4 Pro, VM off): the MPS allocator's cache reached 9.2 GB
beyond the live tensors (driver 17 076 MB against 8 221 current) and the run ran out of memory with 4.71 GiB of 'other
allocations', where Prism had taken off the vendor's constant `compiled_private_reserve_mb` (3 900 MB) and kept a fixed
0.92 of the rung for the compiled engine (`whole_component_fraction`, its provenance never recorded). The engine's
measured growth (core/runtime/runtime_growth, per model and engine, else the vendor's envelope) is the reserve: the
budget takes it once (`_runtime_growth`), the private reserve is not taken a second time, and the compiled engine's
rung part is the whole rung, as the Triton engine's already is. A device with no growth fact is planned as before.

What this test does if the code is wrong: the first cell fails when the compiled engine keeps 0.92 of the rung beside a
measured growth (red on main); the second when the private reserve is still taken beside it (red on main); the third
when a device with no growth fact changes.

Run: python -m pytest tests/unit/prism/test_a_measured_compiled_growth_is_the_one_reserve.py
"""
import pytest

from tests.unit.prism._pinned_machine import APPLE_M4_PRO, V100_16GB, profile, pin_host


def _solver(mode="compiled"):
    from neurobrix.core.prism import PrismSolver
    s = PrismSolver()
    s._mode = mode
    s._growth_model = "mochi-1-preview"
    s._load_copies_mb = 0.0
    return s


def _measured(monkeypatch, mb):
    from neurobrix.core.runtime import runtime_growth
    monkeypatch.setattr(runtime_growth, "load", lambda model, engine, arch: mb if engine == "compiled" else None)


def test_the_compiled_rung_is_whole_beside_a_measured_growth(monkeypatch):
    _measured(monkeypatch, 9200.0)
    s = _solver()
    spec = profile(APPLE_M4_PRO).devices[0]
    assert s._part_of_rung_mb(12288.0, spec) == pytest.approx(12288.0)


def test_the_private_reserve_is_not_taken_beside_a_measured_growth(monkeypatch):
    pin_host(monkeypatch, 24576, 18186, "the Mac, idle")
    p = profile(APPLE_M4_PRO)
    base = _solver()._prepare_devices(p)[0].capacity_mb
    _measured(monkeypatch, 9200.0)
    with_growth = _solver()._prepare_devices(p)[0].capacity_mb
    # the reserve given back (3 900) and the measured growth taken once by the budget (9 200): one reserve, not two
    assert with_growth == pytest.approx(base + 3900.0 - 9200.0), (base, with_growth)


def test_a_device_without_a_growth_fact_is_planned_as_before(monkeypatch):
    _measured(monkeypatch, None)
    s = _solver()
    spec = profile(V100_16GB).devices[0]
    assert s._part_of_rung_mb(12288.0, spec) == pytest.approx(12288.0 * s.whole_component_fraction)
