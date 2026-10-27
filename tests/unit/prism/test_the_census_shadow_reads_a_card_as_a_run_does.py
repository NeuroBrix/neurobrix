"""The census shadow plans a dedicated card as a RUN reads it — the runtime's total less its own context, measured into
the profile (`runtime_total_mb`, `runtime_context_mb`) — never the nominal `memory_mb`. Gate ebm 16g (2026-10-27): Sana
4K triton died on a conv2d band key the coverage check never met — the shadow had planned the V100-16GB at 16 384 MB, the
run at 15 845, so the bands, and their keys, differed. What this test would do if the code were wrong: the nominal
figure read, or a profile without the facts silently planned at it — RED."""
import copy

import pytest

from neurobrix.core.host_memory import MemoryState
from neurobrix.core.prism import solver as S
from tests.unit.prism._pinned_machine import V100_16GB, profile


def _reading(monkeypatch, spec):
    monkeypatch.setattr(S, "_census_shadow_active", lambda: True)
    from neurobrix.kernels import census
    monkeypatch.setattr(census, "active", lambda: True)
    p = profile(spec)
    dev = p.devices[0]
    host = MemoryState(total_mb=257530, available_mb=200000, source="test")
    return S.PrismSolver()._device_reading(dev, float(dev.memory_mb), host, p)


def test_a_dedicated_card_in_the_shadow_is_the_runtimes_total_less_its_context(monkeypatch):
    spec = copy.deepcopy(V100_16GB)
    spec["devices"][0].update(runtime_total_mb=16151, runtime_context_mb=306)
    r = _reading(monkeypatch, spec)
    assert r.capacity_mb == 16151 and r.own_context_mb == 306, (r.capacity_mb, r.own_context_mb)


def test_a_dedicated_card_without_the_measured_facts_is_refused_by_name(monkeypatch):
    with pytest.raises(RuntimeError, match="runtime_total_mb"):
        _reading(monkeypatch, copy.deepcopy(V100_16GB))
