"""S6, Prism's price of slicing (the supervisor 2026-10-28 06:12 point 1): a layer-streaming plan that runs stretches in
slices of a token axis adds to its estimate (1) each slice's halo recompute — the stretch's own work times the share its
slices read beyond their rows — and (2) for a stretch whose inputs wait on the host, its slices cut from host tensors
and its outputs assembled, at the host's measured slab rate (`host_slab_gbps`). Measured on mochi-1-preview native on
the M4 Pro (stable, 2026-10-28): calm (1-frame temporal chunks) 207.5 s, at an imposed 8 GB rung (6 sliced stretches,
2 reading from the host) 226.1 s; the halo +2.7 s, the host slabs 9.9 GiB per host stretch in 3.3-3.6 s. Before: both
plans priced the same. Without the host fact the sliced plan is not timed, said by name. No card."""
import pytest

import neurobrix.core.prism.solver as S
from tests.unit.prism._pinned_machine import pin_host
from tests.unit.prism.test_a_component_over_the_rung_streams_inside_itself import _plan, no_door


def _mochi(monkeypatch, rung=None):
    no_door(monkeypatch)
    pin_host(monkeypatch, 24576, 17582, "the calm Mac")
    if rung:
        monkeypatch.setenv("NBX_PRISM_BUDGET_MB", str(rung))
    terms = []
    real = S.PrismSolver._sliced_stretch_seconds
    monkeypatch.setattr(S.PrismSolver, "_sliced_stretch_seconds",
                        lambda self, *a, **k: (terms.append(real(self, *a, **k)), terms[-1])[1])
    p, s = _plan("mochi-1-preview", "compiled")
    return p, s, terms


def test_the_sliced_plan_pays_its_halo_and_its_host_slabs(monkeypatch):
    calm, _, t0 = _mochi(monkeypatch)
    assert t0 == [0.0], t0                                    # no stretch sliced: nothing added
    sliced, s, t1 = _mochi(monkeypatch, 8192)
    assert sliced.layer_stream_chunks.get("vae"), sliced.layer_stream_chunks
    assert 5.0 < t1[-1] < 30.0, t1                            # halo (~2 s) + 2 x 9.9 GiB at 3.06 GB/s (~7 s)
    assert sliced.estimated_s > calm.estimated_s              # the slower plan is priced slower (226 vs 207.5 s)


def test_without_the_host_slab_rate_the_sliced_plan_is_not_timed(monkeypatch):
    import neurobrix.core.prism.time_estimate as TE
    real = TE.host_facts
    import dataclasses
    monkeypatch.setattr(TE, "host_facts", lambda m: dataclasses.replace(real(m), slab_bps=None))
    p, _, _ = _mochi(monkeypatch, 8192)
    assert p.estimated_s is None and "host_slab_gbps" in str((p.time_estimates or {}).get("layer_streaming")), \
        p.time_estimates
