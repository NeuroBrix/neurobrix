"""S6, Prism's rank (the supervisor 2026-10-27 22:20): slicing mochi-1-preview's VAE along H/W at its group norms is the plan
of a rung where nothing faster fits, never the plan of a rung that holds a faster one. Measured on Metal through the
entry point (stable, alone): on the calm Mac the VAE in 1-frame temporal chunks, 213.6 s; at an imposed 8 GB rung the VAE
sliced in 6 stretches, 249 s, the same video (PSNR 46.4 dB) — and at 8 GB the temporal chunks do not fit (the stream is
refused, the slices plan). RED if the calm plan slices (a slower plan chosen while a faster one fits) or the 8 GB rung
refuses. No card."""
import pytest

from tests.unit.prism._pinned_machine import pin_host
from tests.unit.prism.test_a_component_over_the_rung_streams_inside_itself import _plan, no_door


def _mochi(monkeypatch, rung=None):
    no_door(monkeypatch)
    pin_host(monkeypatch, 24576, 17582, "the calm Mac (its measured system share)")
    if rung:
        monkeypatch.setenv("NBX_PRISM_BUDGET_MB", str(rung))
    p, _ = _plan("mochi-1-preview", "compiled")
    return p


def test_on_the_calm_mac_the_vae_streams_in_time_and_is_not_sliced(monkeypatch):
    p = _mochi(monkeypatch)
    assert (p.component_stream or {}).get("vae", {}).get("schedule"), p.component_stream
    assert "vae" not in (p.layer_stream_chunks or {}), p.layer_stream_chunks


def test_at_an_8_gb_rung_the_vae_is_sliced_and_the_plan_is_timed(monkeypatch):
    p = _mochi(monkeypatch, 8192)
    assert (p.layer_stream_chunks or {}).get("vae"), (p.strategy, p.selection_reason[:300])
    assert "vae" not in (p.component_stream or {}), p.component_stream
    assert p.estimated_s is not None and p.estimated_s > 0
