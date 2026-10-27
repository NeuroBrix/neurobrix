"""A component the flow RE-ENTERS (a language model per token, a denoiser per step) that the rung cannot hold whole keeps
loaded, between its runs, every piece the rung holds beside the streaming window, and streams only the rest — a few
hundred MB short costs a few hundred MB reloaded per use, never the whole component (the supervisor's S5, 2026-10-27
22:20: "garder le cœur résident et ne streamer que le nécessaire"; rule 13, the fastest plan that fits).

Before: Janus-Pro-7B triton on the Mac with 16 732 MB free (lazy_sequential's host side 17 031 MB, 190 MB over) re-planned
to layer_streaming with its language model in 2 pieces, both re-read every token: 716 reloads in 38 min, estimated
3 648 s honestly (d9cf2d19), where the facts' plan runs in 41 s (stable f8def0b9, 2026-10-27 19:3x). RED: no piece held,
the estimate ~1 h. No card."""
import pytest

from tests.unit.prism._pinned_machine import pin_host
from tests.unit.prism.test_a_component_over_the_rung_streams_inside_itself import _plan, no_door


def test_janus_triton_on_a_busy_mac_holds_its_language_model_and_streams_only_the_overflow(monkeypatch):
    no_door(monkeypatch)
    pin_host(monkeypatch, 24576, 16732, "the Mac with 677 MB taken by other applications")
    p, s = _plan("Janus-Pro-7B", "triton")
    assert p.strategy == "layer_streaming", p.strategy
    held = (p.layer_stream_resident or {}).get("language_model")
    assert held, ("no piece of the language model is held between tokens", p.layer_stream_plan)
    part = s._layer_stream_partitions["language_model"]
    streamed = sum(seg.weight_bytes for i, seg in enumerate(part.segments) if i not in set(held))
    assert streamed < 0.25 * part.total_weight_bytes, (streamed / 2**20, part.total_weight_bytes / 2**20)
    # what is reloaded per token is priced: far under the all-streamed hour, still above the calm 41 s
    assert p.estimated_s is not None and 41 < p.estimated_s < 900, (p.estimated_s, p.selection_reason[:300])
    assert p.device_window_mb <= 16732, p.device_window_mb


def test_a_component_run_once_keeps_the_plain_cut(monkeypatch):
    """A piece held between runs buys nothing for a component run once (an encoder, a decoder): its cut is the one
    it had, so no streamed plan moves where nothing is gained."""
    from neurobrix.core.prism.layer_partition import LayerPartitioner
    assert hasattr(LayerPartitioner, "hybrid")
    no_door(monkeypatch)
    pin_host(monkeypatch, 24576, 16732, "the Mac with 677 MB taken by other applications")
    p, _ = _plan("Janus-Pro-7B", "triton")
    once = [c for c in (p.layer_stream_resident or {}) if c != "language_model"]
    assert not once, once


def test_a_denoiser_run_per_step_keeps_the_plain_cut(monkeypatch):
    """mochi-1-preview's transformer (2 steps) held in 61 pieces on the calm Mac entered its VAE with 1.2 GB more driver
    memory than the plain cut and the VAE's stream ran out of memory (2026-10-27 23:47): only a per-token component is
    held, until the per-piece cost is measured."""
    no_door(monkeypatch)
    pin_host(monkeypatch, 24576, 17582, "the calm Mac (its measured system share)")
    p, _ = _plan("mochi-1-preview", "compiled")
    assert not (p.layer_stream_resident or {}), p.layer_stream_resident
