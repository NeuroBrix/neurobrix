"""An audio clip reaches the graph at its own length on every axis the graph carries
symbolically; the fit to the trace extent stays for the axes the graph froze.

granite-speech-3.3-8b, 2026-09-26: the encoder's frame axis is symbolic in the container, and
the flow zero-padded or cut every clip to the 700 frames of the trace ("Expected input shape
(1, 700, 160)"): an 11 s clip of 550 stacked frames was padded, a 4 s clip of 400 padded, a
clip beyond 14 s would have been cut — and a pad-to-window defect of the encoder (right only
for frame counts congruent to 100 mod 200) stayed invisible to every arm, because no clip ever
reached the graph at another length. The same fit was written twice (the ATen flow and the
triton mirror); it is one brick now. Seen RED on main bd56bf90: the 400-frame clip came back
at 700.

Run: python -m pytest tests/unit/flow/test_the_audio_flow_feeds_a_symbolic_frame_axis_unpadded.py
"""
from __future__ import annotations

import torch

from neurobrix.core.flow.audio_utils import component_input_axes, fit_features_to_trace

S0 = {"type": "symbol", "id": "s0", "trace": 1}
S1 = {"type": "symbol", "id": "s1", "trace": 700}
# The container's own encoding: the concrete trace shape in `shape`, the symbolic dims beside it.
GRANITE_DAG = {"tensors": {"input::hidden_states": {"type": "input", "shape": [1, 700, 160],
                                                     "symbolic_shape": {"dims": [S0, S1, 160]}}}}
WHISPER_DAG = {"tensors": {"input::input_features": {"type": "input", "shape": [1, 128, 3000],
                                                      "symbolic_shape": {"dims": [S0, 128, 3000]}}}}
# The older encoding, dicts inside `shape` itself, is still read.
LEGACY_DAG = {"tensors": {"input::x": {"type": "input", "shape": [S0, {"type": "symbol", "id": "s1", "trace_value": 700}, 160]}}}


def test_the_reader_tells_a_symbolic_axis_from_a_frozen_one():
    assert component_input_axes(GRANITE_DAG) == ((1, 700, 160), frozenset({0, 1}))
    assert component_input_axes(WHISPER_DAG) == ((1, 128, 3000), frozenset({0}))
    assert component_input_axes(LEGACY_DAG) == ((1, 700, 160), frozenset({0, 1}))
    assert component_input_axes({"tensors": {}}) is None and component_input_axes(None) is None


def test_a_symbolic_frame_axis_is_fed_at_the_clips_own_extent():
    shape, axes = component_input_axes(GRANITE_DAG)
    for frames in (400, 550, 750, 1100):
        clip = torch.randn(1, frames, 160)
        fed = fit_features_to_trace(clip, shape, axes)
        assert fed.shape == (1, frames, 160) and fed is clip


def test_a_frozen_axis_keeps_the_vendors_extent():
    shape, axes = component_input_axes(WHISPER_DAG)
    short = torch.ones(1, 128, 2000)
    fed = fit_features_to_trace(short, shape, axes)
    assert fed.shape == (1, 128, 3000) and float(fed[..., 2000:].abs().sum()) == 0.0
    long = torch.ones(1, 128, 3500)
    assert fit_features_to_trace(long, shape, axes).shape == (1, 128, 3000)


def test_without_symbolic_axes_the_old_fit_is_byte_identical():
    """The triton mirror and older call sites pass no axes: every non-batch axis is fitted."""
    clip = torch.arange(1 * 400 * 160, dtype=torch.float32).reshape(1, 400, 160)
    fed = fit_features_to_trace(clip, (1, 700, 160))
    assert fed.shape == (1, 700, 160) and torch.equal(fed[:, :400], clip) and float(fed[:, 400:].abs().sum()) == 0.0
    assert fit_features_to_trace(clip, None).shape == (1, 400, 160)
