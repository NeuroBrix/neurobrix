"""A request prices only the modality towers it runs: a VLM's vision tower runs only on a request with an image or a
video, its audio tower only on a request with audio.

Devstral-Small-2-24B-Instruct-2512 (a vlm container: a Pixtral-style vision tower, its projector, a 24B language
model), a text-only coding request on the Dell's four-card pool (2026-11-17): the plan walked the vision tower's
activations and refused — "MissingRuntimeValue: the plan's request names no height for symbol s1 (height)" — though
neither engine's VLM handler loads a tower on a request without its modality (core/flow/vlm.py, triton/flow/vlm.py:
`_ensure_weights_loaded(vision_name)` under `has_visual`; the text-only form, `VLMSteps._text_only_request`).
"""
from __future__ import annotations

import inspect

from neurobrix.core.flow.base import TOWER_INPUTS, unloaded_by_request
from neurobrix.pytorch.prism import InputConfig, PrismSolver

FLOW = {"flow": {"type": "vlm",
                 "vlm": {"vision_component": "vision", "vision_projection_component": "projector",
                         "audio_component": "audio", "audio_projection_component": "audio_proj",
                         "lm_component": "lm", "head_component": "head"}}}
VISION = {"vision", "projector"}
AUDIO = {"audio", "audio_proj"}


def test_a_text_request_never_loads_a_modality_tower():
    assert unloaded_by_request(FLOW, "text", inputs=frozenset()) == VISION | AUDIO


def test_an_image_or_a_video_runs_the_vision_tower_alone():
    assert unloaded_by_request(FLOW, "text", inputs=frozenset({"image"})) == AUDIO
    assert unloaded_by_request(FLOW, "text", inputs=frozenset({"video"})) == AUDIO
    assert unloaded_by_request(FLOW, "text", inputs=frozenset({"audio"})) == VISION


def test_inputs_not_known_or_a_served_session_price_every_tower():
    assert unloaded_by_request(FLOW, "text") == set()
    assert unloaded_by_request(FLOW, None) == set()
    assert unloaded_by_request(FLOW, "text", served=True, inputs=frozenset()) == set()


def test_a_tower_the_decode_reads_stays():
    flow = {"flow": {"type": "vlm", "vlm": {"vision_component": "lm", "lm_component": "lm"}}}
    assert unloaded_by_request(flow, "text", inputs=frozenset()) == set()


def test_every_tower_key_is_one_the_flow_phases_name():
    from neurobrix.core.flow import base
    assert set(TOWER_INPUTS) <= set(inspect.getsource(base._vlm_phases).split('"'))


def test_the_plan_reads_the_request_inputs(monkeypatch):
    s = PrismSolver()
    monkeypatch.setattr(s, "_flow_topology", lambda c: FLOW)
    monkeypatch.setattr(s, "_input_config", InputConfig(batch_size=1, mode="text", inputs=frozenset()), raising=False)
    monkeypatch.setattr(s, "_serve_requested", False, raising=False)
    assert s._unloaded_by_request(object()) == VISION | AUDIO


def test_the_activation_walk_skips_an_unloaded_component():
    from neurobrix.pytorch.prism import solver
    src = inspect.getsource(solver.PrismSolver._compute_memory)
    assert "comp.name not in unloaded" in src


def test_the_run_states_the_request_inputs():
    from neurobrix.cli.commands import run
    src = inspect.getsource(run)
    assert '("image", "input_image"), ("video", "input_video"), ("audio", "audio")' in src


def test_an_unloaded_tower_hands_no_output_to_the_later_phases(monkeypatch):
    # the phase check keeps a tower's outputs until the language model's phase; a tower that never runs has none, and
    # its output walk would ask the request for the image size it does not carry
    s = PrismSolver()
    monkeypatch.setattr(s, "_unloaded_by_request", lambda c: {"vision"}, raising=False)
    monkeypatch.setattr(s, "_flow_phases", lambda c: [{"vision"}, {"lm"}], raising=False)

    def output_bytes(name):
        assert name != "vision", "the unloaded tower's outputs were walked"
        return 0
    monkeypatch.setattr(s, "_component_output_bytes", output_bytes, raising=False)
    mb = 1024 * 1024
    assert s._phase_peak_carried_mb(object(), {"vision": 7 * mb, "lm": 5 * mb}) == 5.0


def test_no_pass_of_the_plan_follows_an_unloaded_component():
    # the request-size scaler of the memory pass and the work table of the time estimate skip it too
    from neurobrix.pytorch.prism import solver
    mem = inspect.getsource(solver.PrismSolver._compute_memory)
    assert "activation_bytes, activation_profiled = (0, True) if comp.name in unloaded else (" in \
        mem.split("self._scale_activations_to_request(")[0][-400:]
    assert "if c.name not in unloaded):" in inspect.getsource(solver.PrismSolver._estimate_times)
    assert "comp.name in unloaded" in inspect.getsource(solver.PrismSolver._detect_op_level_tiling_pairs)
