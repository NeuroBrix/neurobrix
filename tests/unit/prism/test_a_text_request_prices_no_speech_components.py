"""A request prices only the components it loads: an omni VLM's speech leg runs only on a speech request.

Qwen3-Omni-30B-A3B-Instruct, `--mode text` (an image and a prompt, 32 tokens) on the Mac's profile
(2026-10-04): layer_streaming declined "no room for a single segment: the usable 11305 MB of the
rung is filled by what stays resident beside the streamed ['thinker.model'] — whole components
11607 MB" — talker.model (6 345 MB), code2wav and the talker pieces were priced beside the thinker's
segments. Neither engine loads them on that request: weights load on demand, and the only callers of
the speech components are the speech legs, run only when the request `requests_speech`
(core/flow/vlm.py and triton/flow/vlm.py, the one predicate of core/flow/base.py). On f531e076 the
plan held, but cut the 57 GB thinker in 48 segments against 2 381 MB; with the speech leg left out,
6 segments against 9 568 MB (door 12288, triton, the Mac's profile).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from neurobrix.core.flow.base import requests_speech, resident_together, unloaded_by_request
from neurobrix.core.prism import InputConfig, PrismSolver
from neurobrix.core.prism.solver import ComponentMemory
from tests.unit.prism._pinned_machine import container_root

MB = 1024 * 1024

#: An omni VLM's flow with a generative-speech contract, shaped as Qwen3-Omni's topology declares it
#: (keys only).
FLOW = {"flow": {"type": "vlm",
                 "vlm": {"vision_component": "vision", "audio_component": "audio",
                         "lm_component": "lm", "head_component": "head"},
                 "speech": {"components": {"backbone": "talker", "predictor": "predictor",
                                           "vocoder": "vocoder"}}}}
#: The request's image: the repository's own asset (the Mac's request read apple_448.png).
IMAGE = Path(__file__).resolve().parents[3] / "benchmarks" / "assets" / "apple_448.png"
SPEECH = {"talker", "predictor", "vocoder"}
WEIGHTS = {"vision": 1027, "audio": 1239, "lm": 57048, "head": 594,
           "talker": 6035, "predictor": 210, "vocoder": 412}


def _comps():
    return [(n, ComponentMemory(n, w * MB, 10 * MB, 0)) for n, w in WEIGHTS.items()]


def _total(names):
    return sum(m.total_bytes for n, m in _comps() if n in names)


def _solver(monkeypatch, mode, served=False):
    s = PrismSolver()
    monkeypatch.setattr(s, "_flow_topology", lambda c: FLOW)
    monkeypatch.setattr(s, "_input_config", InputConfig(batch_size=1, mode=mode), raising=False)
    monkeypatch.setattr(s, "_serve_requested", served, raising=False)
    return s


def test_only_a_speech_request_runs_the_speech_leg():
    assert requests_speech("audio")
    assert not any(requests_speech(m) for m in ("text", "chat", "image", None, ""))


def test_a_text_request_never_loads_the_speech_components():
    assert unloaded_by_request(FLOW, "text") == SPEECH
    assert unloaded_by_request(FLOW, "audio") == set()


@pytest.mark.parametrize("mode,served", [(None, False), ("text", True)])
def test_an_unknown_mode_or_a_served_session_holds_every_leg(mode, served):
    """A served session may take a speech request later; a request whose mode is not known may be
    one. Both are planned with the leg."""
    assert unloaded_by_request(FLOW, mode, served) == set()


def test_a_flow_without_a_speech_contract_unloads_nothing():
    flow = {"flow": {"type": "vlm", "vlm": {"vision_component": "vision", "lm_component": "lm"}}}
    assert unloaded_by_request(flow, "text") == set()
    assert unloaded_by_request({"flow": {"type": "iterative_process"}}, "text") == set()


def test_a_streamed_lm_on_a_text_request_reserves_its_head_alone(monkeypatch):
    got = _solver(monkeypatch, "text")._resident_beside_streamed(None, _comps(), {"lm"})
    assert got == _total({"head"}), (got / MB, _total({"head"}) / MB)


def test_a_streamed_lm_on_a_speech_request_reserves_its_head_and_speech_leg(monkeypatch):
    got = _solver(monkeypatch, "audio")._resident_beside_streamed(None, _comps(), {"lm"})
    assert got == _total({"head"} | SPEECH), (got / MB, _total({"head"} | SPEECH) / MB)


def test_a_text_request_s_peak_holds_no_speech_component(monkeypatch):
    """The lifecycle peak a plan that loads on demand is held to: the decode phase without the leg —
    and the speech components are not loose either (a component no phase names counts everywhere)."""
    costs = {n: int(m.total_bytes) for n, m in _comps()}
    assert _solver(monkeypatch, "text")._phase_peak(None, costs) == _total({"lm", "head"})
    assert _solver(monkeypatch, "audio")._phase_peak(None, costs) == _total({"lm", "head"} | SPEECH)


def test_the_phases_still_hold_the_leg_for_a_speech_request():
    """The phases are the handlers' lifecycle; what a request never loads is said beside them."""
    assert {"lm", "head"} | SPEECH in resident_together(FLOW, "triton", served=False)


def test_the_run_plans_with_the_request_s_mode():
    """`run.request_input_config` — the InputConfig `cmd_run` plans with and the derived census binds
    from — carries the mode the flow will read, resolved by the same `resolve_mode`."""
    from neurobrix.cli import create_parser
    from neurobrix.cli.commands.run import request_input_config
    model = "Qwen3-Omni-30B-A3B-Instruct"
    root = container_root(model)
    man = json.loads((root / "manifest.json").read_text())
    for mode in ("text", "audio"):
        args = create_parser().parse_args(["run", "--model", model, "--prompt", "p", "--mode", mode,
                                           "--input-image", str(IMAGE), "--max-tokens", "32"])
        ic = request_input_config(args, man, man.get("family"), root)
        assert ic.mode == mode
        topo = json.loads((root / "topology.json").read_text())
        speech = set(topo["flow"]["speech"]["components"].values())
        assert unloaded_by_request(topo, ic.mode) == (speech if mode == "text" else set())
