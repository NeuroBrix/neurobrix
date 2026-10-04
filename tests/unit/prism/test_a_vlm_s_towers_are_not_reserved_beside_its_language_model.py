"""A VLM's towers are not reserved beside its language model: the vlm flow declares its phases.

The vlm flow declared none, so Prism held every component concurrent with every other — beside a
streamed language model's segments (`_resident_beside_streamed`) and in the KV check of a plan that
loads on demand (`_phase_peak`). Both engines' handlers (core/flow/vlm.py, triton/flow/vlm.py, the
legacy, staged and M-RoPE masked splices) run each modality tower and projection once, before the
language model, and unload it right after unless the session is persistent (an EAGER served plan
only, which reads no phases); the LM and its head then decode together, and the speech leg loads
its talker groups beside the still-loaded LM (CFM) or right after it (deepstack). MiniCPM-o-4_5
reserved its 1 872 MB of towers beside its streamed LM (the Mac, 2026-10-04; the owner's 04:03 rule
puts the lifecycle in the check). `core/flow/base.py _vlm_phases` reads the phases from the
topology: each tower and projection alone, then {LM, head, every speech component}.

SEEN RED (2026-10-04) on three injections into `_vlm_phases` / RESIDENT_PHASES:
  * the "vlm" registration removed (every component concurrent again);
  * the towers put back into the decode phase;
  * the speech components left out of the decode phase (an under-count: the CFM leg loads them
    beside the LM).

Run: CUDA_VISIBLE_DEVICES= PYTHONPATH=src:. python -m pytest -q \
     tests/unit/prism/test_a_vlm_s_towers_are_not_reserved_beside_its_language_model.py
"""
from __future__ import annotations

import pytest

from neurobrix.core.flow.base import resident_together
from neurobrix.core.prism import InputConfig, PrismSolver
from neurobrix.core.prism.solver import ComponentMemory
from neurobrix.nbx import NBXContainer
from tests.unit.prism._pinned_machine import APPLE_M4_PRO, container_root, pin_host, profile

MB = 1024 * 1024

#: A staged omni VLM's flow, shaped as MiniCPM-o-4_5's topology declares it (keys only).
FLOW = {"flow": {"type": "vlm",
                 "vlm": {"vision_component": "vision", "vision_projection_component": "resampler",
                         "audio_component": "audio", "audio_projection_component": "audio_proj",
                         "lm_component": "lm", "head_component": "head"},
                 "speech": {"components": {"backbone": "talker", "vocoder": "vocoder"},
                            "condition_type": "hidden_text_merge"}}}
WEIGHTS = {"vision": 800, "resampler": 300, "audio": 600, "audio_proj": 40,
           "lm": 13000, "head": 1200, "talker": 380, "vocoder": 90}


def _comps():
    return [(n, ComponentMemory(n, w * MB, 10 * MB, 0)) for n, w in WEIGHTS.items()]


def _total(names):
    return sum(m.total_bytes for n, m in _comps() if n in names)


@pytest.mark.parametrize("engine", ["compiled", "triton"])
@pytest.mark.parametrize("served", [False, True])
def test_each_tower_runs_alone_and_the_lm_decodes_with_its_head_and_speech_leg(engine, served):
    phases = resident_together(FLOW, engine, served)
    assert sorted(map(sorted, phases)) == sorted(
        [["vision"], ["resampler"], ["audio"], ["audio_proj"], ["head", "lm", "talker", "vocoder"]]), phases


def test_a_flow_that_names_no_language_model_declares_no_phases():
    flow = {"flow": {"type": "vlm", "vlm": {"vision_component": "vision"}}}
    assert resident_together(flow, "triton") is None


def test_a_streamed_lm_reserves_its_head_and_speech_leg_not_the_towers(monkeypatch):
    s = PrismSolver()
    monkeypatch.setattr(s, "_flow_topology", lambda c: FLOW)
    got = s._resident_beside_streamed(None, _comps(), {"lm"})
    assert got == _total({"head", "talker", "vocoder"}), (got / MB, _total({"head", "talker", "vocoder"}) / MB)


def test_a_streamed_tower_reserves_nothing(monkeypatch):
    s = PrismSolver()
    monkeypatch.setattr(s, "_flow_topology", lambda c: FLOW)
    assert s._resident_beside_streamed(None, _comps(), {"vision"}) == 0


def test_minicpm_streamed_lm_is_cut_beside_its_decode_phase_only(monkeypatch):
    """The container: MiniCPM-o-4_5 on the Mac's profile, its LM streamed — what is reserved beside
    its segments is its head and its speech leg, never the vision and audio towers."""
    pin_host(monkeypatch, 24576, 18186, "the Mac, idle")
    monkeypatch.delenv("NBX_FORCE_STRATEGY", raising=False)
    monkeypatch.delenv("NBX_PRISM_BUDGET_MB", raising=False)
    s = PrismSolver()
    seen = {}
    real = s._compute_memory

    def spy(*a, **k):
        out = real(*a, **k)
        seen.update(out)
        return out

    s._compute_memory = spy
    c = NBXContainer.load(str(container_root("MiniCPM-o-4_5")))
    s.solve_smart(c, profile(APPLE_M4_PRO), InputConfig(batch_size=1), mode="triton")
    flow = s._flow_topology(c)["flow"]
    vlm = flow["vlm"]
    beside = {vlm["head_component"]} | set(flow["speech"]["components"].values())
    got = s._resident_beside_streamed(c, sorted(seen.items()), {vlm["lm_component"]})
    want = sum(seen[n].total_bytes for n in beside)
    every = sum(m.total_bytes for n, m in seen.items() if n != vlm["lm_component"])
    assert got == want < every, (got / MB, want / MB, every / MB)
