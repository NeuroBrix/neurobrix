"""A container that declares its voice is cloned from a reference clip (`runtime/defaults.json` voice
`{"reference_clip_required": true}`, builder 1237: s1-mini and VibeVoice-1.5B) is refused by name, and that
declaration is never read as a voice (supervisor 23/11 08:00, order B; qa 1237-3).

Before: three readers took `defaults["voice"]` as truthy — `voices_to_choose` returned no list to choose from,
`requested_voice` returned the dict's text as a voice name, `g2p.request_voice` returned the dict — and the two flows
read no voice at all, so the run spoke in a voice nobody declared.

What each test would do if the code were wrong: the dict read as a voice name or as a declared default (red on main
b2a296e6: `declared_voice` does not exist and the readers return the dict); a run of either flow without a clip not
refused (red); a value that is neither a name nor the declaration accepted (red)."""
from __future__ import annotations

import types

import pytest

from neurobrix.core.flow import audio_utils as AU
from neurobrix.core.module.audio import g2p

CLIP = {"voice": {"reference_clip_required": True}}


def test_the_three_declarations_are_told_apart():
    assert AU.declared_voice({"voice": "tara"}) == ("name", "tara")
    assert AU.declared_voice(CLIP) == ("clip_required", None)
    assert AU.declared_voice({}) == (None, None)


@pytest.mark.parametrize("value", [{"reference_clip_required": False}, {"clip": "x.wav"}, 3, ["tara"]])
def test_a_voice_value_the_engine_does_not_understand_is_refused_by_name(value):
    with pytest.raises(RuntimeError, match="neither a voice name"):
        AU.declared_voice({"voice": value})


def test_the_clip_declaration_is_never_a_voice_name(tmp_path):
    (tmp_path / "modules" / "voices").mkdir(parents=True)
    (tmp_path / "modules" / "voices" / "a.pt").write_bytes(b"")
    assert AU.voices_to_choose(tmp_path, CLIP) == ["a"]                  # no default: the request must choose
    assert g2p.request_voice({}, CLIP) is None
    engine = types.SimpleNamespace(ctx=types.SimpleNamespace(
        variable_resolver=types.SimpleNamespace(resolved={}), pkg=types.SimpleNamespace(defaults=CLIP)))
    with pytest.raises(RuntimeError, match="declares none as its default"):
        AU.requested_voice(engine, ["a"])


def test_a_cloned_voice_run_is_refused_by_name_and_a_named_voice_is_not():
    with pytest.raises(RuntimeError, match="reference_clip_required"):
        AU.refuse_without_reference_clip(CLIP)
    AU.refuse_without_reference_clip({"voice": "tara"})
    AU.refuse_without_reference_clip({})


@pytest.mark.parametrize("module", ["neurobrix.core.flow.dual_ar", "neurobrix.triton.flow.dual_ar",
                                    "neurobrix.core.flow.next_token_diffusion",
                                    "neurobrix.triton.flow.next_token_diffusion"])
def test_both_engines_flows_of_the_two_models_ask_before_anything_runs(module):
    import importlib
    import inspect
    src = inspect.getsource(importlib.import_module(module))
    head = src[src.index("    def execute(self)"):][:600]
    assert "refuse_without_reference_clip(self.ctx.pkg.defaults)" in head, module
