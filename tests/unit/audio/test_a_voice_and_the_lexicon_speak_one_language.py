"""A TTS request whose VOICE speaks a language the container's embedded phoneme
lexicon does not is refused by name, before a phoneme is produced.

Measured 2026-09-16 (vitrine, Kokoro-82M): the French voice `ff_siwis` with the
American-English lexicon synthesised 4.97 s of fluent French-sounding speech in
which faster-whisper heard none of the sentence asked for — word error rate
1.42. No instrument inside the engine could see it: level, duration and silence
were all normal, and a byte gate between the two arms would have read
IDENTICAL, both arms saying the same wrong words.

Injection: `refusal_for_language` returning None for every pair → the first test
read no refusal — RED."""
import pytest

from neurobrix.core.module.audio.g2p import (VOICE_LANGUAGE, language_of,
                                             refusal_for_language)


def test_a_french_voice_on_an_english_lexicon_is_refused_by_name():
    msg = refusal_for_language("ff_siwis", "a")
    assert msg and "ff_siwis" in msg and "French" in msg and "American English" in msg
    assert "--speaker" in msg and "rebuild" in msg          # both ways out are named


def test_the_two_english_accents_are_one_language():
    assert refusal_for_language("bf_emma", "a") is None
    assert refusal_for_language("af_heart", "b") is None
    assert refusal_for_language("af_heart", "a") is None


def test_every_other_language_letter_is_refused_and_named():
    for letter, name in VOICE_LANGUAGE.items():
        msg = refusal_for_language(f"{letter}f_x", "a")
        if letter in ("a", "b"):
            assert msg is None, letter
        else:
            assert msg and name in msg, letter


def test_an_absent_voice_gates_nothing_and_an_unknown_letter_still_refuses():
    assert refusal_for_language(None, "a") is None and refusal_for_language("", "a") is None
    assert "does not" in (refusal_for_language("qq_x", "a") or "")
    assert language_of("f") == "French" and language_of("") is None


def test_both_engines_ask_the_gate_before_phonemising(monkeypatch):
    """R30: the gate exists on the compiled path AND on the Triton mirror. It
    was written on the compiled path alone, and the vitrine's Triton run walked
    past it (2026-09-16 17:47, `[Phonemizer·np]`) — a gate in one mode is no
    gate. Both engines now phonemise through ONE function, `phoneme_ids`, and
    that function refuses before it phonemises: the sources are read for the
    one call, the function is run with a g2p that fails if reached."""
    from pathlib import Path
    from neurobrix.core.module.audio import g2p as G
    root = Path(__file__).resolve().parents[3] / "src" / "neurobrix"
    for rel in ("core/flow/stages/kokoro.py", "triton/audio_frontend.py"):
        src = (root / rel).read_text()
        assert "phoneme_ids(" in src, f"{rel} does not phonemise through phoneme_ids"
        assert "g2p_phonemes(" not in src, f"{rel} phonemises around the gate"

    def reached(*_a, **_k):
        raise AssertionError("phonemised before the language gate")
    monkeypatch.setattr(G, "g2p_phonemes", reached)
    with pytest.raises(RuntimeError, match="ff_siwis"):
        G.phoneme_ids("Bonjour", "/nowhere", {"b": 1}, "a", "ff_siwis")


def test_the_ids_are_framed_and_keep_only_known_phonemes(monkeypatch):
    from neurobrix.core.module.audio import g2p as G
    monkeypatch.setattr(G, "g2p_phonemes", lambda *a, **k: "h?ə")
    phonemes, ids = G.phoneme_ids("x", "/nowhere", {"h": 5, "ə": 7}, "a", "af_heart")
    assert phonemes == "h?ə" and ids == [0, 5, 7, 0]


def test_the_plan_binds_a_phonemizer_request_at_its_phoneme_count(monkeypatch, tmp_path):
    """The one InputConfig a run plans with (`run.request_input_config`) carries the
    phonemizer request's length — len(phoneme_ids) — so every `seq_len` symbol of the text
    components binds to the length the flow will run, not the trace's. Measured 2026-09-28:
    Kokoro planned 23 phonemes and ran 94, and the derived census, binding from this same
    config, derived the trace's keys. Injection: the seq_len line removed -> None, RED."""
    import json
    from neurobrix.cli import create_parser
    from neurobrix.cli.commands.run import request_input_config
    from neurobrix.core.module.audio import g2p as G
    monkeypatch.setattr(G, "g2p_phonemes", lambda *a, **k: "abcab")
    (tmp_path / "runtime").mkdir()
    (tmp_path / "runtime" / "defaults.json").write_text(json.dumps(
        {"phoneme_vocab": {"a": 1, "b": 2, "c": 3}, "phoneme_lang": "a", "dtype": "float32"}))
    manifest = {"family": "tts", "dtype": "float32", "modules": {"g2p": {}, "voices": {}}}
    args = create_parser().parse_args(["run", "--model", "x", "--prompt", "hello"])
    ic = request_input_config(args, manifest, "tts", tmp_path)
    assert ic.seq_len == 7                          # [0] + 5 phonemes + [0]
    manifest["modules"]["tokenizer"] = {}           # a tokenizer container: not this path
    assert request_input_config(args, manifest, "tts", tmp_path).seq_len is None
