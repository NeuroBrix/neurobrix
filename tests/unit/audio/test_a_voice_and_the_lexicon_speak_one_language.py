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
