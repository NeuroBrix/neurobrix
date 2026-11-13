"""The text-to-speech front end writes out what the g2p lexicon cannot read: numbers, years, decimals, ordinals,
percents, currencies, units, multipliers, symbols, brand words and acronyms (core/module/audio/text_normalize.py), and
refuses by name a character it still cannot say.

Before (the Dell, 2026-11-12): the g2p tokenizer took a digit for punctuation and Kokoro's phoneme vocabulary has no
digit, so "2017" vanished from the speech with nothing said; "NeuroBrix" came out "Nair Aubrix", "PyTorch" "P-Torch"
(the letter-to-sound fallback).

What this test does if the code is wrong: a case below is said otherwise than written, a number disappears, a brand
word or an acronym falls to the letter-to-sound guess, or an unspeakable character passes silently — red.

Run: python -m pytest tests/unit/audio/test_the_voice_says_every_number_and_name_it_is_given.py
"""
import gzip
from pathlib import Path

import pytest

from neurobrix.core.module.audio.text_normalize import cardinal, unspeakable, verbalize, year

CASES = [
    ("Founded in 2017.", "Founded in twenty seventeen."),
    ("In 1999 and 1900 and 2000 and 2005 and 2010.",
     "In nineteen ninety nine and nineteen hundred and two thousand and two thousand five and twenty ten."),
    ("It is 3.14 wide.", "It is three point one four wide."),
    ("1,519 tokens", "one thousand five hundred nineteen tokens"),
    ("the 21st and 2nd and 3rd and 12th", "the twenty first and second and third and twelfth"),
    ("50% off", "fifty percent off"),
    ("$5.50 and $1", "five dollars and fifty cents and one dollar"),
    ("16GB and 8 GB and 1 GB", "sixteen gigabytes and eight gigabytes and one gigabyte"),
    ("5.7x faster", "five point seven times faster"),
    ("46.2 s to 8.1 s", "forty six point two seconds to eight point one seconds"),
    ("a 1.1B model", "a one point one billion model"),
    ("-3 degrees", "minus three degrees"),
    ("NeuroBrix runs PyTorch models", "new row bricks runs pie torch models"),
    ("cats & dogs", "cats and dogs"),
    ("version 0.6.13 is out", "version zero point six point thirteen is out"),
]


@pytest.mark.parametrize("text,said", CASES)
def test_each_case_is_said_as_it_is_written(text, said):
    assert verbalize(text) == said


def test_the_number_words_hold_at_their_edges():
    assert cardinal(0) == "zero" and cardinal(100) == "one hundred" and cardinal(1001) == "one thousand one"
    assert cardinal(2_000_000_017) == "two billion seventeen"
    assert year(1905) == "nineteen oh five" and year(2009) == "two thousand nine"


def test_an_acronym_the_lexicon_lacks_is_spelled_and_a_word_it_holds_is_not():
    lexicon = {"ai": "x", "the": "x", "runs": "x", "on": "x"}
    assert verbalize("AI runs on the GPU", lexicon) == "AI runs on the G P U"


def test_a_character_the_voice_cannot_say_is_named_never_dropped():
    assert unspeakable(verbalize("price: 5 ¥ or 3 ♥")) == ["¥", "♥"]
    assert unspeakable(verbalize("Founded in 2017, 50% faster.")) == []


def _lexicon():
    """Kokoro's own lexicon, read from the engine's configured model cache (`neurobrix.core.paths`), never a machine's
    path; skipped by name when the container is not installed here."""
    from neurobrix.core.paths import cache_dir
    for path in (cache_dir() / "Kokoro-82M" / "modules" / "g2p" / "en_lexicon.txt.gz",):
        if path.exists():
            rows = gzip.open(path, "rt").read().splitlines()
            return {k: v for k, v in (r.split("\t", 1) for r in rows if "\t" in r and not r.startswith("#"))}
    pytest.skip("no Kokoro lexicon installed here")


def test_every_word_written_out_is_read_by_the_lexicon():
    """No case falls to the letter-to-sound guess: every word the front end writes is in Kokoro's own lexicon, or a
    regular inflection of a word it holds (the g2p's own rule), or a single letter of an acronym (which it holds)."""
    from neurobrix.core.module.audio.g2p import G2P
    lex = _lexicon()
    g = G2P(lex)
    for text, _ in CASES:
        for word in verbalize(text, lex).replace(".", " ").replace(",", " ").split():
            key = word.lower()
            assert key in lex or g._inflect(key) is not None, \
                f"{word!r} (from {text!r}) is not read by the lexicon: the letter-to-sound guess"


def test_a_year_reaches_the_phonemes():
    from neurobrix.core.module.audio.g2p import G2P
    lex = _lexicon()
    ipa = G2P(lex).convert(verbalize("Founded in 2017.", lex))
    assert lex["twenty"] in ipa and lex["seventeen"] in ipa
