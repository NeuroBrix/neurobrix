"""Text to speakable words before the g2p lexicon: numbers, years, decimals, ordinals, percents, currencies, units,
multipliers, symbols, brand words and acronyms written out, as a speech front-end does before it phonemizes
(espeak's text normalizer, Kokoro's own misaki front-end: a year read as a year, "3.14" as "three point one four").
Stdlib only: the g2p stays Zero Outsider.

Before (the Dell, 2026-11-12): the g2p tokenizer took a digit for punctuation, and Kokoro's phoneme vocabulary has no
digit, so "2017" was dropped from the speech with nothing said; "NeuroBrix" and "PyTorch" went to the letter-to-sound
fallback ("Nair Aubrix", "P-Torch"). A character still unspeakable after this pass is refused by name
(`unspeakable`), never dropped.
"""
from __future__ import annotations

import re
from typing import List

_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
         "seventeen eighteen nineteen").split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_SCALES = ((10 ** 12, "trillion"), (10 ** 9, "billion"), (10 ** 6, "million"), (1000, "thousand"))
_ORDINAL = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth", "nine": "ninth",
            "twelve": "twelfth"}

#: brand and product words the lexicon does not hold, respelled with words it does (each checked by the tests)
BRANDS = {"neurobrix": "new row bricks", "pytorch": "pie torch", "nbxtensor": "N B X tensor", "github": "git hub",
          "macos": "mac O S", "kokoro": "co co row", "huggingface": "hugging face", "nbx": "N B X"}
#: units written after a number, by their usual short form (case-sensitive: "MB" is not "mb")
UNITS = {"KB": "kilobytes", "MB": "megabytes", "GB": "gigabytes", "TB": "terabytes", "GHz": "gigahertz",
         "MHz": "megahertz", "ms": "milli seconds", "s": "seconds", "x": "times", "B": "billion", "M": "million",
         "K": "thousand", "W": "watts", "fps": "frames per second"}
_SINGULAR = {"kilobytes": "kilobyte", "megabytes": "megabyte", "gigabytes": "gigabyte", "terabytes": "terabyte",
             "seconds": "second", "milli seconds": "milli second", "watts": "watt"}
SYMBOLS = {"&": " and ", "+": " plus ", "=": " equals ", "@": " at ", "%": " percent ", "#": " number ",
           "/": " slash ", "*": " times ", "<": " less than ", ">": " more than ", "~": " about "}
_CURRENCY = {"$": ("dollar", "dollars", "cent", "cents"), "€": ("euro", "euros", "cent", "cents"),
             "£": ("pound", "pounds", "penny", "pence")}
#: what Kokoro speaks as a pause or an intonation: kept as written
PUNCTUATION = set(".,!?;:'\"()-—–…")


def cardinal(n: int) -> str:
    if n < 0:
        return "minus " + cardinal(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        return _ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + cardinal(n % 100))
    for value, name in _SCALES:
        if n >= value:
            return cardinal(n // value) + " " + name + ("" if n % value == 0 else " " + cardinal(n % value))
    raise AssertionError(n)


def ordinal(n: int) -> str:
    words = cardinal(n).split()
    last = words[-1]
    if last in _ORDINAL:
        words[-1] = _ORDINAL[last]
    elif last.endswith("y"):
        words[-1] = last[:-1] + "ieth"
    else:
        words[-1] = last + "th"
    return " ".join(words)


def year(n: int) -> str:
    """A year as it is said: 1999 nineteen ninety nine, 1900 nineteen hundred, 2000 two thousand, 2005 two thousand
    five, 2017 twenty seventeen."""
    if 2000 <= n <= 2009:
        return cardinal(n)
    hi, lo = divmod(n, 100)
    if lo == 0:
        return cardinal(hi) + " hundred"
    return cardinal(hi) + " " + (("oh " + _ONES[lo]) if lo < 10 else cardinal(lo))


def decimal(text: str) -> str:
    whole, _, frac = text.partition(".")
    return cardinal(int(whole or "0")) + " point " + " ".join(_ONES[int(d)] for d in frac)


def _number(text: str) -> str:
    """A written number (thousands commas allowed, one decimal point) as words."""
    text = text.replace(",", "")
    return decimal(text) if "." in text else cardinal(int(text))


def _plural(unit_words: str, value: str) -> str:
    return _SINGULAR.get(unit_words, unit_words) if value in ("1", "1.0") else unit_words


_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_RULES = [
    # a version: 0.6.13, 3.11.2 (each part said, "point" between)
    (re.compile(r"(?<![\d.])\d+(?:\.\d+){2,}(?![\d])"),
     lambda m: " point ".join(cardinal(int(p)) for p in m.group(0).split("."))),
    # currency: $5, $5.50, €12
    (re.compile(r"([$€£])(" + _NUM + r")"),
     lambda m: _money(m.group(1), m.group(2))),
    # ordinals: 1st 2nd 3rd 21st 100th
    (re.compile(r"\b(\d+)(st|nd|rd|th)\b", re.I), lambda m: ordinal(int(m.group(1)))),
    # percent: 50%, 2.5 %
    (re.compile(r"(" + _NUM + r")\s?%"), lambda m: _number(m.group(1)) + " percent"),
    # a number with its unit: 16GB, 8 GB, 5.7x, 1.1B, 30 ms
    (re.compile(r"(" + _NUM + r")\s?(" + "|".join(sorted(map(re.escape, UNITS), key=len, reverse=True)) +
                r")(?![A-Za-z])"),
     lambda m: _number(m.group(1)) + " " + _plural(UNITS[m.group(2)], m.group(1))),
    # a year: four digits from 1100 to 2099 standing alone
    (re.compile(r"(?<![\d.,])(1[1-9]\d\d|20\d\d)(?!\d|[.,]\d|\s?%)"), lambda m: year(int(m.group(1)))),
    # any other number
    (re.compile(r"-?(?:" + _NUM + r")"), lambda m: ("minus " if m.group(0).startswith("-") else "")
     + _number(m.group(0).lstrip("-"))),
]


def _money(sign: str, amount: str) -> str:
    one, many, sub, subs = _CURRENCY[sign]
    whole, _, frac = amount.replace(",", "").partition(".")
    words = cardinal(int(whole)) + " " + (one if whole == "1" else many)
    if frac and int(frac[:2].ljust(2, "0")):
        c = int(frac[:2].ljust(2, "0"))
        words += " and " + cardinal(c) + " " + (sub if c == 1 else subs)
    return words


def _acronym(m: re.Match) -> str:
    return " ".join(m.group(0))


def verbalize(text: str, lexicon=None) -> str:
    """The text with every number, symbol, brand word and unknown acronym written as speakable words. `lexicon` (the
    g2p's word -> IPA map), when given, decides which all-capital words are acronyms to spell (those it lacks)."""
    out = text
    for word, said in BRANDS.items():
        out = re.sub(rf"(?<![A-Za-z]){re.escape(word)}(?![A-Za-z])", said, out, flags=re.I)
    for pattern, say in _RULES:
        out = pattern.sub(lambda m, say=say: " " + say(m) + " ", out)
    for sym, said in SYMBOLS.items():
        out = out.replace(sym, said)
    if lexicon is not None:
        out = re.sub(r"\b[A-Z]{2,5}s?\b",
                     lambda m: m.group(0) if m.group(0).lower() in lexicon else _acronym(m), out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return re.sub(r"[ \t]+([.,!?;:])", r"\1", out).strip()                  # punctuation glued back to its word


def unspeakable(text: str) -> List[str]:
    """The characters of a verbalized text that are neither a letter, a space nor a punctuation Kokoro speaks."""
    return sorted({ch for ch in text if not (ch.isalpha() or ch.isspace() or ch in PUNCTUATION)})
