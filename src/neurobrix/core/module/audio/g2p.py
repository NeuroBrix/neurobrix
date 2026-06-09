"""NeuroBrix-internal g2p (grapheme → IPA) for Kokoro — stdlib only, no vendor.

ZO-3 "Zero Outsider": the Kokoro phoneme front-end no longer imports
``phonemizer`` / ``espeakng_loader`` / ``kokoro`` / ``misaki`` at inference.
Instead it loads an espeak-distilled pronunciation lexicon embedded in the .nbx
(``modules/g2p/en_lexicon.txt.gz``) and runs a tiny deterministic runner here.

This mirrors the tokenizer doctrine: vendor data is captured at build time and
embedded in the container; the runtime is a self-contained reader. The embedded
IPA is produced by espeak-ng (GPL-3.0) and retains that license — the .nbx is a
transport container (no relicensing), exactly like the embedded tokenizer vocab.

Imports: json, gzip, re, unicodedata, pathlib — all stdlib. NO torch, NO numpy,
NO vendor g2p. R33-safe (importable from the triton path) and used by the
compiled path too (single source of truth, both modes).

Fidelity vs the live espeak path
--------------------------------
* Dictionary words are byte-identical to espeak's per-word output (the lexicon
  is distilled from espeak in isolation, so lookup == espeak-in-isolation).
* The runner joins per-word IPA with single spaces and glues trailing/leading
  punctuation to the adjacent word with no space, reproducing espeak's word
  separation. This matches espeak's SENTENCE output for ordinary text.
* Sentence-level context features that espeak applies (number expansion,
  abbreviation expansion like "Mr." -> "mister", a/the vowel-reduction by
  following context, hyphen word-fusion) are NOT reproduced — those need the
  espeak text normalizer. They are a measured, bounded divergence (see the ZO-3
  parity report); Kokoro remains intelligible because the per-word phonemes are
  correct. OOV words use a deterministic letter-to-sound fallback whose output
  is filtered to the Kokoro phoneme vocab.
"""
import gzip
import re
import unicodedata
from pathlib import Path
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Deterministic letter-to-sound fallback for OOV words (English).
# Intentionally small: the task does not require correct OOV phonemes, only a
# deterministic, phoneme_vocab-mappable IPA. Digraphs are matched before single
# letters. Every emitted symbol is in the Kokoro 'a' phoneme vocab; anything the
# runtime can't map is dropped by the caller's char->id loop anyway.
# ---------------------------------------------------------------------------
_LTS_DIGRAPHS = [
    ("sch", "sk"),
    ("tch", "ʧ"),
    ("dge", "ʤ"),
    ("igh", "aɪ"),
    ("ough", "ʌf"),
    ("augh", "ɔː"),
    ("sh", "ʃ"),
    ("ch", "ʧ"),
    ("th", "θ"),
    ("ph", "f"),
    ("wh", "w"),
    ("ck", "k"),
    ("ng", "ŋ"),
    ("qu", "kw"),
    ("oo", "uː"),
    ("ee", "iː"),
    ("ea", "iː"),
    ("ou", "aʊ"),
    ("ow", "aʊ"),
    ("ai", "eɪ"),
    ("ay", "eɪ"),
    ("oa", "oʊ"),
    ("oy", "ɔɪ"),
    ("oi", "ɔɪ"),
    ("ar", "ɑːɹ"),
    ("or", "ɔːɹ"),
    ("er", "ɚ"),
    ("ir", "ɜː"),
    ("ur", "ɜː"),
]
_LTS_SINGLE = {
    "a": "æ", "b": "b", "c": "k", "d": "d", "e": "ɛ", "f": "f", "g": "ɡ",
    "h": "h", "i": "ɪ", "j": "ʤ", "k": "k", "l": "l", "m": "m", "n": "n",
    "o": "ɑː", "p": "p", "q": "k", "r": "ɹ", "s": "s", "t": "t", "u": "ʌ",
    "v": "v", "w": "w", "x": "ks", "y": "j", "z": "z",
}

# token split: a run of letters/apostrophes/hyphens is a "word"; everything else
# (punctuation, whitespace) is captured so we can reproduce espeak's join. The
# apostrophe set covers ASCII ' and the unicode right single quote U+2019.
_APOS = "'’"
_WORD_RE = re.compile(r"[a-z%s][a-z%s\-]*" % (_APOS, _APOS), re.IGNORECASE)
_TOKEN_RE = re.compile(r"([a-z%s][a-z%s\-]*)|(\s+)|([^\sa-z%s])"
                       % (_APOS, _APOS, _APOS), re.IGNORECASE)


def _normalize_word(w: str) -> str:
    """NFC + lowercase + ASCII-apostrophe so contractions/possessives match the
    lexicon keys (which were stored from NFC espeak input)."""
    w = unicodedata.normalize("NFC", w)
    w = w.replace("’", "'")
    return w.lower()


class G2P:
    """Embedded-lexicon g2p runner. Stdlib only."""

    def __init__(self, lexicon: Dict[str, str]):
        self._lex = lexicon

    # -- loading ------------------------------------------------------------
    @classmethod
    def from_path(cls, path) -> "G2P":
        """Load the embedded lexicon from a .txt or .txt.gz path (word\\tipa)."""
        p = Path(path)
        lex: Dict[str, str] = {}
        opener = gzip.open if p.suffix == ".gz" else open
        with opener(p, "rt", encoding="utf-8") as f:
            for line in f:
                if not line or line.startswith("#"):
                    continue
                line = line.rstrip("\n")
                if "\t" not in line:
                    continue
                w, ipa = line.split("\t", 1)
                if w:
                    lex[w] = ipa
        return cls(lex)

    @classmethod
    def from_nbx(cls, nbx_path_str: str) -> Optional["G2P"]:
        """Load from a Kokoro .nbx: modules/g2p/en_lexicon.txt(.gz). Returns None
        if no g2p module is embedded (lets the caller raise an actionable error)."""
        base = Path(nbx_path_str) / "modules" / "g2p"
        for name in ("en_lexicon.txt.gz", "en_lexicon.txt"):
            cand = base / name
            if cand.exists():
                return cls.from_path(cand)
        return None

    # -- conversion ---------------------------------------------------------
    def _lts(self, word: str) -> str:
        """Deterministic OOV fallback. Output only Kokoro-vocab IPA symbols."""
        s = word.replace("'", "").replace("-", "")
        out = []
        i = 0
        while i < len(s):
            matched = False
            for g, ph in _LTS_DIGRAPHS:
                if s.startswith(g, i):
                    out.append(ph)
                    i += len(g)
                    matched = True
                    break
            if matched:
                continue
            ch = s[i]
            out.append(_LTS_SINGLE.get(ch, ""))
            i += 1
        return "".join(out)

    def _inflect(self, key: str) -> Optional[str]:
        """Regular-inflection fallback: the misaki word inventory is missing many
        inflected forms (e.g. 'makes' OOV though 'make' is present — ~58% of
        stems lack their '+s' form). When a word is a regular inflection of a
        stem that IS in the lexicon, append the inflection phone with English
        voicing assimilation. Deterministic, intra-word morphology (NOT
        sentence prosody). Returns None if no rule applies."""
        # -s / -es plural / 3sg: voicing from the stem's final phone.
        if key.endswith("es") and len(key) > 3:
            stem = self._lex.get(key[:-2])
            if stem is not None:
                # sibilant-final stem -> 'ɪz', else handled by the -s branch.
                if stem.rstrip("ˈˌ")[-1:] in ("s", "z", "ʃ", "ʒ", "ʧ", "ʤ"):
                    return stem + "ᵻz"
        if key.endswith("s") and len(key) > 2:
            stem = self._lex.get(key[:-1])
            if stem is not None:
                last = stem.rstrip("ˈˌ")[-1:]
                if last in ("s", "z", "ʃ", "ʒ", "ʧ", "ʤ"):
                    return stem + "ᵻz"
                # voiceless obstruents -> 's', everything voiced -> 'z'.
                return stem + ("s" if last in ("p", "t", "k", "f", "θ") else "z")
        # -ed past: 'ɪd' after t/d, 't' after voiceless, 'd' otherwise.
        if key.endswith("ed") and len(key) > 3:
            stem = self._lex.get(key[:-2]) or self._lex.get(key[:-1])
            if stem is not None:
                last = stem.rstrip("ˈˌ")[-1:]
                if last in ("t", "d"):
                    return stem + "ᵻd"
                return stem + ("t" if last in ("p", "k", "f", "θ", "s",
                                               "ʃ", "ʧ") else "d")
        # -ing: append 'ɪŋ' to the stem (drop a trailing 'e' if needed).
        if key.endswith("ing") and len(key) > 4:
            stem = (self._lex.get(key[:-3]) or self._lex.get(key[:-3] + "e"))
            if stem is not None:
                return stem + "ɪŋ"
        return None

    def _word_ipa(self, word: str) -> str:
        key = _normalize_word(word)
        ipa = self._lex.get(key)
        if ipa is not None:
            return ipa
        # hyphenated compound not in the lexicon: phonemize each part, glue with
        # no space (mirrors espeak fusing "well-known" -> one token).
        if "-" in key:
            parts = [p for p in key.split("-") if p]
            if parts:
                return "".join(self._word_ipa(p) for p in parts)
        # possessive 'X's -> stem IPA + plural-style 's'/'z'/'ɪz'.
        if key.endswith("'s") and len(key) > 2:
            infl = self._inflect(key[:-2] + "s")
            if infl is not None:
                return infl
        infl = self._inflect(key)
        if infl is not None:
            return infl
        return self._lts(key)

    def convert(self, text: str) -> str:
        """text -> IPA string, reproducing espeak's word separation:
        words joined with single spaces, punctuation glued to the adjacent word
        with no surrounding space (e.g. 'fox.' -> 'fˈɑːks.')."""
        pieces = []  # list of (kind, ipa) where kind in {"word","punct","space"}
        for m in _TOKEN_RE.finditer(text):
            word, ws, punct = m.group(1), m.group(2), m.group(3)
            if word is not None:
                pieces.append(("word", self._word_ipa(word)))
            elif ws is not None:
                pieces.append(("space", " "))
            else:
                pieces.append(("punct", punct))

        # Assemble: a single space between consecutive "content" tokens (words
        # or punctuation that follow whitespace); punctuation that immediately
        # follows a word with no whitespace is glued (no space).
        out = []
        prev_kind = None
        for kind, val in pieces:
            if kind == "space":
                prev_kind = "space"
                continue
            if not val:
                # empty word ipa (shouldn't happen) — skip but keep separator state
                prev_kind = kind
                continue
            if out:
                if kind == "punct" and prev_kind == "word":
                    pass  # glue trailing punctuation to the word: no space
                else:
                    out.append(" ")
            out.append(val)
            prev_kind = kind
        return "".join(out).strip()


# ---------------------------------------------------------------------------
# drop-in replacement for the old _g2p_phonemes(prompt, lang, kokoro_lang).
# A module-level cache keyed by lexicon path keeps repeated calls cheap.
# ---------------------------------------------------------------------------
_RUNNER_CACHE: Dict[str, G2P] = {}


def load_runner(nbx_path_str: str) -> G2P:
    """Load (and cache) the G2P runner for a given .nbx path. Raises a clean,
    actionable error if the lexicon module is missing from the container."""
    key = str(nbx_path_str)
    runner = _RUNNER_CACHE.get(key)
    if runner is None:
        runner = G2P.from_nbx(nbx_path_str)
        if runner is None:
            raise RuntimeError(
                "ZERO-OUTSIDER g2p: no embedded lexicon found at "
                f"{Path(nbx_path_str) / 'modules/g2p/en_lexicon.txt(.gz)'}. "
                "Rebuild the Kokoro .nbx with the g2p module embedded "
                "(forge build re-runs the ZO-3 embed step).")
        _RUNNER_CACHE[key] = runner
    return runner


def g2p_phonemes(prompt: str, nbx_path_str: str,
                 lang: str = "en-us", kokoro_lang: str = "a") -> str:
    """Embedded-lexicon mirror of the old _g2p_phonemes(): text -> IPA string.
    `lang`/`kokoro_lang` are accepted for signature compatibility; the embedded
    lexicon is American-English (Kokoro phoneme_lang 'a')."""
    return load_runner(nbx_path_str).convert(prompt)
