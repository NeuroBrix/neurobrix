"""The tagged front end's unknown-word reader: the container's espeak-distilled word list read through misaki's E2M map.

misaki's English G2P hands a word its lexicon pass cannot read to `EspeakFallback` (misaki 0.9.4 espeak.py): espeak-ng's
phonemes of the text (tie '^' between the two halves of a diphthong or an affricate), mapped to misaki's phoneme set by
its E2M table, the syllabic mark rewritten, the variant's replacements, the tie removed, rating 2. The engine runs no
espeak: the container carries espeak-ng's phonemes of a word list (`modules/g2p/en_lexicon.txt.gz`, `word<TAB>ipa`, the
variant the container's `phoneme_lang` names) and every step of the map as data (front_end.json `e2m`, `e2m_variant`,
`final_replacements`). A text the list does not hold stays unknown (None): the pass then writes the container's `unk`,
as kokoro does without espeak.

The list may carry espeak's phonemes WITHOUT the tie (the list built before this front end did): read from the data
itself (no entry holds the tie). Then an E2M key that holds the tie also matches its untied spelling ("e^ɪ" -> "A" reads
"eɪ" -> "A"), since the tie is what the key needs; where espeak would not have tied the two characters (t+ʃ across a
syllable, schwa then l) the untied reading maps them as one — the list's loss, not the map's, gone once the list
carries the tie (the map then reads the tied keys first, longest first, as misaki does)."""
import gzip
import re
import unicodedata
from typing import Dict, List, Optional, Tuple


def read_word_list(path) -> Dict[str, str]:
    """`word<TAB>ipa` lines (gzip or plain text, '#' comment lines) -> {word: ipa}."""
    opener = gzip.open if str(path).endswith(".gz") else open
    words: Dict[str, str] = {}
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or "\t" not in line:
                continue
            w, ipa = line.rstrip("\n").split("\t", 1)
            if w:
                words[w] = ipa
    return words


def word_key(text: str) -> str:
    """The list's key of a text: NFC, the right single quote as an apostrophe, lower case (the list's own spelling)."""
    return unicodedata.normalize("NFC", text).replace("’", "'").lower()


class EspeakList:
    """text -> (phonemes, rating) from the word list, or (None, None) when the list does not hold it."""

    def __init__(self, words: Dict[str, str], tables, variant: str):
        e2m = [tuple(p) for p in tables.e2m]
        ev = tables.e2m_variant
        if variant not in ev:
            raise ValueError(f"front_end.json e2m_variant has no {variant!r} replacements (it has "
                             f"{sorted(k for k in ev if isinstance(ev[k], list))})")
        self.tie = ev["tie"]
        self.rating = ev["rating"]
        self.words = words
        self.tied = any(self.tie in ipa for ipa in words.values())
        if not self.tied:
            # each tied key also read untied; longest first, the tied key's own place kept among equals
            e2m = e2m + [(k.replace(self.tie, ""), v) for k, v in e2m if self.tie in k]
            e2m = sorted(e2m, key=lambda kv: -len(kv[0]))
        self.e2m: List[Tuple[str, str]] = e2m
        pattern, repl = ev["syllabic_regex"]
        self.syllabic = re.compile(pattern)
        self.syllabic_repl = repl
        # the mark the syllabic pattern rewrites, removed wherever the rewrite left it (misaki: .replace(chr(809), ''))
        self.syllabic_mark = self.syllabic.pattern.split(")", 1)[1].encode("ascii").decode("unicode_escape")
        self.variant_replacements = [tuple(p) for p in ev[variant]] + [tuple(p) for p in ev["all"]]
        if not self.tied:          # in place: these replacements run in their order
            self.variant_replacements = [q for k, v in self.variant_replacements
                                         for q in ([(k, v), (k.replace(self.tie, ""), v)] if self.tie in k else [(k, v)])]
        self.final_replacements = [tuple(p) for p in tables.final_replacements]

    def __call__(self, text: str) -> Tuple[Optional[str], Optional[int]]:
        ps = self.words.get(word_key(text))
        if ps is None:
            return None, None
        ps = ps.strip()
        for old, new in self.e2m:
            ps = ps.replace(old, new)
        ps = self.syllabic.sub(self.syllabic_repl, ps).replace(self.syllabic_mark, "")
        for old, new in self.variant_replacements:
            ps = ps.replace(old, new)
        for old, new in self.final_replacements:
            ps = ps.replace(old, new)
        return ps.replace(self.tie, ""), self.rating
