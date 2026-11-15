"""Words found in the word list: the entry that applies, its stress, its phonemes, its written form.

Written from the data files, their documentation and observed outputs only.
"""
import os

from . import english
from .data_files import DataFileError, parse_voice, parse_word_list, read_phoneme_source
from .phonemes import Ph, PhonemeTable, Unreadable

def _read(path):
    if not os.path.isfile(path):
        raise DataFileError("the data file %s is absent: fetch the pinned data of the processor" % path)
    with open(path, encoding="utf-8") as f:
        return f.read()


def find_voice(data_dir, language):
    """The voice file whose first `language` line names this language."""
    root = os.path.join(data_dir, english.VOICE_FOLDER)
    for folder, _, names in sorted(os.walk(root)):
        for name in sorted(names):
            voice = parse_voice(_read(os.path.join(folder, name)), name)
            lines = voice.get("language", [])
            if any(not v for v in lines):
                raise DataFileError("the voice file %s has a `language` line with no value: the voice file must name "
                                    "its language" % os.path.join(folder, name))
            if lines and lines[0][0] == language:
                return voice
    raise DataFileError("no voice file under %s declares the language %s" % (root, language))


class Reader(object):
    """Reads a text of one variant into its unit sequence. This lot reads a single word found in the word list."""

    def __init__(self, data_dir, variant, tie=english.DEFAULT_TIE):
        if variant not in english.VARIANT_LANGUAGE:
            raise ValueError('this reader does not read "%s"; it reads: %s' % (
                variant, ", ".join(sorted(english.VARIANT_LANGUAGE))))
        self.tie = tie
        voice = find_voice(data_dir, english.VARIANT_LANGUAGE[variant])
        if "phonemes" in voice and not voice["phonemes"][0]:
            raise DataFileError("the voice of %s has a `phonemes` line with no value: it must name its phoneme table"
                                % english.VARIANT_LANGUAGE[variant])
        table = voice["phonemes"][0][0] if "phonemes" in voice else english.DEFAULT_TABLE
        self.rules = set()
        for line in voice.get("dictrules", []):
            for n in line:
                if not n.isdigit():
                    raise DataFileError("the voice of %s has a `dictrules` value which is no number: %r" % (
                        english.VARIANT_LANGUAGE[variant], n))
                self.rules.add(int(n))
        source = read_phoneme_source(os.path.join(data_dir, english.PHONEME_FOLDER), tables=(table,))
        self.table = PhonemeTable(source, table)
        self.entries = {}
        for e in parse_word_list(_read(os.path.join(data_dir, english.WORD_LIST)), english.WORD_LIST):
            if e.condition is None or (e.condition[0] in self.rules) == e.condition[1]:
                self.entries.setdefault(tuple(w.lower() for w in e.words), []).append(e)
        # the voice's `replace <flags> <phoneme> <replacement>` lines (bit 0: final phoneme only, bit 1: not in
        # a stressed syllable)
        self.replace = []
        for r in voice.get("replace", []):
            if len(r) != 3 or not r[0].isdigit():
                raise DataFileError("the voice of %s has a `replace` line which is not <flags> <phoneme> <replacement>: "
                                    "%s" % (english.VARIANT_LANGUAGE[variant], " ".join(r)))
            for name in r[1:]:
                if name != "NULL" and name not in self.table.defs:
                    raise DataFileError("the voice of %s replaces the phoneme %r, which the table %s does not "
                                        "define" % (english.VARIANT_LANGUAGE[variant], name, table))
            self.replace.append((int(r[0]), r[1], r[2]))

    # -- the entry that applies --------------------------------------------------------------------------------
    def entry(self, word, written=None):
        """The entry of a word read alone: the lowest one in the list whose flags a lone word satisfies."""
        written = word if written is None else written
        for e in reversed(self.entries.get((word,), [])):
            if any(f in english.FLAGS_NEEDING_CONTEXT for f in e.flags):
                continue
            if "capital" in e.flags and not written[:1].isupper():
                continue
            if "allcaps" in e.flags and not (written.isupper() and len(written) > 1):
                continue
            if e.excess:
                raise Unreadable("the list line %d of %r has fields beyond the format (%s): it is not read" % (
                    e.line, word, " ".join(e.excess)))
            return e
        return None

    # -- stress ------------------------------------------------------------------------------------------------
    def stressed(self, defs, flags=()):
        """[definition ...] with stress phonemes -> [Ph ...] without them, each vowel with its stress level."""
        word, vowels, mark = [], [], None
        for d in defs:
            if "stress" in d.words:
                if d.name == english.STRESS_TO_PREVIOUS:
                    if vowels:
                        vowels[-1].stress = english.PRIMARY
                else:
                    if "stress_type" not in d.values:
                        raise DataFileError("the stress phoneme %r has no stress_type: the phoneme file must give it" % d.name)
                    mark = int(d.values["stress_type"])
                continue
            ph = Ph(d)
            if self.table.is_vowel(d):
                ph.stress = mark
                mark = None
                vowels.append(ph)
            word.append(ph)
        if not vowels:
            return word
        for bits, old, new in self.replace:
            for k, ph in enumerate(word):
                if ph.d.name == old and not (bits & 1 and k != len(word) - 1) and not (
                        bits & 2 and ph.stress is not None and ph.stress >= english.PRIMARY):
                    if new == "NULL":
                        raise Unreadable("the voice drops the phoneme %r: this lot does not read that" % old)
                    ph.d = self.table.defs[new]
        if not any(v.stress is not None and v.stress >= english.PRIMARY for v in vowels):
            place = next((int(f) for f in flags if f.isdigit()), None)
            if place is not None:
                if place > len(vowels):
                    raise Unreadable("the entry puts the stress on syllable %d of %d" % (place, len(vowels)))
                vowels[place - 1].stress = english.PRIMARY
            elif any(f in english.FLAGS_STRESSED_AT_END for f in flags) or not any(
                    f in english.FLAGS_UNSTRESSED for f in flags):
                free = [v for v in vowels if v.stress is None and "unstressed" not in v.d.words]
                if free:
                    free[0].stress = english.PRIMARY
        for k, v in enumerate(vowels):
            if v.stress is None:
                near = [vowels[j].stress for j in (k - 1, k + 1) if 0 <= j < len(vowels)]
                lone = all(s is None or s < english.SECONDARY for s in near)
                full = "unstressed" not in v.d.words
                v.stress = english.SECONDARY if full and lone and len(vowels) > 1 and any(
                    x.stress is not None and x.stress >= english.PRIMARY for x in vowels) else english.UNSTRESSED
        return word

    # -- a word ------------------------------------------------------------------------------------------------
    def word_phonemes(self, written):
        word = written.lower()
        e = self.entry(word, written)
        if e is None:
            raise Unreadable("the word %r is not in the word list (the rules are a later lot)" % written)
        if e.phonemes is not None:
            return self.table.change(self.stressed(self.table.read(e.phonemes), e.flags))
        if e.text is None and "abbrev" in e.flags:
            return self.table.change(self.spelled(word, written))
        raise Unreadable("the entry of %r gives no phoneme string (its flags: %s): the rules are a later lot" % (
            written, " ".join(e.flags) or "none"))

    def spelled(self, word, written):
        """The letter names of a word, every letter but the last with its primary stress lowered."""
        out = []
        for k, letter in enumerate(word):
            le = self.entry(english.LETTER_NAME_PREFIX + letter) or self.entry(letter)
            if le is None or le.phonemes is None:
                raise Unreadable("the letter %r of %r has no name in the word list" % (letter, written))
            part = self.stressed(self.table.read(le.phonemes), le.flags)
            lower = [ph for ph in part if k < len(word) - 1 and ph.stress is not None and ph.stress >= english.PRIMARY]
            for ph in lower:
                ph.stress = english.SECONDARY
            out.extend(part)
        return out

    def units(self, text):
        """The unit sequence of a text, as the vendor stack writes it (the written phonemes and one space)."""
        words = text.split()
        if len(words) != 1:
            raise Unreadable("this lot reads one word; the text has %d" % len(words))
        return "".join(self.table.written(ph, self.tie) for ph in self.word_phonemes(words[0])) + " "
