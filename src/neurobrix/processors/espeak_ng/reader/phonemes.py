"""The phoneme table of a variant: reading a phoneme string, the changes the definitions order, the written form.

Written from the data files, their documentation and observed outputs only.
"""
import unicodedata

from . import english
from .data_files import DataFileError


class Unreadable(ValueError):
    """The text needs something this lot of the reader does not read; the message names it."""


class Ph(object):
    """One phoneme of a word being read."""
    __slots__ = ("d", "stress", "written")

    def __init__(self, d, stress=None):
        self.d = d              # its definition
        self.stress = stress    # the stress level of its syllable (vowels only)
        self.written = None     # a written form ordered by an `ipa(...)` instruction, or None


class PhonemeTable(object):
    def __init__(self, source, name):
        self.source = source
        self.name = name
        self.defs = source.table(name)
        for d in list(self.defs.values()):
            self._resolve_import(d)
        self.longest = max(len(n) for n in self.defs)

    def _lookup(self, ref, where):
        """A `table/phoneme` or `phoneme` reference -> its definition."""
        table, name = "", ref
        if "/" in ref and ref.split("/", 1)[0] in self.source.parents:
            table, name = ref.split("/", 1)
        defs = self.source.table(table) if table else self.defs
        if name in defs:
            return defs[name]
        for (_, n), procedure in self.source.procedures.items():
            if n == name:
                return procedure
        raise DataFileError("phoneme %r refers to %r, which no table defines" % (where, ref))

    def _resolve_import(self, d):
        ref = next((n[2] for n in d.program if n[0] == "do" and n[1] == "import_phoneme"), None)
        if ref is None:
            return
        src = self._lookup(ref, d.name)
        if src is not d:
            self._resolve_import(src)
            d.words = list(src.words) + d.words
            merged = dict(src.values)
            merged.update((k, v) for k, v in d.values.items() if k != "import_phoneme")
            d.values = merged
            d.program = list(src.program) + [n for n in d.program if n[1] != "import_phoneme"]

    # -- reading a phoneme string ------------------------------------------------------------------------------
    def read(self, string):
        """A phoneme string -> [definition ...] by the longest name at each place; `|` separates two names."""
        out, i = [], 0
        while i < len(string):
            if string[i] == "|" and string[i:i + 2] != "||":
                i += 1
                continue
            for n in range(min(self.longest, len(string) - i), 0, -1):
                d = self.defs.get(string[i:i + n])
                if d is not None:
                    out.append(d)
                    i += n
                    break
            else:
                raise Unreadable("the phoneme string %r has no phoneme of table %s at %r" % (string, self.name, string[i:]))
        return out

    # -- properties --------------------------------------------------------------------------------------------
    @staticmethod
    def is_vowel(d):
        return "vwl" in d.words

    def has(self, ph, attribute, side):
        """One documented attribute test on a phoneme (None = no phoneme there: a pause)."""
        if attribute == "isPause2":     # "not a vowel or liquid consonant within the same word"
            return ph is None or not ("vwl" in ph.d.words or "liquid" in ph.d.words)
        if ph is None:
            return attribute in ("isPause", "isNotVowel")
        d = ph.d
        w = d.words
        if attribute == "isVowel":
            return "vwl" in w
        if attribute == "isNotVowel":
            return "vwl" not in w
        if attribute == "isPause":
            return "pause" in w
        if attribute == "isLiquid":
            return "liquid" in w
        if attribute == "isNasal":
            return "nas" in w
        if attribute == "isRhotic":
            return "rhotic" in w
        if attribute == "isPalatal":
            return "pal" in w or "palatal" in w
        if attribute == "isVelar":
            return "vel" in w
        if attribute == "isSibilant":
            return "sib" in w
        if attribute == "isVoiced":
            return "vwl" in w or "vcd" in w or "liquid" in w or "nas" in w
        if attribute == "isVFricative":
            return "vcd" in w and "frc" in w
        if attribute in ("isStressed", "isMaxStress"):
            return ph.stress is not None and ph.stress >= english.PRIMARY
        if attribute == "isNotStressed":
            return ph.stress is None or ph.stress < english.PRIMARY
        if attribute == "isUnstressed":
            return ph.stress is not None and ph.stress <= english.UNSTRESSED
        if attribute == "isDiminished":
            return ph.stress is not None and ph.stress <= english.DIMINISHED
        if attribute.startswith("#"):
            return d.values.get("endtype" if side < 0 else "starttype") == attribute
        if attribute in self.defs:
            return d.name == attribute
        raise Unreadable("phoneme %r tests %r, an attribute this lot does not read" % (d.name, attribute))

    def _test(self, word, i, test, attribute):
        def vowel(step):
            j = i + step
            while 0 <= j < len(word):
                if self.is_vowel(word[j].d):
                    return word[j]
                j += step
            return None
        at = lambda j: word[j] if 0 <= j < len(word) else None
        if test in ("KlattSynth", "MbrolaSynth", "PreVoicing"):
            return False
        if test == "thisPh":
            if attribute in ("isWordStart", "notWordStart"):
                return (i == 0) == (attribute == "isWordStart")
            if attribute == "isWordEnd":
                return i == len(word) - 1
            if attribute in ("isFirstVowel", "isFinalVowel"):
                return vowel(-1 if attribute == "isFirstVowel" else 1) is None
            if attribute == "isAfterStress":
                return any(p.stress is not None and p.stress >= english.PRIMARY for p in word[:i])
            return self.has(word[i], attribute, 0)
        steps = {"prevPh": (-1, False), "prevPhW": (-1, True), "prev2PhW": (-2, True), "nextPh": (1, False),
                 "nextPhW": (1, True), "next2Ph": (2, False), "next2PhW": (2, True), "next3PhW": (3, True)}
        if test in steps:
            step, inside = steps[test]
            ph = at(i + step)
            if ph is None and inside:
                return False
            return self.has(ph, attribute, step)
        if test in ("nextVowel", "prevVowel"):
            ph = vowel(1 if test == "nextVowel" else -1)
            return ph is not None and self.has(ph, attribute, 0)
        raise Unreadable("phoneme %r uses the test %r, which this lot does not read" % (word[i].d.name, test))

    def _run(self, word, i, program, calls=()):
        """Runs the first-phase instructions of word[i]; returns True when the phoneme changed (instructions end)."""
        ph = word[i]
        for node in program:
            if node[0] == "if":
                body = node[2]
                for (joiner, tests), then in node[1]:
                    results = (self._test(word, i, t, a) != neg for neg, t, a in tests)
                    if all(results) if joiner == "AND" else any(results):
                        body = then
                        break
                if self._run(word, i, body, calls):
                    return True
                continue
            _, name, arg = node
            change = None
            if name == "ChangePhoneme":
                change = arg
            elif name == "ChangeIfDiminished":
                change = arg if ph.stress is not None and ph.stress <= english.DIMINISHED else None
            elif name == "ChangeIfUnstressed":
                change = arg if ph.stress is not None and ph.stress <= english.UNSTRESSED else None
            elif name == "ChangeIfNotStressed":
                change = arg if ph.stress is None or ph.stress < english.PRIMARY else None
            elif name == "ChangeIfStressed":
                change = arg if ph.stress is not None and ph.stress >= english.PRIMARY else None
            elif name == "ipa":
                ph.written = "" if arg == "NULL" else arg
            elif name == "RETURN":
                return True
            elif name == "CALL":
                if arg in calls:
                    raise DataFileError("the procedure %r calls itself (%s): its phoneme %r cannot be read" % (
                        arg, " -> ".join(calls + (arg,)), ph.d.name))
                if self._run(word, i, self._lookup(arg, ph.d.name).program, calls + (arg,)):
                    return True
            elif name in ("IfNextVowelAppend", "AppendPhoneme", "InsertPhoneme"):
                nxt = word[i + 1] if i + 1 < len(word) else None
                if name != "IfNextVowelAppend" or (nxt is not None and self.is_vowel(nxt.d)):
                    new = Ph(self._lookup(arg, ph.d.name))
                    if name == "InsertPhoneme":
                        word.insert(i, new)
                        return True
                    if nxt is None or nxt.d is not new.d:
                        word.insert(i + 1, new)
            if change is not None:
                if change == "NULL":
                    del word[i]
                else:
                    ph.d = self._lookup(change, ph.d.name)
                    ph.written = None
                return True
        return False

    def change(self, word):
        """The first phase: every phoneme's instructions may replace it, in word order; a replaced phoneme is
        read again under its new definition."""
        i = 0
        while i < len(word):
            seen = []
            while True:
                before = word[i]
                d, n = before.d, len(word)
                if (before, d) in seen:
                    raise Unreadable("the changes of phoneme %r come back to it (%s): a cycle, not read" % (
                        d.name, " -> ".join(x.name for _, x in seen) + " -> " + d.name))
                seen.append((before, d))
                changed = self._run(word, i, d.program)
                if not changed or i >= len(word) or (word[i] is before and before.d is d):
                    break
                if len(word) != n and (i >= len(word) or word[i] is not before):
                    i -= 1 if len(word) < n else 0
                    break
            i += 1
        return word

    # -- the written form --------------------------------------------------------------------------------------
    def written(self, ph, tie):
        text = ph.written
        if text is None:
            text = ph.d.values.get("ipa")
            if text is None:
                name = ph.d.name
                text = english.NAME_TO_WRITTEN.get(name)
            if text is None:
                text = "".join(english.NAME_TO_WRITTEN.get(c, c if c.isalpha() else "")
                               for k, c in enumerate(name) if k == 0 or not c.isdigit())
        text = text.replace("|", "")
        if tie:
            text = text[:1] + "".join(tie + c if _is_letter(c) else c for c in text[1:])
        if ph.stress is not None and ph.stress >= english.PRIMARY:
            return english.PRIMARY_MARK + text
        if ph.stress is not None and ph.stress >= english.SECONDARY:
            return english.SECONDARY_MARK + text
        return text


def _is_letter(c):
    return unicodedata.category(c).startswith("L") and unicodedata.category(c) != "Lm"
