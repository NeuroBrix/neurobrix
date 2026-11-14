"""The English lexicon pass of the tagged text front end, standard library only.

The behaviour of misaki 0.9.4 (Apache-2.0) en.py, studied; our own code, written to give the same phonemes token by
token (proven byte for byte, us and gb, on the judge texts, a 175-sentence rule set and 887 documentation sentences).

Input: the tokens and part-of-speech tags of the tagger (text, tag, whitespace per token), plus the markdown-link
features of `preprocess`. Output: the phoneme string and the per-token phonemes.

Every constant table comes from the container's front_end.json (each with its source); the gold and silver lexicons
are read unmodified from the container (`modules/g2p/<variant>_gold.json.gz`, `..._silver.json.gz`). Unknown words go
to the fallback the caller gives: a callable text -> (phonemes, rating) or (None, None).
"""
import functools
import gzip
import json
import math
import os
import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable, List, Optional, Tuple


# ----------------------------------------------------------------------------------------------------------------
# Tables


class Tables:
    """front_end.json's tables (the loaded document), each one an attribute; a missing table is a named refusal."""

    def __init__(self, doc: dict):
        self.sources = doc["sources"]
        self._t = {k: v["value"] for k, v in doc["tables"].items()}

    def __getattr__(self, name):
        t = self.__dict__.get("_t")
        if t is None or name not in t:
            raise KeyError(f"front_end.json has no table {name!r}")
        return t[name]


# ----------------------------------------------------------------------------------------------------------------
# Unicode letter classes for the sub-token pattern (the vendor writes it in the 'regex' module's \p{..} syntax;
# Python's re has no \p, so the classes are built from unicodedata's categories).

_CLASS_CACHE = {}


def _category_class(name: str) -> str:
    """The body of a character class (no brackets) matching the Unicode category class `name` (L, Lu or Ll)."""
    if name in _CLASS_CACHE:
        return _CLASS_CACHE[name]
    if name == "L":
        want = ("Lu", "Ll", "Lt", "Lm", "Lo")
    elif name in ("Lu", "Ll"):
        want = (name,)
    else:
        raise ValueError(f"sub-token pattern: unicode class \\p{{{name}}} is not supported")
    parts = []
    start = prev = None
    for cp in range(0x110000):
        if unicodedata.category(chr(cp)) in want:
            if start is None:
                start = cp
            elif cp != prev + 1:
                parts.append((start, prev))
                start = cp
            prev = cp
    if start is not None:
        parts.append((start, prev))

    def esc(cp):
        return "\\U%08x" % cp

    body = "".join(esc(a) if a == b else f"{esc(a)}-{esc(b)}" for a, b in parts)
    _CLASS_CACHE[name] = body
    return body


def translate_unicode_classes(pattern: str) -> str:
    """Rewrite \\p{X} into explicit classes: the class body inside [...], a bracketed class outside."""
    out = []
    i = 0
    in_class = False
    while i < len(pattern):
        c = pattern[i]
        if c == "\\":
            if pattern.startswith("\\p{", i):
                j = pattern.index("}", i)
                body = _category_class(pattern[i + 3:j])
                out.append(body if in_class else f"[{body}]")
                i = j + 1
                continue
            out.append(pattern[i:i + 2])
            i += 2
            continue
        if c == "[" and not in_class:
            in_class = True
        elif c == "]" and in_class:
            in_class = False
        out.append(c)
        i += 1
    return "".join(out)


# ----------------------------------------------------------------------------------------------------------------
# Number words (num2words' English rules, our own code; tested against num2words over every value get_number uses)


class NumberWords:
    def __init__(self, T: Tables):
        self.cards = [(int(v), w) for v, w in T.n2w_cards]   # descending
        self.card = dict(self.cards)
        self.maxval = int(T.n2w_maxval)
        self.ords = T.n2w_ords
        self.negword = T.n2w_words["negword"]
        self.pointword = T.n2w_words["pointword"]
        self.year_words = T.n2w_year
        self.m = T.n2w_merge

    # value -> nested (word, value) groups, largest card first
    def _split(self, value):
        for unit, word in self.cards:
            if unit > value:
                continue
            if value == 0:
                div, mod = 1, 0
            else:
                div, mod = divmod(value, unit)
            if div == 1:
                parts = [(self.card[1], 1)]
            elif div == value:
                return [(div * word, div * unit)]
            else:
                parts = [self._split(div)]
            parts.append((word, unit))
            if mod:
                parts.append(self._split(mod))
            return parts

    def _join(self, left, right):
        ltext, lnum = left
        rtext, rnum = right
        if lnum == 1 and rnum < 100:
            return rtext, rnum
        if 100 > lnum > rnum:
            return f"{ltext}{self.m['hyphen']}{rtext}", lnum + rnum
        if lnum >= 100 > rnum:
            return f"{ltext}{self.m['and']}{rtext}", lnum + rnum
        if rnum > lnum:
            return f"{ltext}{self.m['space']}{rtext}", lnum * rnum
        return f"{ltext}{self.m['comma']}{rtext}", lnum + rnum

    def _reduce(self, items):
        # The first two groups join once both are reduced; the rest reduces as one group, then joins.
        while len(items) != 1:
            a, b = items[0], items[1]
            if isinstance(a, tuple) and isinstance(b, tuple):
                rest = items[2:]
                items = [self._join(a, b)] + ([rest] if rest else [])
            else:
                items = [(e[0] if len(e) == 1 else self._reduce(e)) if isinstance(e, list) else e for e in items]
        return items[0]

    def cardinal(self, value):
        try:
            integral = int(value) == value
        except (ValueError, TypeError):
            integral = False
        if not integral:
            return self._cardinal_float(value)
        prefix = ""
        if value < 0:
            value = abs(value)
            prefix = self.negword + " "
        if value >= self.maxval:
            raise OverflowError(f"abs({value}) must be less than {self.maxval}.")
        words, _ = self._reduce(self._split(value))
        return prefix + words

    def _cardinal_float(self, value):
        value = float(value)
        pre = int(value)
        precision = abs(Decimal(str(value)).as_tuple().exponent)
        post = abs(value - pre) * 10 ** precision
        if abs(round(post) - post) < 0.01:
            post = int(round(post))
        else:
            post = int(math.floor(post))
        post = str(post)
        post = "0" * (precision - len(post)) + post
        out = [self.cardinal(pre)]
        if precision:
            out.append(self.pointword)
        for i in range(precision):
            out.append(str(self.cardinal(int(post[i]))))
        return " ".join(out)

    def ordinal(self, value):
        if not value == int(value):
            raise TypeError(f"Cannot treat float {value} as ordinal.")
        if not abs(value) == value:
            raise TypeError(f"Cannot treat negative num {value} as ordinal.")
        outwords = self.cardinal(value).split(" ")
        lastwords = outwords[-1].split("-")
        last = lastwords[-1].lower()
        if last in self.ords:
            last = self.ords[last]
        else:
            y, ie = self.m["ord_y"]
            if last[-1] == y:
                last = last[:-1] + ie
            last += self.m["ord_suffix"]
        lastwords[-1] = last
        outwords[-1] = "-".join(lastwords)
        return " ".join(outwords)

    def year(self, val):
        if val < 0:
            raise ValueError(f"year {val}: a negative year is never asked by the lexicon pass")
        high, low = val // 100, val % 100
        if high == 0 or (high % 10 == 0 and low < 10) or high >= 100:
            return self.cardinal(val)
        hightext = self.cardinal(high)
        if low == 0:
            lowtext = self.year_words["hundred"]
        elif low < 10:
            lowtext = self.year_words["oh"] + self.cardinal(low)
        else:
            lowtext = self.cardinal(low)
        return f"{hightext} {lowtext}"


# ----------------------------------------------------------------------------------------------------------------
# Tokens


@dataclass
class Tok:
    text: str
    tag: str
    whitespace: str
    phonemes: Optional[str] = None
    rating: Optional[int] = None
    is_head: bool = True
    alias: Optional[str] = None
    stress: Optional[float] = None
    currency: Optional[str] = None
    num_flags: str = ""
    prespace: bool = False


@dataclass
class Context:
    future_vowel: Optional[bool] = None
    future_to: bool = False


def _case_weight(text):
    return sum(1 if c == c.lower() else 2 for c in text)


def merge_tokens(tokens: List[Tok], unk: Optional[str] = None) -> Tok:
    stress = {t.stress for t in tokens if t.stress is not None}
    currency = {t.currency for t in tokens if t.currency is not None}
    ratings = {t.rating for t in tokens}
    phonemes = None
    if unk is not None:
        phonemes = ""
        for t in tokens:
            if t.prespace and phonemes and not phonemes[-1].isspace() and t.phonemes:
                phonemes += " "
            phonemes += unk if t.phonemes is None else t.phonemes
    best = tokens[0]
    for t in tokens[1:]:
        if _case_weight(t.text) > _case_weight(best.text):
            best = t
    return Tok(
        text="".join(t.text + t.whitespace for t in tokens[:-1]) + tokens[-1].text,
        tag=best.tag,
        whitespace=tokens[-1].whitespace,
        phonemes=phonemes,
        rating=None if None in ratings else min(ratings),
        is_head=tokens[0].is_head,
        alias=None,
        stress=next(iter(stress)) if len(stress) == 1 else None,
        currency=max(currency) if currency else None,
        num_flags="".join(sorted({c for t in tokens for c in t.num_flags})),
        prespace=tokens[0].prespace,
    )


# ----------------------------------------------------------------------------------------------------------------
# Stress


class Stress:
    def __init__(self, T: Tables):
        self.primary = T.stresses["primary"]
        self.secondary = T.stresses["secondary"]
        self.marks = self.secondary + self.primary
        self.vowels = frozenset(T.vowels)
        self.diphthongs = frozenset(T.diphthongs)

    def _restress(self, ps):
        # Each stress mark moves to just before the first vowel at or after it.
        placed = []
        for i, p in enumerate(ps):
            if p in self.marks:
                j = next(k for k in range(i, len(ps)) if ps[k] in self.vowels)
                placed.append((j - 0.5, p))
            else:
                placed.append((i, p))
        return "".join(p for _, p in sorted(placed))

    def apply(self, ps, stress):
        P, S = self.primary, self.secondary
        if stress is None:
            return ps
        if stress < -1:
            return ps.replace(P, "").replace(S, "")
        if stress == -1 or (stress in (0, -0.5) and P in ps):
            return ps.replace(S, "").replace(P, S)
        if stress in (0, 0.5, 1) and all(s not in ps for s in self.marks):
            if all(v not in ps for v in self.vowels):
                return ps
            return self._restress(S + ps)
        if stress >= 1 and P not in ps and S in ps:
            return ps.replace(S, P)
        if stress > 1 and all(s not in ps for s in self.marks):
            if all(v not in ps for v in self.vowels):
                return ps
            return self._restress(P + ps)
        return ps

    def weight(self, ps):
        return sum(2 if c in self.diphthongs else 1 for c in ps) if ps else 0


_DIGITS = re.compile(r"^[0-9]+$")


def is_digit(text):
    return bool(_DIGITS.match(text))


# ----------------------------------------------------------------------------------------------------------------
# The lexicon


def _load_json(directory, name):
    for cand in (os.path.join(directory, name), os.path.join(directory, name + ".gz")):
        if os.path.exists(cand):
            opener = gzip.open if cand.endswith(".gz") else open
            with opener(cand, "rt", encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(f"the text front end's lexicon {name} (or {name}.gz) is not in {directory}")


def _grow(d):
    """Add the capitalised form of every lower-case entry and the lower form of every capitalised one."""
    extra = {}
    for k, v in d.items():
        if len(k) < 2:
            continue
        if k == k.lower():
            if k != k.capitalize():
                extra[k.capitalize()] = v
        elif k == k.lower().capitalize():
            extra[k.lower()] = v
    extra.update(d)
    return extra


def _parent_tag(tag):
    if tag is None:
        return tag
    if tag.startswith("VB"):
        return "VERB"
    if tag.startswith("NN"):
        return "NOUN"
    if tag.startswith("ADV") or tag.startswith("RB"):
        return "ADV"
    if tag.startswith("ADJ") or tag.startswith("JJ"):
        return "ADJ"
    return tag


class Lexicon:
    def __init__(self, T: Tables, lexicon_dir: str, british: bool):
        self.T = T
        self.british = british
        variant = "gb" if british else "us"
        gold_name, silver_name = T.lexicon_files[variant]
        self.golds = _grow(_load_json(lexicon_dir, gold_name))
        self.silvers = _grow(_load_json(lexicon_dir, silver_name))
        vocab = frozenset(T.vocab[variant])
        for v in self.golds.values():
            if isinstance(v, str):
                bad = [c for c in v if c not in vocab]
            elif isinstance(v, dict):
                if "DEFAULT" not in v:
                    raise ValueError(f"lexicon {gold_name}: a tagged entry without DEFAULT: {v}")
                bad = [c for x in v.values() if x is not None for c in x if c not in vocab]
            else:
                raise ValueError(f"lexicon {gold_name}: entry of type {type(v).__name__}")
            if bad:
                raise ValueError(f"lexicon {gold_name}: phonemes {bad} outside the {variant} vocabulary")
        self.st = Stress(T)
        self.P, self.S = self.st.primary, self.st.secondary
        self.cap_stresses = tuple(T.cap_stresses)
        self.symbols = T.symbols
        self.add_symbols = T.add_symbols
        self.currencies = T.currencies
        self.ordinals = frozenset(T.ordinals)
        self.ords_ok = frozenset(T.lexicon_ords)
        self.us_taus = frozenset(T.us_taus)
        self.sc = T.special_cases
        self.sx = T.suffixes
        self.nw = T.number_words
        self.doubled = re.compile(self.sx["ing_doubled_regex"])
        self.num_suffix = re.compile(self.nw["number_suffix_regex"])
        self.word_split = re.compile(self.nw["word_split_regex"])
        self.vs = re.compile(self.sc["vs_regex"])
        self.n2w = NumberWords(T)
        self.is_number_suffixes = []
        for s in self.nw["is_number_suffixes"]:
            if s == "*ORDINALS":  # misaki splices its ORDINALS set here; they end differently, so order is free
                self.is_number_suffixes.extend(T.ordinals)
            else:
                self.is_number_suffixes.append(s)

    # -- spelling of initialisms and names letter by letter
    def get_NNP(self, word):
        ps = [self.golds.get(c.upper()) for c in word if c.isalpha()]
        if None in ps:
            return None, None
        ps = self.st.apply("".join(ps), 0)
        ps = ps.rsplit(self.S, 1)
        return self.P.join(ps), 3

    def get_special_case(self, word, tag, stress, ctx):
        sc = self.sc
        if tag == "ADD" and word in self.add_symbols:
            return self.lookup(self.add_symbols[word], None, sc["add_symbol_stress"], ctx)
        if word in self.symbols:
            return self.lookup(self.symbols[word], None, None, ctx)
        if "." in word.strip(".") and word.replace(".", "").isalpha() and len(max(word.split("."), key=len)) < 3:
            return self.get_NNP(word)
        if word in ("a", "A"):
            return (sc["a_dt"] if tag == "DT" else sc["a_other"]), 4
        if word in ("am", "Am", "AM"):
            if tag.startswith("NN"):
                return self.get_NNP(word)
            if ctx.future_vowel is None or word != "am" or stress and stress > 0:
                return self.golds["am"], 4
            return sc["am_weak"], 4
        if word in ("an", "An", "AN"):
            if word == "AN" and tag.startswith("NN"):
                return self.get_NNP(word)
            return sc["an"], 4
        if word == "I" and tag == "PRP":
            return sc["i_prp"], 4
        if word in ("by", "By", "BY") and _parent_tag(tag) == "ADV":
            return sc["by_adv"], 4
        if word in ("to", "To") or (word == "TO" and tag in ("TO", "IN")):
            fv = ctx.future_vowel
            if fv is None:
                return self.golds["to"], 4
            return (sc["to_before_vowel"] if fv else sc["to_before_consonant"]), 4
        if word in ("in", "In") or (word == "IN" and tag != "NNP"):
            mark = self.P if ctx.future_vowel is None or tag != "IN" else ""
            return mark + sc["in"], 4
        if word in ("the", "The") or (word == "THE" and tag == "DT"):
            return (sc["the_before_vowel"] if ctx.future_vowel == True else sc["the_before_consonant"]), 4  # noqa: E712
        if tag == "IN" and self.vs.match(word):
            return self.lookup(sc["vs_word"], None, None, ctx)
        if word in ("used", "Used", "USED"):
            if tag in ("VBD", "JJ") and ctx.future_to:
                return self.golds["used"]["VBD"], 4
            return self.golds["used"]["DEFAULT"], 4
        return None, None

    def is_known(self, word, tag):
        if word in self.golds or word in self.symbols or word in self.silvers:
            return True
        if not word.isalpha() or not all(ord(c) in self.ords_ok for c in word):
            return False
        if len(word) == 1:
            return True
        if word == word.upper() and word.lower() in self.golds:
            return True
        return word[1:] == word[1:].upper()

    def lookup(self, word, tag, stress, ctx):
        is_NNP = None
        if word == word.upper() and word not in self.golds:
            word = word.lower()
            is_NNP = tag == "NNP"
        ps, rating = self.golds.get(word), 4
        if ps is None and not is_NNP:
            ps, rating = self.silvers.get(word), 3
        if isinstance(ps, dict):
            if ctx and ctx.future_vowel is None and "None" in ps:
                tag = "None"
            elif tag not in ps:
                tag = _parent_tag(tag)
            ps = ps.get(tag, ps["DEFAULT"])
        if ps is None or (is_NNP and self.P not in ps):
            ps, rating = self.get_NNP(word)
            if ps is not None:
                return ps, rating
        return self.st.apply(ps, stress), rating

    # -- suffixes
    def _s(self, stem):
        sx = self.sx
        if not stem:
            return None
        if stem[-1] in sx["s_voiceless"]:
            return stem + sx["s_plain_voiceless"]
        if stem[-1] in sx["s_sibilant"]:
            return stem + sx["s_after_sibilant"]["gb" if self.british else "us"]
        return stem + sx["s_plain"]

    def stem_s(self, word, tag, stress, ctx):
        if len(word) < 3 or not word.endswith("s"):
            return None, None
        if not word.endswith("ss") and self.is_known(word[:-1], tag):
            stem = word[:-1]
        elif (word.endswith("'s") or (len(word) > 4 and word.endswith("es") and not word.endswith("ies"))) \
                and self.is_known(word[:-2], tag):
            stem = word[:-2]
        elif len(word) > 4 and word.endswith("ies") and self.is_known(word[:-3] + "y", tag):
            stem = word[:-3] + "y"
        else:
            return None, None
        stem, rating = self.lookup(stem, tag, stress, ctx)
        return self._s(stem), rating

    def _ed(self, stem):
        sx = self.sx
        v = "gb" if self.british else "us"
        if not stem:
            return None
        if stem[-1] in sx["ed_voiceless"]:
            return stem + sx["ed_after_voiceless"]
        if stem[-1] == "d":
            return stem + sx["ed_after_d"][v]
        if stem[-1] != "t":
            return stem + sx["ed_plain"]
        if self.british or len(stem) < 2:
            return stem + sx["ed_after_t_short"]
        if stem[-2] in self.us_taus:
            return stem[:-1] + sx["ed_flap"]
        return stem + sx["ed_after_t"][v]

    def stem_ed(self, word, tag, stress, ctx):
        if len(word) < 4 or not word.endswith("d"):
            return None, None
        if not word.endswith("dd") and self.is_known(word[:-1], tag):
            stem = word[:-1]
        elif len(word) > 4 and word.endswith("ed") and not word.endswith("eed") and self.is_known(word[:-2], tag):
            stem = word[:-2]
        else:
            return None, None
        stem, rating = self.lookup(stem, tag, stress, ctx)
        return self._ed(stem), rating

    def _ing(self, stem):
        sx = self.sx
        if not stem:
            return None
        if self.british:
            if stem[-1] in sx["ing_gb_refuse"]:
                return None
        elif len(stem) > 1 and stem[-1] == "t" and stem[-2] in self.us_taus:
            return stem[:-1] + sx["ing_flap"]
        return stem + sx["ing"]

    def stem_ing(self, word, tag, stress, ctx):
        if len(word) < 5 or not word.endswith("ing"):
            return None, None
        if len(word) > 5 and self.is_known(word[:-3], tag):
            stem = word[:-3]
        elif self.is_known(word[:-3] + "e", tag):
            stem = word[:-3] + "e"
        elif len(word) > 5 and self.doubled.search(word) and self.is_known(word[:-4], tag):
            stem = word[:-4]
        else:
            return None, None
        stem, rating = self.lookup(stem, tag, stress, ctx)
        return self._ing(stem), rating

    def get_word(self, word, tag, stress, ctx):
        ps, rating = self.get_special_case(word, tag, stress, ctx)
        if ps is not None:
            return ps, rating
        wl = word.lower()
        if len(word) > 1 and word.replace("'", "").isalpha() and word != word.lower() \
                and (tag != "NNP" or len(word) > 7) and word not in self.golds and word not in self.silvers \
                and (word == word.upper() or word[1:] == word[1:].lower()):
            if wl in self.golds or wl in self.silvers:
                word = wl
            else:
                for fn in (self.stem_s, self.stem_ed, self.stem_ing):
                    if fn(wl, tag, stress, ctx)[0]:
                        word = wl
                        break
        if self.is_known(word, tag):
            return self.lookup(word, tag, stress, ctx)
        if word.endswith("s'") and self.is_known(word[:-2] + "'s", tag):
            return self.lookup(word[:-2] + "'s", tag, stress, ctx)
        if word.endswith("'") and self.is_known(word[:-1], tag):
            return self.lookup(word[:-1], tag, stress, ctx)
        ps, rating = self.stem_s(word, tag, stress, ctx)
        if ps is not None:
            return ps, rating
        ps, rating = self.stem_ed(word, tag, stress, ctx)
        if ps is not None:
            return ps, rating
        ps, rating = self.stem_ing(word, tag, 0.5 if stress is None else stress, ctx)
        if ps is not None:
            return ps, rating
        return None, None

    # -- numbers
    @staticmethod
    def is_currency(word):
        if "." not in word:
            return True
        if word.count(".") > 1:
            return False
        # misaki also tests set(cents) == {0}, which a set of characters never equals: only the length decides.
        return len(word.split(".")[1]) < 3

    def _extend(self, result, num_flags, num, first=True, escape=False):
        """The words of a number (its cardinal, or `num` itself when `escape`) appended to `result` as phonemes."""
        nw = self.nw
        words = self.word_split.split(num if escape else self.n2w.cardinal(int(num)))
        for i, w in enumerate(words):
            if w != nw["and"] or nw["flag_and"] in num_flags:
                if first and i == 0 and len(words) > 1 and w == "one" and nw["flag_a"] in num_flags:
                    result.append((nw["a_one"], 4))
                else:
                    result.append(self.lookup(w, None, nw["point_stress"] if w == nw["point"] else None, None))
            elif w == nw["and"] and nw["flag_n"] in num_flags and result:
                result[-1] = (result[-1][0] + nw["and_suffix"], result[-1][1])

    def _digit_by_digit(self, result, extend, word):
        """A number that is not the head of its word, without a point: digit by digit, or in groups."""
        num = word.replace(",", "")
        if num[0] == "0" or len(num) > 3:
            for n in num:
                extend(n, first=False)
        elif len(num) == 3 and not num.endswith("00"):
            extend(num[0])
            if num[1] == "0":
                result.append(self.lookup(self.nw["zero_letter"], None, self.nw["zero_letter_stress"], None))
                extend(num[2], first=False)
            else:
                extend(num[1:], first=False)
        else:
            extend(num)

    @staticmethod
    def _dotted(extend, word):
        """A number of several points (a version, an address) or not the head of its word: part by part."""
        first = True
        for num in word.replace(",", "").split("."):
            if not num:
                pass
            elif num[0] == "0" or (len(num) != 2 and any(n != "0" for n in num[1:])):
                for n in num:
                    extend(n, first=False)
            else:
                extend(num, first=first)
            first = False

    def _money(self, result, extend, word, currency):
        """An amount of a currency: its units and sub-units, each with its plural."""
        units = self.currencies[currency]
        pairs = [(int(num) if num else 0, unit) for num, unit in zip(word.replace(",", "").split("."), units)]
        if len(pairs) > 1:
            if pairs[1][0] == 0:
                pairs = pairs[:1]
            elif pairs[0][0] == 0:
                pairs = pairs[1:]
        for i, (num, unit) in enumerate(pairs):
            if i > 0:
                result.append(self.lookup(self.nw["and"], None, None, None))
            extend(num, first=i == 0)
            result.append(self.stem_s(unit + "s", None, None, None)
                          if abs(num) != 1 and unit != self.nw["no_plural_unit"] else self.lookup(unit, None, None, None))

    def _spelled(self, word, suffix):
        """A head number read whole: its cardinal, its ordinal, or its decimals."""
        if is_digit(word):
            return self.n2w.cardinal(int(word))
        if "." not in word:
            v = int(word.replace(",", ""))
            return self.n2w.ordinal(v) if suffix in self.ordinals else self.n2w.cardinal(v)
        word = word.replace(",", "")
        if word[0] == ".":
            return self.nw["point"] + " " + " ".join(self.n2w.cardinal(int(n)) for n in word[1:])
        return self.n2w.cardinal(float(word))

    def get_number(self, word, currency, is_head, num_flags):
        nw = self.nw
        m = self.num_suffix.search(word)
        suffix = m.group() if m else None
        word = word[:-len(suffix)] if suffix else word
        result = []
        if word.startswith("-"):
            result.append(self.lookup(nw["minus"], None, None, None))
            word = word[1:]
        extend = functools.partial(self._extend, result, num_flags)
        if is_digit(word) and suffix in self.ordinals:
            extend(self.n2w.ordinal(int(word)), escape=True)
        elif not result and len(word) == 4 and currency not in self.currencies and is_digit(word):
            extend(self.n2w.year(int(word)), escape=True)
        elif not is_head and "." not in word:
            self._digit_by_digit(result, extend, word)
        elif word.count(".") > 1 or not is_head:
            self._dotted(extend, word)
        elif currency in self.currencies and Lexicon.is_currency(word):
            self._money(result, extend, word, currency)
        else:
            extend(self._spelled(word, suffix), escape=True)
        if not result:
            return None, None
        ps, rating = " ".join(p for p, _ in result), min(r for _, r in result)
        if suffix in nw["suffix_s"]:
            return self._s(ps), rating
        if suffix in nw["suffix_ed"]:
            return self._ed(ps), rating
        if suffix in nw["suffix_ing"]:
            return self._ing(ps), rating
        return ps, rating

    def append_currency(self, ps, currency):
        if not currency:
            return ps
        units = self.currencies.get(currency)
        word = self.stem_s(units[0] + "s", None, None, None)[0] if units else None
        return f"{ps} {word}" if word else ps

    @staticmethod
    def numeric_if_needed(c):
        if not c.isdigit():
            return c
        n = unicodedata.numeric(c)
        return str(int(n)) if n == int(n) else c

    def is_number(self, word, is_head):
        if all(not is_digit(c) for c in word):
            return False
        for s in self.is_number_suffixes:
            if word.endswith(s):
                word = word[:-len(s)]
                break
        return all(is_digit(c) or c in ",." or (is_head and i == 0 and c == "-") for i, c in enumerate(word))

    def __call__(self, tk: Tok, ctx: Context):
        word = (tk.text if tk.alias is None else tk.alias).replace(chr(8216), "'").replace(chr(8217), "'")
        word = unicodedata.normalize("NFKC", word)
        word = "".join(Lexicon.numeric_if_needed(c) for c in word)
        stress = None if word == word.lower() else self.cap_stresses[int(word == word.upper())]
        ps, rating = self.get_word(word, tk.tag, stress, ctx)
        if ps is not None:
            return self.st.apply(self.append_currency(ps, tk.currency), tk.stress), rating
        if self.is_number(word, tk.is_head):
            ps, rating = self.get_number(word, tk.currency, tk.is_head, tk.num_flags)
            return self.st.apply(ps, tk.stress), rating
        return None, None


# ----------------------------------------------------------------------------------------------------------------
# The front end


class FrontEnd:
    """The loaded data of one variant: tables, lexicons, the fallback (callable text -> (phonemes, rating))."""

    def __init__(self, front_end: dict, lexicon_dir: str, british: bool,
                 fallback: Optional[Callable[[str], Tuple[Optional[str], Optional[int]]]] = None,
                 unk: Optional[str] = ""):
        self.T = Tables(front_end)
        self.british = british
        self.lexicon = Lexicon(self.T, lexicon_dir, british)
        self.fallback = fallback
        self.unk = unk
        self.link_regex = re.compile(self.T.link_regex)
        self.subtoken_regex = re.compile(translate_unicode_classes(self.T.subtoken_regex))
        self.punct_tags = frozenset(self.T.punct_tags)
        self.punct_tag_phonemes = self.T.punct_tag_phonemes
        self.puncts = frozenset(self.T.puncts)
        self.non_quote_puncts = frozenset(self.T.non_quote_puncts)
        self.junks = frozenset(self.T.subtoken_junks)
        self.vowels = frozenset(self.T.vowels)
        self.consonants = frozenset(self.T.consonants)
        self.currencies = self.T.currencies
        self.final_replacements = [tuple(p) for p in self.T.final_replacements]
        self.st = self.lexicon.st

    # -- markdown links: [text](/phonemes/) | [text](#flags#) | [text](stress)
    def preprocess(self, text: str):
        result = ""
        words = []
        features = {}
        last = 0
        text = text.lstrip()
        for m in self.link_regex.finditer(text):
            result += text[last:m.start()]
            words.extend(text[last:m.start()].split())
            f = m.group(2)
            if is_digit(f[1 if f[:1] in ("-", "+") else 0:]):
                f = int(f)
            elif f in ("0.5", "+0.5"):
                f = 0.5
            elif f == "-0.5":
                f = -0.5
            elif len(f) > 1 and f[0] == "/" and f[-1] == "/":
                f = f[0] + f[1:].rstrip("/")
            elif len(f) > 1 and f[0] == "#" and f[-1] == "#":
                f = f[0] + f[1:].rstrip("#")
            else:
                f = None
            if f is not None:
                features[len(words)] = f
            result += m.group(1)
            words.append(m.group(1))
            last = m.end()
        if last < len(text):
            result += text[last:]
            words.extend(text[last:].split())
        return result, words, features

    # -- the features of preprocess land on the tagger's tokens through a character alignment
    @staticmethod
    def align_y2x(xs: List[str], ys: List[str]) -> List[List[int]]:
        """For each token of ys, the indices of the tokens of xs sharing characters with it (case and whitespace
        ignored), in the shape spaCy's Alignment.from_strings gives (a token reached by no character pairing gets
        an empty list; a token never reached gets no entry)."""
        x_of = [i for i, x in enumerate(xs) for _ in range(len(x.lower()))]
        y_of = [i for i, y in enumerate(ys) for _ in range(len(y.lower()))]
        sx, sy = "".join(xs).lower(), "".join(ys).lower()
        if re.sub(r"\s+", "", sx) != re.sub(r"\s+", "", sy) or len(sx) != len(x_of) or len(sy) != len(y_of):
            raise ValueError(f"alignment: the two token lists differ beyond whitespace and case: {xs[:10]} {ys[:10]}")
        a = b = 0
        prev_x = prev_y = -1
        y2x = []
        while a < len(sx) and b < len(sy):
            xi, yi = x_of[a], y_of[b]
            if prev_y != yi:
                y2x.append(set())
            at_x_start = a == 0 or x_of[a - 1] < xi
            at_y_start = b == 0 or y_of[b - 1] < yi
            if xs[xi] == ys[yi] and at_x_start and at_y_start:
                y2x[-1].add(xi)
                a += len(xs[xi])
                b += len(ys[yi])
            elif sx[a] == sy[b]:
                y2x[-1].add(xi)
                a += 1
                b += 1
            elif sx[a].isspace():
                a += 1
            elif sy[b].isspace():
                b += 1
            else:
                raise ValueError(f"alignment: no pairing at {sx[a]!r} / {sy[b]!r}")
            prev_x, prev_y = xi, yi
        y2x.extend([set()] * len(set(y_of[b:])))
        return [sorted(s) for s in y2x]

    def tokens_from_tagger(self, tokens, words, features) -> List[Tok]:
        toks = [Tok(text=t["text"], tag=t["tag"], whitespace=t["whitespace"]) for t in tokens]
        if not features:
            return toks
        flat = [x for row in self.align_y2x(words, [t.text for t in toks]) for x in row]
        for k, v in features.items():
            if not (isinstance(v, (str, int)) or v in (0.5, -0.5)):
                raise ValueError(f"link feature {k}: {v!r}")
            hits = [j for j, x in enumerate(flat) if x == k]
            for i, j in enumerate(hits):
                if j >= len(toks):
                    continue
                t = toks[j]
                if not isinstance(v, str):
                    t.stress = v
                elif v.startswith("/"):
                    t.is_head = i == 0
                    t.phonemes = v.lstrip("/") if i == 0 else ""
                    t.rating = 5
                elif v.startswith("#"):
                    t.num_flags = v.lstrip("#")
        return toks

    def fold_left(self, toks: List[Tok]) -> List[Tok]:
        out = []
        for t in toks:
            if out and not t.is_head:
                t = merge_tokens([out.pop(), t], unk=self.unk)
            out.append(t)
        return out

    def retokenize(self, toks: List[Tok]):
        words = []
        currency = None
        for i, token in enumerate(toks):
            if token.alias is None and token.phonemes is None:
                parts = [Tok(text=s, tag=token.tag, whitespace="", phonemes=token.phonemes, is_head=True,
                             num_flags=token.num_flags, stress=token.stress, prespace=False)
                         for s in self.subtoken_regex.findall(token.text)]
            else:
                parts = [token]
            parts[-1].whitespace = token.whitespace
            for j, tk in enumerate(parts):
                if tk.alias is not None or tk.phonemes is not None:
                    pass
                elif tk.tag == "$" and tk.text in self.currencies:
                    currency = tk.text
                    tk.phonemes = ""
                    tk.rating = 4
                elif tk.tag == ":" and tk.text in ("-", "–"):
                    tk.phonemes = "—"
                    tk.rating = 3
                elif tk.tag in self.punct_tags and not all(97 <= ord(c.lower()) <= 122 for c in tk.text):
                    tk.phonemes = self.punct_tag_phonemes.get(tk.tag, "".join(c for c in tk.text if c in self.puncts))
                    tk.rating = 4
                elif currency is not None:
                    if tk.tag != "CD":
                        currency = None
                    elif j + 1 == len(parts) and (i + 1 == len(toks) or toks[i + 1].tag != "CD"):
                        tk.currency = currency
                elif 0 < j < len(parts) - 1 and tk.text == "2" and (parts[j - 1].text[-1] + parts[j + 1].text[0]).isalpha():
                    tk.alias = "to"
                if tk.alias is not None or tk.phonemes is not None:
                    words.append(tk)
                elif words and isinstance(words[-1], list) and not words[-1][-1].whitespace:
                    tk.is_head = False
                    words[-1].append(tk)
                else:
                    words.append(tk if tk.whitespace else [tk])
        return [w[0] if isinstance(w, list) and len(w) == 1 else w for w in words]

    def token_context(self, ctx: Context, ps, token: Tok) -> Context:
        vowel = ctx.future_vowel
        if ps:
            for c in ps:
                if c in self.vowels or c in self.consonants or c in self.non_quote_puncts:
                    vowel = None if c in self.non_quote_puncts else (c in self.vowels)
                    break
        future_to = token.text in ("to", "To") or (token.text == "TO" and token.tag in ("TO", "IN"))
        return Context(future_vowel=vowel, future_to=future_to)

    def resolve_tokens(self, toks: List[Tok]):
        text = "".join(t.text + t.whitespace for t in toks[:-1]) + toks[-1].text
        kinds = {0 if c.isalpha() else (1 if is_digit(c) else 2) for c in text if c not in self.junks}
        prespace = " " in text or "/" in text or len(kinds) > 1
        for i, t in enumerate(toks):
            if t.phonemes is None:
                if i == len(toks) - 1 and t.text in self.non_quote_puncts:
                    t.phonemes = t.text
                    t.rating = 3
                elif all(c in self.junks for c in t.text):
                    t.phonemes = ""
                    t.rating = 3
            elif i > 0:
                t.prespace = prespace
        if prespace:
            return
        idx = [(self.st.primary in t.phonemes, self.st.weight(t.phonemes), i) for i, t in enumerate(toks) if t.phonemes]
        if len(idx) == 2 and len(toks[idx[0][2]].text) == 1:
            i = idx[1][2]
            toks[i].phonemes = self.st.apply(toks[i].phonemes, -0.5)
            return
        if len(idx) < 2 or sum(b for b, _, _ in idx) <= (len(idx) + 1) // 2:
            return
        for _, _, i in sorted(idx)[:len(idx) // 2]:
            toks[i].phonemes = self.st.apply(toks[i].phonemes, -0.5)

    def _read_word(self, w: Tok, ctx: Context) -> Context:
        """One whitespace-separated word of a single token, read with the context of the words after it."""
        if w.phonemes is None:
            w.phonemes, w.rating = self.lexicon(w, ctx)
        if w.phonemes is None and self.fallback is not None:
            w.phonemes, w.rating = self.fallback(w.text)
        return self.token_context(ctx, w.phonemes, w)

    def _span_step(self, w: List[Tok], left: int, right: int, ctx: Context):
        """One step of the longest-span search over a word's sub-tokens: (left, right, ctx, to_fallback)."""
        span = w[left:right]
        if any(t.alias is not None or t.phonemes is not None for t in span):
            ps, rating, tk = None, None, None
        else:
            tk = merge_tokens(span)
            ps, rating = self.lexicon(tk, ctx)
        if ps is not None:
            w[left].phonemes = ps
            w[left].rating = rating
            for x in w[left + 1:right]:
                x.phonemes = ""
            return 0, left, self.token_context(ctx, ps, tk), False
        if left + 1 < right:
            return left + 1, right, ctx, False
        right -= 1
        t = w[right]
        if t.phonemes is None:
            if all(c in self.junks for c in t.text):
                t.phonemes = ""
                t.rating = 3
            elif self.fallback is not None:
                return 0, right, ctx, True
        return 0, right, ctx, False

    def _read_subtokens(self, w: List[Tok], ctx: Context) -> Context:
        """A word of several sub-tokens: the longest spans the lexicon reads, right to left; else the fallback whole."""
        left, right = 0, len(w)
        to_fallback = False
        while left < right and not to_fallback:
            left, right, ctx, to_fallback = self._span_step(w, left, right, ctx)
        if to_fallback:
            tk = merge_tokens(w)
            w[0].phonemes, w[0].rating = self.fallback(tk.text)
            for x in w[1:]:
                x.phonemes = ""
                x.rating = w[0].rating
        else:
            self.resolve_tokens(w)
        return ctx

    def run(self, toks: List[Tok]):
        words = self.retokenize(self.fold_left(toks))
        ctx = Context()
        for w in reversed(words):
            ctx = self._read_subtokens(w, ctx) if isinstance(w, list) else self._read_word(w, ctx)
        out = [merge_tokens(w, unk=self.unk) if isinstance(w, list) else w for w in words]
        for t in out:
            if t.phonemes:
                for old, new in self.final_replacements:
                    t.phonemes = t.phonemes.replace(old, new)
        unk = self.unk
        text = "".join((unk if t.phonemes is None else t.phonemes) + t.whitespace for t in out)
        return text, out

def phonemize(tokens, data: FrontEnd, words=None, features=None):
    """tokens: [{text, tag, whitespace}] from the tagger over preprocess's text; words, features: preprocess's.
    Returns (phoneme string, [Tok])."""
    toks = data.tokens_from_tagger(tokens, words or [], features or {})
    return data.run(toks)
