"""Pure-Python interpreter of HuggingFace ``tokenizer.json`` (Zero Outsider / R34).

``PyTokenizer`` is a clean-room, stdlib-only drop-in replacement for the subset
of ``tokenizers.Tokenizer`` that NeuroBrix uses at inference. It reads a
HuggingFace ``tokenizer.json`` and reproduces, byte-for-byte, the Rust library's
``encode(...).ids`` / ``decode(ids)`` for the model families NeuroBrix ships.

The runtime engine must not import the Rust ``tokenizers`` library (R34): the
tokenizer is part of the model and is interpreted from the ``.nbx``-embedded
``tokenizer.json`` by this module. Only Python stdlib is used.

Faithfully implemented components (data-driven, read from the JSON):

* normalizers: ``Sequence``, ``NFC``, ``NFD``, ``NFKC``, ``NFKD``, ``Replace``,
  ``Prepend``, ``Lowercase``, ``Strip``, ``NFC/Replace`` combos (Sana, TinyLlama).
* pre_tokenizers: ``ByteLevel``, ``Split`` (Regex/String, all behaviors),
  ``Sequence``, ``Digits``, ``Whitespace``, ``Metaspace``, ``None``.
* model: ``BPE`` with ``byte_fallback`` (Llama/Sana ``<0xXX>`` + Metaspace ``▁``)
  and byte-level BPE (GPT-2/Qwen ``bytes_to_unicode``), ``ignore_merges``,
  ``fuse_unk``, ``unk_token``.
* decoders: ``ByteLevel``, ``Sequence``, ``Replace``, ``Strip``, ``Fuse``,
  ``ByteFallback``, ``Metaspace``.
* post_processors: ``TemplateProcessing`` (BOS/EOS/special), ``ByteLevel``
  (id no-op), ``Sequence``.
* added_tokens: longest-match verbatim split before normalization, with
  ``lstrip``/``rstrip``/``normalized`` flags; ``special`` flag drives
  ``skip_special_tokens`` in decode.

Every regex of the file (``Split``, ``Replace``) is read with the meaning the
vendor's engine gives it and rewritten into stdlib ``re`` (``translate_regex``):
``\\p{..}`` general categories become ``unicodedata``-derived classes, ``\\s``
follows Unicode White_Space, possessive quantifiers and atomic groups never
backtrack, ``^``/``$`` anchor lines -- the third-party ``regex`` module is never
imported.
"""

from __future__ import annotations

import heapq
import json
import re
import unicodedata
from functools import lru_cache
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# GPT-2 byte<->unicode map (shared by ByteLevel pre-tokenizer/decoder)
# ---------------------------------------------------------------------------
def _bytes_to_unicode() -> Dict[int, str]:
    """GPT-2 reversible byte->unicode map (256 printable code points)."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


_BYTE_ENCODER = _bytes_to_unicode()
_BYTE_DECODER = {v: k for k, v in _BYTE_ENCODER.items()}


# ---------------------------------------------------------------------------
# Vendor regex -> stdlib re. The vendor engines (Oniguruma behind tokenizer.json,
# the `regex` module behind tiktoken / Tekken) take syntax stdlib `re` refuses
# (\p{..} classes, possessive quantifiers, atomic groups) or reads differently
# (\s, line anchors). Each construct is rewritten into stdlib `re` with the
# vendor's meaning; a construct the standard library cannot express (a script
# or binary property) is a named error, never a silent approximation.
# ---------------------------------------------------------------------------
# Unicode general categories by every name the vendors accept (PropertyValueAliases).
_GC_NAMES = {
    "letter": "L", "casedletter": "LC", "uppercaseletter": "Lu", "lowercaseletter": "Ll",
    "titlecaseletter": "Lt", "modifierletter": "Lm", "otherletter": "Lo",
    "mark": "M", "combiningmark": "M", "nonspacingmark": "Mn", "spacingmark": "Mc", "enclosingmark": "Me",
    "number": "N", "decimalnumber": "Nd", "letternumber": "Nl", "othernumber": "No",
    "punctuation": "P", "connectorpunctuation": "Pc", "dashpunctuation": "Pd", "openpunctuation": "Ps",
    "closepunctuation": "Pe", "initialpunctuation": "Pi", "finalpunctuation": "Pf", "otherpunctuation": "Po",
    "symbol": "S", "mathsymbol": "Sm", "currencysymbol": "Sc", "modifiersymbol": "Sk", "othersymbol": "So",
    "separator": "Z", "spaceseparator": "Zs", "lineseparator": "Zl", "paragraphseparator": "Zp",
    "other": "C", "control": "Cc", "format": "Cf", "surrogate": "Cs", "privateuse": "Co", "unassigned": "Cn",
}
_GC_NAMES.update({code.lower(): code for code in set(_GC_NAMES.values())})
# Unicode White_Space: the vendors' \s (stdlib `re` also counts U+001C..U+001F).
_WHITE_SPACE = frozenset((0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20, 0x85, 0xA0, 0x1680, 0x2028, 0x2029,
                          0x202F, 0x205F, 0x3000, *range(0x2000, 0x200B)))
_CLASS_BODIES: Dict[Tuple[str, bool], str] = {}


def _class_body(name: str, negated: bool) -> str:
    """The ``[...]`` body (``\\Uxxxxxxxx`` ranges) of a general category, or of
    White_Space for ``name == "s"``; ``negated`` gives its complement."""
    key = (name, negated)
    if key not in _CLASS_BODIES:
        if name == "s":
            member = lambda cp: cp in _WHITE_SPACE  # noqa: E731
        else:
            code = _GC_NAMES.get(name.lower().replace("_", "").replace(" ", "").replace("-", ""))
            if code is None:
                raise ValueError(
                    f"tokenizer pattern uses the Unicode property \\p{{{name}}}: only general categories "
                    f"(L, Lu, M, Mn, N, P, S, Z, C ... and their long names) can be built from the standard "
                    f"library; a script or binary property needs its code point table added to the regex translator")
            cats = {"LC": ("Lu", "Ll", "Lt")}.get(code, (code,))
            member = lambda cp: unicodedata.category(chr(cp)).startswith(cats)  # noqa: E731
        parts, start = [], None
        for cp in range(0x110001):
            inside = cp <= 0x10FFFF and member(cp) != negated
            if inside and start is None:
                start = cp
            elif not inside and start is not None:
                parts.append("\\U%08x" % start if start == cp - 1 else "\\U%08x-\\U%08x" % (start, cp - 1))
                start = None
        _CLASS_BODIES[key] = "".join(parts)
    return _CLASS_BODIES[key]


_INTERVAL = re.compile(r"\{\d*,?\d*\}")


class _RegexTranslator:
    r"""Rewrites one vendor regex into a stdlib ``re`` pattern with the same matches.

    ``onig=True`` reads it as Oniguruma (tokenizer.json): ``^``/``$`` are line
    anchors, ``\z``/``\Z`` its string ends, and ``{m,n}+`` repeats the interval.
    ``onig=False`` reads it as the ``regex`` module (tiktoken, Tekken): ``^``/``$``
    keep their stdlib meaning and ``{m,n}+`` is possessive. Both: ``\p{X}``/
    ``\P{X}``/``\p{^X}`` become explicit code point classes, ``\s``/``\S`` follow
    Unicode White_Space, and possessive quantifiers and atomic groups ``(?>...)``
    become an atomic look-ahead capture re-read by a back-reference,
    ``(?=(?P<a>X))(?P=a)``, which never backtracks.
    """

    def __init__(self, pattern: str, onig: bool):
        self.pattern, self.onig = pattern, onig
        self.out: List[str] = []              # one entry per atom; a quantifier joins its atom
        self.groups: List[Tuple[int, bool]] = []
        self.quantifier = ""                  # the quantifier that ended out[-1]: "?", "*", "+", "{", "lazy"
        self.atomic_ids = 0

    def translate(self) -> str:
        i = 0
        while i < len(self.pattern):
            i = self._step(i)
        return "".join(self.out)

    def _step(self, i: int) -> int:
        c = self.pattern[i]
        if c in "?*+{" and self.quantifier:
            return self._quantified_again(i, c)
        if c in "?*+" and self.out:
            return self._quantify(i, c, i + 1)
        interval = _INTERVAL.match(self.pattern, i) if c == "{" and self.out else None
        if interval:
            return self._quantify(i, "{", interval.end())
        self.quantifier = ""
        if c == "\\" and i + 1 < len(self.pattern):
            atom, i = self._escape(i, in_class=False)
            self.out.append(atom)
            return i
        if c == "[":
            return self._char_class(i)
        if c == "(":
            return self._open_group(i)
        if c == ")" and self.groups:
            start, is_atomic = self.groups.pop()
            group = "".join(self.out[start:]) + ")"
            del self.out[start:]
            self.out.append(self._atomic(group) if is_atomic else group)
            return i + 1
        if self.onig and c in "^$":
            c = "(?<![^\\n])" if c == "^" else "(?![^\\n])"
        self.out.append(c)
        return i + 1

    def _quantify(self, i: int, kind: str, end: int) -> int:
        self.out[-1] += self.pattern[i:end]
        self.quantifier = kind
        return end

    def _quantified_again(self, i: int, c: str) -> int:
        """A quantifier right after a quantifier: lazy, possessive, or a repeat of the quantified atom."""
        if c == "?" and self.quantifier != "lazy":     # lazy keeps its stdlib meaning
            return self._quantify(i, "lazy", i + 1)
        interval, self.quantifier = self.quantifier == "{", ""
        if c == "+" and not (self.onig and interval):
            self.out[-1] = self._atomic(self.out[-1])  # possessive
            return i + 1
        self.out[-1] = "(?:" + self.out[-1] + ")"      # re-read c as the quantifier of the group
        return i

    def _atomic(self, body: str) -> str:
        self.atomic_ids += 1
        return f"(?=(?P<_atomic{self.atomic_ids}>{body}))(?P=_atomic{self.atomic_ids})"

    def _open_group(self, i: int) -> int:
        """``(``, ``(?>`` (atomic) or a ``(?...`` opener (non-capturing, flags, look-around, named) kept verbatim."""
        p = self.pattern
        is_atomic = p.startswith("(?>", i)
        opener = re.match(r"\((?:\?(?:>|P?<[A-Za-z_]\w*>|<[=!]|[=!:]|[a-zA-Z-]*:))?", p[i:]).group(0)
        self.groups.append((len(self.out), is_atomic))
        self.out.append("(?:" if is_atomic else opener)
        return i + len(opener)

    def _char_class(self, i: int) -> int:
        p = self.pattern
        j = i + 1 + (p[i + 1:i + 2] == "^")
        j += p[j:j + 1] == "]"                          # a ] first is a member
        body = [p[i:j]]
        while p[j] != "]":
            atom, j = self._escape(j, in_class=True) if p[j] == "\\" else (p[j], j + 1)
            body.append(atom)
        self.out.append("".join(body) + "]")
        return j + 1

    def _escape(self, j: int, in_class: bool) -> Tuple[str, int]:
        p = self.pattern
        two, end = p[j:j + 2], j + 2
        if two in ("\\p", "\\P"):
            end = p.index("}", j + 3) + 1
            name = p[j + 3:end - 1]
            negated = (two == "\\P") != name.startswith("^")
            name = name.lstrip("^")
        elif two in ("\\s", "\\S"):
            name, negated = "s", two == "\\S"
        elif self.onig and not in_class and two in ("\\z", "\\Z"):
            return ("\\Z" if two == "\\z" else "(?=\\n?\\Z)"), end
        else:
            return two, end
        if in_class:
            return _class_body(name, negated), end
        return ("[^" if negated else "[") + _class_body(name, False) + "]", end


def translate_regex(pattern: str, onig: bool = True) -> str:
    """A vendor regex rewritten into stdlib ``re`` with the vendor's matches (see :class:`_RegexTranslator`)."""
    return _RegexTranslator(pattern, onig).translate()


@lru_cache(maxsize=None)
def compile_regex(pattern: str) -> "re.Pattern":
    """Compile a vendor regex through :func:`translate_regex`; a construct the
    translation does not express is a named error that quotes the pattern."""
    try:
        return re.compile(translate_regex(pattern))
    except (re.error, IndexError) as exc:
        raise ValueError(
            f"tokenizer pattern {pattern!r} uses a construct the stdlib regex translation does not express "
            f"({exc}); add it to the regex translator of the tokenizer module") from exc



# ---------------------------------------------------------------------------
# Encoding result object (mirrors tokenizers.Encoding subset)
# ---------------------------------------------------------------------------
class _Encoding:
    __slots__ = ("ids", "tokens")

    def __init__(self, ids: List[int], tokens: List[str]):
        self.ids = ids
        self.tokens = tokens


# ---------------------------------------------------------------------------
# Normalizers
# ---------------------------------------------------------------------------
class _Normalizer:
    """Applies a (possibly nested) normalizer spec to a string."""

    def __init__(self, spec: Optional[dict]):
        self.spec = spec

    def __call__(self, text: str, prefix: bool = True) -> str:
        """`prefix` False: the segment does not start the input and the tokenizer prefixes only the input's start
        (`PyTokenizer.prefix_only_at_start`), so a `Prepend` step leaves it as it is."""
        self._prefix = prefix
        return self._apply(self.spec, text)

    def _apply(self, spec: Optional[dict], text: str) -> str:
        if spec is None:
            return text
        t = spec.get("type")
        if t == "Sequence":
            for sub in spec.get("normalizers", []):
                text = self._apply(sub, text)
            return text
        if t == "NFC":
            return unicodedata.normalize("NFC", text)
        if t == "NFD":
            return unicodedata.normalize("NFD", text)
        if t == "NFKC":
            return unicodedata.normalize("NFKC", text)
        if t == "NFKD":
            return unicodedata.normalize("NFKD", text)
        if t == "Lowercase":
            return text.lower()
        if t == "Prepend":
            return spec.get("prepend", "") + text if getattr(self, "_prefix", True) else text
        if t == "Replace":
            return self._replace(spec, text)
        if t == "Strip":
            left = spec.get("strip_left", True)
            right = spec.get("strip_right", True)
            if left:
                text = text.lstrip()
            if right:
                text = text.rstrip()
            return text
        # Unknown normalizer (BertNormalizer, Nmt, ...): identity (data-driven; none of our models hit this)
        return text

    @staticmethod
    def _replace(spec: dict, text: str) -> str:
        pat = spec.get("pattern", {})
        content = spec.get("content", "")
        if "String" in pat:
            return text.replace(pat["String"], content)
        if "Regex" in pat:
            return compile_regex(pat["Regex"]).sub(content, text)
        return text


# ---------------------------------------------------------------------------
# Pre-tokenizers
# ---------------------------------------------------------------------------
class _PreTokenizer:
    """Turns a normalized string into a list of word pieces (pre-tokens).

    Each piece is a plain ``str``. ByteLevel pieces are already byte-mapped into
    the GPT-2 unicode alphabet; byte_fallback (Metaspace) pieces are raw text.
    """

    def __init__(self, spec: Optional[dict]):
        self.spec = spec

    def __call__(self, text: str, at_start: bool = True, prefix_only_at_start: bool = False) -> List[str]:
        """`at_start`: the segment begins the input (offset 0). Metaspace's `first` scheme prefixes only such a segment,
        and so does `always` when the tokenizer prefixes only the input's start (`PyTokenizer.prefix_only_at_start`)."""
        self._at_start, self._prefix_only_at_start = at_start, prefix_only_at_start
        pieces = [text]
        pieces = self._apply(self.spec, pieces)
        return pieces

    # -- dispatch ----------------------------------------------------------
    def _apply(self, spec: Optional[dict], pieces: List[str]) -> List[str]:
        if spec is None:
            return pieces
        t = spec.get("type")
        if t == "Sequence":
            for sub in spec.get("pretokenizers", []):
                pieces = self._apply(sub, pieces)
            return pieces
        if t == "Split":
            return self._split_all(spec, pieces)
        if t == "Digits":
            return self._digits_all(spec, pieces)
        if t == "ByteLevel":
            return self._bytelevel_all(spec, pieces)
        if t == "Whitespace":
            return self._whitespace_all(pieces)
        if t == "WhitespaceSplit":
            out: List[str] = []
            for p in pieces:
                out.extend(p.split())
            return out
        if t == "Metaspace":
            return self._metaspace_all(spec, pieces)
        if t == "BertPreTokenizer":
            return self._bert_all(pieces)
        # Unknown pre-tokenizer: identity
        return pieces

    # -- Split -------------------------------------------------------------
    def _split_compiled(self, spec: dict) -> Tuple[Optional["re.Pattern"], Optional[str]]:
        pat = spec.get("pattern", {})
        if "Regex" in pat:
            return compile_regex(pat["Regex"]), None
        if "String" in pat:
            return None, pat["String"]
        return None, None

    def _split_all(self, spec: dict, pieces: List[str]) -> List[str]:
        behavior = spec.get("behavior", "Isolated")
        invert = spec.get("invert", False)
        rx, literal = self._split_compiled(spec)
        out: List[str] = []
        for p in pieces:
            out.extend(self._split_one(p, rx, literal, behavior, invert))
        return out

    @staticmethod
    def _matches(text: str, rx, literal: Optional[str]):
        """Yield (start, end) match spans (char offsets)."""
        if rx is not None:
            for m in rx.finditer(text):
                s, e = m.start(), m.end()
                if e > s:  # skip zero-width matches (onig behavior)
                    yield s, e
        elif literal is not None and literal != "":
            start = 0
            while True:
                idx = text.find(literal, start)
                if idx < 0:
                    break
                yield idx, idx + len(literal)
                start = idx + len(literal)

    def _split_one(self, text: str, rx, literal, behavior: str, invert: bool) -> List[str]:
        if text == "":
            return []
        spans = list(self._matches(text, rx, literal))
        if invert:
            # matched regions become the "gaps" and vice versa
            spans = self._complement(spans, len(text))
        if not spans:
            return [text]

        out: List[str] = []
        cursor = 0
        if behavior == "Isolated":
            for s, e in spans:
                if s > cursor:
                    out.append(text[cursor:s])
                out.append(text[s:e])
                cursor = e
            if cursor < len(text):
                out.append(text[cursor:])
        elif behavior == "Removed":
            for s, e in spans:
                if s > cursor:
                    out.append(text[cursor:s])
                cursor = e
            if cursor < len(text):
                out.append(text[cursor:])
        elif behavior == "MergedWithPrevious":
            # delimiter attaches to the preceding gap
            for s, e in spans:
                out.append(text[cursor:e])
                cursor = e
            if cursor < len(text):
                out.append(text[cursor:])
        elif behavior == "MergedWithNext":
            # delimiter attaches to the following gap
            for s, e in spans:
                if s > cursor:
                    out.append(text[cursor:s])
                cursor = s
            out.append(text[cursor:])
        elif behavior == "Contiguous":
            # collapse consecutive matches; emit gap, then merged-match run
            for s, e in spans:
                if s > cursor:
                    out.append(text[cursor:s])
                out.append(text[s:e])
                cursor = e
            if cursor < len(text):
                out.append(text[cursor:])
        else:
            return [text]
        return [x for x in out if x != ""]

    @staticmethod
    def _complement(spans: List[Tuple[int, int]], length: int) -> List[Tuple[int, int]]:
        comp: List[Tuple[int, int]] = []
        cursor = 0
        for s, e in spans:
            if s > cursor:
                comp.append((cursor, s))
            cursor = e
        if cursor < length:
            comp.append((cursor, length))
        return comp

    # -- Digits ------------------------------------------------------------
    def _digits_all(self, spec: dict, pieces: List[str]) -> List[str]:
        individual = spec.get("individual_digits", False)
        out: List[str] = []
        for p in pieces:
            out.extend(self._digits_one(p, individual))
        return out

    @staticmethod
    def _digits_one(text: str, individual: bool) -> List[str]:
        if text == "":
            return []
        out: List[str] = []
        buf = []
        mode = None  # "digit" / "other"
        def flush():
            if buf:
                out.append("".join(buf))
                buf.clear()
        for ch in text:
            # Rust Digits uses char::is_numeric() == any Unicode number category
            # (Nd / Nl / No); CJK ideographic numerals (Lo) are NOT digits.
            is_digit = unicodedata.category(ch)[0] == "N"
            if individual:
                if is_digit:
                    flush()
                    out.append(ch)
                    mode = None
                else:
                    if mode != "other":
                        flush()
                        mode = "other"
                    buf.append(ch)
            else:
                cur = "digit" if is_digit else "other"
                if mode is not None and cur != mode:
                    flush()
                mode = cur
                buf.append(ch)
        flush()
        return out

    # -- ByteLevel ---------------------------------------------------------
    @staticmethod
    def _gpt2_split(text: str) -> List[str]:
        """The canonical GPT-2 ByteLevel regex split (use_regex=true)."""
        rx = _GPT2_BYTELEVEL_RX
        return [m.group() for m in rx.finditer(text)]

    def _bytelevel_all(self, spec: dict, pieces: List[str]) -> List[str]:
        add_prefix_space = spec.get("add_prefix_space", False)
        use_regex = spec.get("use_regex", True)
        out: List[str] = []
        first = True
        for p in pieces:
            seg = p
            if add_prefix_space and seg and not seg[0].isspace() and first:
                seg = " " + seg
            first = False
            if use_regex:
                subs = self._gpt2_split(seg) if seg else []
            else:
                subs = [seg] if seg != "" else []
            for sub in subs:
                mapped = "".join(_BYTE_ENCODER[b] for b in sub.encode("utf-8"))
                if mapped != "":
                    out.append(mapped)
        return out

    # -- Whitespace (Rust: \w+|[^\w\s]+) -----------------------------------
    def _whitespace_all(self, pieces: List[str]) -> List[str]:
        rx = _WHITESPACE_RX
        out: List[str] = []
        for p in pieces:
            out.extend(m.group() for m in rx.finditer(p))
        return out

    def _bert_all(self, pieces: List[str]) -> List[str]:
        out: List[str] = []
        for p in pieces:
            out.extend(_BERT_RX.findall(p))
        return out

    # -- Metaspace ---------------------------------------------------------
    def _metaspace_all(self, spec: dict, pieces: List[str]) -> List[str]:
        replacement = spec.get("replacement", "▁")
        prepend_scheme = spec.get("prepend_scheme", "always")
        add_prefix_space = spec.get("add_prefix_space", True)
        split = spec.get("split", True)
        out: List[str] = []
        first = True
        at_start = getattr(self, "_at_start", True)
        if prepend_scheme == "first" or getattr(self, "_prefix_only_at_start", False):
            first = at_start
        for p in pieces:
            seg = p
            if (add_prefix_space or prepend_scheme in ("always", "first")) and first:
                if not seg.startswith(" "):
                    seg = " " + seg
            first = False
            seg = seg.replace(" ", replacement)
            if split:
                # split on replacement, keeping it with the following piece
                parts = re.split(f"(?={re.escape(replacement)})", seg)
                out.extend(x for x in parts if x != "")
            else:
                if seg != "":
                    out.append(seg)
        return out


# GPT-2 ByteLevel regex (Rust ByteLevel uses this exact pattern with onig).
_GPT2_BYTELEVEL_RX = compile_regex(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")

# Rust Whitespace pre-tokenizer: \w+|[^\w\s]+  (Unicode-aware \w), \w read as \p{L} | \p{N} | _.
_WHITESPACE_RX = compile_regex(r"[\p{L}\p{N}_]+|[^\p{L}\p{N}_\s]+")

# Rust BertPreTokenizer: split on whitespace then isolate punctuation.
_BERT_RX = re.compile(r"\w+|[^\w\s]")


# ---------------------------------------------------------------------------
# BPE model
# ---------------------------------------------------------------------------
# The reference's word cache (tokenizers v0.23.3, utils/cache.rs; models/bpe/model.rs `tokenize_with_cache`): a word is
# kept only when it is shorter than MAX_LENGTH bytes and the cache holds fewer than DEFAULT_CACHE_CAPACITY words; a full
# cache keeps what it has. A tokenizer.json does not carry them: they are the reproduced library's, not the model's.
#: the reference's MAX_LENGTH (bytes of the word's UTF-8)
CACHE_MAX_WORD_BYTES = 256
#: the reference's DEFAULT_CACHE_CAPACITY (words)
CACHE_MAX_WORDS = 10_000


class _BPE:
    def __init__(self, model: dict):
        self.vocab: Dict[str, int] = model.get("vocab", {})
        self.id_to_tok: Dict[int, str] = {v: k for k, v in self.vocab.items()}
        merges = model.get("merges", [])
        self.ranks: Dict[Tuple[str, str], int] = {}
        for i, m in enumerate(merges):
            if isinstance(m, (list, tuple)):
                a, b = m[0], m[1]
            else:
                # space-separated; split on the FIRST space only (tokens may
                # themselves contain spaces? GPT-2 merges never do, but the
                # SentencePiece byte-fallback merges are token pairs joined by
                # a single space — split once from the left is correct).
                parts = m.split(" ")
                if len(parts) != 2:
                    # be robust: rsplit fallback
                    parts = m.split(" ", 1)
                a, b = parts[0], parts[1]
            self.ranks[(a, b)] = i
        self.unk_token: Optional[str] = model.get("unk_token")
        self.fuse_unk: bool = bool(model.get("fuse_unk", False))
        self.byte_fallback: bool = bool(model.get("byte_fallback", False))
        self.ignore_merges: bool = bool(model.get("ignore_merges", False))
        self.cont_prefix: str = model.get("continuing_subword_prefix") or ""
        self.end_suffix: str = model.get("end_of_word_suffix") or ""
        self._cache: Dict[str, List[str]] = {}

    def _bpe(self, token: str) -> List[str]:
        cached = self._cache.get(token)
        if cached is not None:
            return cached
        # tokenizers' BPE::merge_word (0.23.2, models/bpe/model.rs): every character
        # after the first carries continuing_subword_prefix, the last one carries
        # end_of_word_suffix (CLIP: "a" -> "a</w>", id 320, not the mid-word "a", 64).
        word = list(token)
        if self.cont_prefix:
            word = word[:1] + [self.cont_prefix + c for c in word[1:]]
        if self.end_suffix:
            word[-1] = word[-1] + self.end_suffix
        # tokenizers' Word::merge_all (v0.23.3, models/bpe/word.rs): a min-heap of (rank, position) over a linked list
        # of symbols, the lowest rank then the leftmost merged first. A text with no pre-tokenizer (Llama's Metaspace
        # normalizer) is one word: rescanning every pair for each merge was quadratic in its length.
        n = len(word)
        nxt, prv = list(range(1, n + 1)), list(range(-1, n - 1))      # n / -1: no neighbour
        heap = [(self.ranks[p], i) for i, p in enumerate(zip(word, word[1:])) if p in self.ranks]
        heapq.heapify(heap)
        while heap:
            rank, i = heapq.heappop(heap)
            j = nxt[i]
            if j == n or word[i] is None or self.ranks.get((word[i], word[j])) != rank:
                continue        # expired: a side of this pair was merged since it was pushed
            # the merged token drops the right side's continuing_subword_prefix
            # (tokenizers' BpeBuilder::build: `&b[prefix_len..]`)
            word[i], word[j], nxt[i] = word[i] + word[j][len(self.cont_prefix):], None, nxt[j]
            if nxt[i] < n:
                prv[nxt[i]] = i
            for a, b in ((prv[i], i), (i, nxt[i])):       # the two pairs the merged symbol forms with its neighbours
                if a >= 0 and b < n and (word[a], word[b]) in self.ranks:
                    heapq.heappush(heap, (self.ranks[(word[a], word[b])], a))
        word = [w for w in word if w is not None]
        # bounded as the reference's: a Llama-style text is one word, which an unbounded cache kept for the process's life
        if len(self._cache) < CACHE_MAX_WORDS and len(token.encode("utf-8")) < CACHE_MAX_WORD_BYTES:
            self._cache[token] = word
        return word

    def tokenize_piece(self, piece: str) -> List[str]:
        """Tokenize a single pre-token piece into vocab tokens (as strings)."""
        if piece == "":
            return []
        if self.ignore_merges and piece in self.vocab:
            return [piece]
        symbols = self._bpe(piece)
        out: List[str] = []
        pending_unk = False
        for sym in symbols:
            if sym in self.vocab:
                out.append(sym)
                pending_unk = False
                continue
            # symbol not in vocab
            if self.byte_fallback:
                replaced = self._byte_fallback(sym)
                if replaced is not None:
                    out.extend(replaced)
                    pending_unk = False
                    continue
            if self.unk_token is not None:
                if self.fuse_unk and pending_unk:
                    # fuse consecutive unks into one
                    pass
                else:
                    out.append(self.unk_token)
                    pending_unk = True
            # if no unk_token, the symbol is dropped (matches tokenizers when
            # there is genuinely no representation — should not happen for our
            # byte-level models since every byte maps).
        return out

    def _byte_fallback(self, sym: str) -> Optional[List[str]]:
        """Replace an unknown symbol by its ``<0xXX>`` byte tokens if all
        such tokens exist in the vocab; else None."""
        raw = sym.encode("utf-8")
        toks = [f"<0x{b:02X}>" for b in raw]
        if all(t in self.vocab for t in toks):
            return toks
        return None


# ---------------------------------------------------------------------------
# Decoders
# ---------------------------------------------------------------------------
class _Decoder:
    def __init__(self, spec: Optional[dict]):
        self.spec = spec

    def decode(self, tokens: List[str]) -> str:
        return self._decode_chain(self.spec, tokens)

    def _decode_chain(self, spec: Optional[dict], tokens: List[str]) -> str:
        if spec is None:
            # No decoder configured: the Rust default joins tokens with a
            # single space (WordLevel-style), as used by chatterbox.
            return " ".join(tokens)
        t = spec.get("type")
        if t == "Sequence":
            for sub in spec.get("decoders", []):
                tokens = self._decode_step(sub, tokens)
            return "".join(tokens)
        tokens = self._decode_step(spec, tokens)
        return "".join(tokens)

    def _decode_step(self, spec: dict, tokens: List[str]) -> List[str]:
        t = spec.get("type")
        if t == "ByteLevel":
            joined = "".join(tokens)
            try:
                bs = bytearray(_BYTE_DECODER[c] for c in joined)
                return [bs.decode("utf-8", errors="replace")]
            except KeyError:
                # any char not in the byte map: best-effort
                bs = bytearray()
                for c in joined:
                    if c in _BYTE_DECODER:
                        bs.append(_BYTE_DECODER[c])
                    else:
                        bs.extend(c.encode("utf-8"))
                return [bs.decode("utf-8", errors="replace")]
        if t == "Replace":
            pat = spec.get("pattern", {})
            content = spec.get("content", "")
            if "String" in pat:
                return [tok.replace(pat["String"], content) for tok in tokens]
            if "Regex" in pat:
                return [compile_regex(pat["Regex"]).sub(content, tok) for tok in tokens]
            return tokens
        if t == "ByteFallback":
            return self._byte_fallback_decode(tokens)
        if t == "Fuse":
            return ["".join(tokens)]
        if t == "Strip":
            content = spec.get("content", " ")
            start = spec.get("start", 0)
            stop = spec.get("stop", 0)
            out = []
            for tok in tokens:
                s = tok
                for _ in range(start):
                    if s.startswith(content):
                        s = s[len(content):]
                for _ in range(stop):
                    if s.endswith(content):
                        s = s[:-len(content)] if content else s
                out.append(s)
            return out
        if t == "Metaspace":
            replacement = spec.get("replacement", "▁")
            out = [tok.replace(replacement, " ") for tok in tokens]
            # Metaspace decoder strips a single leading space on the first token
            if out and out[0].startswith(" "):
                out[0] = out[0][1:]
            return out
        if t == "WordPiece":
            prefix = spec.get("prefix", "##")
            cleanup = spec.get("cleanup", True)
            text_parts = []
            for i, tok in enumerate(tokens):
                if i == 0:
                    text_parts.append(tok)
                elif tok.startswith(prefix):
                    text_parts.append(tok[len(prefix):])
                else:
                    text_parts.append(" " + tok)
            s = "".join(text_parts)
            if cleanup:
                s = (s.replace(" .", ".").replace(" ?", "?").replace(" !", "!")
                     .replace(" ,", ",").replace(" ' ", "'").replace(" n't", "n't")
                     .replace(" 'm", "'m").replace(" 's", "'s").replace(" 've", "'ve")
                     .replace(" 're", "'re"))
            return [s]
        # Unknown decoder step: identity
        return tokens

    def token_bytes(self, tok: str) -> bytes:
        """The bytes one token stands for inside a text (constrained decoding's vocabulary): the chain's per-token steps
        applied, up to `Fuse` — Fuse and the steps after it act on the whole text, and a first token's stripped space
        (Metaspace's, a Strip after Fuse) is a sequence rule, not the token's."""
        steps = (self.spec.get("decoders", []) if self.spec and self.spec.get("type") == "Sequence"
                 else ([self.spec] if self.spec else []))
        text = tok
        for st in steps:
            t = st.get("type")
            if t == "Fuse":
                break
            if t == "ByteLevel":        # as decode: a char outside the byte map is its own UTF-8
                return b"".join(bytes([_BYTE_DECODER[c]]) if c in _BYTE_DECODER else c.encode("utf-8") for c in text)
            if t == "ByteFallback" and len(text) == 6 and text.startswith("<0x") and text.endswith(">"):
                return bytes([int(text[3:5], 16)])
            if t == "Replace" and "String" in st.get("pattern", {}):
                text = text.replace(st["pattern"]["String"], st.get("content", ""))
            elif t == "Metaspace":
                text = text.replace(st.get("replacement", "▁"), " ")
            elif t != "ByteFallback":       # a regex Replace, a Strip before Fuse, any other step: never skipped silently
                raise RuntimeError(f"ZERO FALLBACK: token_bytes has no per-token rule for the decoder step {t!r} "
                                   f"({st}) before Fuse")
        return text.encode("utf-8")

    @staticmethod
    def _byte_fallback_decode(tokens: List[str]) -> List[str]:
        """Reassemble ``<0xXX>`` runs into UTF-8 text; pass through others."""
        out: List[str] = []
        byte_buf = bytearray()

        def flush_bytes():
            if byte_buf:
                out.append(byte_buf.decode("utf-8", errors="replace"))
                byte_buf.clear()

        for tok in tokens:
            if len(tok) == 6 and tok.startswith("<0x") and tok.endswith(">"):
                try:
                    byte_buf.append(int(tok[3:5], 16))
                    continue
                except ValueError:
                    pass
            flush_bytes()
            out.append(tok)
        flush_bytes()
        return out


# ---------------------------------------------------------------------------
# Added-token (special / verbatim) handling
# ---------------------------------------------------------------------------
class _AddedToken:
    __slots__ = ("content", "id", "special", "lstrip", "rstrip", "normalized", "single_word")

    def __init__(self, entry: dict):
        self.content = entry["content"]
        self.id = entry["id"]
        self.special = bool(entry.get("special", False))
        self.lstrip = bool(entry.get("lstrip", False))
        self.rstrip = bool(entry.get("rstrip", False))
        self.normalized = bool(entry.get("normalized", False))
        self.single_word = bool(entry.get("single_word", False))


# ---------------------------------------------------------------------------
# PyTokenizer
# ---------------------------------------------------------------------------
class PyTokenizer:
    """Stdlib-only drop-in for the subset of ``tokenizers.Tokenizer`` used by
    NeuroBrix. See module docstring for the supported component matrix.
    """

    def __init__(self, data: dict):
        self._data = data
        model = data.get("model", {})
        self._bpe = _BPE(model)
        self._vocab = self._bpe.vocab
        self._id_to_tok = self._bpe.id_to_tok

        self._normalizer = _Normalizer(data.get("normalizer"))
        self._pretok = _PreTokenizer(data.get("pre_tokenizer"))
        self._decoder = _Decoder(data.get("decoder"))

        # Added tokens (override vocab id mapping for their content), and the character trie the split walks: one node
        # per prefix of an added token, the tokens whose content ends at a node listed under its "" key (no character
        # is ""), in declaration order
        self._added: List[_AddedToken] = []
        self._added_trie: dict = {}
        self._special_ids: set = set()
        self._added_content_to_id: Dict[str, int] = {}
        for entry in data.get("added_tokens", []):
            self._add_token(_AddedToken(entry))

        # The vendor's non-legacy SentencePiece tokenizers (tokenizer_config `legacy: false`) put the word-start prefix
        # only at the start of the input, never again after an added token (transformers `_get_prepend_scheme`:
        # "first"); the tokenizer.json alone says "Prepend" on every segment. Set by the owner that reads the config.
        self.prefix_only_at_start = False

        # Post-processor
        self._post = data.get("post_processor")
        self._post_special = self._collect_post_special()

    def _add_token(self, at: _AddedToken) -> None:
        self._added.append(at)
        self._vocab.setdefault(at.content, at.id)
        self._id_to_tok.setdefault(at.id, at.content)
        if at.special:
            self._special_ids.add(at.id)
        self._added_content_to_id[at.content] = at.id
        if at.content:     # an empty content never matches
            node = self._added_trie
            for ch in at.content:
                node = node.setdefault(ch, {})
            node.setdefault("", []).append(at)

    def register_added_token(self, entry: dict) -> bool:
        """Register an added token post-construction (idempotent).

        Replicates PreTrainedTokenizerFast merging tokenizer_config.json's
        `added_tokens_decoder` entries into the matcher: some vendors (GLM-4V
        class) declare tokens like <|image|> ONLY there, not in
        tokenizer.json's added_tokens — without this they BPE-split into
        pieces and placeholder scans miss them. Returns True if added."""
        content = entry.get("content")
        if not content or content in self._added_content_to_id:
            return False
        self._add_token(_AddedToken(entry))
        return True

    # -- construction ------------------------------------------------------
    @classmethod
    def from_file(cls, path: str) -> "PyTokenizer":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(data)


    # -- vocab API ---------------------------------------------------------
    def token_to_id(self, tok: str) -> Optional[int]:
        return self._vocab.get(tok)


    def token_bytes(self, i: int) -> Optional[bytes]:
        """Token `i`'s bytes inside a text; None for a special token (an added token marked special)."""
        if i in self._special_ids:
            return None
        tok = self._id_to_tok.get(i)
        if tok is None:
            return None
        return self._decoder.token_bytes(tok)

    def get_vocab_size(self) -> int:
        """The vocabulary with its added tokens (every caller's question)."""
        return len(self._vocab)


    # -- encode ------------------------------------------------------------
    def encode(self, text: str, add_special_tokens: bool = True) -> _Encoding:
        # 1) split out added/special tokens verbatim (before normalization)
        segments = self._split_added(text)

        tokens: List[str] = []
        ids: List[int] = []
        at_start = True        # no added token before this segment: it begins the input
        for seg_text, seg_added in segments:
            if seg_added is not None:
                tokens.append(seg_added.content)
                ids.append(seg_added.id)
                at_start = False
                continue
            if seg_text == "":
                continue
            # 2) normalize, 3) pre-tokenize, 4) BPE
            norm = self._normalizer(seg_text, prefix=at_start or not self.prefix_only_at_start)
            pieces = self._pretok(norm, at_start=at_start, prefix_only_at_start=self.prefix_only_at_start)
            at_start = False
            for piece in pieces:
                for tok in self._bpe.tokenize_piece(piece):
                    tid = self._vocab.get(tok)
                    if tid is None:
                        # unk fallback already produced vocab tokens; if still
                        # missing, skip (defensive)
                        continue
                    tokens.append(tok)
                    ids.append(tid)

        # 5) post-processing (template / bytelevel id no-op)
        if add_special_tokens:
            ids, tokens = self._post_process(ids, tokens)
        return _Encoding(ids, tokens)

    def _split_added(self, text: str) -> List[Tuple[str, Optional[_AddedToken]]]:
        """Split ``text`` into alternating (plain, None) and ("", AddedToken)
        segments, matching added tokens verbatim, longest-first, left-to-right,
        honoring lstrip/rstrip whitespace consumption.

        The tokens that start at a position are found by walking the added tokens' character trie from it: the cost
        is the longest added token's length, not the number of added tokens (orpheus declares 28 940). The reference
        (tokenizers v0.23.3, tokenizer/added_vocabulary.rs, `find_matches`) runs one leftmost-longest Aho-Corasick
        automaton; the engine's rules differ from it on two points, kept as they were: a `single_word` token that
        fails its boundary check gives way to the next shorter token at the same position (the reference discards
        the match and resumes after it), and `normalized` tokens are matched on the raw text (the reference matches
        them in a second pass on the normalized text)."""
        if not self._added_trie:
            return [(text, None)]
        out: List[Tuple[str, Optional[_AddedToken]]] = []
        i = 0
        n = len(text)
        buf_start = 0
        while i < n:
            node, j, ends = self._added_trie, i, []     # (end, the tokens ending there) for each prefix walked
            while j < n and text[j] in node:
                node, j = node[text[j]], j + 1
                ends.append((j, node.get("", ())))
            # the longest first, the tokens of one content in declaration order: the per-token loop's stable sort
            matched = next((at for end, ats in reversed(ends) for at in ats
                            if not at.single_word or self._is_word_boundary(text, i, end)), None)
            if matched is not None:
                start = i
                end = i + len(matched.content)
                # lstrip: consume whitespace before the token from the buffer
                left_cut = start
                if matched.lstrip:
                    while left_cut > buf_start and text[left_cut - 1].isspace():
                        left_cut -= 1
                # rstrip: consume whitespace after the token
                if matched.rstrip:
                    while end < n and text[end].isspace():
                        end += 1
                if left_cut > buf_start:
                    out.append((text[buf_start:left_cut], None))
                out.append(("", matched))
                buf_start = end
                i = end
            else:
                i += 1
        if buf_start < n:
            out.append((text[buf_start:], None))
        return out

    @staticmethod
    def _is_word_boundary(text: str, start: int, end: int) -> bool:
        def is_word(ch: str) -> bool:
            return ch.isalnum() or ch == "_"
        left_ok = (start == 0) or (not is_word(text[start - 1]))
        right_ok = (end == len(text)) or (not is_word(text[end]))
        return left_ok and right_ok

    # -- post-processing ---------------------------------------------------
    def _collect_post_special(self) -> Dict[str, List[int]]:
        mapping: Dict[str, List[int]] = {}
        spec = self._post
        if spec is None:
            return mapping
        self._walk_post_special(spec, mapping)
        return mapping

    def _walk_post_special(self, spec: dict, mapping: Dict[str, List[int]]):
        t = spec.get("type")
        if t == "Sequence":
            for sub in spec.get("processors", []):
                self._walk_post_special(sub, mapping)
        elif t == "TemplateProcessing":
            for name, info in spec.get("special_tokens", {}).items():
                mapping[name] = list(info.get("ids", []))

    def _post_process(self, ids: List[int], tokens: List[str]) -> Tuple[List[int], List[str]]:
        spec = self._post
        if spec is None:
            return ids, tokens
        return self._apply_post(spec, ids, tokens)

    def _apply_post(self, spec: dict, ids: List[int], tokens: List[str]):
        t = spec.get("type")
        if t == "Sequence":
            for sub in spec.get("processors", []):
                ids, tokens = self._apply_post(sub, ids, tokens)
            return ids, tokens
        if t == "ByteLevel":
            # post-processor ByteLevel only adjusts offsets — id no-op
            return ids, tokens
        if t == "TemplateProcessing":
            return self._apply_template(spec, ids, tokens)
        # RobertaProcessing / BertProcessing: add cls/sep
        if t in ("RobertaProcessing", "BertProcessing"):
            return self._apply_bert_like(spec, ids, tokens)
        return ids, tokens

    def _apply_template(self, spec: dict, ids: List[int], tokens: List[str]):
        template = spec.get("single", [])
        special = spec.get("special_tokens", {})
        out_ids: List[int] = []
        out_toks: List[str] = []
        for item in template:
            if "SpecialToken" in item:
                name = item["SpecialToken"]["id"]
                info = special.get(name, {})
                tids = info.get("ids", [])
                ttoks = info.get("tokens", [name] * len(tids))
                out_ids.extend(tids)
                out_toks.extend(ttoks)
            elif "Sequence" in item:
                out_ids.extend(ids)
                out_toks.extend(tokens)
        return out_ids, out_toks

    def _apply_bert_like(self, spec: dict, ids: List[int], tokens: List[str]):
        cls = spec.get("cls")
        sep = spec.get("sep")
        out_ids, out_toks = [], []
        if cls:
            out_toks.append(cls[0])
            out_ids.append(cls[1])
        out_ids.extend(ids)
        out_toks.extend(tokens)
        if sep:
            out_toks.append(sep[0])
            out_ids.append(sep[1])
        return out_ids, out_toks

    # -- decode ------------------------------------------------------------
    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        toks: List[str] = []
        for i in ids:
            if skip_special_tokens and i in self._special_ids:
                continue
            tok = self._id_to_tok.get(i)
            if tok is None:
                continue
            toks.append(tok)
        return self._decoder.decode(toks)
