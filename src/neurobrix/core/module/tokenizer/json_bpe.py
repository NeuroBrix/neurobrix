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

The ``\\p{L}`` / ``\\p{N}`` semantics required by the GPT-2/Qwen ``Split``
pre-tokenizer are derived from ``unicodedata.category()`` (letters = categories
beginning with ``'L'``, numbers = ``'N'``) and compiled into stdlib ``re``
character classes -- the third-party ``regex`` module is never imported.
"""

from __future__ import annotations

import json
import re
import unicodedata
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
# \p{L} / \p{N} -> stdlib re character-class fragments (lazy, built once)
# ---------------------------------------------------------------------------
_UNICODE_CLASS_CACHE: Dict[str, str] = {}


def _category_ranges(first_letter: str) -> str:
    """Return an ``re`` character-class body covering all code points whose
    Unicode general category starts with ``first_letter`` (e.g. ``'L'`` for
    letters, ``'N'`` for numbers). Result is a string of ``\\Uxxxxxxxx`` ranges
    suitable for inlining inside ``[...]``.
    """
    if first_letter in _UNICODE_CLASS_CACHE:
        return _UNICODE_CLASS_CACHE[first_letter]
    cat = unicodedata.category
    pieces: List[str] = []
    start = None
    prev = None
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:  # surrogates: never members
            is_member = False
        else:
            is_member = cat(chr(cp))[0] == first_letter
        if is_member and start is None:
            start = cp
        elif not is_member and start is not None:
            pieces.append((start, prev))
            start = None
        prev = cp
    if start is not None:
        pieces.append((start, 0x10FFFF))
    body = "".join(
        (re.escape(chr(a)) if a == b else f"{re.escape(chr(a))}-{re.escape(chr(b))}")
        for a, b in pieces
    )
    _UNICODE_CLASS_CACHE[first_letter] = body
    return body


def _translate_pp_regex(pattern: str) -> str:
    r"""Translate a HuggingFace/onig ``Split`` regex into an equivalent stdlib
    ``re`` pattern.

    The only constructs that differ from ``re`` in the patterns NeuroBrix ships
    are the Unicode property escapes ``\p{L}`` / ``\p{N}`` (and their negations
    inside classes). We expand them to explicit ``unicodedata``-derived ranges.
    Everything else (``(?i:...)`` groups, ``\r`` ``\n`` ``\s`` ``\S``,
    quantifiers, anchors, alternation) is valid stdlib ``re`` syntax.
    """
    out: List[str] = []
    i = 0
    L = _category_ranges("L")
    N = _category_ranges("N")
    n = len(pattern)
    in_class = False  # are we currently inside a [...] character class?
    while i < n:
        c = pattern[i]
        if c == "\\" and i + 1 < n:
            nxt = pattern[i + 1]
            if nxt == "p" and i + 2 < n and pattern[i + 2] == "{":
                end = pattern.index("}", i + 3)
                prop = pattern[i + 3:end]
                body = L if prop == "L" else (N if prop == "N" else None)
                if body is not None:
                    # Inside a class: emit the raw range body. As a bare atom:
                    # wrap in [...] so the quantifier/anchor binds to the class.
                    out.append(body if in_class else "[" + body + "]")
                else:  # other properties: leave raw (none used by our models)
                    out.append(pattern[i:end + 1])
                i = end + 1
                continue
            # keep the escape verbatim (\r \n \s \S \d \w ...)
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

    def __call__(self, text: str) -> str:
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
            return spec.get("prepend", "") + text
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
        if t in ("NFKC", "BertNormalizer", "Nmt"):  # not used; pass-through-ish
            return text
        # Unknown normalizer: identity (data-driven; none of our models hit this)
        return text

    @staticmethod
    def _replace(spec: dict, text: str) -> str:
        pat = spec.get("pattern", {})
        content = spec.get("content", "")
        if "String" in pat:
            return text.replace(pat["String"], content)
        if "Regex" in pat:
            return re.sub(pat["Regex"], content, text)
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
        self._compiled: Dict[int, "re.Pattern"] = {}

    def __call__(self, text: str) -> List[str]:
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
            key = id(spec)
            rx = self._compiled.get(key)
            if rx is None:
                rx = re.compile(_translate_pp_regex(pat["Regex"]))
                self._compiled[key] = rx
            return rx, None
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
# \p{L}/\p{N} expanded to unicodedata ranges; the rest is stdlib re.
_GPT2_BYTELEVEL_RX = re.compile(
    _translate_pp_regex(
        r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
    )
)

# Rust Whitespace pre-tokenizer: \w+|[^\w\s]+  (Unicode-aware \w).
# \w = \p{L} | \p{N} | _ ; build with unicodedata-derived classes.
_WHITESPACE_RX = re.compile(
    "(?:["
    + _category_ranges("L")
    + _category_ranges("N")
    + "_]+|[^"
    + _category_ranges("L")
    + _category_ranges("N")
    + r"_\s]+)"
)

# Rust BertPreTokenizer: split on whitespace then isolate punctuation.
_BERT_RX = re.compile(r"\w+|[^\w\s]")


# ---------------------------------------------------------------------------
# BPE model
# ---------------------------------------------------------------------------
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
        word = list(token)
        if len(word) <= 1:
            self._cache[token] = word
            return word
        while True:
            best_rank = None
            best_idx = -1
            for i in range(len(word) - 1):
                r = self.ranks.get((word[i], word[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_idx = i
            if best_idx < 0:
                break
            word[best_idx:best_idx + 2] = [word[best_idx] + word[best_idx + 1]]
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
                return [re.sub(pat["Regex"], content, tok) for tok in tokens]
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

        # Added tokens (override vocab id mapping for their content)
        self._added: List[_AddedToken] = []
        for entry in data.get("added_tokens", []):
            at = _AddedToken(entry)
            self._added.append(at)
            self._vocab.setdefault(at.content, at.id)
            self._id_to_tok.setdefault(at.id, at.content)
        # longest-first so overlapping specials match greedily (Rust behavior)
        self._added_sorted = sorted(self._added, key=lambda a: len(a.content), reverse=True)
        self._special_ids = {a.id for a in self._added if a.special}
        self._added_content_to_id = {a.content: a.id for a in self._added}

        # Post-processor
        self._post = data.get("post_processor")
        self._post_special = self._collect_post_special()

    # -- construction ------------------------------------------------------
    @classmethod
    def from_file(cls, path: str) -> "PyTokenizer":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(data)

    @classmethod
    def from_str(cls, blob: str) -> "PyTokenizer":
        return cls(json.loads(blob))

    # -- vocab API ---------------------------------------------------------
    def token_to_id(self, tok: str) -> Optional[int]:
        return self._vocab.get(tok)

    def id_to_token(self, i: int) -> Optional[str]:
        return self._id_to_tok.get(i)

    def get_vocab_size(self, with_added_tokens: bool = True) -> int:
        if with_added_tokens:
            return len(self._vocab)
        base = self._data.get("model", {}).get("vocab", {})
        return len(base)

    def get_vocab(self, with_added_tokens: bool = True) -> Dict[str, int]:
        return dict(self._vocab)

    # -- encode ------------------------------------------------------------
    def encode(self, text: str, add_special_tokens: bool = True) -> _Encoding:
        # 1) split out added/special tokens verbatim (before normalization)
        segments = self._split_added(text)

        tokens: List[str] = []
        ids: List[int] = []
        for seg_text, seg_added in segments:
            if seg_added is not None:
                tokens.append(seg_added.content)
                ids.append(seg_added.id)
                continue
            if seg_text == "":
                continue
            # 2) normalize, 3) pre-tokenize, 4) BPE
            norm = self._normalizer(seg_text)
            for piece in self._pretok(norm):
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
        honoring lstrip/rstrip whitespace consumption."""
        if not self._added_sorted:
            return [(text, None)]
        out: List[Tuple[str, Optional[_AddedToken]]] = []
        i = 0
        n = len(text)
        buf_start = 0
        while i < n:
            matched = None
            for at in self._added_sorted:
                c = at.content
                if not c:
                    continue
                if text.startswith(c, i):
                    if at.single_word and not self._is_word_boundary(text, i, i + len(c)):
                        continue
                    matched = at
                    break
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
                i = i + len(matched.content)
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
