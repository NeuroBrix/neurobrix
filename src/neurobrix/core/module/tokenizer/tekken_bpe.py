# core/module/tokenizer/tekken_bpe.py
"""
PyTekken — pure-Python, stdlib-only replacement for ``mistral_common`` Tekken.

Zero Outsider chantier (R34): the NeuroBrix runtime engine must not import any
vendor ML library at inference time. ``mistral_common`` provides the Tekken
byte-level BPE tokenizer used by Voxtral-Mini-3B-2507. This module reproduces
``Tekkenizer.encode`` / ``decode`` bit for bit using only the Python standard
library.

Imports: ``json``, ``base64``, ``re``, ``unicodedata``, ``typing`` — NOT
``mistral_common``.

``tekken.json`` structure
-------------------------
- ``config``: ``pattern`` (the pre-tokenizer regex), ``default_vocab_size``
  (131072), ``default_num_special_tokens`` (1000), ``version``.
- ``vocab``: list of ``{"rank": int, "token_bytes": base64, "token_str": str}``.
- ``special_tokens``: list of ``{"rank": int, "token_str": str, "is_control":
  bool}``.

Tekken == tiktoken byte-level BPE with three model-specific id rules, verified
empirically against ``mistral_common==1.9.1``:

1. **Vocab truncation.** Only the first ``inner_vocab_size = default_vocab_size
   - default_num_special_tokens`` (= 131072 - 1000 = 130072) entries of
   ``vocab`` are used as mergeable ranks. The rest are discarded so that a
   high-rank merge that the reference never makes cannot fire.
2. **Special-token id space.** Special tokens occupy ids ``0 .. num_special-1``.
   The list is padded with synthetic ``<SPECIAL_i>`` controls up to
   ``num_special`` so the offset is exact.
3. **Offset.** Every regular BPE id is shifted by ``+num_special`` (= +1000):
   ``Tekkenizer.encode`` does ``[t + num_special for t in model.encode(s)]``.
   So byte ``0x61`` ('a', rank 97) -> id 1097.

``encode`` uses plain byte-level BPE — special-token *strings* inside the input
text are NOT treated as special (they are byte-BPE'd), matching
``Tekkenizer.encode`` which calls the inner tiktoken model with no
``allowed_special``. BOS / EOS are added only on request via the ``bos`` / ``eos``
flags. ``decode`` follows ``SpecialTokenPolicy.IGNORE`` by default (special ids
are dropped), which is the ``mistral_common>=1.9`` default.

stdlib regex note
-----------------
The Tekken pattern uses Unicode property classes (``\p{Lu}``, ``\p{Ll}``,
``\p{Lm}``, ``\p{Lo}``, ``\p{M}``, ``\p{L}``, ``\p{N}``) which stdlib ``re`` does
not understand. We translate every ``\p{X}`` into an explicit character-class
body by enumerating codepoints with ``unicodedata.category()`` at import time,
and substitute a fixed Unicode-White_Space set for ``\s`` / ``\S``. The
translation is validated to produce chunk-identical splits to the ``regex``
module across an 8000-string Unicode fuzz battery.
"""

from typing import Dict, List, Optional, Sequence, Tuple
import base64
import json
import re
import unicodedata


# ---------------------------------------------------------------------------
# Unicode property-class translation (stdlib `re` has no \p{...} support).
# ---------------------------------------------------------------------------

_PROP_PREDICATES = {
    "L": lambda c: c[0] == "L",
    "N": lambda c: c[0] == "N",
    "M": lambda c: c[0] == "M",
    "Lu": lambda c: c == "Lu",
    "Ll": lambda c: c == "Ll",
    "Lt": lambda c: c == "Lt",
    "Lm": lambda c: c == "Lm",
    "Lo": lambda c: c == "Lo",
}

# Unicode White_Space property codepoints. stdlib `re`'s \s additionally matches
# 0x1C-0x1F which are NOT Unicode White_Space; the `regex` module (used by
# tiktoken under mistral_common) follows White_Space. We substitute this exact
# set for \s / \S to match the reference.
_WHITE_SPACE = {
    0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20, 0x85, 0xA0, 0x1680,
    0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007,
    0x2008, 0x2009, 0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000,
}


def _category_table() -> List[str]:
    """Enumerate the unicodedata general category for every codepoint once."""
    return [unicodedata.category(chr(cp)) for cp in range(0x110000)]


_CATS = _category_table()


def _ranges_for_predicate(pred) -> List[Tuple[int, int]]:
    """Contiguous (start, end) codepoint ranges where pred(category) is True."""
    out: List[Tuple[int, int]] = []
    start: Optional[int] = None
    prev = -1
    for cp in range(0x110000):
        if pred(_CATS[cp]):
            if start is None:
                start = cp
            prev = cp
        elif start is not None:
            out.append((start, prev))
            start = None
    if start is not None:
        out.append((start, prev))
    return out


def _ranges_for_set(codepoints) -> List[Tuple[int, int]]:
    """Contiguous (start, end) ranges covering an explicit codepoint set."""
    out: List[Tuple[int, int]] = []
    start: Optional[int] = None
    prev = -1
    for cp in sorted(codepoints):
        if start is None:
            start = prev = cp
        elif cp == prev + 1:
            prev = cp
        else:
            out.append((start, prev))
            start = prev = cp
    if start is not None:
        out.append((start, prev))
    return out


def _class_body(ranges: Sequence[Tuple[int, int]]) -> str:
    """Render ranges as a `re` character-class body (codepoints as \\Uxxxxxxxx)."""
    parts: List[str] = []
    for a, b in ranges:
        pa = "\\U%08x" % a
        if a == b:
            parts.append(pa)
        else:
            parts.append(pa + "-" + "\\U%08x" % b)
    return "".join(parts)


_PROP_BODY = {name: _class_body(_ranges_for_predicate(pred))
              for name, pred in _PROP_PREDICATES.items()}
_WS_BODY = _class_body(_ranges_for_set(_WHITE_SPACE))


def _translate_pattern(pat: str) -> str:
    """Translate a `regex`-style pat_str into an equivalent stdlib-`re` pattern.

    Replaces ``\\p{X}`` / ``\\P{X}`` (both standalone and inside ``[...]``) with
    explicit character-class bodies, and ``\\s`` / ``\\S`` with the Unicode
    White_Space set / its negation.
    """
    out: List[str] = []
    i = 0
    n = len(pat)
    while i < n:
        c = pat[i]
        if c == "[":
            j = i + 1
            body = ""
            if j < n and pat[j] == "^":
                body += "^"
                j += 1
            if j < n and pat[j] == "]":
                body += "]"
                j += 1
            while j < n and pat[j] != "]":
                if pat[j] == "\\" and j + 1 < n:
                    two = pat[j:j + 2]
                    if two in ("\\p", "\\P"):
                        k = pat.index("}", j + 3)
                        body += _PROP_BODY[pat[j + 3:k]]
                        j = k + 1
                        continue
                    if two == "\\s":
                        body += _WS_BODY
                        j += 2
                        continue
                    body += two
                    j += 2
                    continue
                body += pat[j]
                j += 1
            out.append("[" + body + "]")
            i = j + 1
            continue
        if c == "\\" and i + 1 < n:
            two = pat[i:i + 2]
            if two in ("\\p", "\\P"):
                k = pat.index("}", i + 3)
                name = pat[i + 3:k]
                out.append(("[^" if two == "\\P" else "[") + _PROP_BODY[name] + "]")
                i = k + 1
                continue
            if two == "\\s":
                out.append("[" + _WS_BODY + "]")
                i += 2
                continue
            if two == "\\S":
                out.append("[^" + _WS_BODY + "]")
                i += 2
                continue
            out.append(two)
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


# ---------------------------------------------------------------------------
# Byte-level BPE (tiktoken algorithm, shared with PyTiktoken).
# ---------------------------------------------------------------------------

_MAX_RANK = (1 << 63) - 1


# Surrogate sanitizer. mistral_common wraps tiktoken, which converts the Python
# ``str`` to Rust at the PyO3 boundary: a valid surrogate PAIR (high U+D800..
# U+DBFF immediately followed by low U+DC00..U+DFFF) is combined into its astral
# scalar, and any remaining LONE surrogate is mapped to U+FFFD. Both arise only
# from corrupted / non-UTF-8 input. We replicate the exact conversion so our
# pure-Python ``str.encode("utf-8")`` (which rejects surrogates) matches the
# oracle bit for bit.
_SURROGATE_RE = re.compile("[\ud800-\udfff]")


def _sanitize_surrogates(text: str) -> str:
    if not _SURROGATE_RE.search(text):
        return text
    out: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        o = ord(text[i])
        if 0xD800 <= o <= 0xDBFF:  # high surrogate
            if i + 1 < n and 0xDC00 <= ord(text[i + 1]) <= 0xDFFF:
                lo = ord(text[i + 1])
                out.append(chr(0x10000 + ((o - 0xD800) << 10) + (lo - 0xDC00)))
                i += 2
                continue
            out.append("�")
            i += 1
            continue
        if 0xDC00 <= o <= 0xDFFF:  # lone low surrogate
            out.append("�")
            i += 1
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def _byte_pair_merge(piece: bytes, ranks: Dict[bytes, int]) -> List[bytes]:
    """tiktoken's byte-pair merge: split `piece` into minimal-rank pieces."""
    parts: List[List[int]] = [[i, _MAX_RANK] for i in range(len(piece) + 1)]

    def get_rank(parts_list: List[List[int]], idx: int) -> int:
        if idx + 3 < len(parts_list):
            sub = piece[parts_list[idx][0]:parts_list[idx + 3][0]]
            return ranks.get(sub, _MAX_RANK)
        return _MAX_RANK

    for i in range(len(parts) - 2):
        parts[i][1] = ranks.get(piece[parts[i][0]:parts[i + 2][0]], _MAX_RANK)

    while len(parts) > 1:
        min_rank = _MAX_RANK
        min_idx = -1
        for i in range(len(parts) - 1):
            if parts[i][1] < min_rank:
                min_rank = parts[i][1]
                min_idx = i
        if min_rank == _MAX_RANK:
            break
        i = min_idx
        parts[i][1] = get_rank(parts, i)
        if i > 0:
            parts[i - 1][1] = get_rank(parts, i - 1)
        parts.pop(i + 1)

    return [piece[parts[i][0]:parts[i + 1][0]] for i in range(len(parts) - 1)]


class PyTekken:
    """Pure-Python Tekken byte-level BPE encoder.

    Build via :meth:`from_file` on a ``tekken.json``. Ids follow the Tekken
    convention: special tokens occupy ``0 .. num_special-1`` and regular BPE
    tokens are offset by ``+num_special``.
    """

    def __init__(
        self,
        mergeable_ranks: Dict[bytes, int],
        special_tokens: List[Tuple[int, str]],
        pat_str: str,
        num_special_tokens: int,
        vocab_size: int,
    ):
        self._ranks = mergeable_ranks
        self._decoder: Dict[int, bytes] = {v: k for k, v in mergeable_ranks.items()}
        self._num_special = num_special_tokens
        self._vocab_size = vocab_size
        self._pat = re.compile(_translate_pattern(pat_str))

        # special id -> string, and string -> id
        self._special_id_to_str: Dict[int, str] = {}
        self._special_str_to_id: Dict[str, int] = {}
        for rank, token_str in special_tokens:
            self._special_id_to_str[rank] = token_str
            self._special_str_to_id[token_str] = rank

        # Conventional Mistral control ids.
        self._unk_id = self._special_str_to_id.get("<unk>", 0)
        self._bos_id = self._special_str_to_id.get("<s>", 1)
        self._eos_id = self._special_str_to_id.get("</s>", 2)
        self._pad_id = self._special_str_to_id.get("<pad>", 11)

        self._cache: Dict[str, List[int]] = {}

    # ----- construction -----------------------------------------------------

    @classmethod
    def from_file(cls, tekken_path: str) -> "PyTekken":
        """Build a PyTekken from a ``tekken.json`` file.

        Reproduces ``Tekkenizer.from_file`` id semantics: truncate ``vocab`` to
        ``default_vocab_size - default_num_special_tokens`` regular ranks, build
        the special-token table padded to ``default_num_special_tokens``.
        """
        with open(tekken_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        config = data["config"]
        pattern = config["pattern"]
        vocab_size = config["default_vocab_size"]
        num_special = config["default_num_special_tokens"]
        inner_vocab_size = vocab_size - num_special

        vocab = data["vocab"]
        if len(vocab) > inner_vocab_size:
            vocab = vocab[:inner_vocab_size]

        ranks: Dict[bytes, int] = {}
        for i, entry in enumerate(vocab):
            merge = base64.b64decode(entry["token_bytes"])
            # invariant from mistral_common: rank == position; first 256 ranks
            # are the single bytes 0x00..0xFF.
            ranks[merge] = entry["rank"]

        # Special tokens: defined ones, then synthetic <SPECIAL_i> fillers up to
        # num_special so the +offset is exact.
        defined = data.get("special_tokens", []) or []
        special_tokens: List[Tuple[int, str]] = [
            (st["rank"], st["token_str"]) for st in defined
        ]
        defined_count = len(special_tokens)
        for r in range(defined_count, num_special):
            special_tokens.append((r, "<SPECIAL_%d>" % r))

        return cls(ranks, special_tokens, pattern, num_special, vocab_size)

    # ----- core encoding ----------------------------------------------------

    def _encode_ordinary(self, text: str) -> List[int]:
        """Byte-level BPE over `text`, before the +num_special offset."""
        ids: List[int] = []
        for match in self._pat.finditer(text):
            piece = match.group()
            cached = self._cache.get(piece)
            if cached is not None:
                ids.extend(cached)
                continue
            piece_bytes = piece.encode("utf-8")
            token = self._ranks.get(piece_bytes)
            if token is not None:
                result = [token]
            else:
                result = [self._ranks[p]
                          for p in _byte_pair_merge(piece_bytes, self._ranks)]
            self._cache[piece] = result
            ids.extend(result)
        return ids

    def encode(self, text: str, bos: bool = False, eos: bool = False) -> List[int]:
        """Encode text to Tekken token ids.

        Mirrors ``Tekkenizer.encode(s, bos, eos)``: plain byte-level BPE (no
        special-string splitting), each regular id offset by ``+num_special``,
        with optional BOS / EOS.
        """
        text = _sanitize_surrogates(text)
        tokens = [t + self._num_special for t in self._encode_ordinary(text)]
        if bos:
            tokens = [self._bos_id] + tokens
        if eos:
            tokens = tokens + [self._eos_id]
        return tokens

    # ----- decoding ---------------------------------------------------------

    def decode(self, ids: Sequence[int], keep_special: bool = False,
               errors: str = "replace") -> str:
        """Decode Tekken token ids to a string.

        Default behaviour follows ``SpecialTokenPolicy.IGNORE`` (special ids are
        dropped). With ``keep_special=True`` special tokens are rendered as their
        string form (``SpecialTokenPolicy.KEEP``). Regular ids are un-offset and
        their bytes concatenated, then UTF-8 decoded as a whole so multi-token
        characters reassemble correctly.
        """
        out: List[str] = []
        buf = bytearray()
        for i in ids:
            if i < self._num_special:
                if buf:
                    out.append(buf.decode("utf-8", errors=errors))
                    buf = bytearray()
                if keep_special:
                    sp = self._special_id_to_str.get(i)
                    if sp is not None:
                        out.append(sp)
                # else IGNORE: drop the special token
                continue
            piece = self._decoder.get(i - self._num_special)
            if piece is not None:
                buf.extend(piece)
        if buf:
            out.append(buf.decode("utf-8", errors=errors))
        return "".join(out)

    # ----- lookups / metadata ----------------------------------------------

    def token_to_id(self, token: bytes) -> Optional[int]:
        """Tekken id (with +num_special offset) for a regular token's bytes."""
        rank = self._ranks.get(token)
        return None if rank is None else rank + self._num_special

    def id_to_token(self, token_id: int) -> Optional[bytes]:
        """Bytes for a regular Tekken id (None for special / unknown ids)."""
        if token_id < self._num_special:
            return None
        return self._decoder.get(token_id - self._num_special)

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size (``default_vocab_size`` = 131072)."""
        return self._vocab_size

    @property
    def n_words(self) -> int:
        """Alias for :attr:`vocab_size` (mistral_common naming)."""
        return self._vocab_size

    @property
    def bos_id(self) -> int:
        return self._bos_id

    @property
    def eos_id(self) -> int:
        return self._eos_id

    @property
    def pad_id(self) -> int:
        return self._pad_id

    @property
    def unk_id(self) -> int:
        return self._unk_id
