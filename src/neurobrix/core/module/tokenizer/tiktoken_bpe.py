# core/module/tokenizer/tiktoken_bpe.py
"""
PyTiktoken — pure-Python, stdlib-only replacement for the ``tiktoken`` library.

Zero Outsider chantier (R34): the NeuroBrix runtime engine must not import any
vendor ML library at inference time. ``tiktoken`` is the byte-level BPE
tokenizer used by openaudio-s1-mini (and other fish-speech / Qwen2-derived
models). This module reproduces its ``encode`` / ``decode`` behaviour bit for
bit using only the Python standard library.

Imports: ``base64``, ``re``, ``unicodedata``, ``json``, ``typing`` — NOT
``tiktoken``.

Format of a ``*.tiktoken`` file (one merge per line)::

    base64(token_bytes) <space> rank

Algorithm (standard tiktoken byte-level BPE):

1. Split special-token strings out of the text first (they are emitted as their
   ids with no BPE), mirroring ``Encoding.encode(text, allowed_special="all")``.
2. Split each remaining text segment into chunks with the model's pre-tokenizer
   regex (``PAT_STR`` below).
3. Within each chunk, byte-pair-merge by rank: repeatedly merge the adjacent
   pair whose merged bytes have the lowest rank, until no mergeable pair
   remains. The resulting pieces map directly to ids via ``mergeable_ranks``.

Resolved pre-tokenizer pattern for openaudio-s1-mini
----------------------------------------------------
openaudio-s1-mini's text tokenizer carries exactly 151643 base ranks — the
Qwen2 BPE fingerprint — with semantic / control special tokens appended from
rank 151643 upward. The source-true Qwen2 pre-tokenizer pattern is the GPT-4
(cl100k) pattern with the digit clause ``\p{N}{1,3}`` reduced to a single
``\p{N}``. Because the openaudio vocabulary contains **no multi-byte
all-ASCII-digit merge token**, the ``\p{N}`` vs ``\p{N}{1,3}`` distinction is
unobservable on this vocab (verified: 509/509 digit-heavy strings byte-identical
through the full BPE). We use the source-true single-``\p{N}`` Qwen2 pattern.

stdlib regex note
-----------------
The pattern uses Unicode property classes (``\p{L}``, ``\p{N}``, …) which the
stdlib ``re`` module does not understand (it raises ``bad escape \p``). We
translate every ``\p{X}`` into an explicit character-class body by enumerating
codepoints with ``unicodedata.category()`` at import time, and we substitute a
fixed Unicode-White_Space set for ``\s`` / ``\S`` (stdlib ``re``'s ``\s``
differs from the ``regex`` module's on a handful of control codepoints). The
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

# `regex`/PCRE Unicode general-category property -> predicate on the two-letter
# unicodedata.category() code. We support the property names that appear in the
# tiktoken-family pre-tokenizer patterns.
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
# 0x1C-0x1F (file/group/record/unit separators) which are NOT Unicode
# White_Space; the `regex` module (used by tiktoken) follows White_Space. To
# match tiktoken exactly we substitute this explicit set for \s / \S.
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
    """Render ranges as a character-class body (no surrounding []) for `re`.

    Codepoints are emitted as \\Uxxxxxxxx escapes so they remain valid inside a
    character class regardless of the codepoint (avoids needing to escape ``-``,
    ``]``, ``^``, ``\\`` literals).
    """
    parts: List[str] = []
    for a, b in ranges:
        pa = "\\U%08x" % a
        if a == b:
            parts.append(pa)
        else:
            parts.append(pa + "-" + "\\U%08x" % b)
    return "".join(parts)


# Precompute the character-class bodies once at import.
_PROP_BODY = {name: _class_body(_ranges_for_predicate(pred))
              for name, pred in _PROP_PREDICATES.items()}
_WS_BODY = _class_body(_ranges_for_set(_WHITE_SPACE))


def _translate_pattern(pat: str) -> str:
    """Translate a `regex`-style pat_str into an equivalent stdlib-`re` pattern.

    Replaces ``\\p{X}`` / ``\\P{X}`` (both standalone and inside ``[...]``) with
    explicit character-class bodies, and ``\\s`` / ``\\S`` with the Unicode
    White_Space set / its negation. All other constructs (``(?i:...)``,
    ``(?!\\S)``, alternation, quantifiers) are valid stdlib `re` already.
    """
    out: List[str] = []
    i = 0
    n = len(pat)
    while i < n:
        c = pat[i]
        if c == "[":
            # Parse a character class; inline \p/\P/\s bodies (no nesting in re).
            j = i + 1
            body = ""
            if j < n and pat[j] == "^":
                body += "^"
                j += 1
            if j < n and pat[j] == "]":  # literal ] as first class member
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


# Source-true Qwen2 pre-tokenizer pattern (cl100k with single \p{N}). See the
# module docstring for why this is the resolved openaudio-s1-mini pat_str.
PAT_STR = (
    r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)"""
    r"""|[^\r\n\p{L}\p{N}]?\p{L}+"""
    r"""|\p{N}"""
    r"""| ?[^\s\p{L}\p{N}]+[\r\n]*"""
    r"""|\s*[\r\n]+"""
    r"""|\s+(?!\S)"""
    r"""|\s+"""
)


# ---------------------------------------------------------------------------
# Byte-level BPE (tiktoken algorithm).
# ---------------------------------------------------------------------------

def _byte_pair_merge(piece: bytes, ranks: Dict[bytes, int]) -> List[bytes]:
    """tiktoken's byte-pair merge: split `piece` into the minimal-rank pieces.

    Identical algorithm to ``tiktoken``'s ``_byte_pair_merge``: maintain a list
    of part boundaries, each annotated with the rank of merging the part at that
    boundary with the next one; repeatedly merge the globally lowest-rank
    adjacent pair, recomputing only the two affected boundary ranks.
    """
    # parts[i] = (start_index_into_piece, rank_of_merging_part_i_with_part_(i+1))
    # The sentinel parts at the end carry MAX rank so they are never chosen.
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


_MAX_RANK = (1 << 63) - 1


# Surrogate sanitizer. tiktoken receives a Python ``str`` and converts it to Rust
# at the PyO3 boundary, which reproduces UCS-2→scalar semantics: a valid
# surrogate PAIR (high U+D800..U+DBFF immediately followed by low U+DC00..U+DFFF)
# is combined into its astral scalar, and any remaining LONE surrogate is mapped
# to U+FFFD. Both arise only from corrupted / non-UTF-8 input — never from valid
# text. We replicate the exact conversion so our pure-Python
# ``str.encode("utf-8")`` (which rejects surrogates) matches the oracle bit for
# bit. Verified against tiktoken on lone-high, lone-low, hi-hi, low-high, and
# valid-pair (with/without surrounding ASCII) inputs.
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


class PyTiktoken:
    """Pure-Python tiktoken byte-level BPE encoder.

    Parameters
    ----------
    mergeable_ranks : dict[bytes, int]
        token bytes -> rank, as loaded from the ``*.tiktoken`` file.
    special_tokens : dict[str, int]
        special token string -> id.
    pat_str : str
        the `regex`-style pre-tokenizer pattern (``\\p{...}`` allowed).
    """

    def __init__(
        self,
        mergeable_ranks: Dict[bytes, int],
        special_tokens: Optional[Dict[str, int]] = None,
        pat_str: str = PAT_STR,
    ):
        self._ranks = mergeable_ranks
        self._decoder: Dict[int, bytes] = {v: k for k, v in mergeable_ranks.items()}
        self._special_tokens = dict(special_tokens or {})
        self._special_decoder: Dict[int, str] = {
            v: k for k, v in self._special_tokens.items()
        }
        self._pat = re.compile(_translate_pattern(pat_str))

        # Regex matching any special-token string (longest-first to avoid a
        # prefix special masking a longer one).
        if self._special_tokens:
            keys = sorted(self._special_tokens, key=len, reverse=True)
            self._special_pat = re.compile("|".join(re.escape(k) for k in keys))
        else:
            self._special_pat = None

        self._cache: Dict[str, List[int]] = {}

    # ----- construction -----------------------------------------------------

    @classmethod
    def from_file(
        cls,
        tiktoken_path: str,
        special_tokens_path: Optional[str] = None,
        special_tokens: Optional[Dict[str, int]] = None,
        pat_str: str = PAT_STR,
    ) -> "PyTiktoken":
        """Build from a ``*.tiktoken`` file (+ optional ``special_tokens.json``).

        ``special_tokens`` may be passed directly; otherwise, if
        ``special_tokens_path`` is given it is read as a JSON ``{str: int}``
        mapping (the openaudio ``special_tokens.json`` format). If neither is
        given, the sibling ``special_tokens.json`` next to the ``.tiktoken``
        file is loaded when present.
        """
        ranks: Dict[bytes, int] = {}
        with open(tiktoken_path, "rb") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                token_b64, rank = line.split()
                ranks[base64.b64decode(token_b64)] = int(rank)

        specials: Dict[str, int] = {}
        if special_tokens is not None:
            specials = dict(special_tokens)
        else:
            path = special_tokens_path
            if path is None:
                import os
                sibling = os.path.join(
                    os.path.dirname(tiktoken_path), "special_tokens.json"
                )
                if os.path.exists(sibling):
                    path = sibling
            if path is not None:
                with open(path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                for name, info in raw.items():
                    if isinstance(info, dict):
                        tid = info.get("id")
                        content = info.get("content", name)
                        if tid is not None:
                            specials[content] = tid
                    elif isinstance(info, int):
                        specials[name] = info

        return cls(ranks, specials, pat_str)

    # ----- core encoding ----------------------------------------------------

    def _encode_ordinary(self, text: str) -> List[int]:
        """Encode text containing NO special tokens (plain byte-level BPE)."""
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

    def encode(self, text: str, allowed_special: str = "all") -> List[int]:
        """Encode text to token ids.

        Mirrors ``tiktoken.Encoding.encode(text, allowed_special="all")``:
        special-token strings present in the text are emitted as their ids with
        no BPE; the remaining segments are byte-level BPE encoded. Pass
        ``allowed_special="none"`` to disable special-token splitting (every
        character, including ``<``/``>``, is BPE'd).
        """
        if not text:
            return []
        text = _sanitize_surrogates(text)
        if allowed_special != "all" or self._special_pat is None:
            return self._encode_ordinary(text)

        ids: List[int] = []
        pos = 0
        for m in self._special_pat.finditer(text):
            if m.start() > pos:
                ids.extend(self._encode_ordinary(text[pos:m.start()]))
            ids.append(self._special_tokens[m.group()])
            pos = m.end()
        if pos < len(text):
            ids.extend(self._encode_ordinary(text[pos:]))
        return ids

    # ----- decoding ---------------------------------------------------------

    def decode(self, ids: Sequence[int], errors: str = "replace") -> str:
        """Decode token ids back to a string.

        Special-token ids are rendered as their string form (matching
        ``tiktoken.Encoding.decode``, which keeps special tokens). Regular byte
        pieces are concatenated and UTF-8 decoded as a whole so multi-token
        characters reassemble correctly.
        """
        out: List[str] = []
        buf = bytearray()
        for i in ids:
            piece = self._decoder.get(i)
            if piece is not None:
                buf.extend(piece)
                continue
            # flush byte buffer before emitting a special token's text
            if buf:
                out.append(buf.decode("utf-8", errors=errors))
                buf = bytearray()
            sp = self._special_decoder.get(i)
            if sp is not None:
                out.append(sp)
            # unknown id -> skipped (no byte content)
        if buf:
            out.append(buf.decode("utf-8", errors=errors))
        return "".join(out)

    def decode_bytes(self, ids: Sequence[int]) -> bytes:
        """Decode regular token ids to raw bytes (special ids contribute none)."""
        buf = bytearray()
        for i in ids:
            piece = self._decoder.get(i)
            if piece is not None:
                buf.extend(piece)
        return bytes(buf)

    # ----- lookups / metadata ----------------------------------------------

    def token_to_id(self, token: bytes) -> Optional[int]:
        """Rank/id for a regular token's bytes (None if absent)."""
        return self._ranks.get(token)

    def id_to_token(self, token_id: int) -> Optional[bytes]:
        """Bytes for a regular token id (None if it is a special / unknown id)."""
        return self._decoder.get(token_id)

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size: regular ranks + special tokens."""
        return len(self._ranks) + len(self._special_tokens)

    @property
    def n_vocab(self) -> int:
        """Alias for :attr:`vocab_size` (tiktoken naming)."""
        return self.vocab_size
