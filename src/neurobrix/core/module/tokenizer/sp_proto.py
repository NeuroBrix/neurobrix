"""Pure-Python SentencePiece ``.model`` parser and inference engine.

Zero-Outsider runtime replacement for ``sentencepiece.SentencePieceProcessor``.
The NeuroBrix inference engine must not import vendor ML libraries (R34), so
this module re-implements the subset of SentencePiece that the runtime relies
on using *stdlib only*.

What is implemented, by hand:

* A minimal protobuf wire-format reader (varint / length-delimited / fixed32)
  to parse the serialized ``ModelProto`` directly from the ``.model`` bytes.
  No ``protobuf`` dependency, no generated ``_pb2`` module.
* The SentencePiece text normalizer, including the ``precompiled_charsmap``
  Darts double-array trie used by the ``nmt_nfkc`` / ``nmt_nfkc_cf`` rule sets.
  Applying the model's own trie (rather than ``unicodedata``) is what makes the
  normalization byte-exact across Unicode versions and case-folding rule sets.
* Both segmentation algorithms in production use:

  - ``UNIGRAM`` (Viterbi best-segmentation by piece score) for T5 / PixArt
    ``spiece.model``.
  - ``BPE`` (greedy merge by piece score, highest score merges first) for
    NeMo / parakeet, Llama-family and Sana ``tokenizer.model``.

* ``add_dummy_prefix``, ``remove_extra_whitespaces``, ``escape_whitespaces``
  (space -> U+2581 "lower one eighth block"), USER_DEFINED atomic pieces and
  ``byte_fallback`` to ``<0xXX>`` BYTE pieces.

The public surface mirrors the ``SentencePieceProcessor`` subset NeuroBrix
uses: :meth:`from_bytes`, :meth:`encode` / :meth:`encode_as_ids`,
:meth:`decode`, :meth:`piece_to_id`, :meth:`id_to_piece`,
:meth:`get_piece_size`, and :meth:`bos_id` / :meth:`eos_id` / :meth:`pad_id` /
:meth:`unk_id`.
"""

# Stdlib only (R34): no sentencepiece, no protobuf, no third-party package.
# The .model protobuf, the Darts double-array charsmap, and both the UNIGRAM
# (Viterbi) and BPE segmenters are all parsed/implemented by hand below.
import struct
from typing import Dict, List, Optional, Tuple

__all__ = ["PySentencePiece"]

# SentencePiece piece type enum (ModelProto.SentencePiece.Type).
_TYPE_NORMAL = 1
_TYPE_UNKNOWN = 2
_TYPE_CONTROL = 3
_TYPE_USER_DEFINED = 4
_TYPE_BYTE = 6
_TYPE_UNUSED = 5

# TrainerSpec.ModelType enum.
_MODEL_UNIGRAM = 1
_MODEL_BPE = 2
_MODEL_WORD = 3
_MODEL_CHAR = 4

# The whitespace meta symbol used by SentencePiece ("LOWER ONE EIGHTH BLOCK").
_SPACE_SYMBOL = "▁"


# --------------------------------------------------------------------------- #
# Minimal protobuf wire-format reader (stdlib only).
# --------------------------------------------------------------------------- #
class _ProtoReader:
    """Reads a serialized protobuf message field-by-field from raw bytes.

    Only the four wire types that appear in a SentencePiece ``ModelProto`` are
    supported: varint (0), fixed64 (1), length-delimited (2) and fixed32 (5).
    """

    __slots__ = ("buf", "pos", "end")

    def __init__(self, buf: bytes, start: int = 0, end: Optional[int] = None):
        self.buf = buf
        self.pos = start
        self.end = len(buf) if end is None else end

    def eof(self) -> bool:
        return self.pos >= self.end

    def _read_varint(self) -> int:
        buf = self.buf
        result = 0
        shift = 0
        while True:
            b = buf[self.pos]
            self.pos += 1
            result |= (b & 0x7F) << shift
            if not (b & 0x80):
                break
            shift += 7
        return result

    def read_field(self) -> Tuple[int, int, object]:
        """Return ``(field_number, wire_type, value)`` for the next field.

        ``value`` is an ``int`` for varint / fixed types and a ``bytes`` slice
        for length-delimited fields.
        """
        key = self._read_varint()
        field_number = key >> 3
        wire_type = key & 0x07
        if wire_type == 0:  # varint
            value: object = self._read_varint()
        elif wire_type == 2:  # length-delimited
            length = self._read_varint()
            value = self.buf[self.pos:self.pos + length]
            self.pos += length
        elif wire_type == 5:  # fixed32
            value = struct.unpack_from("<I", self.buf, self.pos)[0]
            self.pos += 4
        elif wire_type == 1:  # fixed64
            value = struct.unpack_from("<Q", self.buf, self.pos)[0]
            self.pos += 8
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire_type}")
        return field_number, wire_type, value


def _decode_float32(bits: int) -> float:
    """Decode an IEEE-754 fixed32 (already a little-endian uint32) to float."""
    return struct.unpack("<f", struct.pack("<I", bits))[0]


# --------------------------------------------------------------------------- #
# Darts double-array trie (verbatim port of darts_clone bit layout).
# --------------------------------------------------------------------------- #
class _DartsTrie:
    """Common-prefix search over a darts_clone double-array (uint32 units).

    Bit layout per unit (see ``third_party/darts_clone/darts.h``):

    * ``has_leaf``  = ``(unit >> 8) & 1``
    * ``value``     = ``unit & 0x7FFFFFFF``                 (leaf units only)
    * ``label``     = ``unit & ((1 << 31) | 0xFF)``
    * ``offset``    = ``(unit >> 10) << ((unit & (1 << 9)) >> 6)``
    """

    __slots__ = ("array",)

    def __init__(self, units: List[int]):
        self.array = units

    @staticmethod
    def _has_leaf(unit: int) -> bool:
        return ((unit >> 8) & 1) == 1

    @staticmethod
    def _value(unit: int) -> int:
        return unit & 0x7FFFFFFF

    @staticmethod
    def _label(unit: int) -> int:
        return unit & ((1 << 31) | 0xFF)

    @staticmethod
    def _offset(unit: int) -> int:
        return (unit >> 10) << ((unit & (1 << 9)) >> 6)

    def common_prefix_search(self, key: bytes) -> List[Tuple[int, int]]:
        """Return ``(value, length)`` for every key prefix present in the trie."""
        results: List[Tuple[int, int]] = []
        array = self.array
        node_pos = 0
        unit = array[node_pos]
        node_pos ^= self._offset(unit)
        for i, byte in enumerate(key):
            node_pos ^= byte
            unit = array[node_pos]
            if self._label(unit) != byte:
                return results
            node_pos ^= self._offset(unit)
            if self._has_leaf(unit):
                results.append((self._value(array[node_pos]), i + 1))
        return results


# --------------------------------------------------------------------------- #
# SentencePiece precompiled-charsmap normalizer.
# --------------------------------------------------------------------------- #
class _Normalizer:
    """Applies the ``precompiled_charsmap`` trie + SP whitespace rules.

    When ``charsmap`` is empty (the ``identity`` rule used by Llama / Sana) the
    trie pass is skipped and the input passes through verbatim (only literal
    U+0020 spaces are escaped).
    """

    __slots__ = (
        "trie",
        "normalized",
        "add_dummy_prefix",
        "remove_extra_whitespaces",
        "escape_whitespaces",
    )

    def __init__(
        self,
        charsmap: bytes,
        add_dummy_prefix: bool,
        remove_extra_whitespaces: bool,
        escape_whitespaces: bool,
    ):
        self.add_dummy_prefix = add_dummy_prefix
        self.remove_extra_whitespaces = remove_extra_whitespaces
        self.escape_whitespaces = escape_whitespaces
        if charsmap:
            trie_size = struct.unpack_from("<I", charsmap, 0)[0]
            trie_bytes = charsmap[4:4 + trie_size]
            units = list(struct.unpack(f"<{trie_size // 4}I", trie_bytes))
            self.trie: Optional[_DartsTrie] = _DartsTrie(units)
            self.normalized: bytes = charsmap[4 + trie_size:]
        else:
            self.trie = None
            self.normalized = b""

    def _normalize_prefix(self, data: bytes) -> Tuple[bytes, int]:
        """Return ``(replacement_bytes, consumed)`` for the next input prefix.

        Mirrors ``Normalizer::NormalizePrefix``: longest trie match wins; when
        no rule matches, one UTF-8 character is consumed unchanged (or one byte
        replaced by U+FFFD on invalid UTF-8).
        """
        if self.trie is not None:
            matches = self.trie.common_prefix_search(data)
            if matches:
                longest_value = 0
                longest_length = 0
                for value, length in matches:
                    if length > longest_length:
                        longest_length = length
                        longest_value = value
                # Replacement is a NUL-terminated string in the blob.
                end = self.normalized.index(b"\x00", longest_value)
                return self.normalized[longest_value:end], longest_length

        # No trie match: emit one valid UTF-8 character unchanged.
        length = _utf8_char_len(data)
        if length == 0:
            # Invalid lead byte -> U+FFFD, consume one byte.
            return b"\xef\xbf\xbd", 1
        return bytes(data[:length]), length

    def normalize(self, text: str) -> str:
        """Full SentencePiece normalization, returning the escaped string."""
        if not text:
            return ""

        data = text.encode("utf-8")
        space_sym = _SPACE_SYMBOL if self.escape_whitespaces else " "
        out: List[str] = []

        # Strip leading whitespace from the *input* (remove_extra_whitespaces).
        if self.remove_extra_whitespaces:
            while data:
                rep, consumed = self._normalize_prefix(data)
                if rep != b" ":
                    break
                data = data[consumed:]
            if not data:
                return ""  # all-whitespace input -> empty, no dummy prefix.

        # Dummy prefix (prefix variant only; suffix variant is unused here).
        if self.add_dummy_prefix:
            out.append(space_sym)

        is_prev_space = self.remove_extra_whitespaces  # collapse against the prefix.
        while data:
            rep, consumed = self._normalize_prefix(data)
            data = data[consumed:]
            piece = rep.decode("utf-8")
            # Replace literal spaces with the meta symbol, with optional dedup.
            chunk_chars: List[str] = []
            for ch in piece:
                if ch == " ":
                    if self.remove_extra_whitespaces and is_prev_space:
                        continue
                    chunk_chars.append(space_sym)
                    is_prev_space = True
                else:
                    chunk_chars.append(ch)
                    is_prev_space = False
            out.append("".join(chunk_chars))

        result = "".join(out)

        # Strip trailing meta-space (remove_extra_whitespaces).
        if self.remove_extra_whitespaces:
            while result.endswith(space_sym):
                result = result[: -len(space_sym)]

        return result


def _utf8_char_len(data: bytes) -> int:
    """Length in bytes of the leading UTF-8 character, or 0 if invalid."""
    if not data:
        return 0
    b0 = data[0]
    if b0 < 0x80:
        return 1
    if 0xC2 <= b0 <= 0xDF:
        n = 2
    elif 0xE0 <= b0 <= 0xEF:
        n = 3
    elif 0xF0 <= b0 <= 0xF4:
        n = 4
    else:
        return 0
    if len(data) < n:
        return 0
    for i in range(1, n):
        if not (0x80 <= data[i] <= 0xBF):
            return 0
    return n


# --------------------------------------------------------------------------- #
# The public SentencePiece engine.
# --------------------------------------------------------------------------- #
class PySentencePiece:
    """Pure-Python re-implementation of the SentencePiece inference subset.

    Construct via :meth:`from_bytes` with the raw ``.model`` bytes.
    """

    def __init__(self):
        # piece tables
        self.pieces: List[str] = []
        self.scores: List[float] = []
        self.types: List[int] = []
        self.piece_to_id_map: Dict[str, int] = {}
        # special ids (from trainer_spec; -1 = unset)
        self._unk_id: int = 0
        self._bos_id: int = -1
        self._eos_id: int = -1
        self._pad_id: int = -1
        # The surface rendered for an unknown token at decode (default " ⁇ ").
        self._unk_surface: str = " ⁇ "
        self._unk_piece: str = "<unk>"
        # model + normalizer config
        self._model_type: int = _MODEL_UNIGRAM
        self._byte_fallback: bool = False
        self._add_dummy_prefix: bool = True
        self._remove_extra_whitespaces: bool = True
        self._normalizer: Optional[_Normalizer] = None
        # byte-fallback lookup: byte value -> piece id of "<0xXX>"
        self._byte_id: List[int] = [-1] * 256
        # user-defined / control pieces matched atomically before segmentation
        self._user_defined: List[str] = []

    # ------------------------------------------------------------------ #
    # Parsing
    # ------------------------------------------------------------------ #
    @classmethod
    def from_bytes(cls, model_bytes: bytes) -> "PySentencePiece":
        """Parse a serialized ``ModelProto`` from raw ``.model`` bytes."""
        self = cls()

        charsmap = b""
        add_dummy_prefix = True
        remove_extra_whitespaces = True
        escape_whitespaces = True

        reader = _ProtoReader(model_bytes)
        piece_idx = 0
        while not reader.eof():
            field_number, wire_type, value = reader.read_field()
            if field_number == 1 and wire_type == 2:  # repeated SentencePiece pieces
                piece, score, ptype = cls._parse_piece(value)  # type: ignore[arg-type]
                self.pieces.append(piece)
                self.scores.append(score)
                self.types.append(ptype)
                self.piece_to_id_map.setdefault(piece, piece_idx)
                if ptype == _TYPE_BYTE:
                    byte_val = cls._byte_piece_value(piece)
                    if byte_val is not None:
                        self._byte_id[byte_val] = piece_idx
                elif ptype == _TYPE_USER_DEFINED:
                    # USER_DEFINED pieces are matched atomically from raw text
                    # by SentencePiece's PrefixMatcher. CONTROL pieces (<s>,
                    # </s>, <pad>) are NEVER extracted from text — they are only
                    # inserted explicitly — so they are excluded here.
                    self._user_defined.append(piece)
                piece_idx += 1
            elif field_number == 2 and wire_type == 2:  # trainer_spec
                self._parse_trainer_spec(value)  # type: ignore[arg-type]
            elif field_number == 3 and wire_type == 2:  # normalizer_spec
                (
                    charsmap,
                    add_dummy_prefix,
                    remove_extra_whitespaces,
                    escape_whitespaces,
                ) = cls._parse_normalizer_spec(value)  # type: ignore[arg-type]
            # other ModelProto fields (self_test_data, denormalizer_spec) ignored.

        self._add_dummy_prefix = add_dummy_prefix
        self._remove_extra_whitespaces = remove_extra_whitespaces
        self._normalizer = _Normalizer(
            charsmap,
            add_dummy_prefix=add_dummy_prefix,
            remove_extra_whitespaces=remove_extra_whitespaces,
            escape_whitespaces=escape_whitespaces,
        )
        return self

    @staticmethod
    def _parse_piece(buf: bytes) -> Tuple[str, float, int]:
        piece = ""
        score = 0.0
        ptype = _TYPE_NORMAL
        reader = _ProtoReader(buf)
        while not reader.eof():
            fn, wt, val = reader.read_field()
            if fn == 1 and wt == 2:  # piece (string)
                piece = val.decode("utf-8")  # type: ignore[union-attr]
            elif fn == 2 and wt == 5:  # score (float, fixed32)
                score = _decode_float32(val)  # type: ignore[arg-type]
            elif fn == 3 and wt == 0:  # type (enum, varint)
                ptype = val  # type: ignore[assignment]
        return piece, score, ptype

    def _parse_trainer_spec(self, buf: bytes) -> None:
        reader = _ProtoReader(buf)
        while not reader.eof():
            fn, wt, val = reader.read_field()
            if fn == 3 and wt == 0:  # model_type
                self._model_type = val  # type: ignore[assignment]
            elif fn == 35 and wt == 0:  # byte_fallback
                self._byte_fallback = bool(val)
            elif fn == 40 and wt == 0:  # unk_id
                self._unk_id = _zigzag_none(val)  # type: ignore[arg-type]
            elif fn == 41 and wt == 0:  # bos_id
                self._bos_id = _zigzag_none(val)  # type: ignore[arg-type]
            elif fn == 42 and wt == 0:  # eos_id
                self._eos_id = _zigzag_none(val)  # type: ignore[arg-type]
            elif fn == 43 and wt == 0:  # pad_id
                self._pad_id = _zigzag_none(val)  # type: ignore[arg-type]
            elif fn == 44 and wt == 2:  # unk_surface (string)
                self._unk_surface = val.decode("utf-8")  # type: ignore[union-attr]
            elif fn == 45 and wt == 2:  # unk_piece (string)
                self._unk_piece = val.decode("utf-8")  # type: ignore[union-attr]

    @staticmethod
    def _parse_normalizer_spec(buf: bytes) -> Tuple[bytes, bool, bool, bool]:
        charsmap = b""
        # SentencePiece proto3 defaults: these fields are present in practice,
        # but default to the SP library defaults when absent.
        add_dummy_prefix = True
        remove_extra_whitespaces = True
        escape_whitespaces = True
        reader = _ProtoReader(buf)
        while not reader.eof():
            fn, wt, val = reader.read_field()
            if fn == 2 and wt == 2:  # precompiled_charsmap (bytes)
                charsmap = bytes(val)  # type: ignore[arg-type]
            elif fn == 3 and wt == 0:  # add_dummy_prefix
                add_dummy_prefix = bool(val)
            elif fn == 4 and wt == 0:  # remove_extra_whitespaces
                remove_extra_whitespaces = bool(val)
            elif fn == 5 and wt == 0:  # escape_whitespaces
                escape_whitespaces = bool(val)
        return charsmap, add_dummy_prefix, remove_extra_whitespaces, escape_whitespaces

    @staticmethod
    def _byte_piece_value(piece: str) -> Optional[int]:
        """Parse ``<0xXX>`` BYTE piece -> integer byte value, else None."""
        if len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">"):
            try:
                return int(piece[3:5], 16)
            except ValueError:
                return None
        return None

    # ------------------------------------------------------------------ #
    # Vocab queries
    # ------------------------------------------------------------------ #
    def get_piece_size(self) -> int:
        return len(self.pieces)

    def piece_to_id(self, piece: str) -> int:
        return self.piece_to_id_map.get(piece, self._unk_id)

    def id_to_piece(self, idx: int) -> str:
        return self.pieces[idx]

    def bos_id(self) -> int:
        return self._bos_id

    def eos_id(self) -> int:
        return self._eos_id

    def pad_id(self) -> int:
        return self._pad_id

    def unk_id(self) -> int:
        return self._unk_id

    # ------------------------------------------------------------------ #
    # Encoding
    # ------------------------------------------------------------------ #
    def encode(self, text: str) -> List[int]:
        """Tokenize ``text`` into piece ids (a.k.a. ``encode_as_ids``)."""
        normalized = self._normalizer.normalize(text)  # type: ignore[union-attr]
        if not normalized:
            return []
        if self._model_type == _MODEL_BPE:
            return self._encode_bpe(normalized)
        return self._encode_unigram(normalized)

    # ``sentencepiece`` alias.
    encode_as_ids = encode

    def _split_on_user_defined(self, text: str) -> List[Tuple[str, bool]]:
        """Split ``text`` into ``(chunk, is_user_defined)`` segments.

        USER_DEFINED / CONTROL pieces are matched atomically (longest-first)
        before any character-level segmentation, exactly as SentencePiece's
        ``PrefixMatcher`` does.
        """
        if not self._user_defined:
            return [(text, False)]
        # Longest pieces first so greedy left-to-right matching is maximal.
        symbols = sorted(self._user_defined, key=len, reverse=True)
        segments: List[Tuple[str, bool]] = []
        i = 0
        n = len(text)
        buf_start = 0
        while i < n:
            matched = None
            for sym in symbols:
                if sym and text.startswith(sym, i):
                    matched = sym
                    break
            if matched is not None:
                if buf_start < i:
                    segments.append((text[buf_start:i], False))
                segments.append((matched, True))
                i += len(matched)
                buf_start = i
            else:
                i += 1
        if buf_start < n:
            segments.append((text[buf_start:], False))
        return segments

    def _finalize_symbols(self, symbols: List[Tuple[str, Optional[int]]]) -> List[int]:
        """Convert ``(piece, id-or-None)`` symbols to final ids.

        Mirrors ``SentencePieceProcessor::PopulateSentencePieceText``:

        * Known pieces emit their id.
        * Unknown pieces with ``byte_fallback`` decompose into ``<0xXX>`` BYTE
          ids (one per UTF-8 byte).
        * Unknown pieces *without* ``byte_fallback`` emit the unk id, and a
          *consecutive run* of unknown pieces collapses into a SINGLE unk id
          (``if (is_prev_unk && is_unk)`` appends to the previous piece rather
          than adding a new one).
        """
        ids: List[int] = []
        is_prev_unk = False
        for piece, pid in symbols:
            is_known = pid is not None and self.types[pid] not in (
                _TYPE_UNUSED,
                _TYPE_UNKNOWN,
            )
            if is_known:
                ids.append(pid)  # type: ignore[arg-type]
                is_prev_unk = False
            elif self._byte_fallback:
                for byte in piece.encode("utf-8"):
                    bid = self._byte_id[byte]
                    ids.append(bid if bid >= 0 else self._unk_id)
                is_prev_unk = False  # byte pieces are known, not unk.
            else:
                if not is_prev_unk:
                    ids.append(self._unk_id)
                # else: merge into the previous unk run (emit nothing).
                is_prev_unk = True
        return ids

    # ---- BPE -------------------------------------------------------------- #
    def _encode_bpe(self, normalized: str) -> List[int]:
        symbols: List[Tuple[str, Optional[int]]] = []
        for chunk, is_user in self._split_on_user_defined(normalized):
            if is_user:
                symbols.append((chunk, self.piece_to_id_map[chunk]))
            else:
                symbols.extend(self._bpe_segment(chunk))
        return self._finalize_symbols(symbols)

    def _bpe_segment(self, text: str) -> List[Tuple[str, Optional[int]]]:
        """Greedy SentencePiece BPE: repeatedly merge the highest-score pair."""
        if not text:
            return []
        # Initial symbols are single Unicode characters.
        symbols: List[str] = list(text)

        while len(symbols) > 1:
            best_score = None
            best_pos = -1
            best_merged = ""
            for i in range(len(symbols) - 1):
                merged = symbols[i] + symbols[i + 1]
                pid = self.piece_to_id_map.get(merged)
                if pid is None:
                    continue
                score = self.scores[pid]
                if best_score is None or score > best_score:
                    best_score = score
                    best_pos = i
                    best_merged = merged
            if best_pos < 0:
                break
            symbols[best_pos:best_pos + 2] = [best_merged]

        return [(sym, self.piece_to_id_map.get(sym)) for sym in symbols]

    # ---- UNIGRAM ---------------------------------------------------------- #
    def _encode_unigram(self, normalized: str) -> List[int]:
        symbols: List[Tuple[str, Optional[int]]] = []
        for chunk, is_user in self._split_on_user_defined(normalized):
            if is_user:
                symbols.append((chunk, self.piece_to_id_map[chunk]))
            else:
                symbols.extend(self._unigram_segment(chunk))
        return self._finalize_symbols(symbols)

    def _unigram_segment(self, text: str) -> List[Tuple[str, Optional[int]]]:
        """Viterbi best-segmentation over the SentencePiece unigram lattice.

        Faithful port of ``Model::PopulateNodes`` + ``Lattice::Viterbi``:

        * Nodes are visited in INSERTION order — begin position ascending, and
          within a begin position the trie ``commonPrefixSearch`` results in
          ascending length, then (only when no length-1 vocab piece matched at
          that begin position, ``has_single_node``) a single UNK node of length
          1 with score ``min_score - kUnkPenalty`` (``kUnkPenalty = 10.0``).
        * The Viterbi update uses a STRICT ``>`` comparison, so on a score tie
          the FIRST-inserted predecessor wins — exactly SP's tie-break, which
          differs from a per-end-position best-only DP.

        ``pid == None`` in the result marks an UNK node (resolved to byte
        fallback / unk id at emit time).

        Known residual (inert): when two segmentations of a repeated-glyph run
        reach the lattice exit with a *numerically equal* total score (e.g. the
        ``′``/``′′`` prime pieces over ``▁′′′``), the choice is an equal-score
        Viterbi tie. SentencePiece resolves it through its C++ ``float``
        accumulation order, which has no canonical answer to match and is not
        bit-faithfully reproducible from stdlib Python (verified: neither a
        strict/non-strict comparison nor ``struct`` float32 rounding reproduces
        the C++ choice). It is inert because:

        1. the two segmentations carry the IDENTICAL total score — SP itself
           assigns them equal cost, so neither is "more correct";
        2. the triggering inputs are pathological repeated-glyph runs (stacked
           primes, stacked fractions) absent from real text/caption prompts;
        3. only the UNIGRAM model is affected; all BPE models are byte-exact.
        """
        if not text:
            return []
        n = len(text)
        neg_inf = float("-inf")
        # best_score[pos] = best backtrace score of any node ending at pos.
        # best_start[pos] / best_pid[pos] reconstruct the chosen node.
        best_score = [neg_inf] * (n + 1)
        best_start = [-1] * (n + 1)
        best_pid = [-1] * (n + 1)  # piece id, or -1 for an UNK node
        best_score[0] = 0.0

        min_score = min(self.scores) if self.scores else 0.0
        unk_penalty = min_score - 10.0

        # Visit begin positions in ascending order (matches PopulateNodes).
        for begin in range(n):
            base = best_score[begin]
            if base == neg_inf:
                continue
            has_single_node = False
            max_len = min(n - begin, self._max_piece_len)
            # Trie results in ascending length (commonPrefixSearch order).
            for length in range(1, max_len + 1):
                sub = text[begin:begin + length]
                pid = self.piece_to_id_map.get(sub)
                if pid is None:
                    continue
                ptype = self.types[pid]
                # UNUSED pieces are not lattice nodes; UNKNOWN / CONTROL pieces
                # are never matched as normal nodes either.
                if ptype in (_TYPE_UNUSED, _TYPE_UNKNOWN, _TYPE_CONTROL):
                    continue
                end = begin + length
                cand = base + self.scores[pid]
                # Strict '>' => first-inserted predecessor wins on a tie.
                if cand > best_score[end]:
                    best_score[end] = cand
                    best_start[end] = begin
                    best_pid[end] = pid
                if length == 1:
                    has_single_node = True
            # UNK node, only when no single-character vocab piece matched.
            if not has_single_node:
                end = begin + 1
                cand = base + unk_penalty
                if cand > best_score[end]:
                    best_score[end] = cand
                    best_start[end] = begin
                    best_pid[end] = -1

        # Walk back to recover the segmentation.
        pieces_rev: List[Tuple[int, int, int]] = []  # (start, end, pid)
        pos = n
        while pos > 0:
            start = best_start[pos]
            if start < 0:
                # Disconnected node (should not happen): consume one char.
                start = pos - 1
                pieces_rev.append((start, pos, -1))
            else:
                pieces_rev.append((start, pos, best_pid[pos]))
            pos = start
        pieces_rev.reverse()

        symbols: List[Tuple[str, Optional[int]]] = []
        for start, end, pid in pieces_rev:
            symbols.append((text[start:end], pid if pid >= 0 else None))
        return symbols

    # ------------------------------------------------------------------ #
    # Decoding
    # ------------------------------------------------------------------ #
    def decode(self, ids: List[int]) -> str:
        """Detokenize piece ids back to text (a.k.a. ``decode_ids``).

        Mirrors ``SentencePieceProcessor::Decode``:

        * NORMAL / USER_DEFINED pieces render their string with the meta-space
          ``U+2581`` converted to a literal space.
        * UNKNOWN pieces whose string equals the canonical ``unk_piece`` render
          as ``unk_surface`` (default `` ⁇ ``); an UNKNOWN piece with a custom
          string renders that string verbatim.
        * CONTROL pieces (``<s>``, ``</s>``, ``<pad>`` …) render nothing.
        * BYTE pieces are accumulated and decoded together as a UTF-8 byte run.
        * Exactly ONE leading meta-space is consumed from the very first
          rendered piece, but only when ``add_dummy_prefix`` or
          ``remove_extra_whitespaces`` is enabled (``is_bos_ws``).
        """
        strip_bos_ws = self._add_dummy_prefix or self._remove_extra_whitespaces
        out: List[str] = []
        pending_bytes = bytearray()
        is_first = True  # tracks the bos position for the leading-space strip.

        def flush_bytes() -> None:
            nonlocal is_first
            if pending_bytes:
                surface = bytes(pending_bytes).decode("utf-8", errors="replace")
                surface = surface.replace(_SPACE_SYMBOL, " ")
                if is_first and strip_bos_ws and surface.startswith(" "):
                    surface = surface[1:]
                out.append(surface)
                is_first = False
                pending_bytes.clear()

        for idx in ids:
            if idx < 0 or idx >= len(self.pieces):
                continue
            ptype = self.types[idx]
            piece = self.pieces[idx]

            if ptype == _TYPE_BYTE:
                byte_val = self._byte_piece_value(piece)
                if byte_val is not None:
                    pending_bytes.append(byte_val)
                    continue

            # Any non-byte piece terminates a pending byte run.
            flush_bytes()

            if ptype == _TYPE_CONTROL:
                continue

            if ptype == _TYPE_UNKNOWN:
                surface = self._unk_surface if piece == self._unk_piece else piece
            else:
                surface = piece.replace(_SPACE_SYMBOL, " ")

            if is_first and strip_bos_ws and surface.startswith(" "):
                surface = surface[1:]
            out.append(surface)
            is_first = False

        flush_bytes()
        return "".join(out)

    decode_ids = decode

    # ------------------------------------------------------------------ #
    # Lazy cached longest-piece length (used by the unigram lattice bound).
    # ------------------------------------------------------------------ #
    @property
    def _max_piece_len(self) -> int:
        cached = getattr(self, "_max_piece_len_cache", None)
        if cached is None:
            cached = max((len(p) for p in self.pieces), default=1)
            self._max_piece_len_cache = cached
        return cached


def _zigzag_none(value: int) -> int:
    """Decode a protobuf int32 stored as a 2's-complement varint.

    SentencePiece encodes the special ids as plain ``int32`` (not zigzag).
    Negative ids (``-1`` for unset) are serialized as a 10-byte varint of the
    64-bit 2's complement; map them back to a signed value.
    """
    if value >= 0x8000000000000000:
        value -= 0x10000000000000000
    return value
