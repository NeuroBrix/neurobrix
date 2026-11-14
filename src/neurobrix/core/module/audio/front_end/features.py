"""The tagger's per-token embedding rows, computed from the token texts — standard library only.

What spaCy 3.8 / thinc 8 compute before the tagger network, re-written (MIT; behaviour read from spacy/lang/lex_attrs.py,
spacy/util.py _get_attr_unless_lookup, spacy/strings hash_string (murmurhash hash64 = MurmurHash64A, seed 1), thinc
NumpyOps.hash (its MurmurHash3 of one 8-byte key, seed per table)):
  NORM     the token's own norm (a tokenizer special case's NORM), else the lexeme norm: the pipeline's `lexeme_norm`
           lookup table (keyed by the string's hash), else the language's norm exceptions, else lower();
  PREFIX   string[0];  SUFFIX  string[-3:];  SHAPE  word_shape (X/x/d, a run kept to 4, LONG at 100+ chars);
  SPACY    1 when the token is followed by whitespace;  IS_SPACE  1 when the text is all whitespace;
  a string attribute's id is its hash (the empty string 0, spaCy's reserved strings their fixed ids); the table row of
  each of the murmur keys is key % rows.
Every value is the container's (`modules/g2p/tagger_meta.json`: attributes, seeds, rows, keys; `norms.json`: the
lexeme norms by hash, the exceptions, the reserved symbols). Proven equal to spaCy's rows on 23 014 tokens."""

M64 = (1 << 64) - 1
M32 = (1 << 32) - 1

#: the attributes this module knows how to compute, in the names the container's tagger_meta.json gives them
_ATTRS = ("NORM", "PREFIX", "SUFFIX", "SHAPE", "SPACY", "IS_SPACE")


def hash64a(data: bytes, seed: int = 1) -> int:
    """MurmurHash64A (Austin Appleby, public domain) — spaCy's string hash."""
    m, r = 0xC6A4A7935BD1E995, 47
    h = (seed ^ (len(data) * m)) & M64
    n = len(data) // 8
    for i in range(n):
        k = int.from_bytes(data[8 * i:8 * i + 8], "little")
        k = (k * m) & M64
        k ^= k >> r
        k = (k * m) & M64
        h ^= k
        h = (h * m) & M64
    tail = data[8 * n:]
    if tail:
        for i in range(len(tail) - 1, -1, -1):
            h ^= tail[i] << (8 * i)
        h = (h * m) & M64
    h ^= h >> r
    h = (h * m) & M64
    h ^= h >> r
    return h


def _fmix64(k: int) -> int:
    k ^= k >> 33
    k = (k * 0xFF51AFD7ED558CCD) & M64
    k ^= k >> 33
    k = (k * 0xC4CEB9FE1A85EC53) & M64
    k ^= k >> 33
    return k


def thinc_hash(val: int, seed: int):
    """thinc 8 NumpyOps.hash of one 64-bit key (numpy_ops.pyx MurmurHash3_x86_128_uint64: MurmurHash3_x64_128's
    8-byte tail case, public domain) -> 4 uint32."""
    h1 = (val * 0x87C37B91114253D5) & M64
    h1 = ((h1 << 31) | (h1 >> 33)) & M64
    h1 = (h1 * 0x4CF5AD432745937F) & M64
    h1 ^= seed
    h1 ^= 8
    h2 = (seed ^ 8) & M64
    h1 = (h1 + h2) & M64
    h2 = (h2 + h1) & M64
    h1 = _fmix64(h1)
    h2 = _fmix64(h2)
    h1 = (h1 + h2) & M64
    h2 = (h2 + h1) & M64
    return h1 & M32, h1 >> 32, h2 & M32, h2 >> 32


def word_shape(text: str) -> str:
    if len(text) >= 100:
        return "LONG"
    shape, last, seq = [], "", 0
    for ch in text:
        c = ("X" if ch.isupper() else "x") if ch.isalpha() else ("d" if ch.isdigit() else ch)
        if c == last:
            seq += 1
        else:
            seq, last = 0, c
        if seq < 4:
            shape.append(c)
    return "".join(shape)


class Features:
    """The tagger's input rows from the container's tagger_meta.json and norms.json (both as loaded dicts)."""

    def __init__(self, meta: dict, norms: dict):
        attrs = list(meta["attrs"])
        if attrs != list(_ATTRS):
            raise ValueError(f"tagger_meta.json: attributes {attrs}, this engine computes {list(_ATTRS)} in that order")
        self.seeds, self.rows, self.keys = list(meta["seeds"]), list(meta["rows"]), int(meta["keys"])
        if len(self.seeds) != len(attrs) or len(self.rows) != len(attrs):
            raise ValueError(f"tagger_meta.json: {len(attrs)} attributes, {len(self.seeds)} seeds, {len(self.rows)} rows")
        if self.keys != len(thinc_hash(0, 0)):
            raise ValueError(f"tagger_meta.json: {self.keys} keys per attribute, the hash gives {len(thinc_hash(0, 0))}")
        self.lookup = {int(k): v for k, v in norms["lexeme_norm"].items()}
        self.exceptions = norms["exceptions"]
        self.symbols = dict(norms["symbols"])       # spaCy's reserved strings -> their fixed ids (asked first)

    def string_id(self, s: str) -> int:
        if s in self.symbols:
            return self.symbols[s]
        return 0 if s == "" else hash64a(s.encode("utf8"), 1)

    def norm(self, s):
        v = self.lookup.get(self.string_id(s))
        if v is not None:
            return v
        return self.exceptions.get(s, s.lower())

    def ids(self, tok):
        s = tok["text"]
        norm = tok.get("norm") or self.norm(s)
        sid = self.string_id
        return [sid(norm), sid(s[0]), sid(s[-3:]), sid(word_shape(s)),
                1 if tok.get("whitespace") else 0, 1 if s.isspace() else 0]

    def rows_of(self, tokens):
        """[n][attributes][keys] row indices, the tagger component's input."""
        out = []
        for tok in tokens:
            out.append([[k % self.rows[a] for k in thinc_hash(key, self.seeds[a])]
                        for a, key in enumerate(self.ids(tok))])
        return out
