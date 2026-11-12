"""The BPE model's word cache is bounded as the reference bounds it: a word of 256 bytes or more is not kept, and the
cache never holds more than 10 000 words; the ids are those of an uncached tokenizer.

Before (json_bpe.py at 8599b0f08), `_BPE._bpe` kept every word it merged, forever. A Llama-style tokenizer.json (no
pre-tokenizer, Metaspace written by the normalizer) hands the whole text to the BPE model as ONE word, so every prompt
a `serve` process received stayed in its memory for the life of the process. The reference caches a word only when it
is shorter than 256 bytes and the cache holds fewer than 10 000 entries (tokenizers v0.23.3,
tokenizers/src/models/bpe/model.rs `tokenize_with_cache`, tokenizers/src/utils/cache.rs `MAX_LENGTH`,
`DEFAULT_CACHE_CAPACITY`): a full cache keeps what it has and stores nothing more.

What this test would do if the code were wrong: the unbounded cache keeps the 256-byte word and holds 10 050 words
after 10 050 distinct ones (seen failing).
"""
from __future__ import annotations

import itertools

from neurobrix.core.module.tokenizer.json_bpe import PyTokenizer

MERGES = ["a b", "ab c", "b c", "c a"]


def _tokenizer() -> PyTokenizer:
    """A BPE model behind a whitespace split: each word of a text reaches the cache on its own."""
    vocab = {"<unk>": 0, "a": 1, "b": 2, "c": 3, "ab": 4, "abc": 5, "bc": 6, "ca": 7, "é": 8}
    return PyTokenizer({"added_tokens": [], "normalizer": None, "pre_tokenizer": {"type": "WhitespaceSplit"},
                        "post_processor": None, "decoder": None,
                        "model": {"type": "BPE", "unk_token": "<unk>", "vocab": vocab, "merges": MERGES}})


def test_a_word_of_256_bytes_or_more_is_not_kept():
    t = _tokenizer()
    short, long, wide = "abc" * 85, "abc" * 85 + "a", "é" * 128      # 255 bytes; 256 bytes; 128 characters, 256 bytes
    t.encode(" ".join([short, long, wide]))
    assert short in t._bpe._cache
    assert long not in t._bpe._cache, "a 256-byte word was kept in the BPE cache"
    assert wide not in t._bpe._cache, "a 256-byte word (128 characters) was kept in the BPE cache"


def test_the_cache_never_holds_more_than_ten_thousand_words():
    words = ["".join(w) for w in itertools.islice(itertools.product("abcé", repeat=7), 10_050)]
    t = _tokenizer()
    ids = t.encode(" ".join(words)).ids
    assert len(t._bpe._cache) == 10_000, f"the BPE cache holds {len(t._bpe._cache)} words after 10 050 distinct ones"
    assert t.encode(" ".join(words)).ids == ids        # a full cache: hits for the first 10 000, merges for the rest
    assert [t.encode(w).ids for w in words[-5:]] == [_tokenizer().encode(w).ids for w in words[-5:]]
