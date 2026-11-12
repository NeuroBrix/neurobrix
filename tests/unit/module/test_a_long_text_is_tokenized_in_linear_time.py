"""A long text is tokenized in linear time by a Llama-style tokenizer.json (a BPE model behind Metaspace's `▁`), with the
same ids as before.

Measured on 2026-11-12 (Mac, CPU, the published 0.6.9): TinyLlama-1.1B-Chat-v1.0's tokenizer took 3.0 s for a prompt of
4 078 tokens and 25.7 s for 12 265 (x8.6 for x3); granite's byte-level BPE took 0.04 s for 12 K. Such a tokenizer.json
has no pre-tokenizer (the normalizer writes `▁` for every space), so the whole text reaches the BPE model as ONE word,
and the merge loop rescanned every pair of the word to apply each single merge: n merges x n pairs. cProfile: one
`_bpe` call, 445 714 051 dict lookups, for the 12 K prompt.

The reference merges a word with a min-heap of (rank, position) and a linked list of symbols (tokenizers v0.23.3,
tokenizers/src/models/bpe/word.rs, `Word::merge_all`): each merge pops one entry, skips it when a side of the pair has
been merged since it was pushed, and pushes the two pairs the new symbol forms with its neighbours. The lowest rank,
then the leftmost position, is merged first: the order the rescan chose, so the ids are those of before.

What this test would do if the code were wrong: the rescanning loop counts 225 450 rank lookups for 100 words and 3 601 800
for 400 (x16.0, seen failing); a heap merge counts about four times more for four times the text.
"""
from __future__ import annotations

import random

from neurobrix.core.module.tokenizer.json_bpe import PyTokenizer

MERGES = ["▁ a", "a b", "▁a b", "b c", "▁ ▁", "▁▁ ▁▁", "c ▁", "▁ab c", "a a", "aa a", "1 2", "▁ 1"]


def _llama_style_json() -> dict:
    """A small tokenizer.json shaped as TinyLlama's: Prepend + Replace normalizer, no pre-tokenizer, BPE with byte
    fallback (`<0xXX>`), and merges that cross a `▁` (`c ▁`), so no split at `▁` could give the same ids."""
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
    for b in range(256):
        vocab[f"<0x{b:02X}>"] = len(vocab)
    for tok in ["▁", "a", "b", "c", "1", "2"] + ["".join(m.split(" ")) for m in MERGES]:
        vocab.setdefault(tok, len(vocab))
    return {"version": "1.0",
            "added_tokens": [{"id": 1, "content": "<s>", "special": True}, {"id": 2, "content": "</s>", "special": True}],
            "normalizer": {"type": "Sequence", "normalizers": [{"type": "Prepend", "prepend": "▁"},
                                                               {"type": "Replace", "pattern": {"String": " "},
                                                                "content": "▁"}]},
            "pre_tokenizer": None, "post_processor": None, "decoder": None,
            "model": {"type": "BPE", "dropout": None, "unk_token": "<unk>", "fuse_unk": True, "byte_fallback": True,
                      "continuing_subword_prefix": None, "end_of_word_suffix": None, "vocab": vocab, "merges": MERGES}}


class _CountingRanks(dict):
    """The merge table, counting every lookup the merge loop makes in it."""
    lookups = 0

    def get(self, key, default=None):
        self.lookups += 1
        return super().get(key, default)

    def __contains__(self, key):
        self.lookups += 1
        return super().__contains__(key)

    def __getitem__(self, key):
        self.lookups += 1
        return super().__getitem__(key)


def _lookups(words: int) -> int:
    t = PyTokenizer(_llama_style_json())
    t._bpe.ranks = _CountingRanks(t._bpe.ranks)
    t.encode("abc ab " * words, add_special_tokens=False)
    return t._bpe.ranks.lookups


def test_four_times_the_text_costs_about_four_times_the_merge_work():
    small, large = _lookups(100), _lookups(400)
    assert large < 5 * small, (f"the merge loop made {small} rank lookups for 100 words and {large} for 400 "
                               f"(x{large / small:.1f}): quadratic in the text's length")


# the ids the rescanning merge loop gave (json_bpe.py at ae4b65a09), on the fixture above
BEFORE = {
    "": [],
    "abc ab": [272, 267],
    "ab abc abcabc": [267, 272, 272, 266, 262],
    "   leading,  and trailing   ": [270, 111, 104, 260, 103, 108, 113, 106, 47, 259, 265, 113, 103, 259, 119, 117, 260,
                                     108, 111, 108, 113, 106, 269, 259],
    "a\tb\nc\r\n  a": [265, 12, 261, 13, 262, 16, 13, 259, 265],
    "aaaa aaa 12 12a ▁▁": [265, 274, 265, 273, 259, 275, 259, 275, 260, 269, 259],
    "héllo 😀 1 2": [259, 107, 198, 172, 111, 111, 114, 259, 243, 162, 155, 131, 276, 259, 264],
    "<s>abc</s> <s> ab": [1, 272, 2, 269, 1, 259, 267],
    "ccc cab": [259, 262, 262, 271, 262, 266],
}


def test_the_ids_are_those_of_the_rescanning_merge_loop():
    t = PyTokenizer(_llama_style_json())
    assert {text: t.encode(text, add_special_tokens=False).ids for text in BEFORE} == BEFORE


def _rescan(word: list, ranks: dict) -> list:
    """The textbook merge (the engine's before the heap): the lowest-ranked pair, the leftmost first, until none."""
    word = list(word)
    while True:
        pairs = [(ranks[(a, b)], i) for i, (a, b) in enumerate(zip(word, word[1:])) if (a, b) in ranks]
        if not pairs:
            return word
        _, i = min(pairs)
        word[i:i + 2] = [word[i] + word[i + 1]]


def test_the_heap_merges_any_word_as_the_rescan_does():
    t = PyTokenizer(_llama_style_json())
    rng = random.Random(20261112)
    for _ in range(2000):
        word = "".join(rng.choice("▁▁abc12") for _ in range(rng.randint(1, 40)))
        assert t._bpe._bpe(word) == _rescan(list(word), t._bpe.ranks), word
