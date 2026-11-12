"""A text is split at its added tokens in time linear in the text, whatever the number of added tokens, with the same
splits as before.

Measured on 2026-11-12 (Mac, CPU): canopylabs/orpheus-3b-0.1-ft's tokenizer.json declares 28 940 added tokens (its
audio codes), and `PyTokenizer._split_added` tried every one of them at every character of the text (`startswith` in a
loop over all added tokens, for each position): 1.2 ms per character, 42 s for a 35 000-character prompt, before the
first token reached the model. The reference matches all the added tokens at once with an Aho-Corasick automaton
(tokenizers v0.23.3, tokenizers/src/tokenizer/added_vocabulary.rs, `find_matches`).

The engine's split now walks a character trie of the added tokens from each position: the cost at a position is the
length of the longest added token that starts there, not the number of added tokens. The engine's own rules are kept
as they were (they differ from the reference's on two points, recorded in the module): the longest token that starts at
a position is tried first, and a `single_word` token that fails its word-boundary check gives way to the next shorter
one at the same position; `lstrip` / `rstrip` take the whitespace around the token.

What this test would do if the code were wrong: the per-token loop executes 136 969 lines of the module for a text with
50 added tokens and 751 829 for the same text with 400 (x5.5, seen failing); the trie walk executes 7 429 and 6 709.
"""
from __future__ import annotations

import sys

from neurobrix.core.module.tokenizer import json_bpe
from neurobrix.core.module.tokenizer.json_bpe import PyTokenizer


def _tokenizer_json(codes: int) -> dict:
    """A tiny tokenizer.json whose added tokens exercise every rule of the split, plus `codes` audio-code-like tokens."""
    added = [{"id": 1, "content": "<s>", "special": True}, {"id": 2, "content": "</s>", "special": True},
             {"id": 3, "content": "<mask>", "single_word": True}, {"id": 4, "content": "<m"},
             {"id": 5, "content": "[L]", "lstrip": True}, {"id": 6, "content": "[R]", "rstrip": True},
             {"id": 7, "content": "[B]", "lstrip": True, "rstrip": True},
             {"id": 8, "content": "<dup>", "single_word": True}, {"id": 9, "content": "<dup>"},
             {"id": 10, "content": "ab"}, {"id": 11, "content": "abc"}, {"id": 12, "content": ""}]
    added += [{"id": 100 + i, "content": f"<code_{i}>", "special": True} for i in range(codes)]
    return {"added_tokens": added, "normalizer": None, "pre_tokenizer": None, "post_processor": None, "decoder": None,
            "model": {"type": "BPE", "unk_token": "<unk>", "vocab": {"<unk>": 0, "a": 13, "b": 14, "c": 15},
                      "merges": []}}


def _lines_executed(t: PyTokenizer, text: str) -> int:
    """The lines of json_bpe.py the split executes: a deterministic count of its work, not a timer."""
    count = 0

    def local(frame, event, arg):
        nonlocal count
        if event == "line":
            count += 1
        return local

    def tracer(frame, event, arg):
        return local if frame.f_code.co_filename == json_bpe.__file__ else None

    previous = sys.gettrace()
    sys.settrace(tracer)
    try:
        t._split_added(text)
    finally:
        sys.settrace(previous)
    return count


TEXT = "Say <code_7> then <s>hello</s> to [L] the <mask> world, ab abc <code_42> <code_399>. " * 10


def test_the_split_costs_the_same_for_eight_times_the_added_tokens():
    few, many = _lines_executed(PyTokenizer(_tokenizer_json(50)), TEXT), _lines_executed(
        PyTokenizer(_tokenizer_json(400)), TEXT)
    assert many < 1.2 * few, (f"the split executed {few} lines with 50 added tokens and {many} with 400 "
                              f"(x{many / few:.1f}): it grows with the number of added tokens")


def test_four_times_the_text_costs_about_four_times_the_split():
    t = PyTokenizer(_tokenizer_json(400))
    small, large = _lines_executed(t, TEXT), _lines_executed(t, TEXT * 4)
    assert large < 4.5 * small, f"{small} lines for the text, {large} for four times it (x{large / small:.1f})"


# the splits the per-token loop gave (json_bpe.py at 8599b0f08), on the fixture above with 20 codes: (text, None) for
# plain text, ("", id) for an added token
BEFORE = {
    '': [],
    'plain text': [('plain text', None)],
    '<s>abc</s>': [('', 1), ('', 11), ('', 2)],
    '<s><s></s>': [('', 1), ('', 1), ('', 2)],
    'x<mask> <mask> <mask>y': [('x', None), ('', 4), ('ask> ', None), ('', 3), (' ', None), ('', 4), ('ask>y', None)],
    '_<mask>_': [('_', None), ('', 4), ('ask>_', None)],
    'é<mask>1': [('é', None), ('', 4), ('ask>1', None)],
    '  [L]  [R]  [B]  ': [('', 5), ('  ', None), ('', 6), ('', 7)],
    'a [L][L] [R] [R]b': [('a', None), ('', 5), ('', 5), (' ', None), ('', 6), ('', 6), ('b', None)],
    '\t[B]\n[B]\n': [('', 7), ('', 7)],
    'x<dup> <dup> <dup>y': [('x', None), ('', 9), (' ', None), ('', 8), (' ', None), ('', 9), ('y', None)],
    'abcab abab': [('', 11), ('', 10), (' ', None), ('', 10), ('', 10)],
    '<code_1><code_10><code_19><code_2': [('', 101), ('', 110), ('', 119), ('<code_2', None)],
    '<code_<code_3>>': [('<code_', None), ('', 103), ('>', None)],
    '<<mask>>': [('<', None), ('', 3), ('>', None)],
    ' <m <mask>': [(' ', None), ('', 4), (' ', None), ('', 3)],
}


def test_the_splits_are_those_of_the_per_token_loop():
    t = PyTokenizer(_tokenizer_json(20))
    got = {text: [(s, a.id if a else None) for s, a in t._split_added(text)] for text in BEFORE}
    assert got == BEFORE


def test_a_token_registered_after_construction_is_split_out():
    t = PyTokenizer(_tokenizer_json(0))
    assert t.register_added_token({"id": 50, "content": "<|image|>", "special": True})
    assert not t.register_added_token({"id": 51, "content": "<|image|>"})
    assert [(s, a.id if a else None) for s, a in t._split_added("a<|image|>b<|ima")] == [
        ("a", None), ("", 50), ("b<|ima", None)]
