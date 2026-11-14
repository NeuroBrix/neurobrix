"""A tokenizer.json whose regexes use the vendor's syntax loads and splits as the vendor does.

The tokenizers library reads every ``Split`` / ``Replace`` regex with Oniguruma: Unicode general
categories by short or long name (``\\p{M}``, ``\\p{Letter}``, ``\\P{L}``), possessive quantifiers
(``?+``, ``++``), ``{m,n}+`` as a repeated interval, ``\\s`` as Unicode White_Space (U+001C..U+001F
are not in it), and ``$`` as the end of a line. Python's ``re`` refuses the first two ("bad escape
\\p", "multiple repeat") and reads the last two differently; the reader rewrites each into ``re``
with the vendor's meaning. First field case: a 2026 Qwen tokenizer whose pre-tokenizer adds
``\\p{M}`` to the letter class, refused at load ("bad escape \\p at position 3629").

Reference pieces and ids: tokenizers 0.23.3, ``pre_tokenizers.Split(Regex(p), "isolated")
.pre_tokenize_str(text)``, and ``Tokenizer.encode(text).ids`` on the tokenizer built below.
What this test does if the code is wrong: every \\p{M} / possessive / long-name pattern raises,
and the tokenizer refuses to load (seen on the reader before the change).
"""
import pytest

from neurobrix.core.module.tokenizer.json_bpe import PyTokenizer, compile_regex

QWEN_MARKS = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}"
              r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")

PATTERNS = {
    "qwen marks": QWEN_MARKS,
    "possessive": r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+",
    "possessive gives nothing back": r"a++a|a{2}+|\S",
    "long names": r"\p{Letter}+|\p{Nd}+|\P{L}",
    "line end": r"\s+$",
}

TEXTS = ['\u0928\u092e\u0938\u094d\u0924\u0947 \u0915\u094d\u0937\u0924\u094d\u0930\u093f\u092f',
         '\xe9t\u0301\xe9 \u0301abc', "It's 12 KM!!\n\n  x", 'ctrl\x1cfs\x1dgs end', 'aaab aaaa',
         'ab  \ncd  \nef  ']

VENDOR_PIECES = {
    "qwen marks": [
        ['\u0928\u092e\u0938\u094d\u0924\u0947', ' \u0915\u094d\u0937\u0924\u094d\u0930\u093f\u092f'],
        ['\xe9t\u0301\xe9', ' \u0301abc'], ['It', "'s", ' ', '1', '2', ' KM', '!!\n\n', ' ', ' x'],
        ['ctrl', '\x1cfs', '\x1dgs', ' end'], ['aaab', ' aaaa'], ['ab', '  \n', 'cd', '  \n', 'ef', '  ']],
    "possessive": [
        ['\u0928\u092e\u0938', '\u094d\u0924', '\u0947', ' \u0915', '\u094d\u0937\u0924', '\u094d\u0930',
         '\u093f\u092f'],
        ['\xe9t', '\u0301\xe9', ' \u0301', 'abc'], ['It', "'s", ' ', '1', '2', ' KM', '!!\n\n', ' ', ' x'],
        ['ctrl', '\x1cfs', '\x1dgs', ' end'], ['aaab', ' aaaa'], ['ab', '  \n', 'cd', '  \n', 'ef', '  ']],
    "possessive gives nothing back": [
        ['\u0928', '\u092e', '\u0938', '\u094d', '\u0924', '\u0947', ' ', '\u0915', '\u094d', '\u0937', '\u0924',
         '\u094d', '\u0930', '\u093f', '\u092f'],
        ['\xe9', 't', '\u0301', '\xe9', ' ', '\u0301', 'a', 'b', 'c'],
        ['I', 't', "'", 's', ' ', '1', '2', ' ', 'K', 'M', '!', '!', '\n\n  ', 'x'],
        ['c', 't', 'r', 'l', '\x1c', 'f', 's', '\x1d', 'g', 's', ' ', 'e', 'n', 'd'],
        ['aa', 'a', 'b', ' ', 'aaaa'], ['a', 'b', '  \n', 'c', 'd', '  \n', 'e', 'f', '  ']],
    "long names": [
        ['\u0928\u092e\u0938', '\u094d', '\u0924', '\u0947', ' ', '\u0915', '\u094d', '\u0937\u0924', '\u094d',
         '\u0930', '\u093f', '\u092f'],
        ['\xe9t', '\u0301', '\xe9', ' ', '\u0301', 'abc'],
        ['It', "'", 's', ' ', '12', ' ', 'KM', '!', '!', '\n', '\n', ' ', ' ', 'x'],
        ['ctrl', '\x1c', 'fs', '\x1d', 'gs', ' ', 'end'], ['aaab', ' ', 'aaaa'],
        ['ab', ' ', ' ', '\n', 'cd', ' ', ' ', '\n', 'ef', ' ', ' ']],
    "line end": [
        ['\u0928\u092e\u0938\u094d\u0924\u0947 \u0915\u094d\u0937\u0924\u094d\u0930\u093f\u092f'],
        ['\xe9t\u0301\xe9 \u0301abc'], ["It's 12 KM!!", '\n', '\n  x'], ['ctrl\x1cfs\x1dgs end'], ['aaab aaaa'],
        ['ab', '  ', '\ncd', '  ', '\nef', '  ']],
}


def _isolated(pattern, text):
    """The vendor's Split "isolated": every match and every gap between matches, in order."""
    pieces, cursor = [], 0
    for m in compile_regex(pattern).finditer(text):
        if m.start() > cursor:
            pieces.append(text[cursor:m.start()])
        if m.end() > m.start():
            pieces.append(m.group(0))
        cursor = m.end()
    return pieces + ([text[cursor:]] if cursor < len(text) else [])


@pytest.mark.parametrize("name", sorted(PATTERNS))
def test_a_vendor_regex_splits_every_text_as_the_vendor_does(name):
    assert [_isolated(PATTERNS[name], t) for t in TEXTS] == VENDOR_PIECES[name]


def _byte_level_alphabet():
    keep = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) \
        + list(range(ord("\xae"), ord("\xff") + 1))
    shifted, n = [], 0
    for b in range(256):
        if b not in keep:
            shifted.append(chr(256 + n))
            n += 1
    return [chr(b) for b in keep] + shifted


def _qwen_marks_tokenizer_json():
    vocab = {c: i for i, c in enumerate(sorted(_byte_level_alphabet()))}
    merges = [["\xe0", "\xa4"], ["\xe0\xa4", "\xa8"], ["\u0120", "\xe0\xa4\xa8"], ["\xcc", "\u0123"],
              ["\u0120", "\xcc"]]
    for a, b in merges:
        vocab[a + b] = len(vocab)
    return {"version": "1.0", "added_tokens": [], "normalizer": None, "post_processor": None,
            "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": QWEN_MARKS}, "behavior": "Isolated", "invert": False},
                {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False}]},
            "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
            "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
                      "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False,
                      "ignore_merges": False, "vocab": vocab, "merges": merges}}


VENDOR_IDS = {
    '\u0928\u092e\u0938\u094d\u0924\u0947 \u0928 \u0301x':
        [257, 256, 106, 256, 116, 156, 98, 235, 256, 97, 156, 98, 229, 258, 220, 259, 87],
    'e\u0301\u0301 n\u0301': [68, 259, 259, 220, 77, 259],
}


@pytest.mark.parametrize("text", sorted(VENDOR_IDS))
def test_a_tokenizer_json_with_a_mark_class_loads_and_gives_the_vendor_ids(text):
    tok = PyTokenizer(_qwen_marks_tokenizer_json())
    assert tok.encode(text).ids == VENDOR_IDS[text]
    assert tok.decode(VENDOR_IDS[text]) == text
