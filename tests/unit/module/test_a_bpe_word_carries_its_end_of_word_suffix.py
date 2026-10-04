"""The tokenizer.json interpreter applies the BPE model's end_of_word_suffix and
continuing_subword_prefix as tokenizers' BPE::merge_word does (0.23.2, models/bpe/model.rs):
every character after the first carries the prefix, the last one carries the suffix, and a
merge drops the right side's prefix (BpeBuilder::build, `&b[prefix_len..]`).

The interpreter read both values and applied neither. Every CLIP tokenizer declares
end_of_word_suffix "</w>": Open-Sora v2's text_encoder_2 was fed "a" -> 64 (the mid-word
piece) where CLIP's vocabulary has "a</w>" -> 320, twelve pieces for ten words, and its pooled
prompt vector came out at cos 0.566 against the reference pipeline's (2026-10-04, PyTorch-
sequential raw dump: the CLIP model itself matched at 0.9999996 on the ids it was given).
Reference ids: tokenizers 0.22.2 on the installed container's tokenizer.json. What this test
would do if the code were wrong: the suffix ignored gives 64 for "a" (seen failing on injection).
"""
import json
from pathlib import Path

import pytest

from neurobrix.core.module.tokenizer.json_bpe import PyTokenizer


def _bpe_json(model):
    return {"version": "1.0", "added_tokens": [], "normalizer": None,
            "pre_tokenizer": {"type": "Whitespace"}, "post_processor": None, "decoder": None,
            "model": dict({"type": "BPE", "dropout": None, "unk_token": "<unk>", "fuse_unk": False}, **model)}


def test_the_last_character_carries_the_end_of_word_suffix():
    vocab = {"<unk>": 0, "a": 1, "b": 2, "a</w>": 3, "b</w>": 4, "ab</w>": 5, "ab": 6}
    t = PyTokenizer(_bpe_json({"vocab": vocab, "merges": ["a b</w>", "a b"], "end_of_word_suffix": "</w>",
                               "continuing_subword_prefix": None}))
    assert t.encode("a", add_special_tokens=False).ids == [3]           # a single character is a whole word
    assert t.encode("ab", add_special_tokens=False).ids == [5]          # merged with the suffixed b</w>
    assert t.encode("ab a", add_special_tokens=False).ids == [5, 3]


def test_a_merge_drops_the_right_side_s_continuing_prefix():
    vocab = {"<unk>": 0, "a": 1, "##b": 2, "##c": 3, "ab": 4, "abc": 5}
    t = PyTokenizer(_bpe_json({"vocab": vocab, "merges": ["a ##b", "ab ##c"], "end_of_word_suffix": None,
                               "continuing_subword_prefix": "##"}))
    assert t.encode("abc", add_special_tokens=False).ids == [5]
    assert t.encode("ab", add_special_tokens=False).ids == [4]


def test_without_either_value_a_word_is_its_characters_merged_as_before():
    vocab = {"<unk>": 0, "a": 1, "b": 2, "ab": 3}
    t = PyTokenizer(_bpe_json({"vocab": vocab, "merges": ["a b"]}))
    assert t.encode("ab a", add_special_tokens=False).ids == [3, 1]


# tokenizers 0.22.2, Tokenizer.from_file(<Open-Sora-v2>/modules/tokenizer_2/tokenizer.json).encode(p).ids
CLIP_REFERENCE = {
    "a red apple rolling slowly across a wooden table": [49406, 320, 736, 3055, 6347, 9568, 2500, 320, 9057, 2175, 49407],
    "It's 4K: a cat's 2 eyes, glowing!!": [49406, 585, 568, 275, 330, 281, 320, 2368, 568, 273, 3095, 267, 18437, 748, 49407],
    "Héllo   WORLD  x": [49406, 71, 3459, 19293, 1002, 343, 49407],
}


def _clip_tokenizer_json():
    from neurobrix.core.paths import cache_dir
    p = Path(cache_dir()) / "Open-Sora-v2" / "modules" / "tokenizer_2" / "tokenizer.json"
    return p if p.exists() else None


@pytest.mark.parametrize("prompt", sorted(CLIP_REFERENCE))
def test_a_clip_tokenizer_json_gives_the_reference_ids(prompt):
    p = _clip_tokenizer_json()
    if p is None:
        pytest.skip("Open-Sora-v2 is not installed here")
    data = json.loads(p.read_text())
    assert data["model"]["end_of_word_suffix"] == "</w>"
    assert PyTokenizer(data).encode(prompt).ids == CLIP_REFERENCE[prompt]
