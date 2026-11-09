"""token_bytes(id), one rule per tokenizer class (constrained decoding's vocabulary, reste_apres_064 point 9a; the Dell
2026-11-09 21:35): the bytes a token stands for INSIDE a text — what a grammar engine walks — or None for a token no
grammar may emit (special, control, unknown). Card-less: a small tokenizer of each class built here. Checked on the real
catalogue tokenizers too (campagnes/2026_11_09_constrained_decoding/token_bytes: Qwen3, TinyLlama, Voxtral, PixArt T5)."""
from neurobrix.core.module.tokenizer.json_bpe import PyTokenizer, _BYTE_ENCODER
from neurobrix.core.module.tokenizer.tekken_bpe import PyTekken
from neurobrix.core.module.tokenizer.tiktoken_bpe import PyTiktoken


def _bytelevel(b: bytes) -> str:
    return "".join(_BYTE_ENCODER[x] for x in b)


def test_a_byte_level_token_is_its_bytes():
    vocab = {_bytelevel(b'"'): 0, _bytelevel(b' name'): 1, _bytelevel("é".encode()): 2, "<|eos|>": 3}
    tk = PyTokenizer({"model": {"type": "BPE", "vocab": vocab, "merges": []}, "decoder": {"type": "ByteLevel"},
                      "added_tokens": [{"id": 3, "content": "<|eos|>", "special": True}]})
    assert tk.token_bytes(0) == b'"' and tk.token_bytes(1) == b" name" and tk.token_bytes(2) == "é".encode()
    assert tk.token_bytes(3) is None                                   # special: never a grammar's


def test_a_metaspace_byte_fallback_token_is_its_text_or_its_byte_and_keeps_its_space():
    vocab = {"▁{": 0, "<0x22>": 1, "name": 2, "</s>": 3}
    decoder = {"type": "Sequence", "decoders": [
        {"type": "Replace", "pattern": {"String": "▁"}, "content": " "}, {"type": "ByteFallback"}, {"type": "Fuse"},
        {"type": "Strip", "content": " ", "start": 1, "stop": 0}]}                       # Llama's chain
    tk = PyTokenizer({"model": {"type": "BPE", "vocab": vocab, "merges": [], "byte_fallback": True},
                      "decoder": decoder, "added_tokens": [{"id": 3, "content": "</s>", "special": True}]})
    assert tk.token_bytes(0) == b" {"            # the first-token strip after Fuse is a sequence rule, not the token's
    assert tk.token_bytes(1) == b'"' and tk.token_bytes(2) == b"name" and tk.token_bytes(3) is None


def test_tiktoken_and_tekken_tokens_are_their_ranks_bytes():
    ranks = {b"a": 0, b'{"': 1, b"\xe2\x80": 2}
    tt = PyTiktoken(ranks, special_tokens={"<|endoftext|>": 3})
    assert tt.token_bytes(1) == b'{"' and tt.token_bytes(2) == b"\xe2\x80" and tt.token_bytes(3) is None
    tk = PyTekken(ranks, [(0, "<unk>"), (1, "<s>"), (2, "</s>")], r"\S+", num_special_tokens=3, vocab_size=6)
    assert tk.token_bytes(0) is None and tk.token_bytes(2) is None   # the special ids come first
    assert tk.token_bytes(3) == b"a" and tk.token_bytes(4) == b'{"'
