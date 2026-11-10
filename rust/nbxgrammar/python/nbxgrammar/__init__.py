"""nbxgrammar: NeuroBrix's grammar engine (this repository's Rust crate, std only) through its C ABI, loaded with ctypes - no
binding library. The shared library lives beside this file (built by the package's build from the crate and placed
here). Masks are int32 words, ceil(vocab / 32), bit v % 32 of word v // 32 = token v allowed."""
from __future__ import annotations

import ctypes
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

_NAME = {"darwin": "libnbxgrammar.dylib", "win32": "nbxgrammar.dll"}.get(sys.platform, "libnbxgrammar.so")
_PATH = Path(__file__).with_name(_NAME)
if not _PATH.is_file():
    raise ImportError(f"nbxgrammar: its library {_PATH} is not in the package (install a wheel: pip install nbxgrammar; from a checkout: pip install .)")
_lib = ctypes.CDLL(str(_PATH))
_P, _U32, _SZ = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_size_t
_lib.ng_vocab_new.restype = _P
_lib.ng_vocab_new.argtypes = [ctypes.c_char_p, ctypes.POINTER(_U32), _U32, ctypes.POINTER(_U32), _U32]
_lib.ng_vocab_free.argtypes = [_P]
_lib.ng_grammar_json_schema.restype = _P
_lib.ng_grammar_json_schema.argtypes = [ctypes.c_char_p, _SZ, ctypes.c_char_p, _SZ]
_lib.ng_grammar_description.restype = _P
_lib.ng_grammar_description.argtypes = [ctypes.c_char_p, _SZ, ctypes.c_char_p, _SZ]
_lib.ng_grammar_free.argtypes = [_P]
_lib.ng_matcher_new.restype = _P
_lib.ng_matcher_new.argtypes = [_P]
_lib.ng_matcher_free.argtypes = [_P]
_lib.ng_mask.restype = ctypes.c_int32
_lib.ng_mask.argtypes = [_P, _P, ctypes.POINTER(_U32), _SZ]
_lib.ng_advance.restype = ctypes.c_int32
_lib.ng_advance.argtypes = [_P, _P, _U32]
_lib.ng_accepting.restype = ctypes.c_int32
_lib.ng_accepting.argtypes = [_P]


class Vocab:
    """A tokenizer's tokens as bytes (None: a token no grammar emits) and its end-of-sequence ids; built once."""

    def __init__(self, token_bytes: Sequence[Optional[bytes]], eos: Sequence[int]):
        lens = np.array([0xFFFFFFFF if b is None else len(b) for b in token_bytes], dtype=np.uint32)
        data = b"".join(b for b in token_bytes if b is not None)
        e = np.array(list(eos), dtype=np.uint32)
        self.size = len(lens)
        self.words = (self.size + 31) // 32
        self._h = _lib.ng_vocab_new(data, lens.ctypes.data_as(ctypes.POINTER(_U32)), self.size,
                                    e.ctypes.data_as(ctypes.POINTER(_U32)), len(e))
        if not self._h:
            raise ValueError(f"nbxgrammar: an end-of-sequence id of {list(eos)} lies past the {self.size} tokens")

    def __del__(self):
        if getattr(self, "_h", None):
            _lib.ng_vocab_free(self._h)
            self._h = None


class Grammar:
    def __init__(self, handle):
        self._h = handle

    @classmethod
    def json_schema(cls, text: str) -> "Grammar":
        """A JSON schema compiled; ValueError naming the construct when it cannot be enforced."""
        raw = text.encode("utf-8")
        err = ctypes.create_string_buffer(512)
        h = _lib.ng_grammar_json_schema(raw, len(raw), err, len(err))
        if not h:
            raise ValueError(err.value.decode("utf-8", "replace"))
        return cls(h)

    @classmethod
    def description(cls, text: str) -> "Grammar":
        """A grammar description (JSON: lit / schema / seq / choice / repeat / until / token) compiled; ValueError
        naming what cannot be enforced."""
        raw = text.encode("utf-8")
        err = ctypes.create_string_buffer(512)
        h = _lib.ng_grammar_description(raw, len(raw), err, len(err))
        if not h:
            raise ValueError(err.value.decode("utf-8", "replace"))
        return cls(h)

    def __del__(self):
        if getattr(self, "_h", None):
            _lib.ng_grammar_free(self._h)
            self._h = None


class Matcher:
    """One request's position in its grammar (keeps its grammar alive)."""

    def __init__(self, grammar: Grammar):
        self._grammar = grammar
        self._h = _lib.ng_matcher_new(grammar._h)

    def mask(self, vocab: Vocab) -> np.ndarray:
        out = np.zeros(vocab.words, dtype=np.uint32)
        if _lib.ng_mask(vocab._h, self._h, out.ctypes.data_as(ctypes.POINTER(_U32)), vocab.words) != 0:
            raise ValueError("nbxgrammar: a mask of the wrong size")
        return out.view(np.int32)

    def advance(self, vocab: Vocab, token: int) -> None:
        if _lib.ng_advance(vocab._h, self._h, int(token)) != 0:
            raise ValueError(f"nbxgrammar: token {token} is not allowed here")

    def accepting(self) -> bool:
        return bool(_lib.ng_accepting(self._h))

    def __del__(self):
        if getattr(self, "_h", None):
            _lib.ng_matcher_free(self._h)
            self._h = None
