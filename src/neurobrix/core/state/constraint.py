"""The constraint a request's decode follows (constrained decoding, reste_apres_064 point 9a) — the seam between a
grammar engine and the generators, the same for every grammar engine.

A constraint is made by the serving layer from the request (a JSON schema, a tool-call envelope) and handed to the
flow (`FlowContext.constraint`); both generators ask it, before each step, for the tokens allowed next and tell it the
token chosen after. A request without one pays nothing. The mask is int32 words, ceil(vocab / 32), bit v % 32 of word
v // 32 set when token v is allowed (the layout nbxtensor.ops.token_bitmask applies); a constraint allows the end of
sequence only where its output may end."""
from __future__ import annotations

from typing import Protocol

import numpy as np


class Constraint(Protocol):
    def mask(self, vocab: int) -> np.ndarray:
        """int32 [ceil(vocab / 32)]: the tokens allowed at the next step."""

    def advance(self, token_id: int) -> None:
        """The token chosen at this step; refused by name when the mask did not allow it."""


def step_mask(constraint, vocab: int) -> np.ndarray:
    """The constraint's words for the next step, checked once for both engines: ceil(vocab / 32) int32 words, at
    least one token allowed (an empty mask refused by name, before either engine applies it)."""
    words = np.ascontiguousarray(constraint.mask(int(vocab)), dtype=np.int32).reshape(-1)
    if words.shape[0] != -(-int(vocab) // 32):
        raise RuntimeError(f"ZERO FALLBACK: the constraint gave {words.shape[0]} mask words for {vocab} logits")
    if not words.any():
        raise RuntimeError("ZERO FALLBACK: the constraint allows no token at this step (an empty mask)")
    return words


def allowed_tokens(words: np.ndarray, vocab: int) -> np.ndarray:
    """The words decoded to one bool per token (bit v % 32 of word v // 32) — the compiled engine's mask."""
    return np.unpackbits(words.astype("<i4").view(np.uint8), bitorder="little")[:int(vocab)].astype(bool)


def compile_json_schema(text: str):
    """A JSON schema compiled by our grammar engine (nbxgrammar, its own package: pip install nbxgrammar); ValueError naming the construct it cannot
    enforce; a refusal by name when the engine is not installed."""
    try:
        import nbxgrammar
    except ImportError as e:
        raise RuntimeError(f"ZERO FALLBACK: constrained decoding needs the grammar engine nbxgrammar, not installed here: pip install nbxgrammar ({e})") from e
    return nbxgrammar.Grammar.json_schema(text)


def compile_description(text: str):
    """A grammar description (agent/envelope.py writes the tool-call envelopes in it) compiled by our grammar engine;
    ValueError naming what it cannot enforce."""
    try:
        import nbxgrammar
    except ImportError as e:
        raise RuntimeError(f"ZERO FALLBACK: constrained decoding needs the grammar engine nbxgrammar, not installed here: pip install nbxgrammar ({e})") from e
    return nbxgrammar.Grammar.description(text)


def constraint_for(grammar, tokenizer, defaults, cache: dict):
    """The generators' constraint of a compiled grammar (None for None): it ends on the model's end of sequence, read
    from the container's defaults — refused by name when they declare none — and keeps its vocabulary in `cache`."""
    if grammar is None:
        return None
    eos = defaults.get("eos_token_id")
    if eos is None or eos == []:
        raise RuntimeError("ZERO FALLBACK: a constrained decode ends on the model's end of sequence, and this "
                           "container's defaults declare no eos_token_id")
    return GrammarConstraint(grammar, tokenizer, eos if isinstance(eos, list) else [eos], cache)


class GrammarConstraint:
    """A request's grammar as the generators' constraint. The vocabulary (every token's bytes, `token_bytes` by
    tokenizer class) is built once per tokenizer and logits width and kept in `cache` (the serving engine's)."""

    def __init__(self, grammar, tokenizer, eos_ids, cache: dict):
        import nbxgrammar
        self._grammar, self._tokenizer, self._eos, self._cache = grammar, tokenizer, list(eos_ids), cache
        self._matcher = nbxgrammar.Matcher(grammar)
        self._vocab = None

    def _vocabulary(self, width: int):
        key = (self._tokenizer, int(width), tuple(self._eos))      # the object itself: an id is reused once freed
        if key not in self._cache:
            import nbxgrammar
            # the logits may be wider than the tokenizer (padded heads): an id it does not know has no bytes
            self._cache[key] = nbxgrammar.Vocab([self._tokenizer.token_bytes(i) for i in range(int(width))], self._eos)
        return self._cache[key]

    def mask(self, vocab: int) -> np.ndarray:
        self._vocab = self._vocabulary(vocab)
        return self._matcher.mask(self._vocab)

    def advance(self, token_id: int) -> None:
        self._matcher.advance(self._vocab, token_id)
