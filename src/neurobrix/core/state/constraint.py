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


def compile_json_schema(text: str):
    """A JSON schema compiled by our grammar engine (nbx_grammar, rust/); ValueError naming the construct it cannot
    enforce; a refusal by name when the engine is not installed."""
    try:
        import nbx_grammar
    except ImportError as e:
        raise RuntimeError(f"ZERO FALLBACK: constrained decoding needs the grammar engine nbx_grammar ({e})") from e
    return nbx_grammar.Grammar.json_schema(text)


class GrammarConstraint:
    """A request's grammar as the generators' constraint. The vocabulary (every token's bytes, `token_bytes` by
    tokenizer class) is built once per tokenizer and logits width and kept in `cache` (the serving engine's)."""

    def __init__(self, grammar, tokenizer, eos_ids, cache: dict):
        import nbx_grammar
        self._grammar, self._tokenizer, self._eos, self._cache = grammar, tokenizer, list(eos_ids), cache
        self._matcher = nbx_grammar.Matcher(grammar)
        self._vocab = None

    def _vocabulary(self, width: int):
        key = (id(self._tokenizer), int(width))
        if key not in self._cache:
            import nbx_grammar
            # the logits may be wider than the tokenizer (padded heads): an id it does not know has no bytes
            self._cache[key] = nbx_grammar.Vocab([self._tokenizer.token_bytes(i) for i in range(int(width))], self._eos)
        return self._cache[key]

    def mask(self, vocab: int) -> np.ndarray:
        self._vocab = self._vocabulary(vocab)
        return self._matcher.mask(self._vocab)

    def advance(self, token_id: int) -> None:
        self._matcher.advance(self._vocab, token_id)
