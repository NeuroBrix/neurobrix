"""The engine's grammar constraint (core/state/constraint.py over nbx_grammar): its vocabulary is the tokenizer's
token_bytes at the logits' width (ids the tokenizer does not know never allowed), built once and kept, and its masks
lead only to output that validates. Card-less."""
import json

import numpy as np
import pytest

from tests.unit.grammar_engine import load  # noqa: E402

nbx_grammar = load()
from neurobrix.core.state.constraint import GrammarConstraint, compile_json_schema  # noqa: E402

TOKS = [b"{", b'"ok"', b":", b"true", b"false", b"}", b" ", None]   # id 7: a special token


class Tok:
    def token_bytes(self, i):
        return TOKS[i] if i < len(TOKS) else None


def _allowed(words, width):
    w = np.asarray(words).view(np.uint32)
    return [t for t in range(width) if w[t >> 5] >> (t & 31) & 1]


def test_the_constraint_allows_only_what_the_schema_lets_follow_at_the_logits_width():
    g = compile_json_schema(json.dumps({"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}))
    cache, tok = {}, Tok()
    c = GrammarConstraint(g, tok, [8], cache)
    width = 70                                         # a head wider than the tokenizer: ids 8..69 unknown but EOS
    out = []
    for expect, pick in (([0], 0),           # no whitespace before the root value
                          ([1, 6], 1), ([2, 6], 2), ([3, 4, 6], 3), ([5, 6], 5)):
        allowed = _allowed(c.mask(width), width)
        assert allowed == expect, (out, allowed)
        c.advance(pick)
        out.append(TOKS[pick])
    assert _allowed(c.mask(width), width) == [8]        # the object closed: only the end of sequence
    assert json.loads(b"".join(out)) == {"ok": True}
    assert len(cache) == 1                              # the vocabulary built once for this tokenizer and width
    GrammarConstraint(g, tok, [8], cache).mask(width)  # a second request on the same tokenizer reuses it
    assert len(cache) == 1
    with pytest.raises(ValueError):                     # after the object, nothing but the end of sequence
        c.advance(0)


def test_a_schema_the_engine_cannot_enforce_is_refused_by_name():
    with pytest.raises(ValueError, match="pattern"):
        compile_json_schema(json.dumps({"type": "string", "pattern": "x"}))


def test_an_eos_past_the_logits_and_an_empty_mask_are_refused_by_name():
    """An end-of-sequence id past the logits' width is refused when the vocabulary is built (never a panic across the
    ABI, which would kill the daemon); a constraint that allows nothing is refused before either engine applies it."""
    from neurobrix.core.state.constraint import allowed_tokens, step_mask
    g = compile_json_schema(json.dumps({"type": "boolean"}))
    with pytest.raises(ValueError, match="end-of-sequence"):
        GrammarConstraint(g, Tok(), [500], {}).mask(70)

    class Nothing:
        def mask(self, vocab):
            return np.zeros(-(-vocab // 32), dtype=np.int32)
    with pytest.raises(RuntimeError, match="allows no token"):
        step_mask(Nothing(), 70)
    words = np.zeros(3, dtype=np.int32)
    words[1] = 1 << 4
    assert np.flatnonzero(allowed_tokens(words, 70)).tolist() == [36]
