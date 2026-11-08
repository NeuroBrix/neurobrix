"""The text of a live stream, decoded by the engine (Studio request 2).

The daemon's streaming RPC used to carry token ids only; the client needed a
tokenizer of its own to show text as it arrived — and the tracker's rule is
that Studio ships no detokeniser (R34 from the other side: decoding is the
engine's, the model's tokenizer lives inside the container). This brick turns
the per-token events into text deltas with the engine's own tokenizer.

A token is not a character: a byte-level BPE token can end in the middle of a
multi-byte character, and a token's surface form can depend on the token
before it (a leading space merged, a piece re-joined). So a delta is never the
decode of one id; it is the difference between the decode of the whole
generated prefix and what was already emitted, and a prefix whose decode ends
in U+FFFD (an incomplete sequence) is held until the next token completes it.
When a later token changes text already emitted, the event says how many
characters to take back (`rewind`) beside the text to append — the client
applies both, and never decodes.

Cost: one decode of the generated prefix per token, bounded by the request's
`max_tokens`. Not measured against the step it rides on; measure before
claiming it is small.
"""
from __future__ import annotations

from typing import Any, Dict, List

_INCOMPLETE = "�"


class TextDeltaDecoder:
    """Feeds token ids in order, returns the text to append and how many
    already-emitted characters to take back first."""

    def __init__(self, tokenizer: Any) -> None:
        if not hasattr(tokenizer, "decode"):
            raise TypeError("TextDeltaDecoder needs a tokenizer with `decode`")
        self._tokenizer = tokenizer
        self._ids: List[int] = []
        self._emitted = ""

    @property
    def text(self) -> str:
        """Everything emitted so far (after rewinds)."""
        return self._emitted

    def push(self, token_id: int) -> Dict[str, Any]:
        self._ids.append(int(token_id))
        full = self._tokenizer.decode(self._ids, skip_special_tokens=True)
        if full.endswith(_INCOMPLETE):
            # A multi-byte character is still open: nothing stable to add yet.
            return {"text": "", "rewind": 0}
        common = 0
        for a, b in zip(self._emitted, full):
            if a != b:
                break
            common += 1
        rewind = len(self._emitted) - common
        delta = full[common:]
        self._emitted = full
        return {"text": delta, "rewind": rewind}


class MarkerDecoder:
    """Decodes token ids with the special tokens skipped, EXCEPT the ones whose text is a dialect marker.

    A tool-call dialect may open its block with a special token (`<|tool_call|>` is one in its tokenizer):
    the display decode (`skip_special_tokens=True`) drops it, and the call then reads as prose. This decode keeps
    exactly the special tokens whose own text is one of `keep`, and skips every other special token as the
    display decode does. Which ids are special is read from the tokenizer itself, id by id (a special token
    decodes to nothing when skipped and to its text when not), never from a list.
    """

    def __init__(self, tokenizer: Any, keep) -> None:
        if not hasattr(tokenizer, "decode"):
            raise TypeError("MarkerDecoder needs a tokenizer with `decode`")
        self._tokenizer = tokenizer
        self._keep = frozenset(k for k in keep if k)
        self._kept: Dict[int, Any] = {}

    def _kept_text(self, token_id: int):
        if token_id not in self._kept:
            text = None
            if self._keep and self._tokenizer.decode([token_id], skip_special_tokens=True) == "":
                surface = self._tokenizer.decode([token_id], skip_special_tokens=False)
                if surface in self._keep:
                    text = surface
            self._kept[token_id] = text
        return self._kept[token_id]

    def decode(self, ids: List[int]) -> str:
        if not self._keep:
            return self._tokenizer.decode(list(ids), skip_special_tokens=True)
        out: List[str] = []
        run: List[int] = []
        for token_id in ids:
            kept = self._kept_text(int(token_id))
            if kept is None:
                run.append(int(token_id))
                continue
            if run:
                out.append(self._tokenizer.decode(run, skip_special_tokens=True))
                run = []
            out.append(kept)
        if run:
            out.append(self._tokenizer.decode(run, skip_special_tokens=True))
        return "".join(out)
