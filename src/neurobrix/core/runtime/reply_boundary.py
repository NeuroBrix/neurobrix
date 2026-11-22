"""The boundary between the prompt a chat template renders and the reply the model writes after it.

A template may end its generation prompt BEFORE the space it writes between the assistant marker and the content
(`Assistant:` then `' ' + content`). A model trained on it emits the space as part of its first token (`ĠHello`), so
the reply decoded on its own starts with that space; stored in the history, the template then writes its own space
before it, and the prompt the model sees the next turn holds two. The reply's own copy of the gap is the one to
remove, once, at the boundary of the reply.

The gap is read from the template itself, never from a model name: one history turn with a sentinel content is
rendered, and the text between the end of the generation prompt and the sentinel is the template's gap (the
mechanism `turn.reasoning_gap` reads a reasoning's gap with: `render_one_turn`). A template whose generation prompt
already ends after its gap has an empty gap, and nothing is removed.

ONE function (`without_reply_gap`) gives the text everywhere a reply is shown or kept: the display decode, the
stored history, the API's content, and the streamer's deltas.
"""
from __future__ import annotations

import os
import weakref
from typing import Any, Dict, Optional

_SENTINEL = "reply-0f3b"
_USER = {"role": "user", "content": "u"}


def render_one_turn(tokenizer: Any, assistant: Dict[str, Any]) -> str:
    """One past turn as the chat template writes it: a user message and `assistant`, no generation prompt."""
    return tokenizer.apply_chat_template([dict(_USER), dict(assistant)], tokenize=False,
                                         add_generation_prompt=False)


_gaps: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def reply_gap(tokenizer: Any) -> str:
    """The whitespace the template writes between its generation prompt and an assistant reply ('' when the
    generation prompt already ends after it, when the template shows the turn differently from what it opens (a
    marker written only in history), or when the tokenizer has no chat template)."""
    if tokenizer is None or not callable(getattr(tokenizer, "apply_chat_template", None)):
        return ""
    has = getattr(tokenizer, "has_chat_template", None)
    if callable(has) and not has():
        return ""
    try:
        return _gaps[tokenizer]
    except (KeyError, TypeError):
        pass
    prompt = tokenizer.apply_chat_template([dict(_USER)], tokenize=False, add_generation_prompt=True)
    shown = render_one_turn(tokenizer, {"role": "assistant", "content": _SENTINEL})
    at = shown.find(_SENTINEL)
    gap = ""
    if at >= 0 and isinstance(prompt, str) and shown.startswith(prompt):
        gap = shown[len(prompt):at]
        if gap.strip():                 # a marker, not a space: the reply boundary is whitespace only
            gap = ""
    try:
        _gaps[tokenizer] = gap
    except TypeError:
        pass
    return gap


def without_reply_gap(text: str, gap: Optional[str]) -> str:
    """`text` without the part of the template's gap the reply starts with: the common prefix of the two, once. A
    reply that does not start with the gap, or an empty gap, leaves it as it is; never a blanket strip."""
    if not gap or not text:
        return text
    return text[len(os.path.commonprefix([gap, text])):]
