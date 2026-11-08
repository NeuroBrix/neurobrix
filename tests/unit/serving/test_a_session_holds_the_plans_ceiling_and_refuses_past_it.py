"""The chat session's ceiling is the plan's, and past it the turn is refused by name (docs/design/prism_kv.md §5bis,
§7; the owner's option C, 2026-11-08).

Before: `ConversationSession` reported a context of `defaults.get("max_length", 4096)` that nothing in the plan
supported; it summarised old turns with the loaded model behind an `except Exception: pass`, truncated the history
when the summary came back empty, and counted tokens as characters / 4 when the template failed. Each of those
changed the conversation, or its reported size, without saying so.

What is read here is the module's behaviour, not its text: the ceiling reported against a container that declares
no `max_length`, the refusal naming the ceiling and its plan fact, the conversation left as it was, no generation
asked for a summary, and a template that cannot render surfacing as the error it is.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from nbxtensor.memory.kv_session import KVSession, held_ids
from neurobrix.serving.session import ConversationSession, play_turn, session_ceiling


class _Tokenizer:
    """One id per character of every message, plus one for the generation prompt."""

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        ids = [ord(c) for m in messages for c in m["content"]]
        return ids + [1] if add_generation_prompt else ids


def _plan(reach=None, max_cache_len=300):
    return SimpleNamespace(session_reach=reach, max_cache_len=max_cache_len)


def test_the_ceiling_is_the_plans_session_reach_and_never_a_default():
    assert session_ceiling(_plan(reach=1984, max_cache_len=640)) == (
        1984, "the plan's session reach (KVCachePlan.session_reach)")
    n, source = session_ceiling(_plan(reach=None, max_cache_len=640))
    assert n == 640 and "max_cache_len" in source
    with pytest.raises(RuntimeError, match="no KV cache"):
        session_ceiling(None)
    s = ConversationSession(_Tokenizer(), *session_ceiling(_plan(reach=777)))
    s.add_user_message("hello")
    info = s.get_context_info()
    assert info["max_context"] == 777, "the context reported is the plan's ceiling"
    assert info["tokens"] == len("hello") + 1, "the count is the template's, exactly"


def test_past_the_ceiling_the_turn_is_refused_by_name_and_nothing_is_summarised():
    s = ConversationSession(_Tokenizer(), 40, "the plan's session reach (KVCachePlan.session_reach)")
    asked = []

    def generate(ids, kv):
        asked.append(list(ids))
        return {"text": "ok"}

    play_turn(s, "x" * 10, 8, generate)
    before = [dict(m) for m in s.messages]
    with pytest.raises(RuntimeError) as refused:
        play_turn(s, "y" * 30, 8, generate)
    msg = str(refused.value)
    assert "at most 40" in msg and "session_reach" in msg, msg
    assert s.messages == before, "a refused turn left the conversation changed"
    assert len(asked) == 1, "a generation ran for the refused turn (a summary, a truncated retry)"


def test_a_template_that_cannot_render_is_the_error_never_an_estimate():
    class _Broken(_Tokenizer):
        def apply_chat_template(self, *a, **k):
            raise ValueError("template error")

    s = ConversationSession(_Broken(), 100, "src")
    s.add_user_message("hello")
    with pytest.raises(ValueError, match="template error"):
        s.get_context_info()
    with pytest.raises(RuntimeError, match="apply_chat_template"):
        ConversationSession(object(), 100, "src")


def test_the_session_matches_held_ids_and_always_prefills_the_last_position():
    kv = KVSession(100, "src")
    assert kv.match([5, 6, 7]) == 0
    kv.record([5, 6, 7, 8], 0, 3, 3, 0)
    assert kv.match([5, 6, 7, 8, 9, 10]) == 4
    assert kv.match([5, 6, 9, 9]) == 2, "a changed rendering shortens the match"
    assert kv.match([5, 6, 7, 8]) == 3, "the last position is always prefilled (its logits)"
    kv.forget()
    assert kv.match([5, 6, 7, 8, 9]) == 0
    assert held_ids([1, 2], [3, 4, 5], 4) == [1, 2, 3, 4]
    with pytest.raises(RuntimeError, match="holds 6 positions"):
        held_ids([1, 2], [3], 6)


def test_a_cache_another_request_cleared_holds_nothing_for_the_session():
    owner, cache = object(), SimpleNamespace(holder=None)
    kv = KVSession(100, "src")
    kv.bind(owner, cache, cache)
    kv.record([1, 2, 3], 0, 3, 3, 0)
    assert kv.held(owner) is cache
    cache.holder = None                         # a request without a session cleared it
    assert kv.held(owner) is None
    kv.bind(owner, cache, cache)
    assert kv.token_ids == [], "positions another request may have rewritten were kept"
