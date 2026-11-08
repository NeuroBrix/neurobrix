"""
ConversationSession — multi-turn chat state for LLM models, and the KV cache it owns.

The KV cache belongs to the session (docs/design/prism_kv.md §5bis, scope A): its positions stay between turns, and
turn n prefills only the tokens the cache does not already hold, on both engines. The conversation is rendered by the
model's own chat template at every turn and matched against the held ids by the flow (core/state/kv_session.py), so
nothing assumes the rendering of earlier turns is stable.

The ceiling is the plan's: the positions a session may hold on this device (`session_ceiling`). A turn whose rendered
conversation and generation budget pass it is refused by name — the conversation is never summarised or truncated
behind the user's back (the owner's decision, 2026-11-08).

`play_turn` is the one turn both the daemon (`InferenceEngine.chat`) and `neurobrix run --turns` play.
"""

from typing import Any, Callable, Dict, List, Tuple

from nbxtensor.memory.kv_session import KVSession


def session_ceiling(kv_plan) -> Tuple[int, str]:
    """(positions, the plan fact they come from) a serving session may hold. The plan's session reach — what a serving
    session holds in the room the cache was sized from, within the declared window — when it states one; otherwise its
    cache (`max_cache_len`), which a serve plan sizes up to the window itself. A plan without a KV cache is refused."""
    if kv_plan is None:
        raise RuntimeError(
            "ZERO FALLBACK: the plan carries no KV cache for this decode session. Prism plans the cache of every "
            "flow that decodes (core/runtime/lm_facts); a plan without one did not see this flow.")
    reach = getattr(kv_plan, "session_reach", None)
    if reach:
        return int(reach), "the plan's session reach (KVCachePlan.session_reach)"
    return int(kv_plan.max_cache_len), ("the plan's KV cache (KVCachePlan.max_cache_len; the plan states no session "
                                        "reach: the decoder's state declares no window, or a serving floor does not "
                                        "fit the room)")


class ConversationSession:
    """
    Multi-turn conversation state for LLM models: the messages, and the KV cache record the flows keep between turns.

    ZERO HARDCODE: the ceiling is the plan's (`session_ceiling`), passed in; nothing here knows a model.
    """

    def __init__(self, tokenizer: Any, ceiling: int, ceiling_source: str):
        if not callable(getattr(tokenizer, "apply_chat_template", None)):
            raise RuntimeError("ZERO FALLBACK: a chat session renders the conversation with the model's chat "
                               "template, and this tokenizer has no apply_chat_template()")
        self.messages: List[Dict[str, str]] = []
        self._tokenizer = tokenizer
        self.kv = KVSession(ceiling, ceiling_source)

    @property
    def ceiling(self) -> int:
        return self.kv.ceiling

    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation."""
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant response to the conversation."""
        self.messages.append({"role": "assistant", "content": content})

    def build_token_ids(self) -> List[int]:
        """The conversation rendered by the model's chat template, with the generation prompt: the ids of the turn."""
        from neurobrix.core.runtime.prompt_length import _flatten_tokenizer_output   # the flows' one rule
        ids = _flatten_tokenizer_output(
            self._tokenizer.apply_chat_template(self.messages, tokenize=True, add_generation_prompt=True))
        if not ids:
            raise RuntimeError("ZERO FALLBACK: the chat template rendered the conversation as no token at all")
        return [int(t) for t in ids]

    def check_fits(self, token_ids: List[int], max_tokens: int) -> None:
        """Refuse by name a turn whose conversation and generation budget pass the session's ceiling."""
        need = len(token_ids) + int(max_tokens)
        if need > self.kv.ceiling:
            raise RuntimeError(
                f"this conversation needs {need} positions ({len(token_ids)} of conversation + {int(max_tokens)} to "
                f"generate) and the session holds at most {self.kv.ceiling} — {self.kv.source}. Start a new "
                f"conversation (/new), or ask for fewer tokens.")

    def clear(self) -> None:
        """Reset the conversation; the cache's positions are no longer the session's."""
        self.messages.clear()
        self.kv.forget()

    def get_context_info(self) -> Dict[str, Any]:
        """Context usage: the conversation's exact token count against the ceiling, and what the cache holds."""
        tokens = len(self.build_token_ids()) if self.messages else 0
        info = {
            "turns": len(self.messages),
            "tokens": tokens,
            "max_context": self.kv.ceiling,
            "usage_pct": round(tokens / self.kv.ceiling * 100, 1),
            "cached_tokens": len(self.kv.token_ids),
        }
        last = self.kv.last_turn
        if last is not None:
            info["last_turn"] = dict(last)
        return info


def play_turn(session: ConversationSession, message: str, max_tokens: int,
              generate: Callable[[List[int], KVSession], Dict[str, Any]]) -> Dict[str, Any]:
    """One chat turn: the user message joins the conversation, the template renders it, the ceiling is checked, and
    `generate(token_ids, kv_session)` runs it on the session's cache (it returns the engine's result with its "text").
    A refused or failed turn leaves the conversation as it was. Returns the result, with the turn's ledger."""
    session.add_user_message(message)
    try:
        token_ids = session.build_token_ids()
        session.check_fits(token_ids, max_tokens)
        result = generate(token_ids, session.kv)
    except BaseException:
        session.messages.pop()
        raise
    session.add_assistant_message(result.get("text", ""))
    result = dict(result)
    result["kv"] = dict(session.kv.last_turn or {})
    return result
