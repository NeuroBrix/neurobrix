"""
ConversationSession — Multi-turn chat state for LLM models.

V1 Strategy: Re-prefill entire history each turn.
The full conversation is formatted via chat_template, tokenized,
and prefilled from scratch each turn. KV cache resets between turns.

Context overflow protection: when conversation approaches KV cache limit,
old turns are summarized using the loaded LLM. Truncation fallback if
summarization fails.

Simple, correct, hardware-agnostic. Incremental prefill is V2 optimization.
"""

from typing import Any, Callable, Dict, List


class ConversationSession:
    """
    Multi-turn conversation state for LLM models.

    Tracks chat history and builds full prompts via chat_template.
    ZERO HARDCODE: max_context from model defaults, not hardcoded.
    """

    def __init__(self, tokenizer: Any, defaults: Dict[str, Any]):
        self.messages: List[Dict[str, str]] = []
        self._tokenizer = tokenizer
        self._max_context = int(defaults.get("max_length", 4096))
        self._summarization_count = 0

    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation."""
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant response to the conversation."""
        self.messages.append({"role": "assistant", "content": content})

    def ensure_fits(
        self,
        max_cache_len: int,
        max_tokens: int,
        generate_fn: Callable,
    ) -> bool:
        """
        Ensure conversation fits in KV cache. Summarize old turns if needed.

        Args:
            max_cache_len: KV cache capacity in tokens (from Prism plan)
            max_tokens: per-turn generation budget
            generate_fn: callable to generate summary (engine.generate)

        Returns:
            True if summarization/truncation was triggered
        """
        budget = max_cache_len - max_tokens  # room for history after reserving generation space
        current_tokens = self._estimate_token_count()

        if current_tokens <= budget:
            return False  # fits fine

        # Identify anchored messages (never summarize)
        # - First system message (if any)
        # - First user message (core intent)
        anchor_count = 0
        if self.messages and self.messages[0]["role"] == "system":
            anchor_count = 1
        # First user message after any system message
        for i in range(anchor_count, len(self.messages)):
            if self.messages[i]["role"] == "user":
                anchor_count = i + 1
                break

        # Find summarizable turn pairs (user+assistant) after anchors
        # Never break a pair — always summarize complete user+assistant turns
        # Keep last 2 messages (current turn)
        summarizable = []
        i = anchor_count
        while i + 1 < len(self.messages) - 2:
            if self.messages[i]["role"] == "user" and self.messages[i + 1]["role"] == "assistant":
                summarizable.append((i, i + 1))
                i += 2
            else:
                i += 1

        if not summarizable:
            return False  # nothing to summarize (only anchors + current)

        # Build text of turns to summarize
        turns_text = []
        for ui, ai in summarizable:
            turns_text.append(f"User: {self.messages[ui]['content']}")
            turns_text.append(f"Assistant: {self.messages[ai]['content']}")

        # Generate summary using the loaded LLM
        summary_prompt = (
            "Summarize the following conversation concisely, preserving key facts, "
            "decisions, and user preferences:\n\n" + "\n".join(turns_text)
        )

        summary_text = None
        try:
            result = generate_fn(prompt=summary_prompt, max_tokens=200, chat_mode=False)
            summary_text = result.get("text", "").strip()
        except Exception:
            pass  # Summarization failed — fall through to truncation

        if not summary_text:
            # Truncation fallback: drop oldest summarizable turns until it fits
            while summarizable and self._estimate_token_count() > budget:
                ui, ai = summarizable.pop(0)
                self.messages.pop(ui)  # remove user
                self.messages.pop(ui)  # remove assistant (shifted down)
                # Re-index remaining summarizable pairs
                summarizable = [(u - 2, a - 2) for u, a in summarizable]
            self._summarization_count += 1
            return True

        # Replace summarizable turns with summary message
        # Remove old turns (in reverse to preserve indices)
        indices_to_remove = []
        for ui, ai in summarizable:
            indices_to_remove.extend([ui, ai])
        for idx in sorted(indices_to_remove, reverse=True):
            self.messages.pop(idx)

        # Insert summary as system message after anchors
        summary_msg = {
            "role": "system",
            "content": f"[Conversation summary: {summary_text}]",
        }
        self.messages.insert(anchor_count, summary_msg)
        self._summarization_count += 1

        return True

    def build_prompt(self) -> str:
        """
        Format full conversation history via chat_template.

        Returns the complete formatted string ready for tokenization.
        The caller should tokenize this WITHOUT applying chat_mode
        (the template is already applied here).
        """
        if hasattr(self._tokenizer, 'apply_chat_template'):
            return self._tokenizer.apply_chat_template(
                self.messages, tokenize=False, add_generation_prompt=True
            )
        return self._format_basic()

    def build_token_ids(self) -> List[int]:
        """
        Build tokenized conversation history via chat_template.

        Returns token IDs ready for prefill.
        """
        if hasattr(self._tokenizer, 'apply_chat_template'):
            return self._tokenizer.apply_chat_template(
                self.messages, tokenize=True, add_generation_prompt=True
            )
        # Fallback: encode the basic format
        prompt = self._format_basic()
        if hasattr(self._tokenizer, 'encode'):
            return self._tokenizer.encode(prompt, padding=False, add_special_tokens=True)
        raise RuntimeError(
            "ZERO FALLBACK: Tokenizer has no apply_chat_template() or encode()."
        )

    def _format_basic(self) -> str:
        """Basic chat format when no chat_template is available."""
        parts = []
        for msg in self.messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        parts.append("Assistant:")
        return "\n".join(parts)

    def clear(self) -> None:
        """Reset conversation history."""
        self.messages.clear()
        self._summarization_count = 0

    def get_context_info(self) -> Dict[str, Any]:
        """Return context usage statistics."""
        est_tokens = self._estimate_token_count()
        info = {
            "turns": len(self.messages),
            "estimated_tokens": est_tokens,
            "max_context": self._max_context,
            "usage_pct": round(est_tokens / self._max_context * 100, 1) if self._max_context > 0 else 0,
        }
        if self._summarization_count > 0:
            info["summarizations"] = self._summarization_count
        return info

    def _estimate_token_count(self) -> int:
        """Estimate current context size in tokens."""
        try:
            ids = self.build_token_ids()
            return len(ids)
        except Exception:
            # Rough estimate: ~4 chars per token
            total_chars = sum(len(m["content"]) for m in self.messages)
            return total_chars // 4
