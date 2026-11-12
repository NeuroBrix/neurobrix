"""One assistant turn, read as it is decoded: the answer, the reasoning, the tool calls.

The decode loop hands token ids; this reader turns them into the three channels of an OpenAI assistant message:

  * `reasoning_content` — the text between the thinking markers, when the chat template declares them (the template
    of a thinking model opens the generation prompt with the start marker, or the model writes it first);
  * `content` — the answer, and nothing else;
  * `tool_calls` — the blocks of the template's tool-call dialect (neurobrix.agent.parsers), parsed when each closes
    (at the end of the turn for a dialect without a closing marker).

Text is only released once it is STABLE: the decode of the ids so far and the decode one id earlier agree on it, and
it ends on a complete character. So the only text held back is what the last token may still change (a byte-level
piece, a merged space) plus any suffix that could be the start of a marker — a few tokens, never the turn. A block
that opened like a tool call but does not parse is released as content, never dropped (an opening marker the turn
ended on, with no body, is not text and is not released).

Which markers: read from the chat template's text, never from a model name (the parser's own `start_marker` /
`end_marker`; the thinking pairs in `REASONING_MARKERS`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from neurobrix.serving.text_stream import MarkerDecoder

#: The thinking-marker pairs a chat template may declare. A template that writes both markers of a pair is a
#: thinking template; its text between them is the turn's reasoning.
REASONING_MARKERS: Tuple[Tuple[str, str], ...] = (("<think>", "</think>"),)

_INCOMPLETE = "�"


def select_reasoning(template_text: str) -> Optional[Tuple[str, str]]:
    """The thinking-marker pair the template declares, or None."""
    for start, end in REASONING_MARKERS:
        if start in (template_text or "") and end in (template_text or ""):
            return start, end
    return None


def reasoning_gap(tokenizer: Any, reasoning: Tuple[str, str]) -> Optional[str]:
    """What the chat template writes between a reasoning's end marker and the answer, read from the template itself:
    one past turn rendered with a reasoning and an answer ("\\n\\n" for Qwen3's templates, nothing for GLM-4.7-Flash's).
    None when the template does not show that turn's reasoning before its answer (nothing to read)."""
    said, answered = "reasoning-0f3b", "answer-0f3b"
    messages = [{"role": "user", "content": "u"},
                {"role": "assistant", "reasoning_content": said, "content": answered}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    end = text.find(reasoning[1], text.find(said))
    at = text.find(answered, end) if end >= 0 and said in text else -1
    return text[end + len(reasoning[1]):at] if at >= 0 else None


@dataclass
class Delta:
    """What one step released: text on a channel, or one parsed tool call."""
    channel: str                       # "content" | "reasoning" | "tool_call"
    text: str = ""
    call: Any = None                   # agent.parsers.base.ToolCall for "tool_call"


@dataclass
class Turn:
    """The whole turn, once finished."""
    content: str = ""
    reasoning: str = ""
    calls: List[Any] = field(default_factory=list)
    malformed: List[str] = field(default_factory=list)


def _partial_suffix(text: str, marker: str) -> int:
    """Length of the longest suffix of `text` that is a proper prefix of `marker`."""
    for n in range(min(len(marker) - 1, len(text)), 0, -1):
        if marker.startswith(text[-n:]):
            return n
    return 0


class TurnReader:
    """Feed token ids with `push`; read the released deltas; `finish` releases the rest and returns the Turn."""

    def __init__(self, tokenizer: Any, parser=None, reasoning: Optional[Tuple[str, str]] = None,
                 opens_in_reasoning: bool = False) -> None:
        self._parser = parser
        self._reasoning = reasoning
        keep = []
        if parser is not None:
            keep += list(parser.kept_markers())
        if reasoning is not None:
            keep += list(reasoning)
        self._decoder = MarkerDecoder(tokenizer, keep)
        self._ids: List[int] = []
        self._previous = ""
        self._released = 0          # characters of the decoded text already consumed by the state machine
        self._pending = ""          # stable text not yet consumed
        self._state = "reasoning" if (reasoning is not None and opens_in_reasoning) else "content"
        self._seen_content = False
        self._block = ""
        self._after_block = False
        self.turn = Turn()

    # -- feeding -----------------------------------------------------------
    def push(self, token_id: int) -> List[Delta]:
        self._ids.append(int(token_id))
        full = self._decoder.decode(self._ids)
        stable_len = 0
        for a, b in zip(self._previous, full):
            if a != b:
                break
            stable_len += 1
        self._previous = full
        stable = full[:stable_len]
        while stable.endswith(_INCOMPLETE):
            stable = stable[:-1]
        if len(stable) <= self._released:
            return []
        self._pending += stable[self._released:]
        self._released = len(stable)
        return self._consume(final=False)

    def finish(self) -> List[Delta]:
        """Release everything left (the turn ended); returns the last deltas. `self.turn` holds the whole turn."""
        full = self._decoder.decode(self._ids) if self._ids else ""
        if len(full) > self._released:
            self._pending += full[self._released:]
            self._released = len(full)
        return self._consume(final=True)

    # -- the state machine ---------------------------------------------------
    def _emit(self, out: List[Delta], channel: str, text: str) -> None:
        if not text:
            return
        if channel == "content":
            if self._after_block:
                text = text.lstrip()
                if not text:
                    return
                self._after_block = False
            self._seen_content = True
            self.turn.content += text
        elif channel == "reasoning":
            self.turn.reasoning += text
        out.append(Delta(channel, text=text))

    def _consume(self, final: bool) -> List[Delta]:
        out: List[Delta] = []
        while self._pending:
            if self._state == "reasoning":
                end = self._reasoning[1]
                i = self._pending.find(end)
                if i >= 0:
                    self._emit(out, "reasoning", self._pending[:i])
                    self._pending = self._pending[i + len(end):]
                    self._state = "content"
                    continue
                hold = 0 if final else _partial_suffix(self._pending, end)
                self._emit(out, "reasoning", self._pending[:len(self._pending) - hold])
                self._pending = self._pending[len(self._pending) - hold:]
                break
            if self._state == "content":
                if self._reasoning is not None and not self._seen_content and not self.turn.reasoning:
                    start = self._reasoning[0]
                    stripped = self._pending.lstrip()
                    if stripped.startswith(start):
                        self._pending = stripped[len(start):]
                        self._state = "reasoning"
                        continue
                    if not final and start.startswith(stripped):
                        break      # could still be the opening of the thinking block
                if self._parser is None:
                    self._emit(out, "content", self._pending)
                    self._pending = ""
                    break
                hits = [(self._pending.find(m), m) for m in self._parser.openings() if m in self._pending]
                if hits:
                    i, start = min(hits)
                    self._emit(out, "content", self._pending[:i])
                    self._block = start
                    self._pending = self._pending[i + len(start):]
                    self._state = "tool"
                    continue
                hold = 0 if final else max(_partial_suffix(self._pending, m) for m in self._parser.openings())
                self._emit(out, "content", self._pending[:len(self._pending) - hold])
                self._pending = self._pending[len(self._pending) - hold:]
                break
            # tool block
            end = self._parser.end_marker
            if end is not None:
                i = self._pending.find(end)
                if i >= 0:
                    self._block += self._pending[:i + len(end)]
                    self._pending = self._pending[i + len(end):]
                    self._close_block(out)
                    continue
            self._block += self._pending
            self._pending = ""
            break
        if final:
            if self._state == "tool" and self._block:
                self._close_block(out)
            elif self._pending:
                self._emit(out, "content" if self._state != "reasoning" else "reasoning", self._pending)
                self._pending = ""
        return out

    def _close_block(self, out: List[Delta]) -> None:
        parsed = self._parser.parse(self._block)
        block, self._block = self._block, ""
        self._state = "content"
        if parsed.calls and not parsed.malformed:
            for call in parsed.calls:
                self.turn.calls.append(call)
                out.append(Delta("tool_call", call=call))
            self._after_block = True
            return
        # Opened like a tool call and did not parse: the model's text, kept as the answer — unless the turn ended on
        # the opening marker alone: a marker with no body is protocol, not text (recorded, not released).
        self.turn.malformed.append(block)
        self._emit(out, "content", "" if block.strip() in self._parser.openings() else block)


def typed_arguments(arguments: Dict[str, Any], name: str, tools: Optional[list]) -> Dict[str, Any]:
    """A dialect that writes every value as text (the XML one): each value whose declared JSON-schema type is not
    `string` is read as JSON when it reads as that type; otherwise it stays the text the model wrote (the client
    validates against its own schema — nothing is guessed or dropped here)."""
    props = {}
    for tool in tools or []:
        fn = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(fn, dict) and fn.get("name") == name:
            props = ((fn.get("parameters") or {}).get("properties") or {})
            break
    checks = {"integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
              "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
              "boolean": lambda v: isinstance(v, bool), "object": lambda v: isinstance(v, dict),
              "array": lambda v: isinstance(v, list), "null": lambda v: v is None}
    out = {}
    for key, value in arguments.items():
        declared = (props.get(key) or {}).get("type")
        types = declared if isinstance(declared, list) else [declared]
        if isinstance(value, str) and declared is not None and "string" not in types:
            try:
                parsed = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                parsed = None
            else:
                if any(t in checks and checks[t](parsed) for t in types):
                    value = parsed
        out[key] = value
    return out
