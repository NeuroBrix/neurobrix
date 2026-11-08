"""JSON-list tool-call dialect.

Matches templates that instruct the model to answer with one marker token followed by a JSON list of calls, the
rest of the turn being that list:

    <|tool_call|>[{"name": "NAME", "arguments": {...}}, ...]

(the template's own words: "respond with <|tool_call|> followed by a JSON list of tools used"). The marker has no
closing tag: the block runs to the end of the turn. Text before the marker is prose.

The model also writes the marker as plain text, `<tool_call>` (measured 2026-11-08: granite-3.1-1b-a400m-instruct
answered `<tool_call>[{"arguments": {"city": "Paris"}, "name": "get_weather"}]` through the entry point), and the
vendor's own parser accepts both openings: vLLM's granite parser strips `bot_token = "<|tool_call|>"` and
`bot_string = "<tool_call>"` (vllm/entrypoints/openai/tool_parsers/granite_tool_parser.py at v0.8.0, lines 42-51). So does this
one: either opening, the first that appears.
"""

import json

from neurobrix.agent.parsers.base import ParseResult, ToolCall, ToolCallParser

_MARKER = "<|tool_call|>"
_MARKERS = (_MARKER, "<tool_call>")


class JsonListToolCallParser(ToolCallParser):
    format_name = "json_list"
    start_marker = _MARKER
    start_markers = _MARKERS
    end_marker = None

    def parse(self, text: str) -> ParseResult:
        result = ParseResult()
        hits = [(text.find(m), m) for m in _MARKERS if m in text]
        if not hits:
            result.prose = text.strip()
            return result
        at, found = min(hits)
        head, tail = text[:at], text[at + len(found):]
        result.prose = head.strip()
        block = tail.strip()
        try:
            payload = json.loads(block)
            if isinstance(payload, dict):
                payload = [payload]
            if not isinstance(payload, list) or not payload:
                raise ValueError("tool call payload shape")
            calls = []
            for item in payload:
                name = item["name"]
                arguments = item.get("arguments")
                if arguments is None:
                    arguments = {}
                if not isinstance(name, str) or not isinstance(arguments, dict):
                    raise ValueError("tool call payload shape")
                calls.append(ToolCall(name=name, arguments=arguments))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError):
            result.malformed.append(found + tail)
            return result
        result.calls.extend(calls)
        return result
