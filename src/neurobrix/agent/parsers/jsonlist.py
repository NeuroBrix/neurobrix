"""JSON-list tool-call dialect.

Matches templates that instruct the model to answer with one marker token followed by a JSON list of calls, the
rest of the turn being that list:

    <|tool_call|>[{"name": "NAME", "arguments": {...}}, ...]

(the template's own words: "respond with <|tool_call|> followed by a JSON list of tools used"). The marker has no
closing tag: the block runs to the end of the turn. Text before the marker is prose.
"""

import json

from neurobrix.agent.parsers.base import ParseResult, ToolCall, ToolCallParser

_MARKER = "<|tool_call|>"


class JsonListToolCallParser(ToolCallParser):
    format_name = "json_list"
    start_marker = _MARKER
    end_marker = None

    def parse(self, text: str) -> ParseResult:
        result = ParseResult()
        head, found, tail = text.partition(_MARKER)
        if not found:
            result.prose = text.strip()
            return result
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
            result.malformed.append(_MARKER + tail)
            return result
        result.calls.extend(calls)
        return result
