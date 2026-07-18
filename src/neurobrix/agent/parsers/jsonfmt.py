"""JSON tool-call dialect.

Matches templates that instruct the model to emit:

    <tool_call>
    {"name": "NAME", "arguments": {...}}
    </tool_call>

(the Hermes-style convention used by several instruct templates).
"""

import json
import re

from neurobrix.agent.parsers.base import ParseResult, ToolCall, ToolCallParser

_BLOCK_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


class JsonToolCallParser(ToolCallParser):
    format_name = "json"

    def parse(self, text: str) -> ParseResult:
        result = ParseResult()

        def _consume(match: "re.Match[str]") -> str:
            body = match.group(1).strip()
            try:
                payload = json.loads(body)
                name = payload["name"]
                arguments = payload.get("arguments") or {}
                if not isinstance(name, str) or not isinstance(arguments, dict):
                    raise ValueError("tool call payload shape")
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                result.malformed.append(match.group(0))
                return ""
            result.calls.append(ToolCall(name=name, arguments=arguments))
            return ""

        prose = _BLOCK_RE.sub(_consume, text)
        if "<tool_call>" in prose:
            head, _, tail = prose.partition("<tool_call>")
            result.malformed.append("<tool_call>" + tail)
            prose = head
        result.prose = prose.strip()
        return result
