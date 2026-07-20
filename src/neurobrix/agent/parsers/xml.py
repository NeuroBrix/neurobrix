"""XML tool-call dialect.

Matches templates that instruct the model to emit:

    <tool_call>
    <function=NAME>
    <parameter=KEY>
    VALUE
    </parameter>
    </function>
    </tool_call>

Free prose is allowed around the blocks. Parameter values keep their
exact content: the dialect surrounds values with single newlines, so
exactly one leading and one trailing newline are stripped — inner
whitespace (indentation, trailing newlines inside code) is preserved.
"""

import re

from neurobrix.agent.parsers.base import ParseResult, ToolCall, ToolCallParser

_BLOCK_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_FUNCTION_RE = re.compile(r"<function=([^>\n]+)>(.*)</function>", re.DOTALL)
_PARAMETER_RE = re.compile(r"<parameter=([^>\n]+)>(.*?)</parameter>", re.DOTALL)


def _trim_value(raw: str) -> str:
    """Strip exactly one dialect newline on each side, nothing more."""
    if raw.startswith("\n"):
        raw = raw[1:]
    if raw.endswith("\n"):
        raw = raw[:-1]
    return raw


class XmlToolCallParser(ToolCallParser):
    format_name = "xml"

    def parse(self, text: str) -> ParseResult:
        result = ParseResult()

        def _consume(match: "re.Match[str]") -> str:
            body = match.group(1)
            fn = _FUNCTION_RE.search(body)
            if fn is None:
                result.malformed.append(match.group(0))
                return ""
            name = fn.group(1).strip()
            arguments = {
                pm.group(1).strip(): _trim_value(pm.group(2))
                for pm in _PARAMETER_RE.finditer(fn.group(2))
            }
            result.calls.append(ToolCall(name=name, arguments=arguments))
            return ""

        prose = _BLOCK_RE.sub(_consume, text)
        # An opening tag without its closing tag is a truncated block —
        # surfaced as malformed (usually a stop-token failure upstream).
        if "<tool_call>" in prose:
            head, _, tail = prose.partition("<tool_call>")
            result.malformed.append("<tool_call>" + tail)
            prose = head
        # A <function=...> block or a dangling </tool_call> that never sat
        # inside a well-formed <tool_call>...</tool_call> pair is a broken
        # tool-call emission, NOT clean prose. The template contract is
        # explicit: "an inner <function=...></function> block MUST be nested
        # within <tool_call>." Silently returning it as prose makes the loop
        # read a broken call as a final answer and terminate mid-task.
        # Surface it as malformed so the format-reminder recovery fires
        # (proven cross-engine: a decode near-tie can drop the <tool_call>
        # opener while keeping the function block + closer intact).
        for _marker in ("<function=", "</tool_call>"):
            if _marker in prose:
                head, _, tail = prose.partition(_marker)
                result.malformed.append((_marker + tail).strip())
                prose = head
                break
        result.prose = prose.strip()
        return result
