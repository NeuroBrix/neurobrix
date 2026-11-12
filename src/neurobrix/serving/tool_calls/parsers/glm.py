"""GLM's key/value tool-call dialect (docs/design/tool-call-dialects.md).

Matches templates that instruct the model to emit (GLM-4.7-Flash's own words: "output the function name and
arguments within the following XML format"):

    <tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>

A string value is written bare and any other value as its JSON (the template's `tojson(ensure_ascii=False) if v is
not string else v`). So values come out as text, exactly as written, and `typed_arguments` reads a value as JSON where
the tool's schema declares another type, as for the xml dialect. Whitespace between the pairs is allowed (other GLM
templates put newlines there), but nothing else. A broken block, or a dialect marker outside a block, is malformed,
never prose.
"""
import re

from neurobrix.serving.tool_calls.parsers.base import ParseResult, ToolCall, ToolCallParser

_BLOCK_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_PAIR_RE = re.compile(r"\s*<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL)
_NAME_RE = re.compile(r"[^\s<>]+")
_INNER = ("<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>")


def _call(body: str):
    """The call a block's body holds, or None when it does not read as one."""
    name, _, rest = body.partition("<arg_key>")
    name = name.strip()
    if not _NAME_RE.fullmatch(name):
        return None
    rest = ("<arg_key>" + rest) if _ else ""
    arguments, at = {}, 0
    while (pair := _PAIR_RE.match(rest, at)) is not None:
        key = pair.group(1).strip()
        if not key or key in arguments:
            return None
        arguments[key], at = pair.group(2), pair.end()
    return ToolCall(name=name, arguments=arguments) if not rest[at:].strip() else None


class GlmToolCallParser(ToolCallParser):
    format_name = "glm"
    start_marker = "<tool_call>"
    end_marker = "</tool_call>"
    inner_markers = _INNER

    def parse(self, text: str) -> ParseResult:
        result = ParseResult()

        def _consume(match: "re.Match[str]") -> str:
            call = _call(match.group(1))
            (result.calls.append(call) if call is not None else result.malformed.append(match.group(0)))
            return ""

        prose = _BLOCK_RE.sub(_consume, text)
        hits = [i for i in (prose.find(m) for m in ("<tool_call>", "</tool_call>", *_INNER)) if i >= 0]
        if hits:                       # a block cut before its end, or a marker that sat in no block
            result.malformed.append(prose[min(hits):].strip())
            prose = prose[:min(hits)]
        result.prose = prose.strip()
        if result.malformed:
            result.calls = []          # never half-taken: the loop re-asks for the whole turn
        return result
