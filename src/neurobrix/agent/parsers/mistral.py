"""Mistral's `[TOOL_CALLS]NAME[ARGS]` tool-call dialect (docs/design/tool-call-dialects.md).

Matches templates that write each call of a turn as (Devstral Small 2's template: `'[TOOL_CALLS]'`, then
`name + '[ARGS]' + arguments`, with the arguments `tojson` when they are not already a string, `'{}'` when empty):

    [TOOL_CALLS]NAME[ARGS]{...}[TOOL_CALLS]NAME[ARGS]{...}

The calls have no closing marker: they run to the end of the turn. Text before the first `[TOOL_CALLS]` is prose.
Both markers are special tokens in the vendor's tokenizer, so `[ARGS]` is declared inside the block, and the decode
keeps it. The older list form (`[TOOL_CALLS][{"name": ...}]`, no `[ARGS]`) is not this dialect. If any call of the
block fails to read, the whole block is malformed and nothing of it is taken.
"""
import json
import re

from neurobrix.agent.parsers.base import ParseResult, ToolCall, ToolCallParser

_OPEN = "[TOOL_CALLS]"
_ARGS = "[ARGS]"
_NAME_RE = re.compile(r"[^\s\[\]{}]+")


def _call(piece: str):
    """One `NAME[ARGS]{...}` piece read as a call, or None (a piece with no `[ARGS]` has no JSON to read)."""
    name, _, raw = piece.partition(_ARGS)
    name = name.strip()
    if not _NAME_RE.fullmatch(name):
        return None
    try:
        arguments, end = json.JSONDecoder().raw_decode(raw.lstrip())
    except json.JSONDecodeError:
        return None
    if not isinstance(arguments, dict) or raw.lstrip()[end:].strip():
        return None
    return ToolCall(name=name, arguments=arguments)


class MistralArgsToolCallParser(ToolCallParser):
    format_name = "mistral_args"
    start_marker = _OPEN
    end_marker = None
    inner_markers = (_ARGS,)

    def parse(self, text: str) -> ParseResult:
        result = ParseResult()
        at = text.find(_OPEN)
        if at < 0:
            at = text.find(_ARGS)          # a call whose opener was lost: malformed, never prose
            if at < 0:
                result.prose = text.strip()
                return result
            result.prose, result.malformed = text[:at].strip(), [text[at:]]
            return result
        result.prose, block = text[:at].strip(), text[at + len(_OPEN):]
        calls = [_call(piece) for piece in block.split(_OPEN)]
        if any(c is None for c in calls):
            result.malformed.append(_OPEN + block)
            return result
        result.calls.extend(calls)
        return result
