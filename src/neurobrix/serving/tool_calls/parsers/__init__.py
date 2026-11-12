"""Data-driven parser selection.

The tool-call dialect is a property of the chat template embedded in the
build. The caller reads the template text engine-side and passes it in;
this package never opens model files itself. No model-name branches —
the DIALECT MARKERS in the template decide (R15).
"""

from typing import Optional

from neurobrix.serving.tool_calls.parsers.base import ParseResult, ToolCall, ToolCallParser
from neurobrix.serving.tool_calls.parsers.glm import GlmToolCallParser
from neurobrix.serving.tool_calls.parsers.jsonfmt import JsonToolCallParser
from neurobrix.serving.tool_calls.parsers.jsonlist import JsonListToolCallParser
from neurobrix.serving.tool_calls.parsers.mistral import MistralArgsToolCallParser
from neurobrix.serving.tool_calls.parsers.xml import XmlToolCallParser

__all__ = [
    "ParseResult",
    "ToolCall",
    "ToolCallParser",
    "GlmToolCallParser",
    "XmlToolCallParser",
    "JsonToolCallParser",
    "MistralArgsToolCallParser",
    "JsonListToolCallParser",
    "select_parser",
]


def select_parser(template_text: str) -> Optional[ToolCallParser]:
    """Pick the dialect the embedded template declares, or None.

    None means the template declares no tool contract — agent mode is
    refused upstream by the mode gate (ZERO FALLBACK, no guessing).
    """
    if not template_text:
        return None
    if GlmToolCallParser.inner_markers[0] in template_text:     # before <tool_call>: GLM's template holds both
        return GlmToolCallParser()
    if XmlToolCallParser.inner_markers[0] in template_text:
        return XmlToolCallParser()
    if "<tool_call>" in template_text:
        return JsonToolCallParser()
    if MistralArgsToolCallParser.start_marker in template_text and MistralArgsToolCallParser.inner_markers[0] in template_text:
        return MistralArgsToolCallParser()
    if JsonListToolCallParser.start_marker in template_text:
        return JsonListToolCallParser()
    return None
