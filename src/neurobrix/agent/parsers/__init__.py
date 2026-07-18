"""Data-driven parser selection.

The tool-call dialect is a property of the chat template embedded in the
build. The caller reads the template text engine-side and passes it in;
this package never opens model files itself. No model-name branches —
the DIALECT MARKERS in the template decide (R15).
"""

from typing import Optional

from neurobrix.agent.parsers.base import ParseResult, ToolCall, ToolCallParser
from neurobrix.agent.parsers.jsonfmt import JsonToolCallParser
from neurobrix.agent.parsers.xml import XmlToolCallParser

__all__ = [
    "ParseResult",
    "ToolCall",
    "ToolCallParser",
    "XmlToolCallParser",
    "JsonToolCallParser",
    "select_parser",
]


def select_parser(template_text: str) -> Optional[ToolCallParser]:
    """Pick the dialect the embedded template declares, or None.

    None means the template declares no tool contract — agent mode is
    refused upstream by the mode gate (ZERO FALLBACK, no guessing).
    """
    if not template_text:
        return None
    if "<function=" in template_text:
        return XmlToolCallParser()
    if "<tool_call>" in template_text:
        return JsonToolCallParser()
    return None
