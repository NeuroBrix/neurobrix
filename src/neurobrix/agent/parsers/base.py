"""Tool-call parsing contract shared by all template dialects.

A parser turns one raw assistant completion into structured tool calls.
Parsers are FORMAT-named (xml, jsonfmt), never model-named: the dialect
is a property of the chat template embedded in the build, and selection
is data-driven from that template text (see parsers.__init__).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class ToolCall:
    """One parsed tool invocation."""

    name: str
    arguments: Dict[str, Any]


@dataclass
class ParseResult:
    """Outcome of parsing one assistant completion.

    prose      — assistant text with tool-call blocks removed.
    calls      — successfully parsed tool calls, in emission order.
    malformed  — raw block texts that opened like a tool call but did not
                 parse; the loop surfaces them and re-asks once (bounded),
                 never silently drops them.
    """

    prose: str = ""
    calls: List[ToolCall] = field(default_factory=list)
    malformed: List[str] = field(default_factory=list)


class ToolCallParser:
    """Dialect contract. Subclasses implement parse()."""

    format_name: str = "base"

    def parse(self, text: str) -> ParseResult:
        raise NotImplementedError
