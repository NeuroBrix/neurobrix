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
    #: The text that opens a tool-call block, and the one that closes it (None: the block runs to the end of
    #: the turn). A streaming reader holds back from the prose any suffix that could begin `start_marker`.
    start_marker: str = ""
    #: Every text that opens a block (a dialect whose model writes its marker two ways); `start_marker` first.
    start_markers: tuple = ()
    end_marker = None

    def openings(self) -> tuple:
        return tuple(self.start_markers) or (self.start_marker,)

    def parse(self, text: str) -> ParseResult:
        raise NotImplementedError
