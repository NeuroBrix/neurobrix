"""A turn that ends right after a tool-call opening marker, with no body, shows no marker in its answer.

Seen 2026-11-11 on a served coding model, calls `auto`: after its last sentence the model wrote the dialect's opening
marker and then ended the turn. The reader keeps a block that opened like a call and does not parse as content — the
model's text is never dropped — and so released the bare marker as the end of the answer. A marker with nothing after
it is protocol, not text: it is recorded in `turn.malformed` and not released. A block with ANY body that does not
parse is still kept whole (test_a_malformed_block_is_kept_as_content_never_dropped).

What this test would do if the code were wrong: the content ends with "<tool_call>" — RED (seen)."""
from __future__ import annotations

import pytest

from neurobrix.agent.parsers import JsonToolCallParser, XmlToolCallParser
from neurobrix.serving.openai_api.turn import TurnReader
from tests.unit.serving.test_an_assistant_turn_is_read_into_answer_reasoning_and_calls import PieceTokenizer, _run, _text


@pytest.mark.parametrize("parser", [JsonToolCallParser, XmlToolCallParser])
@pytest.mark.parametrize("tail", [[], ["\n"]])
def test_the_bare_marker_is_not_released(parser, tail):
    pieces = ["The tests pass now.", "\n\n", "<tool_call>", "<|im_end|>", *tail]
    tok = PieceTokenizer(pieces, special=[3])
    reader = TurnReader(tok, parser=parser())
    deltas = _run(reader, [0, 1, 2, *range(4, len(pieces)), 3])
    assert reader.turn.calls == []
    assert "<tool_call>" not in reader.turn.content and "<tool_call>" not in _text(deltas, "content")
    assert reader.turn.content.startswith("The tests pass now.")
    assert [m.strip() for m in reader.turn.malformed] == ["<tool_call>"]


def test_an_opened_block_with_a_body_is_still_kept_whole():
    pieces = ["ok ", "<tool_call>", "{not json"]
    reader = TurnReader(PieceTokenizer(pieces), parser=JsonToolCallParser())
    _run(reader, range(len(pieces)))
    assert reader.turn.content == "ok <tool_call>{not json"
