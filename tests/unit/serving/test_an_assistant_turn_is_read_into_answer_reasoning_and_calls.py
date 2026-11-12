"""serving/openai_api/turn.py: one decoded turn read into content, reasoning_content and tool_calls, as it streams.

A fake tokenizer whose ids are text pieces (some special, as in a real tokenizer) drives the reader token by token.

Injections (each RED, then GREEN; see the commit message):
  * the marker decode skipping every special token (MarkerDecoder's keep ignored) — the special-marker dialect's
    call reads as prose;
  * no hold-back of a partial marker — the split-marker stream leaks `<tool` into content;
  * a malformed block dropped instead of kept as content — the malformed test fails.
"""
from __future__ import annotations

from neurobrix.serving.tool_calls.parsers import JsonListToolCallParser, JsonToolCallParser, XmlToolCallParser, select_parser
from neurobrix.serving.openai_api.turn import TurnReader, select_reasoning, typed_arguments
from neurobrix.serving.text_stream import MarkerDecoder


class PieceTokenizer:
    """ids index `pieces`; ids in `special` vanish under skip_special_tokens, as a tokenizer's special tokens do."""

    def __init__(self, pieces, special=()):
        self.pieces = list(pieces)
        self.special = set(special)

    def ids(self, *texts):
        return [self.pieces.index(t) for t in texts]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.pieces[i] for i in ids if not (skip_special_tokens and i in self.special))


def _run(reader, ids):
    deltas = []
    for i in ids:
        deltas += reader.push(i)
    deltas += reader.finish()
    return deltas


def _text(deltas, channel):
    return "".join(d.text for d in deltas if d.channel == channel)


def test_a_special_marker_token_is_kept_only_when_it_is_a_dialect_marker():
    tok = PieceTokenizer(["<|end|>", "<|tool_call|>", "hi"], special=[0, 1])
    assert MarkerDecoder(tok, ["<|tool_call|>"]).decode([2, 1, 0]) == "hi<|tool_call|>"
    assert MarkerDecoder(tok, []).decode([2, 1, 0]) == "hi"


def test_the_special_marker_dialect_yields_its_calls_not_prose():
    pieces = ["<|end_of_text|>", "<|tool_call|>", '[{"name": "get_weather", ', '"arguments": {"city": "Paris"}}]']
    tok = PieceTokenizer(pieces, special=[0, 1])
    reader = TurnReader(tok, parser=JsonListToolCallParser())
    deltas = _run(reader, [1, 2, 3, 0])
    calls = [d.call for d in deltas if d.channel == "tool_call"]
    assert [(c.name, c.arguments) for c in calls] == [("get_weather", {"city": "Paris"})]
    assert reader.turn.content == ""


def test_a_split_start_marker_is_held_back_from_the_content():
    pieces = ["Let me", " check.", "\n<tool", "_call>", '\n{"name": "ls", "arguments": {}}\n', "</tool_call>"]
    tok = PieceTokenizer(pieces)
    reader = TurnReader(tok, parser=JsonToolCallParser())
    streamed = []
    for i in range(len(pieces)):
        for d in reader.push(i):
            streamed.append(d)
            assert "<tool" not in d.text, f"a partial marker leaked into {d.channel}: {d.text!r}"
    streamed += reader.finish()
    assert _text(streamed, "content") == "Let me check.\n"
    assert [d.call.name for d in streamed if d.channel == "tool_call"] == ["ls"]


def test_only_the_last_token_is_held_back_never_the_turn():
    pieces = [f"w{i} " for i in range(20)]
    reader = TurnReader(PieceTokenizer(pieces))
    released = []
    for i in range(20):
        released.append(_text(reader.push(i), "content"))
    assert "".join(released) == "".join(pieces[:19]), "more than the last token was held back"
    assert _text(reader.finish(), "content") == pieces[19]


def test_reasoning_is_split_from_the_answer_when_the_prompt_opens_the_thinking_block():
    pieces = ["The user", " wants 4.", "</think>", "\n\n", "4"]
    reader = TurnReader(PieceTokenizer(pieces), reasoning=("<think>", "</think>"), opens_in_reasoning=True)
    deltas = _run(reader, range(len(pieces)))
    assert _text(deltas, "reasoning") == "The user wants 4."
    assert reader.turn.content == "\n\n4"
    assert "</think>" not in reader.turn.content + reader.turn.reasoning


def test_reasoning_opened_by_the_model_itself_is_split_too():
    pieces = ["<think>", "hmm", "</think>", "yes"]
    reader = TurnReader(PieceTokenizer(pieces), reasoning=("<think>", "</think>"))
    _run(reader, range(len(pieces)))
    assert (reader.turn.reasoning, reader.turn.content) == ("hmm", "yes")


def test_a_malformed_block_is_kept_as_content_never_dropped():
    pieces = ["ok ", "<tool_call>", "{not json}", "</tool_call>"]
    reader = TurnReader(PieceTokenizer(pieces), parser=JsonToolCallParser())
    _run(reader, range(len(pieces)))
    assert reader.turn.calls == []
    assert reader.turn.content == "ok <tool_call>{not json}</tool_call>"
    assert reader.turn.malformed == ["<tool_call>{not json}</tool_call>"]


def test_two_xml_calls_and_their_schema_types():
    block = "<tool_call>\n<function=read>\n<parameter=path>\na.py\n</parameter>\n<parameter=limit>\n40\n</parameter>\n</function>\n</tool_call>"
    pieces = [block, "\n", block.replace("a.py", "b.py")]
    reader = TurnReader(PieceTokenizer(pieces), parser=XmlToolCallParser())
    _run(reader, range(3))
    assert [c.arguments["path"] for c in reader.turn.calls] == ["a.py", "b.py"]
    assert reader.turn.content == ""
    tools = [{"type": "function", "function": {"name": "read", "parameters": {"properties": {
        "path": {"type": "string"}, "limit": {"type": "integer"}}}}}]
    assert typed_arguments(reader.turn.calls[0].arguments, "read", tools) == {"path": "a.py", "limit": 40}
    # a value that does not read as its declared type stays the model's text
    assert typed_arguments({"limit": "forty"}, "read", tools) == {"limit": "forty"}


def test_the_template_text_selects_the_dialects():
    assert isinstance(select_parser("... respond with <|tool_call|> followed by a JSON list ..."), JsonListToolCallParser)
    assert select_reasoning("{{ '<think>' }} ... </think>") == ("<think>", "</think>")
    assert select_reasoning("no thinking here") is None


def test_the_json_list_dialect_opened_by_the_plain_text_marker_yields_its_call():
    """granite-3.1 writes its marker as text too (`<tool_call>[...]`, measured through the entry point 2026-11-08),
    as the vendor's parser accepts (vLLM granite: bot_token and bot_string)."""
    pieces = ["<tool", "_call>", '[{"arguments": {"city": "Paris"}, "name": "get_weather"}]']
    reader = TurnReader(PieceTokenizer(pieces), parser=JsonListToolCallParser())
    deltas = _run(reader, range(3))
    assert [(d.call.name, d.call.arguments) for d in deltas if d.channel == "tool_call"] == [("get_weather", {"city": "Paris"})]
    assert reader.turn.content == ""
