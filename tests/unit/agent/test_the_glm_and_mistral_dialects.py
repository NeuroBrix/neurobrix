"""The GLM and Mistral tool-call dialects (docs/design/tool-call-dialects.md), tested against each vendor's own chat
template (data/templates/, byte for byte at a pinned revision; SOURCES.md).

The round trip: the vendor's template, rendered by the engine's own environment on an assistant message with calls,
writes the text the model was trained to write; the dialect's parser reads back the same calls. The round trip also
goes through `typed_arguments`, where the dialect writes values as text. Then the selection by the template's markers,
every malformed form, and the streaming reader with the markers as the tokenizer gives them. Devstral's `[ARGS]` is a
special token, which the decode keeps only because the parser declares it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("jinja2")

from neurobrix.agent.parsers import (  # noqa: E402
    GlmToolCallParser, JsonToolCallParser, MistralArgsToolCallParser, XmlToolCallParser, select_parser)
from neurobrix.core.module.tokenizer.chat_template import environment  # noqa: E402
from neurobrix.serving.openai_api.turn import TurnReader, typed_arguments  # noqa: E402
from neurobrix.serving.text_stream import MarkerDecoder  # noqa: E402

TEMPLATES = Path(__file__).parent / "data" / "templates"
GLM = (TEMPLATES / "GLM-4.7-Flash.jinja").read_text()
DEVSTRAL = (TEMPLATES / "Devstral-Small-2-24B-Instruct-2512.jinja").read_text()

TOOLS = [{"type": "function", "function": {"name": "edit", "parameters": {"type": "object", "properties": {
    "path": {"type": "string"}, "text": {"type": "string"}, "line": {"type": "integer"},
    "flags": {"type": "array"}, "opts": {"type": "object"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "shell", "parameters": {"type": "object", "properties": {
        "cmd": {"type": "string"}}, "required": ["cmd"]}}}]
CALLS = [("edit", {"path": "src/a.py", "text": "if a < b:\n    return 'é & ok'\n", "line": 12, "flags": ["-x", 2],
                   "opts": {"dry": True, "n": None}}),
         ("shell", {"cmd": "pytest -q tests/ && echo done"})]


def _render(template, messages, **context):
    return environment().from_string(template).render(messages=messages, bos_token="<s>", eos_token="</s>",
                                                     add_generation_prompt=False, tools=TOOLS, **context)


def _assistant(calls, content="Let me fix it.", arguments_as=dict):
    return {"role": "assistant", "content": content, "tool_calls": [
        {"id": f"c{i}", "type": "function",
         "function": {"name": n, "arguments": a if arguments_as is dict else json.dumps(a)}}
        for i, (n, a) in enumerate(calls)]}


def _glm_turn(calls, content="Let me fix it."):
    """What GLM wrote in that turn: the text between `<|assistant|>` and the next role marker, after `</think>`."""
    text = _render(GLM, [{"role": "user", "content": "fix the bug"}, _assistant(calls, content),
                         {"role": "user", "content": "thanks"}])
    turn = text.split("<|assistant|>", 1)[1].split("<|user|>", 1)[0]
    return turn.split("</think>", 1)[1]


def _devstral_turn(calls, content="Let me fix it.", arguments_as=dict):
    text = _render(DEVSTRAL, [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "fix the bug"},
                              _assistant(calls, content, arguments_as)])
    return text.split("[/INST]", 1)[1].rsplit("</s>", 1)[0]


def _read_back(parser, turn):
    parsed = parser.parse(turn)
    assert not parsed.malformed, parsed.malformed
    return parsed.prose, [(c.name, typed_arguments(c.arguments, c.name, TOOLS)) for c in parsed.calls]


# -- selection -------------------------------------------------------------------------------------------------------

def test_each_vendor_template_selects_its_own_dialect():
    assert isinstance(select_parser(GLM), GlmToolCallParser), "GLM's template also holds <tool_call>: <arg_key> first"
    assert isinstance(select_parser(DEVSTRAL), MistralArgsToolCallParser)
    assert isinstance(select_parser("... <tool_call>\n<function=NAME> ..."), XmlToolCallParser)
    assert isinstance(select_parser('... <tool_call>\n{"name": ...} ...'), JsonToolCallParser)
    assert select_parser("[TOOL_CALLS][{...}]") is None, "the older list form without [ARGS] is not this dialect"


# -- the round trip through the vendor's template ---------------------------------------------------------------------

def test_glm_reads_back_what_its_template_writes():
    turn = _glm_turn(CALLS)
    assert "<tool_call>edit<arg_key>path</arg_key><arg_value>src/a.py</arg_value>" in turn, turn
    assert _read_back(GlmToolCallParser(), turn) == ("Let me fix it.", CALLS)


def test_glm_reads_a_call_with_no_argument_and_a_turn_with_no_prose():
    assert _read_back(GlmToolCallParser(), _glm_turn([("shell", {})], content="")) == ("", [("shell", {})])


@pytest.mark.parametrize("arguments_as", [dict, str], ids=["arguments as an object", "arguments as a JSON string"])
def test_mistral_reads_back_what_its_template_writes(arguments_as):
    turn = _devstral_turn(CALLS, arguments_as=arguments_as)
    assert turn.startswith("Let me fix it.[TOOL_CALLS]edit[ARGS]{"), turn
    assert _read_back(MistralArgsToolCallParser(), turn) == ("Let me fix it.", CALLS)


def test_mistral_reads_an_empty_arguments_string_as_no_argument():
    turn = _devstral_turn([("shell", "")], arguments_as=dict)          # the template writes '{}' for ''
    assert _read_back(MistralArgsToolCallParser(), turn) == ("Let me fix it.", [("shell", {})])


# -- malformed: surfaced, never read as prose, never half-taken -------------------------------------------------------

@pytest.mark.parametrize("turn", [
    "ok <tool_call>edit<arg_key>path</arg_key><arg_value>a",                                  # cut before its end
    "ok <tool_call><arg_key>path</arg_key><arg_value>a</arg_value></tool_call>",                 # no name
    "ok <tool_call>my tool<arg_key>p</arg_key><arg_value>a</arg_value></tool_call>",             # a space in the name
    "ok <tool_call>edit<arg_key>p</arg_key><arg_value>a</arg_value><arg_key>p</arg_key><arg_value>b</arg_value></tool_call>",
    "ok <tool_call>edit<arg_key>p</arg_key>junk<arg_value>a</arg_value></tool_call>",            # text between key and value
    "ok <tool_call>edit<arg_key>p</arg_key><arg_value>a</arg_value>junk</tool_call>",            # text after a pair
    "ok <tool_call>edit<arg_key>p</arg_key></tool_call>",                                        # a key with no value
    "ok edit<arg_key>p</arg_key><arg_value>a</arg_value></tool_call>",                           # the opener lost
    "ok <arg_key>p</arg_key><arg_value>a</arg_value>",                                           # pairs outside a block
    "ok <tool_call>shell<arg_key>cmd</arg_key><arg_value>ls</arg_value></tool_call><tool_call>edit<arg_key>p</arg_key></tool_call>",
    "ok <tool_call>shell<arg_key>cmd</arg_key><arg_value>ls</arg_value></tool_call><tool_call>edit<arg_key>p",
], ids=["cut", "no-name", "space-in-name", "repeated-key", "text-between", "text-after", "no-value",
        "no-opener", "outside", "good-then-broken", "good-then-cut"])
def test_a_broken_glm_call_is_malformed(turn):
    parsed = GlmToolCallParser().parse(turn)
    assert parsed.malformed and not parsed.calls, parsed
    assert "<arg_key>" not in parsed.prose and "</tool_call>" not in parsed.prose


@pytest.mark.parametrize("turn", [
    "ok [TOOL_CALLS]edit[ARGS]{\"path\": ",                                    # cut
    "ok [TOOL_CALLS]edit{\"path\": \"a\"}",                                    # no [ARGS]
    "ok [TOOL_CALLS][ARGS]{\"path\": \"a\"}",                                  # no name
    "ok [TOOL_CALLS]my tool[ARGS]{}",                                          # a space in the name
    "ok [TOOL_CALLS]edit[ARGS][1, 2]",                                         # not an object
    "ok [TOOL_CALLS]edit[ARGS]{} trailing words",                              # text after the object
    "ok [TOOL_CALLS]shell[ARGS]{\"cmd\": \"ls\"}[TOOL_CALLS]edit[ARGS]{",      # the second call broken
    "ok [TOOL_CALLS][{\"name\": \"edit\", \"arguments\": {}}]",                # the older list form
], ids=["cut", "no-args-marker", "no-name", "space-in-name", "not-an-object", "trailing", "second-broken", "list-form"])
def test_a_broken_mistral_call_is_malformed_and_nothing_of_it_is_taken(turn):
    parsed = MistralArgsToolCallParser().parse(turn)
    assert parsed.malformed and not parsed.calls, parsed
    assert parsed.prose == "ok"


def test_a_mistral_call_that_lost_its_opener_is_malformed_not_prose():
    parsed = MistralArgsToolCallParser().parse('ok edit[ARGS]{"path": "a"}')
    assert parsed.malformed and not parsed.calls and "[ARGS]" not in parsed.prose, parsed


def test_prose_alone_is_prose():
    assert GlmToolCallParser().parse("Done: 3 files.").prose == "Done: 3 files."
    assert MistralArgsToolCallParser().parse("Done: 3 files.").prose == "Done: 3 files."


# -- the decode keeps the markers inside a block ----------------------------------------------------------------------

class PieceTokenizer:
    """ids index `pieces`; ids in `special` vanish under skip_special_tokens, as a tokenizer's special tokens do."""

    def __init__(self, pieces, special=()):
        self.pieces, self.special = list(pieces), set(special)

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.pieces[i] for i in ids if not (skip_special_tokens and i in self.special))


def test_every_marker_a_dialect_reads_is_kept_by_the_decode():
    assert {"[TOOL_CALLS]", "[ARGS]"} <= set(MistralArgsToolCallParser().kept_markers())
    assert {"<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>"} <= \
        set(GlmToolCallParser().kept_markers())
    # Devstral's tokenizer: [TOOL_CALLS] and [ARGS] are special (ids 9 and 32 there)
    tok = PieceTokenizer(["</s>", "[TOOL_CALLS]", "[ARGS]", "shell", '{"cmd": "ls"}'], special=[0, 1, 2])
    assert MarkerDecoder(tok, MistralArgsToolCallParser().kept_markers()).decode([1, 3, 2, 4, 0]) == \
        '[TOOL_CALLS]shell[ARGS]{"cmd": "ls"}'


def _stream(parser, tok, ids):
    reader = TurnReader(tok, parser=parser)
    deltas = []
    for i in ids:
        deltas += reader.push(i)
    deltas += reader.finish()
    return "".join(d.text for d in deltas if d.channel == "content"), \
        [(d.call.name, d.call.arguments) for d in deltas if d.channel == "tool_call"]


def test_the_streaming_reader_yields_mistrals_calls():
    pieces = ["</s>", "[TOOL_CALLS]", "[ARGS]", "Checking.", "shell", '{"cmd": ', '"ls"}', "edit", '{"path": "a"}']
    tok = PieceTokenizer(pieces, special=[0, 1, 2])
    content, calls = _stream(MistralArgsToolCallParser(), tok, [3, 1, 4, 2, 5, 6, 1, 7, 2, 8, 0])
    assert (content, calls) == ("Checking.", [("shell", {"cmd": "ls"}), ("edit", {"path": "a"})])


def test_the_streaming_reader_yields_glms_calls():
    pieces = ["<|observation|>", "<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>",
              "</arg_value>", "Checking.", "shell", "cmd", "ls -la", "edit", "path", "a.py"]
    tok = PieceTokenizer(pieces, special=[0])
    ids = [7, 1, 8, 3, 9, 4, 5, 10, 6, 2, 1, 11, 3, 12, 4, 5, 13, 6, 2, 0]
    assert _stream(GlmToolCallParser(), tok, ids) == ("Checking.", [("shell", {"cmd": "ls -la"}),
                                                                    ("edit", {"path": "a.py"})])
