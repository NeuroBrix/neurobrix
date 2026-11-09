"""The tool-call envelopes (agent/envelope.py) compiled by our grammar engine: for each dialect the canonical call text
the template defines is accepted step by step (markers as token steps), the dialect's own parser reads back the call
with its arguments, and what the envelope forbids is refused (an undeclared function, a second call when parallel is
off, prose before the call, a call missing a required argument)."""
import json

import pytest

from tests.unit.grammar_engine import load  # noqa: E402

nbx_grammar = load()
from neurobrix.agent.envelope import envelope  # noqa: E402
from neurobrix.agent.parsers import JsonListToolCallParser, JsonToolCallParser, XmlToolCallParser  # noqa: E402

MARKERS = {"<tool_call>": 1000, "</tool_call>": 1001, "<|tool_call|>": 1002}
EOS = 1003
TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {
            "type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
            "required": ["city", "days"]}}},
         {"type": "function", "function": {"name": "get_time", "parameters": {
            "type": "object", "properties": {"tz": {"type": "string"}}, "required": ["tz"]}}}]


def encode(text):
    return [MARKERS[text]] if text in MARKERS else list(text.encode())


def _vocab():
    toks = [bytes([b]) for b in range(256)] + [None] * (EOS - 256 + 1)
    return nbx_grammar.Vocab(toks, [EOS])


def _walk(desc, text):
    """Feed `text` as single bytes, its markers as their token steps; True when every step was allowed and the end
    of the turn is allowed after it."""
    v, m = _vocab(), nbx_grammar.Matcher(nbx_grammar.Grammar.description(json.dumps(desc)))
    i = 0
    while i < len(text):
        hit = next((k for k in MARKERS if text.startswith(k, i)), None)
        if hit:
            ids, i = [MARKERS[hit]], i + len(hit)
        else:
            ch = text[i].encode()
            ids, i = list(ch), i + 1
        for t in ids:
            w = m.mask(v).view("uint32")
            if not (w[t >> 5] >> (t & 31)) & 1:
                return False
            m.advance(v, t)
    w = m.mask(v).view("uint32")
    return bool((w[EOS >> 5] >> (EOS & 31)) & 1)


CASES = {
    "json": (JsonToolCallParser(),
             '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris", "days": 2}}\n</tool_call>'),
    "xml": (XmlToolCallParser(),
            "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n2\n"
            "</parameter>\n</function>\n</tool_call>"),
    "json_list": (JsonListToolCallParser(),
                  '<|tool_call|>[{"name": "get_weather", "arguments": {"city": "Paris", "days": 2}}]'),
}


@pytest.mark.parametrize("dialect", sorted(CASES))
def test_the_canonical_call_is_accepted_and_read_back_by_its_parser(dialect):
    parser, text = CASES[dialect]
    assert _walk(envelope(CASES[dialect][0], TOOLS, encode), text)
    calls = parser.parse(text).calls
    assert [c.name for c in calls] == ["get_weather"]
    assert {k: str(v) for k, v in calls[0].arguments.items()} == {"city": "Paris", "days": "2"}


@pytest.mark.parametrize("dialect", sorted(CASES))
def test_what_the_envelope_forbids_is_refused(dialect):
    _, text = CASES[dialect]
    d = envelope(CASES[dialect][0], TOOLS, encode)
    assert not _walk(d, "Sure! " + text)                                   # prose before the call
    assert not _walk(d, text.replace("get_weather", "get_wether"))          # an undeclared function
    assert not _walk(envelope(CASES[dialect][0], TOOLS, encode, only="get_time"), text)   # tool_choice named another one
    if dialect == "json_list":
        two = text[:-1] + ', {"name": "get_time", "arguments": {"tz": "UTC"}}]'
    else:
        two = text + "\n" + text
    assert _walk(d, two)                                                    # parallel calls allowed by default
    assert not _walk(envelope(CASES[dialect][0], TOOLS, encode, parallel=False), two)  # one call when parallel is off
    missing = text.replace('"city": "Paris", ', "").replace("<parameter=city>\nParis\n</parameter>\n", "")
    assert not _walk(d, missing)                                            # a required argument missing



def test_an_xml_string_parameter_with_a_constraint_free_text_cannot_enforce_is_refused_by_name():
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {
        "code": {"type": "string", "pattern": "^[A-Z]{3}$"}}, "required": ["code"]}}}]
    with pytest.raises(ValueError, match="pattern"):
        envelope(XmlToolCallParser(), tools, encode)



@pytest.mark.parametrize("dialect", sorted(CASES))
def test_a_named_function_is_exactly_one_call(dialect):
    """tool_choice naming a function = exactly one call of it (OpenAI, vLLM; the rack's granite cell made two)."""
    parser, _ = CASES[dialect]
    one = {"json": '<tool_call>\n{"name": "get_time", "arguments": {"tz": "UTC"}}\n</tool_call>',
           "xml": "<tool_call>\n<function=get_time>\n<parameter=tz>\nUTC\n</parameter>\n</function>\n</tool_call>",
           "json_list": '<|tool_call|>[{"name": "get_time", "arguments": {"tz": "UTC"}}]'}[dialect]
    two = one[:-1] + ', {"name": "get_time", "arguments": {"tz": "UTC"}}]' if dialect == "json_list" else one + "\n" + one
    d = envelope(parser, TOOLS, encode, only="get_time")
    assert _walk(d, one) and not _walk(d, two)
