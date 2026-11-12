"""The tool-call envelopes of the GLM and Mistral dialects (agent/envelope.py; docs/design/tool-call-dialects.md),
compiled by our grammar engine. The call the vendor's template writes is accepted step by step, with its markers as
the token steps of the vendor's tokenizer, and read back by the dialect's parser. What the envelope forbids is refused:
prose first, an undeclared name, another function than the one named, a missing required argument, and a second
call when parallel calls are off. Two names where one is a prefix of the other cannot be told apart once the name is
followed by a marker token, so they are refused by name before compiling.
"""
import json

import pytest

from tests.unit.grammar_engine import load  # noqa: E402

nbxgrammar = load()
from neurobrix.agent.envelope import envelope  # noqa: E402
from neurobrix.agent.parsers import GlmToolCallParser, MistralArgsToolCallParser  # noqa: E402

# the markers as one id each, as in the vendors' tokenizers (GLM: added tokens; Devstral: special tokens)
MARKERS = {"<tool_call>": 1000, "</tool_call>": 1001, "<arg_key>": 1002, "</arg_key>": 1003, "<arg_value>": 1004,
           "</arg_value>": 1005, "[TOOL_CALLS]": 1006, "[ARGS]": 1007}
EOS = 1010
TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {
            "type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"},
                                             "unit": {"enum": ["c", "f"]}},
            "required": ["city", "days"]}}},
         {"type": "function", "function": {"name": "get_time", "parameters": {
            "type": "object", "properties": {"tz": {"type": "string"}}, "required": ["tz"]}}}]


def encode(text):
    return [MARKERS[text]] if text in MARKERS else list(text.encode())


def spelled(text):
    return list(text.encode())


def _walk(desc, text, markers=MARKERS):
    """Feed `text` as single bytes, its markers as their token steps; True when every step was allowed and the end
    of the turn is allowed after it."""
    toks = [bytes([b]) for b in range(256)] + [None] * (EOS - 256 + 1)
    v, m = nbxgrammar.Vocab(toks, [EOS]), nbxgrammar.Matcher(nbxgrammar.Grammar.description(json.dumps(desc)))
    i = 0
    while i < len(text):
        hit = next((k for k in markers if text.startswith(k, i)), None)
        ids, i = ([markers[hit]], i + len(hit)) if hit else (list(text[i].encode()), i + 1)
        for t in ids:
            w = m.mask(v).view("uint32")
            if not (w[t >> 5] >> (t & 31)) & 1:
                return False
            m.advance(v, t)
    w = m.mask(v).view("uint32")
    return bool((w[EOS >> 5] >> (EOS & 31)) & 1)


GLM_CALL = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value><arg_key>days</arg_key><arg_value>2</arg_value></tool_call>"
GLM_TIME = "<tool_call>get_time<arg_key>tz</arg_key><arg_value>UTC</arg_value></tool_call>"
MISTRAL_CALL = '[TOOL_CALLS]get_weather[ARGS]{"city": "Paris", "days": 2}'
MISTRAL_TIME = '[TOOL_CALLS]get_time[ARGS]{"tz": "UTC"}'
CASES = {"glm": (GlmToolCallParser(), GLM_CALL, GLM_TIME), "mistral_args": (MistralArgsToolCallParser(), MISTRAL_CALL, MISTRAL_TIME)}


@pytest.mark.parametrize("dialect", sorted(CASES))
def test_the_vendors_call_is_accepted_and_read_back(dialect):
    parser, call, time = CASES[dialect]
    d = envelope(parser, TOOLS, encode)
    assert _walk(d, call) and _walk(d, call + time), "one call, and two back to back"
    calls = parser.parse(call + time).calls
    assert [(c.name, {k: str(v) for k, v in c.arguments.items()}) for c in calls] == \
        [("get_weather", {"city": "Paris", "days": "2"}), ("get_time", {"tz": "UTC"})]


@pytest.mark.parametrize("dialect", sorted(CASES))
def test_what_the_envelope_forbids_is_refused(dialect):
    parser, call, time = CASES[dialect]
    d = envelope(parser, TOOLS, encode)
    assert not _walk(d, "Sure! " + call), "prose first"
    assert not _walk(d, call.replace("get_weather", "get_wether")), "an undeclared function"
    assert not _walk(envelope(parser, TOOLS, encode, only="get_time"), call), "tool_choice named another one"
    assert _walk(envelope(parser, TOOLS, encode, only="get_time"), time)
    assert not _walk(envelope(parser, TOOLS, encode, only="get_time"), time + time), "a named function: one call"
    assert not _walk(envelope(parser, TOOLS, encode, parallel=False), call + time), "parallel off: one call"
    missing = call.replace("<arg_key>days</arg_key><arg_value>2</arg_value>", "").replace(', "days": 2', "")
    assert missing != call and not _walk(d, missing), "a required argument missing"


def test_glm_writes_a_string_bare_up_to_its_closing_marker_and_a_number_as_json():
    d = envelope(GlmToolCallParser(), TOOLS, encode)
    multiline = GLM_CALL.replace("Paris", "Paris\n<b> & é")
    assert _walk(d, multiline), "a string value is free text up to </arg_value>"
    assert not _walk(d, GLM_CALL.replace("<arg_value>2<", "<arg_value>two<")), "an integer is the schema's JSON"


def test_a_glm_enum_value_is_written_bare_and_its_values_must_be_told_apart():
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {
        "unit": {"enum": ["c", "f"]}}, "required": ["unit"]}}}]
    d = envelope(GlmToolCallParser(), tools, encode)
    call = "<tool_call>f<arg_key>unit</arg_key><arg_value>{}</arg_value></tool_call>"
    assert _walk(d, call.format("c")) and _walk(d, call.format("f"))
    assert not _walk(d, call.format('"c"')) and not _walk(d, call.format("k"))
    tools[0]["function"]["parameters"]["properties"]["unit"]["enum"] = ["c", "celsius"]
    with pytest.raises(ValueError, match="'c' is a prefix of 'celsius'"):
        envelope(GlmToolCallParser(), tools, encode)


def test_glm_markers_spelled_in_bytes_where_the_tokenizer_has_no_single_id():
    d = envelope(GlmToolCallParser(), TOOLS, spelled)
    assert _walk(d, GLM_CALL, markers={}), "each marker its text, when it is not one id"


class _GlmWithOtherInnerMarkers(GlmToolCallParser):
    inner_markers = ("<k>", "</k>", "<v>", "</v>")


def test_the_glm_envelope_writes_the_inner_markers_its_parser_carries():
    """The envelope reads GLM's key/value markers from the parser, never from its own copy: a parser carrying other
    markers gets an envelope written with them, for a string, an integer and an enum value alike."""
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {
        "city": {"type": "string"}, "days": {"type": "integer"}, "unit": {"enum": ["c", "f"]}},
        "required": ["city", "days", "unit"]}}}]
    call = "<tool_call>f{0}city{1}{2}Paris{3}{0}days{1}{2}2{3}{0}unit{1}{2}c{3}</tool_call>"
    d = envelope(_GlmWithOtherInnerMarkers(), tools, spelled)
    assert _walk(d, call.format("<k>", "</k>", "<v>", "</v>"), markers={}), "the parser's markers"
    assert not _walk(d, call.format(*GlmToolCallParser.inner_markers), markers={}), "not GLM's own"


def test_a_tool_with_no_required_argument_is_its_name_and_the_close():
    tools = [{"type": "function", "function": {"name": "now", "parameters": {"type": "object", "properties": {}}}}]
    for enc, markers in ((encode, MARKERS), (spelled, {})):
        d = envelope(GlmToolCallParser(), tools, enc)
        assert _walk(d, "<tool_call>now</tool_call>", markers) and not _walk(d, "<tool_call>no</tool_call>", markers)
    assert _walk(envelope(MistralArgsToolCallParser(), tools, encode), "[TOOL_CALLS]now[ARGS]{}")


def test_two_names_one_a_prefix_of_the_other_are_refused_by_name_where_a_marker_token_follows():
    tools = TOOLS + [{"type": "function", "function": {"name": "get_weather_now", "parameters": {
        "type": "object", "properties": {}}}}]
    for parser in (GlmToolCallParser(), MistralArgsToolCallParser()):
        with pytest.raises(ValueError, match="'get_weather' is a prefix of 'get_weather_now'"):
            envelope(parser, tools, encode)
    # spelled markers make each literal the name and its marker's text: told apart, so not refused
    assert _walk(envelope(GlmToolCallParser(), tools, spelled), GLM_CALL, markers={})
