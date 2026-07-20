"""Parser dialect tests — XML and JSON tool-call formats + data-driven selection."""

from neurobrix.agent.parsers import (
    JsonToolCallParser,
    XmlToolCallParser,
    select_parser,
)

_XML_TURN = """I'll read the file first.
<tool_call>
<function=read_file>
<parameter=path>
src/mod.py
</parameter>
<parameter=limit>
40
</parameter>
</function>
</tool_call>"""

_XML_MULTILINE_VALUE = """<tool_call>
<function=write_file>
<parameter=path>
a.py
</parameter>
<parameter=content>
def f():
    return 1
</parameter>
</function>
</tool_call>"""


def test_xml_parses_prose_and_arguments():
    result = XmlToolCallParser().parse(_XML_TURN)
    assert result.prose == "I'll read the file first."
    assert len(result.calls) == 1 and not result.malformed
    call = result.calls[0]
    assert call.name == "read_file"
    assert call.arguments == {"path": "src/mod.py", "limit": "40"}


def test_xml_preserves_inner_whitespace_of_values():
    result = XmlToolCallParser().parse(_XML_MULTILINE_VALUE)
    assert result.calls[0].arguments["content"] == "def f():\n    return 1"


def test_xml_truncated_block_is_malformed_not_dropped():
    result = XmlToolCallParser().parse("done?\n<tool_call>\n<function=bash>")
    assert result.calls == []
    assert len(result.malformed) == 1
    assert result.prose == "done?"


def test_xml_final_answer_has_no_calls():
    result = XmlToolCallParser().parse("All tests pass. Done.")
    assert result.calls == [] and result.malformed == []
    assert result.prose == "All tests pass. Done."


def test_xml_unwrapped_function_block_is_malformed_not_final_answer():
    # A decode near-tie can drop the <tool_call> opener while keeping the
    # function block + closer. The template contract requires the wrapper,
    # so this is a broken tool call (loop must recover), NOT a final answer.
    text = ("Sure.\n<function=write_file>\n<parameter=path>\nx.txt\n"
            "</parameter>\n</function>\n</tool_call>")
    result = XmlToolCallParser().parse(text)
    assert result.calls == []
    assert len(result.malformed) == 1  # would else read as final answer → loop stops
    assert result.prose == "Sure."


def test_xml_dangling_closer_is_malformed():
    result = XmlToolCallParser().parse("here you go\n</tool_call>")
    assert result.calls == []
    assert len(result.malformed) == 1


def test_json_dialect_parses_and_rejects():
    parser = JsonToolCallParser()
    good = parser.parse('<tool_call>\n{"name": "bash", "arguments": {"command": "ls"}}\n</tool_call>')
    assert good.calls[0].name == "bash"
    assert good.calls[0].arguments == {"command": "ls"}
    bad = parser.parse("<tool_call>\nnot json\n</tool_call>")
    assert bad.calls == [] and len(bad.malformed) == 1


def test_select_parser_is_template_marker_driven():
    assert select_parser("... <tool_call> <function=x> ...").format_name == "xml"
    assert select_parser('... <tool_call> {"name": ... ').format_name == "json"
    assert select_parser("no tools here") is None
    assert select_parser("") is None
