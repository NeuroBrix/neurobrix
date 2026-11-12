"""A constrained answer that follows a reasoning starts with what the template writes between the reasoning's end and
the answer, read from the template itself, never a fixed text.

Measured 2026-11-12: the catalogue's thinking templates (Qwen3-30B-A3B-Thinking, Qwen3-VL-Thinking, Qwen3-Omni,
MiniCPM-o, canary-qwen) write "\\n\\n" after `</think>`; GLM-4.7-Flash's writes nothing (`</think>` then the content). The
grammar held a fixed "\\n\\n" there (core/state/constraint.py), right for the first five and wrong for GLM: a forced
call would have made GLM write two newlines it was never trained to write.

The gap is read by rendering one past turn with a reasoning and an answer through the model's own template. A template
that shows no past reasoning gives no gap, and a constraint after such a reasoning is refused by name.
"""
import json
from pathlib import Path

import pytest

pytest.importorskip("jinja2")

from tests.unit.grammar_engine import load  # noqa: E402

nbxgrammar = load()
from neurobrix.agent.parsers import GlmToolCallParser, JsonToolCallParser  # noqa: E402
from neurobrix.core.module.tokenizer.sp_tokenizer import HFTokenizer  # noqa: E402
from neurobrix.core.state.constraint import Constrain, ForcedCall, compile_constraint  # noqa: E402
from neurobrix.serving.openai_api.turn import reasoning_gap  # noqa: E402

GLM = (Path(__file__).parents[1] / "agent" / "data" / "templates" / "GLM-4.7-Flash.jinja").read_text()
# Qwen3's own way of writing a past turn's reasoning (Qwen3-30B-A3B-Thinking-2507's template, the assistant branch)
QWEN = ("{% for m in messages %}<|im_start|>{{ m.role }}\n{% if m.reasoning_content %}"
        "{{ '<think>\\n' + m.reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + m.content.lstrip('\\n') }}"
        "{% else %}{{ m.content }}{% endif %}<|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n<think>\n{% endif %}")
SILENT = "{% for m in messages %}<{{ m.role }}>{{ m.content }}{% endfor %}{% if add_generation_prompt %}<think>{% endif %}"

MARKERS = {"<tool_call>": 1000, "</tool_call>": 1001, "<arg_key>": 1002, "</arg_key>": 1003, "<arg_value>": 1004,
           "</arg_value>": 1005, "</think>": 1006}
EOS = 1010
TOOLS = [{"type": "function", "function": {"name": "get_time", "parameters": {
    "type": "object", "properties": {"tz": {"type": "string"}}, "required": ["tz"]}}}]


def _renderer(template):
    t = HFTokenizer.__new__(HFTokenizer)
    t._chat_template, t._bos_token, t._eos_token = template, "", ""
    return t


class _Tok:
    """The grammar's tokenizer: each marker one id, every other text its bytes."""

    def encode(self, text, add_special_tokens=False):
        return [MARKERS[text]] if text in MARKERS else list(text.encode())


def _walk(grammar_text, text):
    toks = [bytes([b]) for b in range(256)] + [None] * (EOS - 256 + 1)
    v, m = nbxgrammar.Vocab(toks, [EOS]), nbxgrammar.Matcher(nbxgrammar.Grammar.description(grammar_text))
    i = 0
    while i < len(text):
        hit = next((k for k in MARKERS if text.startswith(k, i)), None)
        ids, i = ([MARKERS[hit]], i + len(hit)) if hit else (list(text[i].encode()), i + 1)
        for t in ids:
            w = m.mask(v).view("uint32")
            if not (w[t >> 5] >> (t & 31)) & 1:
                return False
            m.advance(v, t)
    w = m.mask(v).view("uint32")
    return bool((w[EOS >> 5] >> (EOS & 31)) & 1)


def _grammar(parser, gap, monkeypatch):
    from neurobrix.core.state import constraint
    seen = []
    monkeypatch.setattr(constraint, "compile_description", lambda text: seen.append(text) or object())
    compile_constraint(Constrain(call=ForcedCall(parser, TOOLS, None, True), after_reasoning="</think>",
                                 reasoning_gap=gap), _Tok())
    return seen[0]


def test_the_gap_is_read_from_each_template():
    assert reasoning_gap(_renderer(QWEN), ("<think>", "</think>")) == "\n\n"
    assert reasoning_gap(_renderer(GLM), ("<think>", "</think>")) == ""
    assert reasoning_gap(_renderer(SILENT), ("<think>", "</think>")) is None, "no past reasoning shown: no gap"


def test_glm_closes_its_reasoning_straight_into_the_call(monkeypatch):
    call = "<tool_call>get_time<arg_key>tz</arg_key><arg_value>UTC</arg_value></tool_call>"
    g = _grammar(GlmToolCallParser(), reasoning_gap(_renderer(GLM), ("<think>", "</think>")), monkeypatch)
    assert _walk(g, "The user wants the time.</think>" + call)
    assert not _walk(g, "The user wants the time.</think>\n\n" + call), "GLM's template writes nothing there"


def test_qwen_keeps_its_two_newlines(monkeypatch):
    call = '<tool_call>\n{"name": "get_time", "arguments": {"tz": "UTC"}}\n</tool_call>'
    g = _grammar(JsonToolCallParser(), reasoning_gap(_renderer(QWEN), ("<think>", "</think>")), monkeypatch)
    assert json.loads(g)["seq"][1] == {"lit": "\n\n"}, "the description Qwen's requests were compiled from, unchanged"
    assert _walk(g, "The user wants the time.</think>\n\n" + call)
    assert not _walk(g, "The user wants the time.</think>" + call)


def test_a_constraint_after_a_reasoning_with_no_known_gap_is_refused_by_name():
    with pytest.raises(ValueError, match="shows no past reasoning"):
        compile_constraint(Constrain(call=ForcedCall(GlmToolCallParser(), TOOLS, None, True),
                                     after_reasoning="</think>", reasoning_gap=None), _Tok())
