"""A chat template that marks its assistant spans with `{% generation %}...{% endgeneration %}` renders as its vendor's
reference renders it (transformers, src/transformers/utils/chat_template_utils.py, the `AssistantTracker` extension):
the tag only marks the span for training masks, and a render gives its body unchanged.

Measured red on 2026-11-13: Qwen3.8-27B's template (Qwen/Qwen3.8-27B, chat_template.jinja, its assistant branch) failed
to parse in the engine's environment, "Encountered unknown tag 'generation'": no chat request could be rendered. The
expected strings below are the reference's own renders (transformers 5.23.0, jinja2 3.1.6).
The cells render through the engine's own renderer with a tokenizer stub: no model, no GPU.
"""
from __future__ import annotations

import pytest

jinja2 = pytest.importorskip("jinja2")

from neurobrix.core.module.tokenizer.sp_tokenizer import HFTokenizer  # noqa: E402


def _render(template: str, messages, add_generation_prompt: bool) -> str:
    t = HFTokenizer.__new__(HFTokenizer)
    t._chat_template = template
    t._bos_token = "<s>"
    t._eos_token = "</s>"
    return t.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)


# The shape of Qwen3.8-27B's assistant branch: the span opens and closes with whitespace control, the loop's own
# variables are read inside it.
MARKED = ("{% for m in messages %}{{ '<' + m.role + '>' }}{% if m.role == 'assistant' %}{%- generation %}\n"
          "    {{- m.content + '<end>' }}\n    {%- if loop.last %}!{% endif %}\n{%- endgeneration %}\n"
          "{% else %}{{ m.content }}{% endif %}\n{% endfor %}"
          "{% if add_generation_prompt %}<assistant>{% endif %}")
TURNS = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"},
         {"role": "user", "content": "x"}, {"role": "assistant", "content": "z"}]


def test_a_marked_assistant_span_renders_its_body_unchanged():
    assert _render(MARKED, TURNS, True) == "<user>hi<assistant>yo<end><user>x<assistant>z<end>!<assistant>"
    assert _render(MARKED, TURNS, False) == "<user>hi<assistant>yo<end><user>x<assistant>z<end>!"


def test_the_tag_keeps_the_references_block_trimming():
    assert _render("{% generation %}\n  a {{ 1 + 1 }}\n{% endgeneration %}\nb", TURNS, False) == "  a 2\nb"


def test_an_unclosed_span_is_the_templates_own_error():
    with pytest.raises(jinja2.exceptions.TemplateSyntaxError, match="looking for the following tags: 'endgeneration'"):
        _render("{% generation %}a", TURNS, False)
