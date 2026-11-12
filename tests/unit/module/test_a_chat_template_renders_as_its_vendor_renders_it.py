"""A chat template is rendered in the environment its vendor's reference renders it in (transformers,
src/transformers/utils/chat_template_utils.py: `ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
extensions=[AssistantTracker, jinja2.ext.loopcontrols])`, its own `tojson` filter, `raise_exception`, `strftime_now`).

Measured red on 2026-11-12 with the engine's plain `jinja2.Environment`:
- `{{ x|tojson }}` wrote `if a \\u003c b` for `if a < b` and `\\u00e9` for `é`. jinja's built-in filter escapes HTML
  characters, which the reference overrides "because Jinja's default filter escapes HTML characters". So every tool
  schema and past call in a prompt differed from the one the model was trained on.
- GLM-4.7-Flash's template writes a past call's non-string value with `tojson(ensure_ascii=False)`, and the built-in
  filter refused the argument: "do_tojson() got an unexpected keyword argument 'ensure_ascii'".
- The same filter sorted the keys: a tool's schema came out reordered. The catalogue's 17 templates were rendered
  old against new on one conversation with tools: 8 differ and none fails; every plain conversation renders
  byte-identical. The new environment with the old filter put back gives the old text in all 34 renders.
- `{% break %}` is not a tag without `loopcontrols`.
- A template is data carried by a container. Rendered unsandboxed, a template reaches Python's objects through
  attribute access; the sandbox refuses it.
The cells render through the engine's own renderer with a tokenizer stub: no model, no GPU.
"""
from __future__ import annotations

import json

import pytest

jinja2 = pytest.importorskip("jinja2")

from neurobrix.core.module.tokenizer.sp_tokenizer import HFTokenizer  # noqa: E402


class _Enc:
    ids = [1, 2, 3]


def _render(template: str, **context) -> str:
    t = HFTokenizer.__new__(HFTokenizer)
    t._chat_template = template
    t._bos_token = "<s>"
    t._eos_token = "</s>"
    t._tokenizer = type("T", (), {"encode": staticmethod(lambda s: _Enc())})()
    return t.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False, **context)


CODE = {"code": "if a < b and c > d: print('é & ok')"}


def test_tojson_writes_the_text_as_json_dumps_does_never_html_escaped():
    out = _render("{{ x|tojson }}", x=CODE)
    assert out == json.dumps(CODE, ensure_ascii=False)
    assert "\\u003c" not in out and "\\u00e9" not in out
    # the keys in the order the request gave them (jinja's built-in filter sorted them: a tool's schema came out
    # reordered, "function" before "type")
    assert _render("{{ x|tojson }}", x={"type": "function", "function": {"name": "f"}}) == \
        '{"type": "function", "function": {"name": "f"}}'


def test_tojson_takes_the_references_arguments():
    assert _render("{{ x|tojson(ensure_ascii=False) }}", x={"k": "é"}) == '{"k": "é"}'
    assert _render("{{ x|tojson(ensure_ascii=True) }}", x={"k": "é"}) == '{"k": "\\u00e9"}'
    assert _render("{{ x|tojson(indent=2) }}", x={"k": 1}) == json.dumps({"k": 1}, ensure_ascii=False, indent=2)
    assert _render("{{ x|tojson(sort_keys=True) }}", x={"b": 1, "a": 2}) == '{"a": 2, "b": 1}'
    assert _render("{{ x|tojson(separators=(',', ':')) }}", x={"a": [1, 2]}) == '{"a":[1,2]}'


def test_loop_controls_are_tags():
    assert _render("{% for i in [1, 2, 3] %}{% if i == 2 %}{% break %}{% endif %}{{ i }}{% endfor %}") == "1"
    assert _render("{% for i in [1, 2, 3] %}{% if i == 2 %}{% continue %}{% endif %}{{ i }}{% endfor %}") == "13"


def test_a_template_cannot_reach_pythons_objects():
    with pytest.raises(jinja2.exceptions.SecurityError):
        _render("{{ ''.__class__.__mro__[1].__subclasses__() }}")


def test_the_two_helpers_stay():
    assert _render('{{ strftime_now("%Y")|length }}') == "4"
    with pytest.raises(jinja2.exceptions.TemplateError, match="no"):
        _render('{{ raise_exception("no") }}')


def test_glms_way_of_writing_a_past_call_renders():
    # the GLM-4.7-Flash template's own line for one argument (zai-org/GLM-4.7-Flash, chat_template.jinja)
    line = ("{% for k, v in args.items() %}<arg_key>{{ k }}</arg_key><arg_value>"
            "{{ v | tojson(ensure_ascii=False) if v is not string else v }}</arg_value>{% endfor %}")
    out = _render(line, args={"path": "a.py", "limit": 40, "flags": ["é"]})
    assert out == ('<arg_key>path</arg_key><arg_value>a.py</arg_value><arg_key>limit</arg_key><arg_value>40'
                   '</arg_value><arg_key>flags</arg_key><arg_value>["é"]</arg_value>')
