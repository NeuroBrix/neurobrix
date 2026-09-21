"""A vendor chat template may call the helpers transformers injects into its Jinja
environment; the engine's renderer provides the same two.

Seen red on 2026-09-20: granite-3.1-1b-a400m-instruct's template opens with
`{%- set today = strftime_now("%B %d, %Y") %}` and every mode failed before the
first token with "'strftime_now' is undefined". `raise_exception` is the other
helper transformers defines (a template's own refusal of a role it does not
accept); it surfaces as a TemplateError so the message reads as the vendor's.
The cells render a minimal template through the engine's own renderer with a
tokenizer stub — no model, no GPU.
"""
from __future__ import annotations

import datetime

import pytest

jinja2 = pytest.importorskip("jinja2")

from neurobrix.core.module.tokenizer.sp_tokenizer import HFTokenizer  # noqa: E402


class _Enc:
    ids = [1, 2, 3]


def _renderer(template: str) -> HFTokenizer:
    t = HFTokenizer.__new__(HFTokenizer)
    t._chat_template = template
    t._bos_token = "<s>"
    t._eos_token = "</s>"
    t._tokenizer = type("T", (), {"encode": staticmethod(lambda s: _Enc())})()
    return t


def test_strftime_now_renders_todays_date():
    t = _renderer('{%- set today = strftime_now("%Y") %}year:{{ today }}|{{ messages[0]["content"] }}')
    out = t.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False)
    assert out == f"year:{datetime.datetime.now().strftime('%Y')}|hi"


def test_raise_exception_is_the_templates_own_refusal():
    t = _renderer('{% if messages[0]["role"] == "system" %}{{ raise_exception("no system role here") }}{% endif %}ok')
    with pytest.raises(jinja2.exceptions.TemplateError, match="no system role here"):
        t.apply_chat_template([{"role": "system", "content": "x"}], tokenize=False)
    assert t.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False) == "ok"
