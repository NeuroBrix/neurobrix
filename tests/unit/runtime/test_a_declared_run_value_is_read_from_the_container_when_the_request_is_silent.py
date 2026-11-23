"""A run value the container declares is read from it when the request is silent.

orpheus-3b on --compiled stopped at the prompt's tokenization: the PyTorch text processor asked the resolver for
`global.chat_mode` with `default=None`, and `VariableResolver.get` read an explicit None as "no default given" and
raised KeyError for a request that set no flag, before the container's declared chat_mode was ever read. The Triton
branch, reading the same key through the same reader (`declared_chat_mode`), was green on the same container.

The root is the resolver's contract: `get` now tells "no default given" (absence raises) from an explicit None
(absence answers None, the caller handles it). The text processor of the PyTorch branch then reads chat_mode as the
Triton branch does: the request's flag, else the container's, else the refusal by name.

Run: PYTHONPATH=src python -m pytest tests/unit/runtime/test_a_declared_run_value_is_read_from_the_container_when_the_request_is_silent.py
"""
from __future__ import annotations

import pytest

from neurobrix.core.module.text.processor import TextProcessor
from neurobrix.core.runtime.resolution.variable_resolver import VariableResolver

TEMPLATED = [101, 102, 103]   # what the chat template renders
PLAIN = [7]                   # what the plain encode renders


class _Tokenizer:
    """A tokenizer with a chat template: which path the processor took is read from the ids it returns."""

    def has_chat_template(self):
        return True

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        return list(TEMPLATED)

    def encode(self, prompt, add_special_tokens=True):
        return list(PLAIN)


def _resolver(request=None):
    vr = VariableResolver({}, {}, {}, {}, {}, device="cpu", mode="compiled")
    vr.set("global.prompt", "hello")
    for key, value in (request or {}).items():
        vr.set(f"global.{key}", value)
    return vr


def _ids(defaults, request=None):
    tp = TextProcessor(_Tokenizer(), defaults, {}, _resolver(request))
    return tp.tokenize("hello", "cpu").flatten().tolist()


def test_a_silent_request_reads_the_containers_chat_mode():
    assert _ids({"chat_mode": True}) == TEMPLATED
    assert _ids({"chat_mode": False}) == PLAIN


@pytest.mark.parametrize("declared,asked,expected", [
    (True, False, PLAIN),
    (False, True, TEMPLATED),
])
def test_the_requests_own_flag_wins(declared, asked, expected):
    assert _ids({"chat_mode": declared}, {"chat_mode": asked}) == expected


def test_neither_the_request_nor_the_container_is_refused_by_name():
    with pytest.raises(RuntimeError, match="declares no chat_mode"):
        _ids({})


def test_the_resolver_raises_an_absence_only_when_no_default_is_given():
    vr = _resolver()
    with pytest.raises(KeyError, match="global.chat_mode"):
        vr.get("global.chat_mode")


def test_an_explicit_none_default_answers_none():
    assert _resolver().get("global.chat_mode", default=None) is None
    assert _resolver().get("global.chat_mode", None) is None


def test_a_given_default_answers_and_a_set_value_wins_over_it():
    vr = _resolver({"max_tokens": 12})
    assert vr.get("global.top_k", default=0) == 0
    assert vr.get("global.max_tokens", default=None) == 12
    assert vr.get("global.max_tokens") == 12
