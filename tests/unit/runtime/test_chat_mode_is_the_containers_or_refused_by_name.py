"""chat_mode has ONE reader (core/runtime/prompt_length.declared_chat_mode): the request's flag, else the container's,
else a refusal by name — the engine supplies no value of its own (owner reset 22/11)."""
from pathlib import Path

import pytest

from neurobrix.core.runtime.prompt_length import declared_chat_mode

SRC = Path(__file__).resolve().parents[3] / "src" / "neurobrix"


def test_the_request_flag_wins_over_the_container():
    assert declared_chat_mode({"chat_mode": True}, False) is False
    assert declared_chat_mode({}, True) is True


def test_the_container_value_is_read_when_the_request_gives_none():
    assert declared_chat_mode({"chat_mode": True}) is True
    assert declared_chat_mode({"chat_mode": False}, None) is False


@pytest.mark.parametrize("defaults", [{}, None, {"max_tokens": 8}])
def test_a_container_without_chat_mode_is_refused_by_name(defaults):
    with pytest.raises(RuntimeError, match="declares no chat_mode"):
        declared_chat_mode(defaults)


def test_no_reader_keeps_an_engine_default_for_chat_mode():
    offenders = [f"{p.relative_to(SRC)}:{i}" for p in SRC.rglob("*.py")
                 for i, line in enumerate(p.read_text().splitlines(), 1)
                 if p.name != "prompt_length.py"
                 # a container read with an engine default; a request's own flag passed on (kwargs["chat_mode"]) is not
                 and ('get("chat_mode",' in line or "get('chat_mode'," in line
                      or ('["chat_mode"]' in line and "default" in line))]
    assert offenders == []


def test_the_refusal_names_the_remedy():
    """A container predating the key is told its remedy: the corrected container, in the engine's re-import wording."""
    with pytest.raises(RuntimeError, match=r"install the corrected container \(neurobrix remove <model> && "
                                           r"neurobrix import <org>/<model>\)"):
        declared_chat_mode({})
