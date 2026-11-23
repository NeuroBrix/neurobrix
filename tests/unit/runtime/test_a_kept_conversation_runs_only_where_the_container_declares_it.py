"""A conversation kept across turns runs only for a container declaring context.keeps true; false or undeclared is
refused by name — the container decides, not its family (supervisor 23/11, order B)."""
from pathlib import Path

import pytest

from neurobrix.core.runtime.prompt_length import declared_context_keeps

SRC = Path(__file__).resolve().parents[3] / "src" / "neurobrix"


def test_a_container_that_keeps_context_passes():
    declared_context_keeps({"context": {"keeps": True}}, "--turns")


def test_a_container_that_keeps_none_is_refused_by_name():
    with pytest.raises(RuntimeError, match='declares it keeps none'):
        declared_context_keeps({"context": {"keeps": False}}, "--turns")


@pytest.mark.parametrize("defaults", [{}, None, {"context": {}}, {"context": True}])
def test_an_undeclared_context_is_refused_by_name(defaults):
    with pytest.raises(RuntimeError, match="does not declare whether"):
        declared_context_keeps(defaults, "a chat session")


def test_every_kept_conversation_door_reads_the_container():
    run = (SRC / "cli/commands/run.py").read_text().split("def play_turns(", 1)[1].split("\ndef ", 1)[0]
    assert 'declared_context_keeps(pkg.defaults, "--turns")' in run and 'family != "llm"' not in run
    eng = (SRC / "serving/engine.py").read_text()
    assert 'declared_context_keeps(self._pkg.defaults, "a chat session")' in eng
    assert 'declared_context_keeps(self._pkg.defaults, "a request with its history")' in eng
