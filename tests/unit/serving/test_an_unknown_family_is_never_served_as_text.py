"""A family whose output format the engine cannot resolve is refused by name on the warm path, as on the cold path
(run.py: `get_output_format(family)` without a net). Before: the daemon (serving/server.py `_call_engine`) and the warm
client (`_try_warm_path`) read it as "txt" (`except RuntimeError: fmt = "txt"`), so an image or audio answer was dropped
and the user got an empty text — a ZERO FALLBACK (reste_apres_064 point 12, the supervisor 2026-11-09 18:45).

What this test would do if the code were wrong: the daemon answering a request of an unknown family (red), the warm
client going on with a daemon that states no family (red), a known family no longer served (red)."""
from __future__ import annotations

import types

import pytest

from neurobrix.serving.server import ServingDaemon


def _daemon(family):
    d = object.__new__(ServingDaemon)
    d._engine = types.SimpleNamespace(family=family, generate=lambda **kw: {"outputs": {"x": 1}, "text": "t"})
    return d


@pytest.mark.parametrize("family", [None, "unknown", "no_such_family"])
def test_the_daemon_refuses_a_family_without_an_output_format(family):
    with pytest.raises(RuntimeError, match="ZERO FALLBACK"):
        _daemon(family)._call_engine("generate", {"prompt": "p", "output_path": "o.png"})


def test_a_text_family_is_still_answered_in_the_json():
    out = _daemon("llm")._call_engine("generate", {"prompt": "p", "output_path": "o.txt"})
    assert out.get("text") == "t"


@pytest.mark.parametrize("family", [None, "unknown"])
def test_the_warm_client_refuses_a_daemon_that_states_no_family(monkeypatch, family):
    from neurobrix.cli.commands import run as R
    from neurobrix.serving import client as C
    closed = []

    class _Client:
        @staticmethod
        def is_running():
            return True

        @staticmethod
        def get_pid():
            return 1

        def connect(self):
            pass

        def status(self):
            return {"model": "org/m", "family": family, "state": "ready"}

        def close(self):
            closed.append(True)

        def generate(self, **kw):
            raise AssertionError("a request was sent to a daemon whose family is unknown")
    monkeypatch.setattr(C, "DaemonClient", _Client)
    args = types.SimpleNamespace(model="org/m", prompt="p", output=None, mode=None, speaker=None, seed=None,
                                 audio=None, steps=None, cfg=None, temperature=None, repetition_penalty=None,
                                 top_k=None, top_p=None, max_tokens=None, chat_mode=None, height=None, width=None)
    with pytest.raises(RuntimeError, match="states no family"):
        R._try_warm_path(args)
    assert closed
