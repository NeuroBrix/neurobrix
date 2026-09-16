"""The daemon's live stream carries text decoded by the engine's own
tokenizer (Studio request 2): a delta per token, a held multi-byte
character, a rewind when a later token changes what was emitted. The
listener is the same for both engines (they emit at one site), so the
test drives it directly with a fake connection and a fake engine — no
socket, no model.

Injection: with `decoder.push` skipped in the listener, the second test
failed on a missing `text`; with the U+FFFD hold removed, the first test
emitted "\\ufffd" for the split character.
"""
from __future__ import annotations

import json
import struct
from types import SimpleNamespace

from neurobrix.serving.text_stream import TextDeltaDecoder


class _ByteTokenizer:
    """ids → byte pieces; decode joins and replaces incomplete sequences,
    the way a byte-level BPE tokenizer's decode does."""
    PIECES = {1: b"He", 2: b"llo", 3: b"\xc3", 4: b"\xa9", 5: b"", 6: b" world"}

    def decode(self, ids, skip_special_tokens=False):
        return b"".join(self.PIECES[i] for i in ids).decode("utf-8", errors="replace")


def test_a_split_multibyte_character_is_held_until_complete():
    d = TextDeltaDecoder(_ByteTokenizer())
    assert d.push(1) == {"text": "He", "rewind": 0}
    assert d.push(2) == {"text": "llo", "rewind": 0}
    assert d.push(3) == {"text": "", "rewind": 0}      # first byte of "é": held
    assert d.push(4) == {"text": "é", "rewind": 0}     # completed
    assert d.push(5) == {"text": "", "rewind": 0}      # a special token adds nothing
    assert d.push(6) == {"text": " world", "rewind": 0}
    assert d.text == "Helloé world"


def test_a_token_that_rewrites_emitted_text_says_how_much_to_take_back():
    class _Merging:
        def decode(self, ids, skip_special_tokens=False):
            # the pair (7, 8) decodes to a different surface than 7 alone
            return {(7,): "ab", (7, 8): "aXY"}[tuple(ids)]
    d = TextDeltaDecoder(_Merging())
    assert d.push(7) == {"text": "ab", "rewind": 0}
    assert d.push(8) == {"text": "XY", "rewind": 1}
    assert d.text == "aXY"


class _Conn:
    def __init__(self):
        self.sent = []

    def sendall(self, payload: bytes):
        n = struct.unpack(">I", payload[:4])[0]
        self.sent.append(json.loads(payload[4:4 + n]))


def test_the_daemon_s_stream_event_carries_the_text_for_a_text_family(monkeypatch):
    from neurobrix.serving import server
    monkeypatch.setattr(server, "_answers_in_text", lambda engine: True)
    engine = SimpleNamespace(family="llm", tokenizer=_ByteTokenizer())
    conn = _Conn()
    on_token = server.stream_listener(conn, engine)
    on_token(0, 1, 1, False)
    on_token(1, 2, 3, False)
    on_token(2, 3, 4, True)
    events = [m["stream"] for m in conn.sent]
    assert events[0] == {"step": 0, "n": 1, "token": 1, "done": False, "text": "He", "rewind": 0}
    assert events[1]["text"] == "" and events[2]["text"] == "é" and events[2]["done"] is True


def test_a_family_whose_answer_is_not_text_streams_ids_only(monkeypatch):
    from neurobrix.serving import server
    monkeypatch.setattr(server, "_answers_in_text", lambda engine: False)
    engine = SimpleNamespace(family="tts", tokenizer=_ByteTokenizer())
    conn = _Conn()
    server.stream_listener(conn, engine)(0, 1, 1, False)
    assert conn.sent[0]["stream"] == {"step": 0, "n": 1, "token": 1, "done": False}


def test_the_family_s_declaration_decides_not_its_name():
    """`_answers_in_text` reads the family YAML's output format."""
    from neurobrix.serving.server import _answers_in_text
    assert _answers_in_text(SimpleNamespace(family="llm")) is True
    assert _answers_in_text(SimpleNamespace(family="tts")) is False
    assert _answers_in_text(SimpleNamespace(family=None)) is False
