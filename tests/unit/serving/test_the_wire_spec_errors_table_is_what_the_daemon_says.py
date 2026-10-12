"""Every row of the errors table in `docs/reference/wire-protocol.md` (section
4) that needs no engine fault is what a real daemon answers: the `error`
prefix the table prints, and whether the connection stays open after it.

The spec is written for a reader with no access to the code (supervisor queue
item 5, 2026-10-11); this test is what keeps the code and the table from
drifting apart. A real ServingDaemon on a real Unix socket with the fake
engine of `test_the_daemon_answers_while_it_generates.py`.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from neurobrix.serving.protocol import MAX_MESSAGE_SIZE, PROTOCOL_RANGE, recv_message, send_message

from test_the_daemon_answers_while_it_generates import _frame, _raw, daemon  # noqa: F401  (fixture)

SPEC = Path(__file__).resolve().parents[3] / "docs" / "reference" / "wire-protocol.md"
_, HIGH = PROTOCOL_RANGE


def _obj(o) -> bytes:
    return _frame(json.dumps(o).encode())


# (row, raw bytes sent, the prefix the table prints, connection stays open)
CASES = [
    ("too large", struct.pack(">I", MAX_MESSAGE_SIZE + 1), "malformed frame: Message too large", False),
    ("empty", _frame(b""), "malformed frame:", True),
    ("not JSON", _frame(b"{nope"), "malformed frame:", True),
    ("not an object", _frame(b"[1, 2]"), "malformed frame:", True),
    ("out of range", _obj({"protocol": HIGH + 1, "method": "status"}),
     f"protocol {HIGH + 1} is not served here", False),
    ("not an integer", _obj({"protocol": "1", "method": "status"}),
     "malformed request: `protocol` must be an integer", False),
    ("method not a string", _obj({"method": 3}), "malformed request:", True),
    ("params not an object", _obj({"method": "status", "params": [1]}), "malformed request:", True),
    ("unknown method", _obj({"method": "nope"}), "Unknown method: nope", True),
    ("cancel without op_id", _obj({"method": "cancel", "params": {}}), "cancel needs params.op_id", True),
    ("op_id not a string", _obj({"method": "generate", "op_id": 5, "params": {"prompt": "x"}}),
     "refused: op_id ...", True),
]


@pytest.mark.parametrize("row, sent, prefix, stays_open", CASES, ids=[c[0] for c in CASES])
def test_the_daemon_answers_what_the_errors_table_prints(daemon, row, sent, prefix, stays_open):
    printed = prefix.replace(str(HIGH + 1), "<n>").replace(": nope", ": <name>")
    assert f"`{printed}`" in SPEC.read_text(encoding="utf-8"), f"the errors table no longer prints {printed!r}"
    _, engine, path = daemon
    s = _raw(path)
    s.sendall(sent)
    reply = recv_message(s)
    assert "error" in reply and "result" not in reply, reply
    assert reply["error"].startswith(prefix.rstrip(". ")), reply["error"]
    if stays_open:
        send_message(s, {"method": "status"})
        assert recv_message(s)["result"]["loaded"] is True, f"{row}: the connection did not survive"
    else:
        assert recv_message(s) is None, f"{row}: the connection stayed open"
    assert engine.log == [], f"{row}: work ran for a refused request"
