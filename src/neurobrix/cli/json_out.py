"""One JSON record per read command — the contract a client reads.

`--json` on a read command prints exactly one JSON object on stdout and
nothing else there; every human line of the same command goes to stderr.
The object carries `schema` ("neurobrix.<command>/<version>") and `engine`
(the engine's version) so a client can refuse a record it does not know
(Studio request 7, the compatibility handshake) and never guesses (request 1).

A schema version changes when a key is removed or its meaning changes;
adding a key does not change it. The versions live here, in one table, and
`tests/unit/cli/test_read_commands_speak_json.py` parses every command's
output and refuses one that does not parse or lacks its schema.
"""
from __future__ import annotations

import contextlib
import json
import sys
from typing import Any, Dict

SCHEMAS: Dict[str, int] = {
    "info": 1, "list": 1, "hub": 1, "inspect": 1, "coverage": 1, "doctor": 1,
    "autotune.status": 1, "autotune.check": 1, "explain-plan": 1, "validate": 1,
}


def wants_json(args) -> bool:
    return bool(getattr(args, "json", False))


_RECORD_STREAM = None      # the stdout that held before human_lines_to_stderr redirected it


def emit(command: str, record: Dict[str, Any]) -> None:
    """Print the one record. `command` must be a key of SCHEMAS."""
    from neurobrix import __version__
    version = SCHEMAS[command]
    doc = {"schema": f"neurobrix.{command}/{version}", "engine": __version__}
    doc.update(record)
    out = _RECORD_STREAM or sys.stdout      # the record reaches the caller's stdout even inside human_lines_to_stderr
    out.write(json.dumps(doc, indent=1, default=str) + "\n")
    out.flush()


@contextlib.contextmanager
def human_lines_to_stderr(enabled: bool):
    """Under --json every print of the wrapped block lands on stderr, so the
    record is the only thing on stdout (Studio request 5: diagnostics to
    stderr, records to stdout)."""
    global _RECORD_STREAM
    if not enabled:
        yield
        return
    _RECORD_STREAM = sys.stdout
    try:
        with contextlib.redirect_stdout(sys.stderr):
            yield
    finally:
        _RECORD_STREAM = None
