"""The repository's `.env`, loaded the way the build toolchain loads its own.

Same rules as the toolchain's loader: a line per `KEY=value`, comments and
blank lines skipped, surrounding quotes stripped, and a variable already in
the environment always wins. Nothing here prints, logs or returns a value —
only whether a name is present. A tool that needs a variable calls
`require(name)` and gets an explicit refusal naming the variable and the
file, never a silent 401 or a state that waits without saying why.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ENV_FILE = REPO / ".env"


def load(path: Path = ENV_FILE) -> int:
    """Set the file's variables that the environment does not already define. Returns how many."""
    if not path.exists():
        return 0
    n = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ and value:
                os.environ[key] = value
                n += 1
    return n


class MissingVariable(RuntimeError):
    pass


def require(name: str, path: Path = ENV_FILE) -> None:
    """Refuse, by name, when `name` is neither in the environment nor in the file."""
    load(path)
    if not os.environ.get(name):
        raise MissingVariable(
            f"REFUSED: {name} is not set in the environment and not defined in {path} "
            f"(the file the build toolchain loads); nothing was created, nothing was asked for elsewhere.")
