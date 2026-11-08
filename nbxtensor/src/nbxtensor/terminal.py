"""What a command line built on this library prints: one logging handler, configured here and only here.

The library and the engine log through the standard `logging` module, one logger per module, and install no handler
at import: an application that embeds them keeps control of its own output. A command line (`neurobrix`,
`python -m nbxtensor`) installs ONE stderr handler, `configure`, for the loggers it owns:

* ERROR / WARNING — always shown: a failure (one line, with its remedy) or a fact the user must know;
* INFO — the few status lines a user reads by default (the engine in use, the run time, a first launch);
* DEBUG — every developer line, shown with `-v`; `-vv` adds an error's traceback; `NBX_LOG=debug` is `-vv` (the
  knob for daemons, chains and the gate, which parse those lines), `NBX_LOG=info|warning|error` a level.

A command's RESULT (text, transcript, `SAVED:` path, table, JSON record) is not a log record: it is printed on stdout.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Optional, Sequence, Tuple

#: The environment knob; its values are the level names below.
ENV = "NBX_LOG"
LEVELS = {"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR}

_HANDLER: Optional[logging.Handler] = None
_TRACEBACKS = [False]


class LogLevelRefused(ValueError):
    """NBX_LOG names no level: refused by name, never read as the default."""


def env_level(environ=None) -> Optional[int]:
    """The level `NBX_LOG` asks for, None when it asks for none."""
    environ = os.environ if environ is None else environ
    asked = (environ.get(ENV) or "").strip().lower()
    if not asked:
        return None
    if asked not in LEVELS:
        raise LogLevelRefused(f"{ENV}={environ.get(ENV)!r} is not a level: use one of {', '.join(LEVELS)}")
    return LEVELS[asked]


def verbosity(argv: Sequence[str]) -> int:
    """How many -v a command line holds (`-v`, `-vv`, `--verbose`), read before any parser exists."""
    n = 0
    for a in argv:
        if a == "--":
            break
        n += 1 if a == "--verbose" else len(a) - 1 if len(a) > 1 and a[0] == "-" and set(a[1:]) == {"v"} else 0
    return n


def resolve(verbose: int = 0, environ=None, quiet_default: bool = False) -> Tuple[int, bool]:
    """(level, tracebacks): `-v` DEBUG, `-vv` with tracebacks; else NBX_LOG (`debug` with tracebacks); else INFO —
    WARNING when the command's stdout is a machine record (`--json`), so stderr stays empty unless something is wrong."""
    if verbose:
        return logging.DEBUG, verbose >= 2
    asked = env_level(environ)
    if asked is not None:
        return asked, asked <= logging.DEBUG
    return (logging.WARNING if quiet_default else logging.INFO), False


def configure(level: int, tracebacks: bool = False, roots: Sequence[str] = ("nbxtensor",), stream=None):
    """The one stderr handler (idempotent, bound to the process's stderr now) and `roots` at `level`. Records keep their
    own text (`%(message)s`); other libraries keep Python's default: their warnings and errors."""
    global _HANDLER
    if _HANDLER is None:
        _HANDLER = logging.StreamHandler(stream if stream is not None else sys.stderr)
        _HANDLER.setFormatter(logging.Formatter("%(message)s"))
        logging.getLogger().addHandler(_HANDLER)
    else:
        _HANDLER.setStream(stream if stream is not None else sys.stderr)
    logging.getLogger().setLevel(logging.WARNING)
    set_level(level, roots, tracebacks)
    return _HANDLER


def set_level(level: int, roots: Sequence[str], tracebacks: Optional[bool] = None) -> None:
    """`roots` at `level` (a daemon's child: DEBUG, into its log file)."""
    for name in roots:
        logging.getLogger(name).setLevel(level)
    if tracebacks is not None:
        _TRACEBACKS[0] = bool(tracebacks)


def tracebacks() -> bool:
    """True when an error's traceback is printed (`-vv`, `NBX_LOG=debug`)."""
    return _TRACEBACKS[0]
