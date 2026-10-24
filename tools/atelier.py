"""The workshop's roots, for every tool that reads or writes outside the engine (`docs/internal/_atelier.md` v2, §4).

Each machine declares its paths ONCE in `~/.neurobrix/atelier.json` (the column of `_atelier.md` §1 for that machine;
`NBX_ATELIER_JSON` names another file). A role is overridden by `NBX_ATELIER_<ROLE>` (upper case). A tool takes every
path it reads or writes from here — no `/tmp/…`, `validation_outputs/`, `~/…` or absolute path in its code — and a
write outside the roles the file lists as `_writable` stops with an error naming the path (`OutsideTheAtelier`).

    from atelier import root, tmp, campaign, registre, writable, python
    out = writable(campaign("2026_10_22_prism_a") / "plans.json")
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

CONFIG_ENV = "NBX_ATELIER_JSON"
_REGISTRES = ("journaux", "runs", "benchmarks", "verdicts")


class OutsideTheAtelier(RuntimeError):
    """A tool asked to write where the workshop has no root."""


class AtelierUndeclared(RuntimeError):
    """The machine's `atelier.json` is missing, unreadable, or lacks the role a tool asks for."""


def config_path() -> Path:
    return Path(os.environ.get(CONFIG_ENV) or Path.home() / ".neurobrix" / "atelier.json")


def _load() -> Dict[str, Any]:
    p = config_path()
    try:
        return json.loads(p.read_text())
    except FileNotFoundError:
        raise AtelierUndeclared(f"{p}: no atelier.json on this machine — write it from the column of "
                                f"docs/internal/_atelier.md §1 for this machine; no tool guesses a path") from None
    except json.JSONDecodeError as e:
        raise AtelierUndeclared(f"{p}: not JSON ({e})") from None


def root(role: str) -> Path:
    """The path of `role` on this machine (`NBX_ATELIER_<ROLE>` first, then atelier.json)."""
    env = os.environ.get(f"NBX_ATELIER_{role.upper()}")
    if env:
        return Path(env)
    d = _load()
    v = d.get(role)
    if not isinstance(v, str) or not v:
        raise AtelierUndeclared(f"{config_path()}: no role {role!r} (declared: "
                                f"{sorted(k for k, x in d.items() if isinstance(x, str) and not k.startswith('_'))})")
    return Path(v)


def python(name: str) -> str:
    """The interpreter of the environment `name` (`engine`, `oracle`, `competitors`, `forge`: environments.md)."""
    env = os.environ.get(f"NBX_ATELIER_PYTHON_{name.upper()}")
    if env:
        return env
    py = (_load().get("python") or {}).get(name)
    if not py:
        raise AtelierUndeclared(f"{config_path()}: no python {name!r} — the environments are those of environments.md")
    return py


def writable(path) -> Path:
    """`path`, resolved, when it lies under one of the roles atelier.json lists as `_writable`; else refused by name."""
    p = Path(path).expanduser().resolve()
    roles = _load().get("_writable") or []
    roots = []
    for r in roles:
        try:
            roots.append(root(r).resolve())
        except AtelierUndeclared:
            continue
    if any(p == r or p.is_relative_to(r) for r in roots):
        return p
    raise OutsideTheAtelier(f"{p}: outside the workshop's writable roots ({', '.join(str(r) for r in roots)}) — "
                            f"docs/internal/_atelier.md §4: a tool writes in its campaign, the registers, tmp or the repo")


def tmp(*parts: str) -> Path:
    return writable(root("tmp").joinpath(*parts))


def campaign(name: str, *parts: str) -> Path:
    return writable(root("campaigns").joinpath(name, *parts))


def registre(kind: str, *parts: str) -> Path:
    if kind not in _REGISTRES:
        raise OutsideTheAtelier(f"registre {kind!r}: the registers are {', '.join(_REGISTRES)}")
    return writable(root("registres").joinpath(kind, *parts))


def optional_root(role: str):
    """`root(role)` when this machine declares it, else None — for a tool that reads a cache only some machines have
    (the Metal compiler's on the Mac)."""
    try:
        return root(role)
    except AtelierUndeclared:
        return None


class Interpreter(os.PathLike):
    """The interpreter of environment `name`, resolved when it is USED (as a path, a string or an argv element), not
    when a tool is imported: a machine without that environment (the Mac has no Forge python, by design) imports and
    collects every tool, and only a run that needs the interpreter is refused, by name."""

    def __init__(self, name: str):
        self.name = name

    def __fspath__(self) -> str:
        return python(self.name)

    def __str__(self) -> str:
        return python(self.name)

    def __repr__(self) -> str:
        return f"Interpreter({self.name!r})"

    def __add__(self, other):
        return str(self) + other

    def __radd__(self, other):
        return other + str(self)

    def __eq__(self, other):
        # compared as the path it resolves to: a tool's `PY == "/…/bin/python"` reads the interpreter, not the object
        if isinstance(other, (str, os.PathLike)):
            return str(self) == os.fspath(other)
        return NotImplemented

    def __hash__(self):
        return hash(str(self))


def interpreter(name: str) -> Interpreter:
    return Interpreter(name)


if __name__ == "__main__":
    # For the shell chains (the Mac's tools/mac/*.sh): `atelier.py root <role>`, `python <name>`, `writable <path>`,
    # `campaign <name> [part...]`, `registre <kind> [part...]` print the path or stop with the named refusal (rc 2).
    import sys
    cmd, *args = sys.argv[1:] or ["?"]
    fns = {"root": root, "python": python, "writable": writable, "campaign": campaign, "registre": registre,
           "tmp": tmp, "optional_root": optional_root}
    if cmd not in fns:
        sys.exit(f"usage: atelier.py {{{','.join(fns)}}} <args>")
    try:
        print(fns[cmd](*args) or "")
    except (OutsideTheAtelier, AtelierUndeclared) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        sys.exit(2)
