"""THE BRANCH DOOR: which of the engine's two branches this process runs, and that branch's own module of a doubled
brick (owner 23/11/2026, orders row 248; RED_LINES.md lines 16-22).

The two branches share no code: a doubled brick lives at `neurobrix.pytorch.<brick>` and `neurobrix.triton.<brick>`
under the same name. Code common to both (the command line, the serving daemon, the container reader) and the shared
files not yet doubled import NEITHER side at their top: they ask here, and receive the side of the branch this process
runs. A file that belongs to one branch imports its own side directly, never through the door.

Contract:
  * `choose(mode)` — the engine mode this process runs (`compiled`, `sequential`, `triton`, `triton_sequential`),
    said once by the command that chose it (`cli/engine_choice.choose_engine`, the serving engine). A second, different
    choice in the same process is refused by name ("BRANCH ALREADY CHOSEN"): one process runs one branch.
  * `side()` — "triton" or "pytorch": the chosen mode's branch; before any choice, the branch a run that names no
    mode would take (`core/runtime/engines.default_mode`: Triton, the default engine).
How a name resolves (for a reader that follows imports, not strings): `names("prism", "solver", "PrismSolver")` is
`importlib.import_module("neurobrix.<side>.prism.solver").PrismSolver`, <side> being `side()`; a member that is a
submodule is imported as `neurobrix.<side>.prism.<member>`. Only the chosen side is ever imported, never both.
  * `module(brick, sub="")` — `neurobrix.<side>.<brick>[.<sub>]`, imported.
  * `names(brick, sub, *names)` — the named members of that module (a member that is a submodule is imported).
Nothing here holds tensor or memory logic; nothing here imports either side at its top.
"""
from __future__ import annotations

import importlib
from typing import Any, Tuple

#: the modes that run the Triton branch; every other mode runs the PyTorch branch
TRITON_MODES = frozenset({"triton", "triton_sequential"})

_CHOSEN: list = [None]       # the side this process runs, once a command chose it


def side_of(mode: str) -> str:
    """The branch a mode runs: "triton" for the Triton modes, "pytorch" for the others."""
    return "triton" if str(mode) in TRITON_MODES else "pytorch"


def choose(mode: str) -> str:
    """Record the branch this process runs (from its engine mode) and return it; refuse a different second choice."""
    side = side_of(mode)
    if _CHOSEN[0] is not None and _CHOSEN[0] != side:
        raise RuntimeError(f"BRANCH ALREADY CHOSEN: this process runs the {_CHOSEN[0]} branch; mode {mode!r} asks for the {side} "
                                  f"branch: one process runs one branch (start another process for it)")
    _CHOSEN[0] = side
    return side


def side() -> str:
    """The branch this process runs: the chosen one, else the one a run that names no mode would take."""
    if _CHOSEN[0] is not None:
        return _CHOSEN[0]
    from neurobrix.core.runtime.engines import default_mode
    return side_of(default_mode()[0])


def module(brick: str, sub: str = "") -> Any:
    """`neurobrix.<side>.<brick>[.<sub>]` imported, for this process's branch."""
    return importlib.import_module(f"neurobrix.{side()}.{brick}" + (f".{sub}" if sub else ""))


def names(brick: str, sub: str, *members: str) -> Tuple[Any, ...]:
    """The members `members` of `module(brick, sub)`, in order; a member that is a submodule is imported."""
    mod = module(brick, sub)
    out = []
    for name in members:
        if hasattr(mod, name):
            out.append(getattr(mod, name))
        else:
            out.append(importlib.import_module(f"{mod.__name__}.{name}"))
    return tuple(out)
