"""A test that needs the workshop configuration asks for it by name.

Some tools read the workshop configuration (`tools/atelier.py`: `~/.neurobrix/atelier.json`, or the file named by
`NBX_ATELIER_JSON`: each role's path) as they are imported. A test module that imports such a tool cannot run on a
machine without that file (a contributor's clone, a clean host): it is skipped there, saying which file is missing and
how to provide it, instead of stopping the whole folder's collection with an error.

    from tests.unit.workshop_configuration import the_workshop_configuration
    with the_workshop_configuration():
        import regression_matrix as R  # noqa: E402

Only an ABSENT configuration is a skip. A machine whose configuration exists but lacks a role the tool asks for still
fails with the tool's own named error: that is a broken workshop, not a missing one."""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import pytest

_TOOLS = str(Path(__file__).resolve().parents[2] / "tools")
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

import atelier  # noqa: E402


def missing() -> str:
    """Why this machine has no workshop configuration and how to give it one, or '' when it has one."""
    p = atelier.config_path()
    if p.exists():
        return ""
    return (f"needs the workshop configuration, absent on this machine ({p}): write atelier.json (each role's path, "
            f"tools/atelier.py) there, or name one with {atelier.CONFIG_ENV}")


@contextlib.contextmanager
def the_workshop_configuration():
    """Run the block (a tool's import, a call into a tool); when it stops because this machine has no workshop
    configuration, skip the test, or the whole module at import, by that name."""
    try:
        yield
    except Exception as exc:                  # noqa: BLE001 — re-raised unless it is the absent configuration
        if not _is_undeclared(exc):
            raise
        why = missing()
        if not why:
            raise
        _forget_the_half_imported(exc)
        pytest.skip(why, allow_module_level=True)


def _is_undeclared(exc: BaseException) -> bool:
    """`exc` is the workshop's `AtelierUndeclared`, whichever import of `tools/atelier.py` raised it: a test that
    re-imports that module (to read another configuration) leaves the tools imported before it raising the class of
    the first import."""
    return any(c.__name__ == "AtelierUndeclared" and c.__module__ == "atelier" for c in type(exc).__mro__)


def _forget_the_half_imported(exc: BaseException) -> None:
    """Drop from `sys.modules` every module whose import stopped on this error. A loader that registers a module
    before running it (`sys.modules[name] = module; spec.loader.exec_module(module)`) leaves it there half-run, and a
    later `from tool import name` in another test would then fail with "cannot import name" instead of asking for
    the configuration."""
    tb = exc.__traceback__
    while tb is not None:
        frame = tb.tb_frame
        if frame.f_code.co_name == "<module>":
            name = frame.f_globals.get("__name__")
            module = sys.modules.get(name) if name else None
            if module is not None and getattr(module, "__dict__", None) is frame.f_globals:
                del sys.modules[name]
        tb = tb.tb_next


def pinned_roles(monkeypatch, root: Path):
    """A function that pins workshop roles to directories under `root` through their `NBX_ATELIER_<ROLE>` override,
    for a test that reaches a tool only for something the role is a DEFAULT for (an output directory, a catalogue a
    flag names anyway). The tools imported while the roles are pinned are forgotten by `forget_the_tools_imported`,
    so no later test finds a tool already imported against the pinned directories."""
    def pin(*roles: str) -> None:
        for role in roles:
            monkeypatch.setenv(f"NBX_ATELIER_{role.upper()}", str(root / "workshop" / role))
    return pin


def forget_the_tools_imported(since) -> None:
    """Drop from `sys.modules` every tool (a module of the repository's `tools/`) imported after `since` (a set of
    module names)."""
    for name in set(sys.modules) - set(since):
        f = getattr(sys.modules.get(name), "__file__", None) or ""
        if f and Path(f).resolve().parent == Path(_TOOLS).resolve():
            del sys.modules[name]


def off_the_workshop(needs: str) -> None:
    """Skip, naming `needs` (a model container of the catalogue, a machine's generated profile), when this machine has
    no workshop configuration: a contributor's clone holds no catalogue to check. On a workshop machine it returns,
    and the test's own refusal of the missing file stands."""
    if missing():
        pytest.skip(f"needs {needs}: absent here, and this machine has no workshop configuration "
                    f"({atelier.config_path()}), so it holds no catalogue to check: install the file, or run on a "
                    f"workshop machine")
