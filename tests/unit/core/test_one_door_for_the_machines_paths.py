"""Where the models live is ONE answer, configurable, and it refuses when wrong.

Before 2026-09-17 the question "where is this machine's model cache" had four
answers in the tree, reached two ways:

    cli/utils.py                  STORE_DIR / CACHE_DIR, literals
    nbx/cache.py                  DEFAULT_CACHE_DIR, a second literal
    cli/commands/coverage.py      os.environ["NEUROBRIX_CACHE"], with its own default
    tools/stimulus_from_depth.py  the same variable again

and `cli/utils.py`'s own docstring called itself "single source of truth for
paths" while three other places disagreed with it. That is the literal-default
class in its most expensive form: the engine reads one location and a tool
reports on another, and the two never meet to argue.
"""
from __future__ import annotations

import json
import os

import pytest

from neurobrix.core import paths as P


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for var in ("NEUROBRIX_HOME", "NEUROBRIX_CACHE", "NEUROBRIX_STORE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(P, "CONFIG_FILE", tmp_path / "paths.json")
    return tmp_path


def test_every_reader_in_the_engine_asks_the_same_door(clean_env, monkeypatch):
    """Not "they happen to agree today" — they must ask the SAME function."""
    import importlib

    where = clean_env / "cache"
    where.mkdir()
    monkeypatch.setenv("NEUROBRIX_CACHE", str(where))

    from neurobrix.cli.commands.coverage import _cache_root
    from neurobrix.nbx.cache import NBXCache
    # Ask for the module by name and make sure the registry holds it before the
    # reload: inside the whole suite this cell read "module neurobrix.cli.utils
    # not in sys.modules" (another cell had popped it and left the package
    # attribute behind), and passed alone — an order-dependent red, measured on
    # the merged tree 2026-09-20 (alone: 10 passed).
    import sys
    U = importlib.import_module("neurobrix.cli.utils")
    sys.modules.setdefault("neurobrix.cli.utils", U)
    U = importlib.reload(U)

    assert P.cache_dir() == where
    assert NBXCache().cache_dir == where
    assert _cache_root() == where
    assert U.CACHE_DIR == where


def test_the_machines_durable_answer_is_a_file_not_a_symlink(clean_env):
    """A daemon and a cron job read this too, and a symlink cannot be read back
    to say what the configuration IS."""
    mount = clean_env / "mounted-cache"
    mount.mkdir()
    P.CONFIG_FILE.write_text(json.dumps({"cache": str(mount)}))

    assert P.cache_dir() == mount
    assert P.describe()["cache"]["said_by"] == str(P.CONFIG_FILE)
    assert P.describe()["cache"]["path"] == str(mount)


def test_the_environment_wins_over_the_file(clean_env, monkeypatch):
    """One run, one shell: an operator override sits above the machine's answer."""
    mount = clean_env / "from-file"
    mount.mkdir()
    once = clean_env / "from-env"
    once.mkdir()
    P.CONFIG_FILE.write_text(json.dumps({"cache": str(mount)}))
    monkeypatch.setenv("NEUROBRIX_CACHE", str(once))

    assert P.cache_dir() == once
    assert P.describe()["cache"]["said_by"] == "$NEUROBRIX_CACHE"


def test_a_configured_location_that_is_not_there_is_REFUSED(clean_env):
    """The one that matters.

    A machine told to read its models from a mount must STOP when the mount is
    not mounted. Falling back to ~/.neurobrix re-extracts every model onto the
    local disk — 6.3 GB on a volume that is 92% full — and calls it normal.
    """
    P.CONFIG_FILE.write_text(json.dumps({"cache": str(clean_env / "not-mounted")}))

    with pytest.raises(P.PathNotConfigured) as e:
        P.cache_dir()
    said = str(e.value)
    assert "not-mounted" in said
    assert "not mounted" in said or "does not exist" in said
    assert ".neurobrix" in said, "the refusal must say what it refused to fall back to"


def test_the_default_is_created_but_a_configured_one_never_is(clean_env, monkeypatch):
    """The engine owns the default location, so it may make it. It does not own
    a mount point, and creating one would turn an unmounted share into an empty
    directory that looks fine."""
    home = clean_env / "home"
    monkeypatch.setenv("NEUROBRIX_HOME", str(home))
    home.mkdir()
    made = P.cache_dir()
    assert made.exists() and made == home / "cache"

    monkeypatch.delenv("NEUROBRIX_HOME")
    absent = clean_env / "absent-mount"
    P.CONFIG_FILE.write_text(json.dumps({"cache": str(absent)}))
    with pytest.raises(P.PathNotConfigured):
        P.cache_dir()
    assert not absent.exists(), "a configured location must never be created"


def test_a_broken_config_file_refuses_rather_than_defaulting(clean_env):
    P.CONFIG_FILE.write_text("{not json")
    with pytest.raises(P.PathNotConfigured) as e:
        P.cache_dir()
    assert "valid JSON" in str(e.value)


def test_no_module_states_the_location_on_its_own_any_more(clean_env):
    """The structural half: a grep-proof against the four answers coming back.

    Read from the AST so a docstring or a comment explaining the history cannot
    fail it — a lesson from three instruments the same day that grepped text and
    matched their own prose.
    """
    import ast
    import pathlib

    root = pathlib.Path(P.__file__).resolve().parents[1]
    offenders = []
    # `core/optim/sweep.py` is in this list because it was the FIFTH place,
    # found only after the other four were unified — a module constant
    # `DEFAULT_ROOT = Path.home() / ".neurobrix" / "cache"`, resolved at import,
    # which a configured location could never have reached.
    for mod in (root / "cli" / "utils.py", root / "nbx" / "cache.py",
                root / "cli" / "commands" / "coverage.py",
                root / "core" / "optim" / "sweep.py"):
        tree = ast.parse(mod.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if node.value == ".neurobrix":
                offenders.append(f"{mod.name}:{node.lineno}")
    assert not offenders, (
        "these modules name the location themselves again instead of asking "
        f"core.paths: {offenders}")
