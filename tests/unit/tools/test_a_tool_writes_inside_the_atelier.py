"""A tool takes its paths from the machine's atelier.json and refuses to write outside its writable roles
(`tools/atelier.py`, `docs/internal/_atelier.md` v2 §4).
Injections (seen red, 2026-10-22): `writable` accepting any path -> red; a missing role answered by a default -> red."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import atelier as A  # noqa: E402


@pytest.fixture
def atelier(tmp_path, monkeypatch):
    shop = tmp_path / "nbx"
    cfg = {"atelier": str(shop), "tmp": str(shop / "tmp"), "campaigns": str(shop / "campaigns"),
           "registres": str(shop / "registres"), "repo": str(tmp_path / "repo"),
           "python": {"engine": "/venvs/engine/bin/python"},
           "_writable": ["repo", "tmp", "campaigns", "registres"]}
    p = tmp_path / "atelier.json"
    p.write_text(json.dumps(cfg))
    monkeypatch.setenv(A.CONFIG_ENV, str(p))
    return shop


def test_the_roots_come_from_the_machines_file(atelier):
    assert A.root("tmp") == atelier / "tmp"
    assert A.tmp("x", "y.json") == (atelier / "tmp" / "x" / "y.json").resolve()
    assert A.campaign("2026_10_22_a", "RUN.md") == (atelier / "campaigns" / "2026_10_22_a" / "RUN.md").resolve()
    assert A.registre("runs", "r.jsonl") == (atelier / "registres" / "runs" / "r.jsonl").resolve()
    assert A.python("engine") == "/venvs/engine/bin/python"


def test_a_write_outside_the_roots_is_refused_by_name(atelier):
    for bad in ("/tmp/claude-x/out.png", Path.home() / "backups" / "x", "/home/elsewhere/validation_outputs/v.md"):
        with pytest.raises(A.OutsideTheAtelier, match=str(Path(bad).name)):
            A.writable(bad)
    with pytest.raises(A.OutsideTheAtelier):
        A.registre("scratch", "x")


def test_a_missing_role_or_file_is_refused_not_guessed(atelier, monkeypatch, tmp_path):
    with pytest.raises(A.AtelierUndeclared, match="clones"):
        A.root("clones")
    monkeypatch.setenv(A.CONFIG_ENV, str(tmp_path / "absent.json"))
    with pytest.raises(A.AtelierUndeclared, match="absent.json"):
        A.root("tmp")


def test_an_environment_variable_overrides_a_role(atelier, monkeypatch, tmp_path):
    monkeypatch.setenv("NBX_ATELIER_TMP", str(tmp_path / "other"))
    assert A.root("tmp") == tmp_path / "other"


def test_an_interpreter_is_resolved_at_use_not_at_import(atelier):
    """A tool names its interpreter at import (`interpreter("forge")`); a machine that declares no such environment
    imports it and is refused only when a run uses it (the Mac collected 19 tool tests as errors, 2026-10-23).
    Injection (seen red): `interpreter` resolving at construction -> red."""
    import os
    py = A.interpreter("forge")                       # this machine declares no forge python
    with pytest.raises(A.AtelierUndeclared, match="forge"):
        os.fspath(py)
    assert os.fspath(A.interpreter("engine")) == "/venvs/engine/bin/python"
    assert str(A.interpreter("engine")) + " -V" == "/venvs/engine/bin/python -V"
