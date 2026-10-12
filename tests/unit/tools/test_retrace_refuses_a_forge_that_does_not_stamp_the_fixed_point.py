"""retrace_zoo refuses to trace, by name, with a Forge whose single write does not stamp the fixed
point the engine requires — asked BEFORE the trace, never found after the install.

2026-10-12: PixArt-Sigma-XL-2-1024-MS and PixArt-XL-2-1024-MS were retraced through a `forge` link
to Forge main (no single write, so no `symbolic_fixed_point` stamp) and installed with `forge local`;
the engine refused both containers in the shared cache in both modes. The door runs Forge's own
`stamp_fixed_point` on a synthetic topology and lets the ENGINE's `unstamped_components` judge it:
no revision list, no flag name copied into the tool.

The TRUNK door (supervisor, 2026-10-12 01:25: one line only) refuses a Forge whose HEAD is not
the repository's `main`, read from git — checked first, before the stamp.
"""
from __future__ import annotations

import json
import shutil
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import retrace_zoo as R  # noqa: E402
import precision_zoo_campaign as C  # noqa: E402

ENGINE_READER = ROOT / "src" / "neurobrix" / "nbx" / "fixed_point.py"

STAMPING = '''
import json
from pathlib import Path
def install_and_apply(*a, **k):
    pass
def stamp_fixed_point(topology, components, revision, apply):
    t = json.loads(Path(topology).read_text())
    for c in components:
        t["extracted_values"].setdefault(c, {})[FLAG] = revision
    Path(topology).write_text(json.dumps(t))
'''


def _forge(tmp: Path, flag: str | None) -> Path:
    """A Forge checkout whose single write stamps `flag` (None: a Forge without the single write)."""
    f = tmp / "forge"
    (f / "tools").mkdir(parents=True)
    (f / "tools" / "__init__.py").write_text("")
    if flag is not None:
        (f / "tools" / "repropagate_in_place.py").write_text(f"FLAG = {flag!r}\n" + STAMPING)
    return f


def _engine(tmp: Path, requires: bool = True) -> Path:
    src = tmp / "engine" / "src"
    (src / "neurobrix" / "nbx").mkdir(parents=True)
    if requires:
        shutil.copy(ENGINE_READER, src / "neurobrix" / "nbx" / "fixed_point.py")
    return src


@pytest.fixture(autouse=True)
def _fresh():
    R._DOOR_SEEN.clear()
    yield
    R._DOOR_SEEN.clear()


def test_a_forge_whose_single_write_stamps_what_the_engine_reads_passes(tmp_path):
    assert R.forge_fixed_point_door(_forge(tmp_path, "symbolic_fixed_point"), _engine(tmp_path),
                                    tmp_path / "scratch", py=sys.executable) is None
    assert list((tmp_path / "scratch").iterdir()) == []          # the probe leaves nothing behind


def test_a_forge_without_the_single_write_is_refused_by_name(tmp_path):
    why = R.forge_fixed_point_door(_forge(tmp_path, None), _engine(tmp_path), tmp_path / "scratch", py=sys.executable)
    assert why and why.startswith("FIXED-POINT DOOR") and "repropagate_in_place" in why and "nothing traced" in why


def test_a_forge_stamping_another_declaration_is_refused_naming_the_components(tmp_path):
    why = R.forge_fixed_point_door(_forge(tmp_path, "fixed_point_v0"), _engine(tmp_path), tmp_path / "scratch",
                                   py=sys.executable)
    assert why and "does not read" in why and "door_a" in why and "door_b" in why


def test_an_engine_that_requires_no_stamp_requires_nothing_of_the_writer(tmp_path):
    assert R.forge_fixed_point_door(_forge(tmp_path, None), _engine(tmp_path, requires=False),
                                    tmp_path / "scratch", py=sys.executable) is None


def _git(f: Path, *a):
    import subprocess
    subprocess.run(["git", "-C", str(f), *a], check=True, capture_output=True,
                   env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                        "GIT_COMMITTER_EMAIL": "t@t", "PATH": "/usr/bin:/bin", "HOME": str(f)})


def _trunk(f: Path, off_main: bool = False) -> Path:
    """Make `f` a git checkout at `main`; `off_main` leaves HEAD on a branch one commit past it."""
    _git(f, "init", "-q", "-b", "main")
    _git(f, "add", "-A")
    _git(f, "commit", "-q", "--allow-empty", "-m", "trunk")
    if off_main:
        _git(f, "checkout", "-q", "-b", "a-side-line")
        _git(f, "commit", "-q", "--allow-empty", "-m", "side")
    return f


def test_a_forge_at_the_trunk_passes_the_trunk_door(tmp_path):
    assert R.forge_trunk_door(_trunk(_forge(tmp_path, "symbolic_fixed_point"))) is None


def test_a_forge_off_the_trunk_is_refused_by_name(tmp_path):
    why = R.forge_trunk_door(_trunk(_forge(tmp_path, "symbolic_fixed_point"), off_main=True))
    assert why and why.startswith("TRUNK DOOR") and "a-side-line" in why and "not the trunk" in why


def test_a_forge_that_is_no_git_checkout_is_refused_by_name(tmp_path):
    why = R.forge_trunk_door(_forge(tmp_path, "symbolic_fixed_point"))
    assert why and why.startswith("TRUNK DOOR") and "nothing traced" in why


def _model(tmp_path, monkeypatch, forge):
    monkeypatch.setattr(R, "FORGE", forge / "forge.py")
    monkeypatch.setattr(R, "PY", sys.executable)
    monkeypatch.setattr(R, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(C, "family_of", lambda n: "image")
    ran = []
    monkeypatch.setattr(R, "run", lambda *a, **k: ran.append(a) or 0)
    args = types.SimpleNamespace(out=str(tmp_path / "out"), backup=str(tmp_path / "backup"),
                                 models_root=str(tmp_path / "builds"), tmp=str(tmp_path / "tmp"), gpu=None,
                                 src=str(_engine(tmp_path)), extra=[], timeout=10, trace_timeout=10)
    m = R.Model("image-test", args)
    monkeypatch.setattr(m, "snapshot", lambda: tmp_path)
    return m, ran


def test_a_stamping_forge_off_the_trunk_is_refused_before_forge_runs(tmp_path, monkeypatch):
    m, ran = _model(tmp_path, monkeypatch, _trunk(_forge(tmp_path, "symbolic_fixed_point"), off_main=True))
    assert m.step_trace() is False and ran == []
    st = m.state["steps"]["trace"]
    assert st["state"] == "REFUSED" and st["error"].startswith("TRUNK DOOR")
    assert m.step_build() is False and m.state["steps"]["build"]["state"] == "REFUSED" and ran == []


def test_a_stamping_forge_at_the_trunk_traces(tmp_path, monkeypatch):
    m, ran = _model(tmp_path, monkeypatch, _trunk(_forge(tmp_path, "symbolic_fixed_point")))
    assert m.step_trace() is True and len(ran) == 1


def test_the_trace_is_refused_before_forge_runs(tmp_path, monkeypatch):
    forge = _trunk(_forge(tmp_path, None))
    monkeypatch.setattr(R, "FORGE", forge / "forge.py")
    monkeypatch.setattr(R, "PY", sys.executable)
    monkeypatch.setattr(R, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(C, "family_of", lambda n: "image")
    ran = []
    monkeypatch.setattr(R, "run", lambda *a, **k: ran.append(a) or 0)
    args = types.SimpleNamespace(out=str(tmp_path / "out"), backup=str(tmp_path / "backup"),
                                 models_root=str(tmp_path / "builds"), tmp=str(tmp_path / "tmp"), gpu=None,
                                 src=str(_engine(tmp_path)), extra=[], timeout=10, trace_timeout=10)
    m = R.Model("image-test", args)
    monkeypatch.setattr(m, "snapshot", lambda: tmp_path)
    assert m.step_trace() is False
    assert ran == []                                             # forge trace never launched
    st = m.state["steps"]["trace"]
    assert st["state"] == "REFUSED" and st["error"].startswith("FIXED-POINT DOOR")
    assert m.step_build() is False and m.state["steps"]["build"]["state"] == "REFUSED" and ran == []
