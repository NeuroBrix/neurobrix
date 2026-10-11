"""retrace_zoo refuses to trace, by name, with a Forge whose single write does not stamp the fixed
point the engine requires — asked BEFORE the trace, never found after the install.

2026-10-12: PixArt-Sigma-XL-2-1024-MS and PixArt-XL-2-1024-MS were retraced through a `forge` link
to Forge main (no single write, so no `symbolic_fixed_point` stamp) and installed with `forge local`;
the engine refused both containers in the shared cache in both modes. The door runs Forge's own
`stamp_fixed_point` on a synthetic topology and lets the ENGINE's `unstamped_components` judge it:
no revision list, no flag name copied into the tool.
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


def test_the_trace_is_refused_before_forge_runs(tmp_path, monkeypatch):
    forge = _forge(tmp_path, None)
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
