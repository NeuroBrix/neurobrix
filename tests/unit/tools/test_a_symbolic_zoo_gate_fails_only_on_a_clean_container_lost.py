"""The symbolic zoo gate fails only when a container that was clean in its baseline is no longer clean.

`tools/symbolic_zoo_gate.py` runs three scanners over every container under a root. These tests give it a fake
scanner (a graph is clean unless it holds the key `"finding"`) and small containers built here, so what is judged
is the gate's own logic: discovery, the verdict against a baseline, a new container, an unreadable container, and
that nothing is written under the root.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import zipfile
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parents[3] / "tools" / "symbolic_zoo_gate.py"
_spec = importlib.util.spec_from_file_location("symbolic_zoo_gate", _TOOL)
gate = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = gate          # the parallel workers find their task function by its module's name
_spec.loader.exec_module(gate)


def fake_scanner(graph_path: Path) -> dict:
    graph = json.loads(Path(graph_path).read_text())
    n = int(graph.get("finding", 0))
    return {"clean": n == 0, "counts": {"findings": n}}


FAKE = {"fake": fake_scanner}


def _nbx(path: Path, graphs: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("manifest.json", "{}")
        for comp, g in graphs.items():
            z.writestr(f"components/{comp}/graph.json", json.dumps(g))


def _snapshot(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in root.rglob("*")}


def _run(tmp_path, root, name, **kw):
    out = tmp_path / "out" / f"{name}.md"
    code = gate.run_gate(root, out, scanners=FAKE, work=tmp_path / "work", **kw)
    return code, json.loads(out.with_suffix(".json").read_text()), out.read_text()


@pytest.mark.parametrize("jobs", [1, 2])
def test_a_container_clean_in_the_baseline_and_no_longer_clean_fails_the_gate(tmp_path, jobs):
    root = tmp_path / "zoo"
    _nbx(root / "llm" / "a" / "model.nbx", {"model": {}, "lm_head": {}})
    _nbx(root / "llm" / "b" / "model.nbx", {"model": {"finding": 2}})
    base = tmp_path / "base.json"
    code, state, _ = _run(tmp_path, root, "first", jobs=jobs, write_baseline=base)
    assert code == 0
    assert state["containers"]["llm/a"]["clean"] is True
    assert state["containers"]["llm/b"]["scanners"]["fake"]["counts"] == {"findings": 2}

    _nbx(root / "llm" / "a" / "model.nbx", {"model": {}, "lm_head": {"finding": 1}})
    code, state, md = _run(tmp_path, root, "second", jobs=jobs, baseline=base)
    assert code == 1
    assert state["comparison"]["regressions"] == {"llm/a": ["fake"]}
    assert state["containers"]["llm/a"]["scanners"]["fake"]["unclean_components"] == ["lm_head"]
    assert "REGRESSION" in md


def test_a_container_unclean_in_the_baseline_and_still_unclean_holds(tmp_path):
    root = tmp_path / "zoo"
    _nbx(root / "x" / "model.nbx", {"c": {"finding": 1}})
    base = tmp_path / "base.json"
    assert _run(tmp_path, root, "first", jobs=1, write_baseline=base)[0] == 0
    _nbx(root / "x" / "model.nbx", {"c": {"finding": 5}})
    code, state, _ = _run(tmp_path, root, "second", jobs=1, baseline=base)
    assert code == 0 and state["comparison"]["regressions"] == {}


def test_a_new_container_is_reported_not_a_failure(tmp_path):
    root = tmp_path / "zoo"
    _nbx(root / "old" / "model.nbx", {"c": {}})
    base = tmp_path / "base.json"
    _run(tmp_path, root, "first", jobs=1, write_baseline=base)
    _nbx(root / "fresh" / "model.nbx", {"c": {"finding": 3}})
    code, state, md = _run(tmp_path, root, "second", jobs=1, baseline=base)
    assert code == 0
    assert state["comparison"]["new"] == ["fresh"]
    assert "| `fresh` |" in md and "| new |" in md


def test_an_unreadable_container_is_named_and_fails_if_it_was_clean(tmp_path):
    root = tmp_path / "zoo"
    _nbx(root / "a" / "model.nbx", {"c": {}})
    _nbx(root / "nograph" / "model.nbx", {})
    base = tmp_path / "base.json"
    code, state, _ = _run(tmp_path, root, "first", jobs=1, write_baseline=base)
    assert code == 0
    assert state["containers"]["nograph"]["scanned"] is False
    assert "no components/<name>/graph.json" in state["containers"]["nograph"]["reason"]

    (root / "a" / "model.nbx").write_bytes(b"not a zip")
    code, state, md = _run(tmp_path, root, "second", jobs=1, baseline=base)
    rec = state["containers"]["a"]
    assert rec["scanned"] is False and rec["reason"].startswith("unreadable: BadZipFile")
    assert code == 1
    assert state["comparison"]["regressions"]["a"][0].startswith("fake (container unreadable: BadZipFile")
    assert "no — unreadable: BadZipFile" in md


def test_a_scanner_that_raises_is_named_and_never_clean(tmp_path):
    root = tmp_path / "zoo"
    _nbx(root / "a" / "model.nbx", {"c": {}})

    def broken(graph_path):
        raise KeyError("symbols")
    gate.run_gate(root, tmp_path / "o.md", jobs=1, scanners={"broken": broken}, work=tmp_path / "w")
    state = json.loads((tmp_path / "o.json").read_text())
    s = state["containers"]["a"]["scanners"]["broken"]
    assert s["clean"] is False and s["errors"] == ["c: KeyError: 'symbols'"]
    assert "## Scanner errors" in (tmp_path / "o.md").read_text()


def test_the_three_real_scanners_read_a_graph_without_symbols_as_clean(tmp_path, monkeypatch):
    monkeypatch.setenv("NEUROBRIX_CACHE_DIR", str(tmp_path))
    root = tmp_path / "zoo"
    _nbx(root / "a" / "model.nbx", {"c": {"ops": {}, "tensors": {}}})
    code = gate.run_gate(root, tmp_path / "o.md", jobs=1, work=tmp_path / "w")
    state = json.loads((tmp_path / "o.json").read_text())
    assert code == 0
    assert sorted(state["containers"]["a"]["scanners"]) == ["broadcast", "chain", "collision"]
    assert state["containers"]["a"]["clean"] is True


def test_an_absent_container_is_reported(tmp_path):
    root = tmp_path / "zoo"
    _nbx(root / "a" / "model.nbx", {"c": {}})
    _nbx(root / "b" / "model.nbx", {"c": {}})
    base = tmp_path / "base.json"
    _run(tmp_path, root, "first", jobs=1, write_baseline=base)
    (root / "b" / "model.nbx").unlink()
    code, state, _ = _run(tmp_path, root, "second", jobs=1, baseline=base)
    assert code == 0 and state["comparison"]["absent"] == ["b"]


def test_an_unpacked_cache_folder_is_a_container(tmp_path):
    root = tmp_path / "zoo"
    comp = root / "m" / "components" / "c"
    comp.mkdir(parents=True)
    (root / "m" / "manifest.json").write_text("{}")
    (comp / "graph.json").write_text(json.dumps({"finding": 1}))
    code, state, _ = _run(tmp_path, root, "first", jobs=1)
    assert code == 0 and state["containers"]["m"]["kind"] == "folder"
    assert state["containers"]["m"]["clean"] is False


def test_the_gate_writes_nothing_under_its_root(tmp_path):
    root = tmp_path / "zoo"
    _nbx(root / "a" / "model.nbx", {"c": {}, "d": {"finding": 1}})
    before = _snapshot(root)
    _run(tmp_path, root, "first", jobs=2, write_baseline=tmp_path / "b.json")
    assert _snapshot(root) == before
    assert not list((tmp_path / "work").iterdir())          # the extracted graphs are removed
    with pytest.raises(gate.Refused, match="never writes under the root"):
        gate.run_gate(root, root / "table.md", scanners=FAKE, jobs=1, work=tmp_path / "w")
    with pytest.raises(gate.Refused, match="never writes under the root"):
        gate.run_gate(root, tmp_path / "t.md", scanners=FAKE, jobs=1, work=root / "w")


def test_a_root_without_a_container_is_refused(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    with pytest.raises(gate.Refused, match="holds no container"):
        gate.run_gate(root, tmp_path / "t.md", scanners=FAKE, jobs=1, work=tmp_path / "w")
    assert gate.main(["--root", str(root), "--out", str(tmp_path / "t.md")]) == 2


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads every folder")
def test_a_folder_the_gate_cannot_enter_is_named(tmp_path):
    root = tmp_path / "zoo"
    _nbx(root / "a" / "model.nbx", {"c": {}})
    locked = root / "locked"
    locked.mkdir()
    locked.chmod(0)
    try:
        _, state, md = _run(tmp_path, root, "first", jobs=1)
    finally:
        locked.chmod(0o755)
    assert state["unreadable_folders"] == ["locked"]
    assert "`locked`" in md
