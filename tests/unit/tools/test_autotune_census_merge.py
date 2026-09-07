"""Replay caches under the campaign roots merge into one census, the machine's first."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import autotune_census_merge as M  # noqa: E402


def test_the_gates_own_sweeps_join_the_census(tmp_path, monkeypatch):
    machine = tmp_path / "machine"; machine.mkdir()
    (machine / M.CACHE_FILE).write_text(json.dumps({"k::a": {"cfg": 1}, "k::b": {"cfg": 2}}))
    monkeypatch.setattr(M, "MACHINE_CACHE", machine)
    gate = tmp_path / "campaign" / "Sana" / "autotune_replay"; gate.mkdir(parents=True)
    (gate / M.CACHE_FILE).write_text(json.dumps({"k::b": {"cfg": 9}, "k::4k": {"cfg": 3}}))
    paths = M.find_caches([tmp_path / "campaign"])
    assert paths[0] == machine / M.CACHE_FILE and len(paths) == 2
    census, added = M.merge(paths)
    assert set(census) == {"k::a", "k::b", "k::4k"} and census["k::b"] == {"cfg": 2}          # first seen keeps
    assert added[str(gate / M.CACHE_FILE)] == 1
    out = tmp_path / "census.json"
    assert M.main(["--out", str(out), "--roots", str(tmp_path / "campaign")]) == 0
    assert set(json.loads(out.read_text())) == {"k::a", "k::b", "k::4k"}
