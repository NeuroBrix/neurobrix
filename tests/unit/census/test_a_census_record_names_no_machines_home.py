"""A census input record names the files it read relative to the engine's home (`home:`), the catalogue (`cache:`) or
the repository (`repo:`), never by a machine's home directory: the committed records held 266 paths under
/home/mlops/.neurobrix and 49 under /Users/hocine/.neurobrix, shipped in the wheel (the owner, 2026-11-07). The reader
resolves `home:` against this machine's engine home, so a record moves only where a file did.

What this test would do if the code were wrong: a file under ~/.neurobrix recorded absolute (red), `home:` not resolved
back (red), a committed record naming a user's home (red)."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "src"))

import census_inputs as CI                     # noqa: E402


def test_a_file_of_the_engine_home_is_recorded_relative_and_read_back(monkeypatch, tmp_path):
    home = tmp_path / "engine_home"
    (home / "calibration" / "M").mkdir(parents=True)
    f = home / "calibration" / "M" / "transformer.json"
    f.write_text("{}")
    monkeypatch.setenv("NEUROBRIX_HOME", str(home))
    from neurobrix.core import paths
    monkeypatch.setattr(paths, "neurobrix_home", lambda: home)
    monkeypatch.setattr(paths, "cache_dir", lambda: home / "cache")
    assert CI._rel(f) == "home:calibration/M/transformer.json"
    assert CI._abs("home:calibration/M/transformer.json") == f


def test_no_committed_record_names_a_users_home():
    for p in (REPO / "src" / "neurobrix" / "config" / "census").rglob("*.inputs.jsonl"):
        text = p.read_text(encoding="utf-8")
        assert '"/home/' not in text and '"/Users/' not in text, p
