"""A hardware class whose census lacks a table another class carries is RED before a release.

Seen 2026-11-13: the Apple class's census had no prefill-chunk record (`18g.chunks.jsonl`) while both NVIDIA classes
had one, and a served conversation on a clean Apple install swept at run time; nothing before the release noticed.

What each test would do if the tool were wrong: a tool that compared nothing (or only kernel tables) passes the class
without its chunk record and the first test fails; one that called an empty root green fails the third; one that kept
its own list of companions instead of reading `census_table`'s would miss a renamed companion (the fourth reads the
names from the module).
"""
from __future__ import annotations

import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[3] / "tools"
sys.path.insert(0, str(TOOLS))
import census_tables_complete as C  # noqa: E402


def _class(root: Path, vendor: str, profile: str, cls: int, *companions: str) -> Path:
    d = root / vendor / profile
    d.mkdir(parents=True, exist_ok=True)
    table = d / f"{cls}g.jsonl"
    table.write_text("")
    for kind in companions:
        C.COMPANIONS[kind](table).write_text("")
    return table


def test_a_class_without_the_chunk_record_the_others_carry_is_red(tmp_path):
    _class(tmp_path, "nvidia", "volta", 16, "inputs", "chunks")
    _class(tmp_path, "nvidia", "volta", 32, "inputs", "chunks")
    _class(tmp_path, "apple", "apple_m4_pro", 18, "inputs")
    out = tmp_path / "verdict.txt"
    assert C.main(["--root", str(tmp_path), "--out", str(out)]) == 1
    text = out.read_text()
    assert "apple/apple_m4_pro/18g.jsonl: lacks 18g.chunks.jsonl" in text
    assert not any(ln.startswith("RED nvidia") for ln in text.splitlines())


def test_every_class_with_every_companion_is_green(tmp_path):
    for vendor, profile, cls in (("nvidia", "volta", 16), ("apple", "apple_m4_pro", 18)):
        _class(tmp_path, vendor, profile, cls, "inputs", "chunks")
    rc, lines = C.verdict(tmp_path)
    assert rc == 0 and lines[0].startswith("GREEN 2 classes")


def test_an_empty_root_is_refused_by_name_never_green(tmp_path):
    out = tmp_path / "verdict.txt"
    assert C.main(["--root", str(tmp_path / "nothing"), "--out", str(out)]) == 2
    assert out.read_text().startswith("REFUSED")


def test_the_companions_are_the_census_tables_own_names(tmp_path):
    from neurobrix.kernels import census_table
    table = tmp_path / "v" / "p" / "16g.jsonl"
    assert {k: f(table).name for k, f in C.COMPANIONS.items()} == {
        "inputs": census_table.inputs_path(table).name, "chunks": census_table.chunks_path(table).name}


def test_a_companion_without_its_class_table_is_red(tmp_path):
    _class(tmp_path, "nvidia", "volta", 16, "chunks")
    (tmp_path / "nvidia" / "volta" / "32g.chunks.jsonl").write_text("")
    rc, lines = C.verdict(tmp_path)
    assert rc == 1 and any("32g.chunks.jsonl: a companion without its class table" in ln for ln in lines)
