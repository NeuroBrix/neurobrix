"""tools/build_processor.py builds ONE artifact per processor: a deterministic zip of two trees, `code/` (ours, its
licence file) and `data/` (the pinned files, unchanged, their licence file), the equality report, and processor.json
stating each tree's licence and the sha256 of every member. Two builds of the same sources give the same sha256,
whatever the files' times, the folder or where the pinned data was read from; the engine's door runs what it built.

Red without the change: there is no tools/build_processor.py; with a build that writes each member at its own time,
two builds give two sha256."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import zipfile
from pathlib import Path

import pytest

from tests.unit.processors import toy

TOOL = Path(__file__).resolve().parents[3] / "tools" / "build_processor.py"


def tool():
    spec = importlib.util.spec_from_file_location("build_processor", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def data_host():
    h = toy.Host(toy.source_data())
    yield h
    h.close()


def test_two_builds_of_the_same_sources_give_one_sha256(tmp_path, data_host):
    bp = tool()
    sub = toy.write_source(tmp_path / "src", data_host.url)
    first = bp.build(toy.NAME, tmp_path / "one", tmp_path / "src")
    for i, path in enumerate(sorted(sub.rglob("*"))):            # other times, another folder, data read from disk
        os.utime(path, (1_000_000 + i * 7919, 1_000_000 + i * 7919))
    disk = tmp_path / "pinned"
    disk.mkdir()
    for name, body in toy.source_data().items():
        (disk / name).write_bytes(body)
    second = bp.build(toy.NAME, tmp_path / "two", tmp_path / "src", data_from=disk)
    assert first.name == second.name == toy.FILE
    assert hashlib.sha256(first.read_bytes()).hexdigest() == hashlib.sha256(second.read_bytes()).hexdigest()
    assert data_host.gets == {"letters.tsv": 1, "COPYING": 1}


def test_the_artifact_holds_two_trees_each_with_its_licence(tmp_path, data_host):
    bp = tool()
    toy.write_source(tmp_path / "src", data_host.url)
    archive = zipfile.ZipFile(bp.build(toy.NAME, tmp_path / "out", tmp_path / "src"))
    names = archive.namelist()
    assert names == sorted(names) == ["EQUALITY.md", "code/LICENSE", "code/__init__.py", "code/reader.py",
                                      "data/COPYING", "data/letters.tsv", "processor.json"]
    assert {i.date_time for i in archive.infolist()} == {bp.MEMBER_TIME}
    assert {i.compress_type for i in archive.infolist()} == {zipfile.ZIP_STORED}
    own = json.loads(archive.read("processor.json"))
    assert (own["code"]["licence"], own["data"]["licence"], own["data"]["licence_file"]) == \
        ("Apache-2.0", "LicenseRef-toy", "COPYING")
    assert own["members"] == {n: hashlib.sha256(archive.read(n)).hexdigest() for n in names if n != "processor.json"}
    assert archive.read("data/letters.tsv") == toy.source_data()["letters.tsv"]


def test_the_engine_runs_the_artifact_it_built(tmp_path, data_host, places):
    from tests.unit.processors.test_a_declared_processor_runs_from_its_verified_bytes import _Get, _open
    bp = tool()
    toy.write_source(tmp_path / "src", data_host.url)
    data = bp.build(toy.NAME, tmp_path / "out", tmp_path / "src").read_bytes()
    held = _open(places, [toy.declaration(data)], _Get(data))
    assert held.units("Ab", "en-us") == ["A", "B"]


def test_a_pinned_file_that_differs_from_its_pin_is_refused_and_nothing_is_written(tmp_path, data_host):
    bp = tool()
    toy.write_source(tmp_path / "src", data_host.url)
    data_host.files["letters.tsv"] = b"a\tZ\n"
    with pytest.raises(bp.Refused, match=r"the pinned file letters.tsv from .* is 4 bytes, sha256 [0-9a-f]{64}; "
                                         r"the pin says \d+ bytes"):
        bp.build(toy.NAME, tmp_path / "out", tmp_path / "src")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("drop, said", [
    (("code", "licence"), "has no str 'licence'"),
    (("data", "licence_file"), "has no str 'licence_file'"),
    (("entry",), "has no str 'entry'"),
])
def test_a_source_without_a_key_it_needs_is_refused_by_that_key(tmp_path, drop, said):
    bp = tool()
    sub = toy.write_source(tmp_path / "src", "http://127.0.0.1:9")
    rec = json.loads((sub / "processor.json").read_text())
    target = rec
    for key in drop[:-1]:
        target = target[key]
    del target[drop[-1]]
    (sub / "processor.json").write_text(json.dumps(rec))
    with pytest.raises(bp.Refused, match=said):
        bp.build(toy.NAME, tmp_path / "out", tmp_path / "src")
