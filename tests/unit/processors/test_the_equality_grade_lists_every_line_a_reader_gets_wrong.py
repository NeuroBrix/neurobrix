"""The grade tool of the espeak_ng processor, on a tiny corpus written for these tests (no vendor data)."""
import gzip
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

_GRADE = Path(__file__).resolve().parents[3] / "src/neurobrix/processors/espeak_ng/equality/grade.py"
_spec = importlib.util.spec_from_file_location("espeak_equality_grade", _GRADE)
grade = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(grade)

ROWS = [["one two", "wʌn tuː"], ["red", "ɹɛd"], ["", ""]]


def _write(folder, rows_by_layer, variants=("us", "gb")):
    for v in variants:
        (Path(folder) / v).mkdir(parents=True, exist_ok=True)
        for layer in grade.LAYERS:
            rows = rows_by_layer.get(layer, ROWS)
            with gzip.open(Path(folder) / v / (layer + ".jsonl.gz"), "wt", encoding="utf-8") as f:
                f.write("\n".join(json.dumps(r) for r in rows) + "\n")


def test_a_reader_equal_on_every_line_grades_clean(tmp_path):
    _write(tmp_path / "ref", {})
    _write(tmp_path / "cand", {})
    result = grade.grade(str(tmp_path / "ref"), str(tmp_path / "cand"))
    text, ok = grade.report(result)
    assert ok and all(eq == tot == 3 for eq, tot, _ in result.values())
    assert "Overall: 48 / 48" in text


def test_a_reader_unequal_on_a_line_is_listed_with_both_readings(tmp_path):
    _write(tmp_path / "ref", {})
    wrong = [ROWS[0], ["red", "ɹɛt"], ROWS[2]]
    _write(tmp_path / "cand", {"rules": wrong})
    result = grade.grade(str(tmp_path / "ref"), str(tmp_path / "cand"))
    text, ok = grade.report(result)
    assert not ok
    assert result[("us", "rules")][0] == 2 and result[("us", "rules")][1] == 3
    assert "line 2 `red`: vendor `ɹɛd` / reader `ɹɛt`" in text


def test_a_missing_or_shifted_candidate_line_is_unequal(tmp_path):
    _write(tmp_path / "ref", {})
    _write(tmp_path / "cand", {"rules": [ROWS[0]], "numbers": [ROWS[1], ROWS[0], ROWS[2]]})
    result = grade.grade(str(tmp_path / "ref"), str(tmp_path / "cand"))
    assert result[("gb", "rules")][0] == 1
    assert result[("gb", "numbers")][0] == 1  # only the row at its place that reads the same text


def test_a_reference_missing_a_layer_is_a_named_error(tmp_path):
    _write(tmp_path / "ref", {})
    os.remove(tmp_path / "ref" / "us" / "rules.jsonl.gz")
    with pytest.raises(SystemExit, match="the reference has no us/rules"):
        grade.grade(str(tmp_path / "ref"), str(tmp_path / "ref"))


def _archive(tmp_path):
    root = tmp_path / "pack" / "tiny_eq"
    _write(root / "reference", {})
    lines = []
    for p in sorted(root.rglob("*.gz")):
        lines.append("%s  %s" % (hashlib.sha256(p.read_bytes()).hexdigest(), p.relative_to(root)))
    (root / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    out = tmp_path / "tiny_eq.tar.zst"
    tar = subprocess.run(["tar", "-c", "-C", str(tmp_path / "pack"), "tiny_eq"], stdout=subprocess.PIPE, check=True)
    out.write_bytes(subprocess.run(["zstd", "-q", "-c"], input=tar.stdout, stdout=subprocess.PIPE, check=True).stdout)
    pin = {"name": "tiny_eq.tar.zst", "size": out.stat().st_size, "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
    return out, pin


@pytest.mark.skipif(shutil.which("zstd") is None or shutil.which("tar") is None, reason="needs the zstd and tar tools")
def test_an_archive_that_is_absent_or_not_the_pinned_one_is_refused_by_name(tmp_path):
    out, pin = _archive(tmp_path)
    into = tmp_path / "unpacked"
    into.mkdir()
    ref = grade.unpack_reference(str(out), str(into), pin)
    assert grade.grade(ref, ref)[("us", "rules")][0] == 3
    with pytest.raises(SystemExit, match="is absent"):
        grade.check_archive(str(tmp_path / "nope.tar.zst"), pin)
    with pytest.raises(SystemExit, match="the pin says"):
        grade.check_archive(str(out), dict(pin, sha256="0" * 64))
    with pytest.raises(SystemExit, match="has size"):
        grade.check_archive(str(out), dict(pin, size=pin["size"] + 1))


def test_the_pin_of_the_repository_names_its_archive():
    pin = grade.load_pin()
    assert pin["name"].endswith(".tar.zst") and len(pin["sha256"]) == 64 and pin["size"] > 0
